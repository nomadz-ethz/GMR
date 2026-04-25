# Foot-Ground Contact Flags in the Output Pickle

Every pkl produced by `scripts/retarget_no_penetration.py` contains per-frame
left/right foot-ground contact flags computed from the SMPL-X toe kinematics.

The full pkl schema is documented in `docs/pipeline.md` §5.7. This doc focuses
on the contact-flag subset.

## Pickle schema (contact-related keys)

```text
{
  ...
  "foot_ground_contact_flags": bool (T, 2) or None,   # **HERE**
  "foot_contact_meta":         dict      or None,     # how the flags were produced
  ...
}
```

### `foot_ground_contact_flags`

- **Shape**: `(T, 2)`. `T` matches every other time-series array in the pickle.
- **Dtype**: `bool`. `True` ≡ foot is in ground contact on that frame.
- **Column 0**: left foot. **Column 1**: right foot.
- **Fallbacks** (key is still present, value becomes `None`):
  - `--no_foot_contact` was passed on the CLI.
  - The SMPL-X frames don't contain `left_foot` / `right_foot` keys.
  - The clip has fewer than 2 frames (velocity can't be computed).

### `foot_contact_meta`

Self-describes the flags so a downstream consumer doesn't need to re-read CLI
args to know how they were produced:

```python
{
  "source":     "smplx_toe_kinematics",
  "joints":     ["left_foot", "right_foot"],   # SMPL-X joint names used
  "columns":    ["left", "right"],             # axis-1 interpretation
  "z_thresh":   float,   # height threshold above per-motion floor (m)
  "vel_thresh": float,   # toe-speed threshold (m/s)
  "floor_z":    float,   # per-motion floor reference in source frame (m)
}
```

`None` when `foot_ground_contact_flags` is `None`.

## How it is calculated

From `scripts/retarget_no_penetration.py`:

```python
# 1. Pull the SMPL-X toe joint position over time, before any retargeting,
#    so the labels are stable across ground_mode / strict_zero_pen / robot.
l_toe = np.array([f["left_foot"][0]  for f in smplx_data_frames])   # (T, 3)
r_toe = np.array([f["right_foot"][0] for f in smplx_data_frames])   # (T, 3)

# 2. 3D toe speed via central differences.
dt    = 1.0 / aligned_fps
l_vel = np.linalg.norm(np.gradient(l_toe, dt, axis=0), axis=1)      # (T,)
r_vel = np.linalg.norm(np.gradient(r_toe, dt, axis=0), axis=1)

# 3. Per-motion floor reference. AMASS clips and GVHMR post-rotated outputs
#    do not guarantee ground ≈ z=0, so use the lowest observed toe z in the clip.
floor_z = min(l_toe[:, 2].min(), r_toe[:, 2].min())

# 4. A foot is "in contact" iff it is BOTH close to the floor AND moving slowly.
l_stance = (l_toe[:, 2] - floor_z < z_thresh) & (l_vel < vel_thresh)
r_stance = (r_toe[:, 2] - floor_z < z_thresh) & (r_vel < vel_thresh)

foot_ground_contact_flags = np.stack([l_stance, r_stance], axis=1).astype(bool)
```

**Key properties**:

1. **SMPL-X `foot` joint = toe** (not heel). Heel-strike frames have the toe
   still ~3–5 cm above the floor, so `z_thresh` below ~0.04 m will drop heel
   strikes.
2. **Per-motion floor**. Floor_z is the minimum observed toe z across the whole
   clip. On a clip where the subject is airborne at start and end, this bias
   is toward the lowest point of the flight path — usually still correct.
3. **Frame-wise AND of two thresholds**. Height alone is fooled by a foot
   swinging forward at low z; speed alone is fooled by a briefly-stationary
   foot in flight. The AND kills both failure modes.
4. **Source-frame computation**. Flags are derived from the SMPL-X input
   before retargeting, so the same clip run through different `--ground_mode`
   or different robots produces identical flags. Downstream consumers can
   treat the flags as a property of the **human motion**, not the robot.

## Tunable parameters

Expose on the CLI of `scripts/retarget_no_penetration.py`:

| Flag                          | Default | Meaning                                                          |
|-------------------------------|---------|------------------------------------------------------------------|
| `--foot_contact_z_thresh`     | 0.08 m  | Max toe height above per-motion floor for "in contact".          |
| `--foot_contact_vel_thresh`   | 0.50 m/s| Max toe speed for "in contact".                                  |
| `--no_foot_contact`           | off     | Skip detection; `foot_ground_contact_flags` is saved as `None`.  |

Quick reference: **kicking**-heavy motions typically need `--foot_contact_vel_thresh 0.8–1.0`
because the support foot pivots fast during a powerful kick.

A runtime log line confirms each call:

```
[Foot contact] L=53/85, R=41/85 (floor_z=-0.014 m, z<0.080 m, v<0.500 m/s)
```

## How to access the flags

### Python

```python
import pickle
import numpy as np

with open("output/foot_contact_tests/walk_02_01.pkl", "rb") as f:
    data = pickle.load(f)

flags = data["foot_ground_contact_flags"]   # bool (T, 2) or None
meta  = data["foot_contact_meta"]           # dict or None

if flags is None:
    raise RuntimeError("This pkl was produced without foot-contact detection.")

print(flags.shape)                          # (T, 2)
l_contact = flags[:, 0]                     # (T,) left foot
r_contact = flags[:, 1]                     # (T,) right foot

# Duty factors.
print(f"L duty: {l_contact.mean():.2f}  R duty: {r_contact.mean():.2f}")

# Step onsets (rising edges).
print(f"L onsets: {int((np.diff(l_contact.astype(int)) == 1).sum())}")

# Airborne (neither foot in contact).
airborne = ~l_contact & ~r_contact

# Recover the thresholds that were used.
print(meta["z_thresh"], meta["vel_thresh"], meta["floor_z"])
```

### PyTorch / RL training loop

```python
import torch
flags_t = torch.as_tensor(data["foot_ground_contact_flags"])  # (T, 2) bool
# Shape matches your other per-frame targets (dof_pos, root_pos, ...).
contact_loss = ((predicted_contact - flags_t.float()) ** 2).mean()
```

### Quick overlay on a rendered mp4

```bash
python scripts/vis_robot_motion_with_contact.py \
    --robot_motion_path output/foot_contact_tests/walk_02_01.pkl \
    --robot booster_k1
```

Green `L FOOT` / orange `R FOOT` text appears top-left while each foot is
flagged in contact.

## Backward compatibility

Pkls produced **before** the foot-contact feature landed (commit `47d3b5d`)
have none of the new keys. Consumers should probe defensively:

```python
flags = data.get("foot_ground_contact_flags")
if flags is None:
    # Either the key is absent (old pkl) or detection was disabled.
    ...
```

## Coverage

`scripts/retarget_no_penetration.py` is the only entry point in this fork
that produces the contact flags — all per-format converters were removed in
the K1/T1 cleanup. The detection itself lives in
`general_motion_retargeting/retargeting/foot_contact.py::detect_smplx_foot_contact`
and can be reused from any caller that has SMPL-X frames.
