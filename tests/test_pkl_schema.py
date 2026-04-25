"""Pin the output pkl schema for downstream consumers.

Loads the pinned baseline pkls (saved under tests/baseline/ — gitignored;
see tests/check_baseline.py to regenerate them) and asserts every key is
present with the right type / shape / dtype.

Run with the gmr conda env:

    python -m pytest tests/test_pkl_schema.py -v

or, without pytest:

    python tests/test_pkl_schema.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "tests" / "baseline"

REQUIRED_KEYS = {
    # Provenance
    "fps":                       float,
    "robot":                     str,
    "input_format":              str,
    "source_file":               str,
    "ground_mode":               str,
    "strict_zero_pen":           bool,
    "ground_clearance":          float,
    "actual_human_height":       float,
    # Numeric arrays
    "root_pos":                  np.ndarray,
    "root_rot":                  np.ndarray,
    "dof_pos":                   np.ndarray,
    # Optional (None when --no_foot_contact / no FK post)
    "local_body_pos":            (np.ndarray, type(None)),
    "link_body_list":            (list, type(None)),
    "foot_ground_contact_flags": (np.ndarray, type(None)),
    "foot_contact_meta":         (dict, type(None)),
}

FOOT_CONTACT_META_KEYS = {
    "source", "joints", "columns", "z_thresh", "vel_thresh", "floor_z",
}


def _check_one(pkl_path: Path) -> list[str]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    failures: list[str] = []

    for key, expected in REQUIRED_KEYS.items():
        if key not in data:
            failures.append(f"{pkl_path.name}: missing key '{key}'")
            continue
        if isinstance(expected, tuple):
            if not isinstance(data[key], expected):
                failures.append(
                    f"{pkl_path.name}: '{key}' is {type(data[key]).__name__}, "
                    f"expected one of {[t.__name__ for t in expected]}"
                )
        elif not isinstance(data[key], expected):
            failures.append(
                f"{pkl_path.name}: '{key}' is {type(data[key]).__name__}, "
                f"expected {expected.__name__}"
            )

    n = len(data["root_pos"])
    if data["root_pos"].shape != (n, 3):
        failures.append(f"{pkl_path.name}: root_pos shape {data['root_pos'].shape} != ({n}, 3)")
    if data["root_rot"].shape != (n, 4):
        failures.append(f"{pkl_path.name}: root_rot shape {data['root_rot'].shape} != ({n}, 4)")
    if data["dof_pos"].ndim != 2 or data["dof_pos"].shape[0] != n:
        failures.append(f"{pkl_path.name}: dof_pos shape {data['dof_pos'].shape}")

    flags = data.get("foot_ground_contact_flags")
    if flags is not None:
        if flags.shape != (n, 2):
            failures.append(f"{pkl_path.name}: contact flags shape {flags.shape} != ({n}, 2)")
        if flags.dtype != bool:
            failures.append(f"{pkl_path.name}: contact flags dtype {flags.dtype} != bool")

    meta = data.get("foot_contact_meta")
    if meta is not None:
        missing = FOOT_CONTACT_META_KEYS - set(meta.keys())
        if missing:
            failures.append(f"{pkl_path.name}: foot_contact_meta missing keys {sorted(missing)}")

    if data["robot"] not in ("booster_k1", "booster_t1"):
        failures.append(f"{pkl_path.name}: unknown robot {data['robot']!r}")

    return failures


def test_baseline_pkls_have_pinned_schema() -> None:
    if not BASELINE_DIR.exists():
        # The baseline pkls are gitignored, regenerate via tests/check_baseline.py.
        # Skip if they're not available (e.g. on a fresh clone).
        return
    pkls = sorted(BASELINE_DIR.glob("*.pkl"))
    if not pkls:
        return
    all_failures: list[str] = []
    for pkl in pkls:
        all_failures.extend(_check_one(pkl))
    assert not all_failures, "\n".join(all_failures)


if __name__ == "__main__":
    if not BASELINE_DIR.exists() or not list(BASELINE_DIR.glob("*.pkl")):
        print(f"[skip] no baseline pkls under {BASELINE_DIR}; "
              "run tests/check_baseline.py first.")
        sys.exit(0)
    failures: list[str] = []
    for pkl in sorted(BASELINE_DIR.glob("*.pkl")):
        f = _check_one(pkl)
        failures.extend(f)
        print(f"[{'FAIL' if f else ' OK '}] {pkl.name}")
        for msg in f:
            print(f"   - {msg}")
    sys.exit(1 if failures else 0)
