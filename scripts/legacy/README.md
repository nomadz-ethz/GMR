# Legacy single-format scripts

These scripts predate `scripts/retarget_no_penetration.py` and are kept here
for reference and for the few one-off workflows the new pipeline does not
yet cover (e.g. interactive single-file viewing without ground constraints).

| File | Replacement on the unified pipeline |
|---|---|
| `smplx_to_robot.py` | `retarget_no_penetration.py --input_format smplx --ground_mode none` |
| `smplx_to_robot_dataset.py` | `retarget_no_penetration.py --input <dir> --input_format smplx` |
| `gvhmr_to_robot.py` | `retarget_no_penetration.py --input_format gvhmr` |
| `gvhmr_to_robot_batch.py` | `retarget_no_penetration.py --input <dir> --input_format gvhmr` |

The unified pipeline adds:
- Strict zero foot-ground penetration via QP + two-pass IK.
- Per-foot ground-contact labels saved to the output pkl.
- A richer pkl schema (robot, input_format, source_file, ...).
- T1 robot support and a `--robot` flag.

See `docs/pipeline.md` for the full design and CLI reference. Keep using
these legacy scripts only if you need the older interactive playback
behaviour or you are debugging a difference between the two code paths.

`bvh_to_robot.py` and `bvh_to_robot_dataset.py` remain at the scripts/ top
level because BVH support has not yet been merged into
`retarget_no_penetration.py`; the work is on the sibling branch
`feature/add-bvh-support`. Once that lands, the BVH scripts will move
here too.
