"""Pipeline-internal helpers used by scripts/retarget_no_penetration.py.

Each module is a coherent piece extracted from what was previously one
700+ line script. They are also useful on their own for downstream tools
(e.g. `foot_contact.detect_smplx_foot_contact` for re-labelling existing
SMPL-X data without re-running IK).

See docs/pipeline.md for the design walk-through.
"""

from .builder import build_retargeter
from .foot_contact import detect_smplx_foot_contact
from .strict_zero_pen import (
    measure_penetration_depths,
    smooth_penetration_envelope,
)
from .fk_post import apply_fk_post
from .batch_runner import (
    Tee,
    run_batch,
    discover_input_files,
    load_yaml_paths,
    amass_stem,
)

__all__ = [
    "build_retargeter",
    "detect_smplx_foot_contact",
    "measure_penetration_depths",
    "smooth_penetration_envelope",
    "apply_fk_post",
    "Tee",
    "run_batch",
    "discover_input_files",
    "load_yaml_paths",
    "amass_stem",
]
