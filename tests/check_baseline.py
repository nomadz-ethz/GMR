"""Regression check: re-run baseline motions and np.allclose vs pinned pkls.

Used during the K1/T1 cleanup to verify that pruning the registry did not
perturb the pipeline output. Run with the `gmr` conda env active:

    python tests/check_baseline.py

Exits 0 on full match, 1 on any divergence.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "tests" / "baseline"

CASES = [
    {
        "name": "amass_cmu_35_01",
        "input": "/home/shinben0327/nomadz/k1_motion_data/data/AMASS/CMU/35/35_01_stageii.npz",
        "input_format": "amass_cmu",
        "baseline": BASELINE_DIR / "35_01_k1_amass.pkl",
    },
    {
        "name": "gvhmr_freekick",
        "input": "/home/shinben0327/nomadz/archive/video_to_k1_pipeline/out_smplx/11_freekick/hmr4d_results.pt",
        "input_format": "gvhmr",
        "baseline": BASELINE_DIR / "freekick_k1_gvhmr.pkl",
    },
]

NUMERIC_KEYS = ("root_pos", "root_rot", "dof_pos", "local_body_pos")


def _run(case: dict, out_path: Path) -> None:
    cmd = [
        sys.executable, str(REPO / "scripts" / "retarget_no_penetration.py"),
        "--input", case["input"],
        "--input_format", case["input_format"],
        "--robot", "booster_k1",
        "--ground_mode", "qp", "--strict_zero_pen",
        "--headless",
        "--output", str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=REPO)


def _compare(a_path: Path, b_path: Path) -> list[str]:
    with open(a_path, "rb") as f:
        a = pickle.load(f)
    with open(b_path, "rb") as f:
        b = pickle.load(f)
    diffs: list[str] = []
    for k in NUMERIC_KEYS:
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            diffs.append(f"key={k} one side is None (a={va is None}, b={vb is None})")
            continue
        va = np.asarray(va); vb = np.asarray(vb)
        if va.shape != vb.shape:
            diffs.append(f"key={k} shape {va.shape} vs {vb.shape}")
        elif not np.allclose(va, vb, atol=1e-7, rtol=1e-7):
            diffs.append(
                f"key={k} max abs diff {np.max(np.abs(va - vb)):.3e}"
            )
    fc_a = a.get("foot_ground_contact_flags")
    fc_b = b.get("foot_ground_contact_flags")
    if fc_a is None and fc_b is None:
        pass
    elif fc_a is None or fc_b is None:
        diffs.append("foot_ground_contact_flags presence differs")
    elif not np.array_equal(fc_a, fc_b):
        diffs.append(
            f"foot_ground_contact_flags differ in {(fc_a != fc_b).sum()} entries"
        )
    return diffs


def main() -> int:
    if not BASELINE_DIR.exists():
        print(f"[error] baseline dir {BASELINE_DIR} missing", file=sys.stderr)
        return 1
    overall_ok = True
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for case in CASES:
            if not case["baseline"].exists():
                print(f"[skip] no baseline at {case['baseline']}")
                continue
            out = td_path / f"{case['name']}.pkl"
            print(f"[run] {case['name']} -> {out}")
            _run(case, out)
            diffs = _compare(out, case["baseline"])
            if diffs:
                overall_ok = False
                print(f"[FAIL] {case['name']}")
                for d in diffs:
                    print(f"   - {d}")
            else:
                print(f"[ OK ] {case['name']}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
