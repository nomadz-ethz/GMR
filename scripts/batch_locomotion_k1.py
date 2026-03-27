import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="locomotion_k1")
    parser.add_argument("--output_dir", type=str, default="locomotion_k1_retargeted_smooth")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All potential source roots
    source_roots = [
        Path("CMU"),
        Path("kicking_source"),
        Path("data/AMASS/CMU")
    ]

    # Get all .pkl files in the input_dir to use as keys
    target_files = sorted(list(input_dir.glob("*.pkl")))
    print(f"Found {len(target_files)} target motions in {input_dir}")

    processed_count = 0
    missing_count = 0

    for pkl_path in target_files:
        base_name = pkl_path.stem  # e.g., '02_01'
        target_npz = f"{base_name}_stageii.npz"
        
        # Search for the npz source
        found_npz = None
        for root in source_roots:
            # Check direct child (for kicking_source) or subfolders (for CMU)
            potential = root / target_npz
            if potential.exists():
                found_npz = potential
                break
            
            # Check subdirectories (for CMU structure)
            subject_id = base_name.split('_')[0]
            potential_sub = root / subject_id / target_npz
            if potential_sub.exists():
                found_npz = potential_sub
                break

        if not found_npz:
            print(f"[!] Could not find source .npz for {base_name}")
            missing_count += 1
            continue

        output_path = output_dir / f"{base_name}.pkl"
        
        # Construct the command
        cmd = [
            "python", "scripts/retarget_no_penetration.py",
            "--input", str(found_npz),
            "--input_format", "amass_cmu",
            "--ground_mode", "qp",
            "--strict_zero_pen",
            "--no_viz",
            "--output", str(output_path)
        ]

        print(f"[{processed_count+1}/{len(target_files)}] Retargeting {base_name}...")
        try:
            subprocess.run(cmd, check=True)
            processed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to process {base_name}: {e}")

    print(f"\nBatch complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Missing sources: {missing_count}")

if __name__ == "__main__":
    main()
