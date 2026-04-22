"""Batch plumbing: file discovery, YAML parsing, the `Tee` log helper, and the
`run_batch` driver that calls a user-provided per-file callback.
"""

import os
import pathlib
import sys


# ---------------------------------------------------------------------------
# AMASS / file-discovery helpers
# ---------------------------------------------------------------------------

def amass_stem(npz_path: str) -> str:
    """Strip `_stageii` / `_stagei` suffix: '02_01_stageii.npz' -> '02_01'."""
    stem = pathlib.Path(npz_path).stem
    return stem.replace("_stageii", "").replace("_stagei", "")


def load_yaml_paths(yaml_file: str) -> list:
    """Parse a locomotion YAML and return [(rel_path, description), ...]."""
    import yaml
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    entries = []
    for _, items in cfg.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and "path" in item:
                entries.append((item["path"], item.get("description", "")))
    return entries


def discover_input_files(input_dir: str, input_format: str):
    """Walk a directory and find motion files.

    Returns:
        list of (input_path, relative_stem). The relative_stem is used to
        construct each output path as `<output_dir>/<stem>.pkl`.
    """
    results = []
    input_dir = os.path.abspath(input_dir)
    base = pathlib.Path(input_dir)

    if input_format == "gvhmr":
        for pt_path in sorted(base.rglob("hmr4d_results.pt")):
            stem = pt_path.parent.name
            if stem == base.name:
                stem = pt_path.stem
            results.append((str(pt_path), stem))

        if not results:  # fallback: any .pt file in the directory
            for pt_path in sorted(base.glob("*.pt")):
                results.append((str(pt_path), pt_path.stem))

    elif input_format in ("smplx", "amass_cmu"):
        for dirpath, _, filenames in os.walk(input_dir):
            for filename in sorted(filenames):
                if filename.endswith("_stagei.npz"):
                    continue
                if filename.endswith((".npz", ".pkl")):
                    full_path = os.path.join(dirpath, filename)
                    rel = os.path.relpath(full_path, input_dir)
                    if input_format == "amass_cmu":
                        rel_path = pathlib.Path(rel)
                        stem = str(rel_path.parent / amass_stem(rel_path.name))
                    else:
                        stem = os.path.splitext(rel)[0]
                    results.append((full_path, stem))

    return results


# ---------------------------------------------------------------------------
# Logging tee
# ---------------------------------------------------------------------------

class Tee:
    """Duplicate writes to multiple streams (real stdout + a log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(files, output_dir, input_format, args, retarget_fn):
    """Process a list of (src_path, stem) pairs and save each pkl under
    `output_dir`. The retarget callback signature is

        retarget_fn(input_path, output_path, input_format, args) -> bool

    matching the script's `retarget_and_save`. It is passed in to keep this
    module decoupled from any one CLI script.
    """
    from rich import print

    # Suppress interactive viewer in batch, but allow video recording (offscreen render).
    args.headless = True
    args.rate_limit = False

    # Tee stdout to <output_dir>/<top-level-stem-dir>/output.txt so the run's
    # terminal log lives next to the pkl/video outputs.
    log_file = None
    saved_stdout = None
    if files:
        first_parts = pathlib.Path(files[0][1]).parts
        log_subdir = first_parts[0] if len(first_parts) > 1 else ""
        log_dir = pathlib.Path(output_dir) / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "output.txt"
        log_file = open(log_path, "w", buffering=1)
        saved_stdout = sys.stdout
        sys.stdout = Tee(saved_stdout, log_file)
        try:
            import rich
            rich.reconfigure()
        except Exception:
            pass

    try:
        print(
            f"[bold]Found {len(files)} files. "
            f"Ground mode: {args.ground_mode}. "
            f"Output: {output_dir}[/bold]"
        )

        try:
            from tqdm import tqdm
            file_iter = tqdm(files, desc="Retargeting")
        except ImportError:
            file_iter = files

        success = 0
        for src_path, rel_stem in file_iter:
            out_path = os.path.join(output_dir, rel_stem + ".pkl")
            if os.path.exists(out_path) and not args.overwrite:
                continue
            try:
                ok = retarget_fn(src_path, out_path, input_format, args)
                if ok:
                    success += 1
            except Exception as e:
                print(f"[red]Error processing {src_path}: {e}[/red]")

        print(f"[bold green]Done! {success}/{len(files)} files -> {output_dir}[/bold green]")
    finally:
        if log_file is not None:
            sys.stdout = saved_stdout
            log_file.close()
            try:
                import rich
                rich.reconfigure()
            except Exception:
                pass
