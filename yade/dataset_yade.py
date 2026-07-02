"""Batch generation of 2D RVE packings over a grid of continuous clump shape parameters.

Calls yade_woodchip_3D_to_2D_continuous.py for each (r1, r2, nz, seed) combination.
Enforces r1 <= r2 (valid triangle region).
"""

import subprocess
import numpy as np
from pathlib import Path
import argparse
import json

YADE_SCRIPT = "yade_woodchip_3D_to_2D_continuous.py"


def run_single(r1, r2, nz, seed, yade_cmd="yadedaily"):
    """Run a single YADE deposition."""
    cmd = [
        yade_cmd, "-n", "-x", YADE_SCRIPT, "--",
        "--r1", f"{r1:.4f}",
        "--r2", f"{r2:.4f}",
        "--nz", str(nz),
        "--seed", str(seed),
        "--no-gui",
    ]

    print(f"  Running: r1={r1:.3f} r2={r2:.3f} nz={nz} seed={seed}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None

    # Extract volume fraction from output
    for line in result.stdout.split("\n"):
        if "volume fraction:" in line.lower():
            try:
                vf = float(line.split(":")[-1].strip())
                print(f"  Done: vf={vf:.4f}")
                return vf
            except ValueError:
                pass

    print("  Done (no vf parsed)")
    return None


def generate_grid(r1_values, r2_values, nz_values):
    """Generate all valid (r1, r2, nz, seed) combinations where r1 <= r2."""
    configs = []
    for nz in nz_values:
        for r1 in r1_values:
            for r2 in r2_values:
                if r1 > r2:
                    continue
                configs.append((r1, r2, nz))
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RVE generation over continuous shape grid")
    parser.add_argument("--r1", nargs="+", type=float, default=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], help="r1 values to sample")
    parser.add_argument("--r2", nargs="+", type=float, default=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], help="r2 values to sample")
    parser.add_argument("--nz", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8], help="nz values to sample")
    parser.add_argument("--yade", type=str, default="yadedaily", help="YADE executable name")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    args = parser.parse_args()

    configs = generate_grid(args.r1, args.r2, args.nz)

    print(f"Total configurations: {len(configs)}")
    print(f"  r1: {sorted(set(args.r1))}")
    print(f"  r2: {sorted(set(args.r2))}")
    print(f"  nz: {sorted(set(args.nz))}")
    seed = 0

    if args.dry_run:
        for r1, r2, nz in configs:
            print(f"  r1={r1:.2f} r2={r2:.2f} nz={nz} seed={seed}")
        exit(0)

    results = []
    for i, (r1, r2, nz) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")
        vf = run_single(r1, r2, nz, seed, yade_cmd=args.yade)
        results.append({"r1": r1, "r2": r2, "nz": nz, "seed": seed, "vf": vf})

    # Save summary
    output_dir = Path(__file__).parent / "periodic" / "depositions_3d_to_2d_relaxed_cont"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    succeeded = [r for r in results if r["vf"] is not None]
    failed = [r for r in results if r["vf"] is None]
    print(f"Succeeded: {len(succeeded)}/{len(results)}")
    if failed:
        print(f"Failed: {[(r['r1'], r['r2'], r['nz'], r['seed']) for r in failed]}")
    print(f"Results saved to: {summary_file}")
