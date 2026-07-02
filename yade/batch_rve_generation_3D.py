"""Batch 3D RVE generation: run multiple YADE packings and compute volume fractions.

Supports two-population mixing: big bulky chips + small slender chips at varying
volume ratios (0% = only bulky, 100% = only slender).
"""

import subprocess
import sys
import os
import json
import numpy as np
from pathlib import Path

# Chip volume in units of r³, computed via Monte Carlo on the union of overlapping
# spheres.  Multiply by radius³ to get the actual volume of a chip.
CHIP_VOLUME_R3 = {
    1:  21.17,  # flat 4x2
    2:  25.91,  # flat 5x2
    3:  16.47,  # flat 3x2
    4:  29.51,  # flat 4x3
    5:  12.83,  # beam 4
    6:  15.69,  # beam 5
    7:  18.60,  # beam 6
    8:   9.96,  # beam 3
    9:  22.99,  # flat 3x3
    10: 36.05,  # flat 5x3
}

# Big/bulky population defaults (flat rectangular chips)
BULKY_TYPES = [1, 2, 3, 9]  # flat 4x2, 5x2, 3x2, 3x3
BULKY_RADIUS = 0.04
BULKY_NUM_CHIPS = 1000

# Small/slender population defaults (beam chips, smaller radius)
SLENDER_TYPES = [5, 6, 7, 8]  # beam 4, 5, 6, 3
SLENDER_RADIUS = 0.01


def avg_chip_volume(types, radius):
    """Average volume of one chip, accounting for sphere overlaps."""
    return np.mean([CHIP_VOLUME_R3[t] for t in types]) * radius**3


def compute_num_slender_chips(slender_frac, num_bulky, bulky_types, bulky_radius,
                              slender_types, slender_radius):
    """Compute number of slender chips needed for a target volume fraction ratio.

    slender_frac: fraction of total chip volume that should be slender (0 to 1).
    Returns the number of slender chips (int).
    """
    if slender_frac <= 0:
        return 0
    if slender_frac >= 1.0:
        # Pure slender: scale to roughly similar total volume as pure bulky
        total_bulky_vol = num_bulky * avg_chip_volume(bulky_types, bulky_radius)
        return int(total_bulky_vol / avg_chip_volume(slender_types, slender_radius))

    vol_bulky = num_bulky * avg_chip_volume(bulky_types, bulky_radius)
    # slender_frac = V_slender / (V_slender + V_bulky)
    # V_slender = slender_frac * V_bulky / (1 - slender_frac)
    vol_slender_target = slender_frac * vol_bulky / (1 - slender_frac)
    return int(np.ceil(vol_slender_target / avg_chip_volume(slender_types, slender_radius)))


def build_mix_config(slender_pct, num_bulky=BULKY_NUM_CHIPS,
                     bulky_types=BULKY_TYPES, bulky_radius=BULKY_RADIUS,
                     slender_types=SLENDER_TYPES, slender_radius=SLENDER_RADIUS):
    """Build a config dict for a given slender percentage (0-100)."""
    slender_frac = slender_pct / 100.0
    common = {"frictionAngle": 0.5, "damping": 0.4, "small_fraction": 0.0}

    if slender_frac <= 0:
        return {
            **common,
            "num_chips": num_bulky,
            "base_radius": bulky_radius,
            "chip_types": ",".join(str(t) for t in bulky_types),
        }
    elif slender_frac >= 1.0:
        n_slender = compute_num_slender_chips(
            1.0, num_bulky, bulky_types, bulky_radius, slender_types, slender_radius)
        return {
            **common,
            "num_chips": n_slender,
            "base_radius": slender_radius,
            "chip_types": ",".join(str(t) for t in slender_types),
        }
    else:
        n_slender = compute_num_slender_chips(
            slender_frac, num_bulky, bulky_types, bulky_radius, slender_types, slender_radius)
        return {
            **common,
            "num_chips": num_bulky,
            "base_radius": bulky_radius,
            "chip_types": ",".join(str(t) for t in bulky_types),
            "num_chips_2": n_slender,
            "base_radius_2": slender_radius,
            "chip_types_2": ",".join(str(t) for t in slender_types),
        }


# Pre-built configurations
CONFIGS = {}
for pct in [100, 75, 50, 25, 0]:
    CONFIGS[f"mix_{pct}pct"] = build_mix_config(pct)

# Print chip counts for reference
for name, cfg in CONFIGS.items():
    n1 = cfg["num_chips"]
    n2 = cfg.get("num_chips_2", 0)
    r1 = cfg["base_radius"]
    r2 = cfg.get("base_radius_2", "N/A")
    print(f"  {name}: {n1} bulky (r={r1}) + {n2} slender (r={r2})")


def run_yade_simulation_3D(config_name, seed, output_dir, config=None):
    """Run 3D YADE simulation with given config and seed."""
    if config is None:
        config = CONFIGS[config_name]
    output_name = f"sphere_coordinates_3D_{config_name}_seed{seed}.npy"
    output_path = output_dir / output_name

    # -n = no GUI, -x = exit after script
    cmd = [
        "yadedaily", "-n", "-x", "yade_woodchip_3D.py", "--",
        "--output", str(output_path),
        "--seed", str(seed),
        "--small_fraction", str(config["small_fraction"]),
        "--num_chips", str(config["num_chips"]),
        "--base_radius", str(config["base_radius"]),
        "--frictionAngle", str(config["frictionAngle"]),
        "--damping", str(config["damping"]),
        "--chip_types", config.get("chip_types", "1,2,3,4,5,6"),
        "--no-gui",
    ]

    # Second population args
    if config.get("num_chips_2", 0) > 0:
        cmd.extend([
            "--num_chips_2", str(config["num_chips_2"]),
            "--base_radius_2", str(config["base_radius_2"]),
            "--chip_types_2", config["chip_types_2"],
        ])

    print(f"Running YADE 3D: {config_name} seed={seed}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    if result.stdout:
        # Print last few lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"  {line}")

    # Check if output file was created (the real success indicator)
    if output_path.exists():
        return output_path

    # If file doesn't exist, show error info
    # Filter out YADE's TCP prompt from stderr (it's not an error)
    stderr_lines = [line for line in result.stderr.split('\n')
                    if line.strip() and 'TCP python prompt' not in line]
    if stderr_lines:
        print(f"YADE error: {' '.join(stderr_lines[:3])}")
    else:
        print(f"YADE failed: output file not created (return code: {result.returncode})")

    return None


def compute_volume_fraction_3D(npy_file, method="monte_carlo", n_samples=500000,
                               periodic=True):
    """Compute volume fraction of 3D RVE using the volume fraction script."""
    cmd = [
        sys.executable, "woodchip_volume_fraction_3D.py",
        str(npy_file),
        "--method", method,
        "--n_samples", str(n_samples),
    ]
    if periodic:
        cmd.append("--periodic")

    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    # Parse volume fraction from output
    vf = None
    for line in result.stdout.split("\n"):
        if "Volume fraction:" in line:
            # Format: "Volume fraction: 0.1234 (12.34%)"
            try:
                vf = float(line.split(":")[1].split("(")[0].strip())
            except (IndexError, ValueError):
                pass
            break

    if result.stdout:
        # Print relevant output lines
        for line in result.stdout.strip().split('\n'):
            if any(x in line for x in ['Loaded', 'Extracted', 'Merged', 'Volume fraction']):
                print(f"  {line}")

    if result.returncode != 0:
        print(f"Volume fraction computation failed: {result.stderr}")

    return vf


def run_batch_3D(config_names, num_seeds=5, output_dir=None, method="monte_carlo", n_samples=500000):
    """Run batch 3D generation for specified configs."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "batch_output_3D"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    results = {}

    for config_name in config_names:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}, skipping")
            continue

        results[config_name] = []

        for seed in range(num_seeds):
            print(f"\n{'='*60}")
            print(f"Config: {config_name}, Seed: {seed}")
            print(f"{'='*60}")

            # Run YADE 3D
            npy_file = run_yade_simulation_3D(config_name, seed, output_dir)
            if npy_file is None or not npy_file.exists():
                print(f"Skipping {config_name} seed={seed} - YADE failed")
                continue

            # Compute volume fraction
            vf = compute_volume_fraction_3D(npy_file, method=method, n_samples=n_samples)

            results[config_name].append({
                "seed": seed,
                "volume_fraction": vf,
                "npy_file": str(npy_file),
            })

            print(f"  -> Volume fraction: {vf:.4f}" if vf else "  -> Volume fraction: N/A")

    # Save results summary
    results_file = output_dir / "batch_results_3D.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    import numpy as np
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Mean VF':>10} {'Std VF':>10} {'Min':>8} {'Max':>8} {'N':>4}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*4}")
    for config_name, runs in results.items():
        vfs = [r["volume_fraction"] for r in runs if r["volume_fraction"] is not None]
        if vfs:
            mean_vf = np.mean(vfs)
            std_vf = np.std(vfs)
            min_vf = np.min(vfs)
            max_vf = np.max(vfs)
            print(f"{config_name:<15} {mean_vf:>10.4f} {std_vf:>10.4f} {min_vf:>8.4f} {max_vf:>8.4f} {len(vfs):>4}")
        else:
            print(f"{config_name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {0:>4}")

    # Individual runs
    print(f"\nPer-run details:")
    for config_name, runs in results.items():
        for r in runs:
            vf_str = f"{r['volume_fraction']:.4f}" if r['volume_fraction'] is not None else "N/A"
            print(f"  {config_name} seed={r['seed']}: VF = {vf_str}")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch 3D RVE generation")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help=f"Config names to run. Available: {list(CONFIGS.keys())}")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds per config")
    parser.add_argument("--output_dir", type=str, default='generated_meshes_3D/', help="Output directory")
    parser.add_argument("--method", type=str, default="monte_carlo",
                        choices=["monte_carlo", "convex_hull", "spheres"],
                        help="Method for volume fraction computation")
    parser.add_argument("--n_samples", type=int, default=500000,
                        help="Number of Monte Carlo samples (if using monte_carlo method)")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")

    args = parser.parse_args()

    if args.list:
        print("Available configurations:")
        for name, cfg in CONFIGS.items():
            print(f"  {name}: {cfg}")
        sys.exit(0)

    run_batch_3D(
        config_names=args.configs,
        num_seeds=args.seeds,
        output_dir=args.output_dir,
        method=args.method,
        n_samples=args.n_samples,
    )
