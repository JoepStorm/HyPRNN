"""Batch RVE generation: run multiple YADE packings and extract volume fractions."""

import subprocess
import sys
import os
import json
from pathlib import Path

# Configuration presets (same as in yade_woodchip.py)
CONFIGS = {
    # "small": {
    #     "small_fraction": 1.0,
    #     "big_fraction": 0.0,
    #     "num_chips": 1000,
    #     "base_radius": 0.02,
    #     "frictionAngle": 0.5,
    #     "damping": 0.4,
    # },
    # "small_looser": {
    #     "small_fraction": 1.0,
    #     "big_fraction": 0.0,
    #     "num_chips": 1000,
    #     "base_radius": 0.02,
    #     "frictionAngle": 1.0,
    #     "damping": 0.7,
    # },
    # "large": {
    #     "small_fraction": 0.0,
    #     "big_fraction": 1.0,
    #     "num_chips": 300,
    #     "base_radius": 0.02,
    #     "frictionAngle": 0.5,
    #     "damping": 0.4,
    # },
    # "large_looser": {
    #     "small_fraction": 0.0,
    #     "big_fraction": 1.0,
    #     "num_chips": 300,
    #     "base_radius": 0.02,
    #     "frictionAngle": 1.0,
    #     "damping": 0.7,
    # },
    # "mix": {
    #     "small_fraction": 0.5,
    #     "big_fraction": 0.5,
    #     "num_chips": 500,
    #     "base_radius": 0.02,
    #     "frictionAngle": 0.5,
    #     "damping": 0.4,
    # },
    # "mix_looser": {
    #     "small_fraction": 0.5,
    #     "big_fraction": 0.5,
    #     "num_chips": 500,
    #     "base_radius": 0.02,
    #     "frictionAngle": 1.0,
    #     "damping": 0.7,
    # },
    # "mix2": {
    #     "small_fraction": 0.8,
    #     "big_fraction": 0.2,
    #     "num_chips": 600,
    #     "base_radius": 0.02,
    #     "frictionAngle": 0.5,
    #     "damping": 0.4,
    # },
    # "mix2_looser": {
    #     "small_fraction": 0.8,
    #     "big_fraction": 0.2,
    #     "num_chips": 600,
    #     "base_radius": 0.02,
    #     "frictionAngle": 1.0,
    #     "damping": 0.7,
    # },
    "vlarge": {
        "small_fraction": 0.0,
        "big_fraction": 1.0,
        "num_chips": 100,
        "base_radius": 0.04,
        "frictionAngle": 0.5,
        "damping": 0.4,
    },
    "vlarge_looser": {
        "small_fraction": 0.0,
        "big_fraction": 1.0,
        "num_chips": 100,
        "base_radius": 0.04,
        "frictionAngle": 1.0,
        "damping": 0.7,
    },
}


def run_yade_simulation(config_name, seed, output_dir):
    """Run YADE simulation with given config and seed."""
    config = CONFIGS[config_name]
    output_name = f"sphere_coordinates_{config_name}_seed{seed}.npy"
    output_path = output_dir / output_name

    # -n = no GUI, -x = exit after script
    cmd = [
        "yadedaily", "-n", "-x", "yade_woodchip.py", "--",
        "--output", str(output_path),
        "--seed", str(seed),
        "--small_fraction", str(config["small_fraction"]),
        "--num_chips", str(config["num_chips"]),
        "--base_radius", str(config["base_radius"]),
        "--frictionAngle", str(config["frictionAngle"]),
        "--damping", str(config["damping"]),
        "--no-gui",
    ]

    print(f"Running YADE: {config_name} seed={seed}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"YADE failed: {result.stderr}")
        return None

    return output_path


def convert_to_geo_and_get_vf(npy_file, output_dir, mesh_size=0.01):
    """Convert .npy to .geo and return volume fraction."""
    geo_file = output_dir / npy_file.stem.replace("sphere_coordinates_", "woodchip_rve_")
    geo_file = geo_file.with_suffix(".geo")

    cmd = [
        sys.executable, "woodchip_to_gmsh_rve.py",
        str(npy_file), str(geo_file), "--mesh_size", str(mesh_size)
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    # Parse volume fraction from output
    vf = None
    for line in result.stdout.split("\n"):
        if "Volume fraction:" in line:
            vf = float(line.split(":")[1].strip())
            break

    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr}")

    return vf, geo_file


def mesh_geo_file(geo_file, output_dir):
    """Mesh the .geo file using gmsh."""
    msh_file = output_dir / geo_file.stem
    msh_file = msh_file.with_suffix(".msh")

    cmd = ["gmsh", str(geo_file), "-2", "-o", str(msh_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Meshing failed: {result.stderr}")
        return None

    return msh_file


def run_batch(config_names, num_seeds=5, output_dir=None, mesh=True, mesh_size=0.01):
    """Run batch generation for specified configs."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "batch_output"
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

            # Run YADE
            npy_file = run_yade_simulation(config_name, seed, output_dir)
            if npy_file is None or not npy_file.exists():
                print(f"Skipping {config_name} seed={seed} - YADE failed")
                continue

            # Convert to .geo and get volume fraction
            vf, geo_file = convert_to_geo_and_get_vf(npy_file, output_dir, mesh_size)

            # Optionally mesh
            msh_file = None
            if mesh and geo_file and geo_file.exists():
                msh_file = mesh_geo_file(geo_file, output_dir)

            results[config_name].append({
                "seed": seed,
                "volume_fraction": vf,
                "npy_file": str(npy_file),
                "geo_file": str(geo_file) if geo_file else None,
                "msh_file": str(msh_file) if msh_file else None,
            })

            print(f"Volume fraction: {vf:.4f}" if vf else "Volume fraction: N/A")

    # Save results summary
    results_file = output_dir / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for config_name, runs in results.items():
        vfs = [r["volume_fraction"] for r in runs if r["volume_fraction"] is not None]
        if vfs:
            print(f"{config_name}: VF = {sum(vfs)/len(vfs):.4f} ± {max(vfs)-min(vfs):.4f} (n={len(vfs)})")
        else:
            print(f"{config_name}: No successful runs")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch RVE generation")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help=f"Config names to run. Available: {list(CONFIGS.keys())}")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per config")
    parser.add_argument("--output_dir", type=str, default='generated_meshes/', help="Output directory")
    parser.add_argument("--no-mesh", action="store_true", help="Skip meshing step")
    parser.add_argument("--mesh_size", type=float, default=0.01, help="Mesh element size")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")

    args = parser.parse_args()

    if args.list:
        print("Available configurations:")
        for name, cfg in CONFIGS.items():
            print(f"  {name}: {cfg}")
        sys.exit(0)

    run_batch(
        config_names=args.configs,
        num_seeds=args.seeds,
        output_dir=args.output_dir,
        mesh=not args.no_mesh,
        mesh_size=args.mesh_size,
    )
