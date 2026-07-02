"""Batch runner for yade_woodchip_3D_wet.py + volume fraction statistics + plot.

Generates 5 configs varying the small/large chip volume ratio from 0% to 100%.
Chip counts are computed from estimated per-chip volumes to achieve the target ratio.
"""

import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from woodchip_volume_fraction_3D import compute_volume_fraction
from matplotlib import rc
plt.style.use(['science', 'bright'])
rc('text', usetex=True)

# ── Batch settings ──────────────────────────────────────────────────────────
NUM_SEEDS  = 5
OUTPUT_DIR = Path(__file__).parent / "generated_meshes_3D_wet"

# ── Chip geometry definitions (must match yade_woodchip_3D_wet.py) ──────────
CHIP_DEFS = {
    # Large (plate-like, two-row)
    4:  ([-1.5,-0.5,0.5,1.5],           [-0.5, 0.5]),
    9:  ([-1,0,1],                       [-0.5, 0.5]),
    10: ([-2,-1,0,1,2],                  [-0.5, 0.5]),

    1: ([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [-1.5, -0.5, 0.5, 1.5]),
    2: ([-2, -1, 0, 1, 2], [-1.5, -0.5, 0.5, 1.5]),
    3: ([-3, -2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2]),
    # Small (slender, single-row)
    7:  ([-2.5,-1.5,-0.5,0.5,1.5,2.5],  [0]),
    11: ([-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5], [0]),
    12: ([-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5], [0]),
}
# LARGE_TYPES = [4, 9, 10]
LARGE_TYPES = [1, 2, 3]
SMALL_TYPES = [7, 11, 12]

# BASE_RADIUS_LARGE     = 0.07
BASE_RADIUS_LARGE     = 0.045
BASE_RADIUS_SMALL     = 0.02
DAMPING               = 0.8
FRICTION_ANGLE_LARGE  = 1.3    # max = pi/2 = 90 degrees.
FRICTION_ANGLE_SMALL  = 0.8

# Total deposited volume target (tuned so extraction region has enough material)
TARGET_TOTAL_VOLUME = 1.3


def estimate_chip_volume(x_offsets, z_offsets, radius, n_samples=200000):
    """Estimate chip volume via Monte Carlo sampling."""
    centers = np.array([(dx * radius, 0, dz * radius)
                        for dz in z_offsets for dx in x_offsets])
    r = radius
    # Bounding box of all sphere centers ± radius
    lo = centers.min(axis=0) - r
    hi = centers.max(axis=0) + r
    box_vol = np.prod(hi - lo)
    # Random points in bounding box
    pts = np.random.uniform(lo, hi, size=(n_samples, 3))
    # Check if each point is inside any sphere
    inside = np.zeros(n_samples, dtype=bool)
    for c in centers:
        inside |= np.sum((pts - c)**2, axis=1) <= r**2
    return box_vol * inside.sum() / n_samples


def mean_chip_volume(types, radius):
    """Average volume across chip types for a given radius."""
    return np.mean([estimate_chip_volume(*CHIP_DEFS[t], radius) for t in types])


# ── Build configs from volume ratios ───────────────────────────────────────
vol_large = mean_chip_volume(LARGE_TYPES, BASE_RADIUS_LARGE)
vol_small = mean_chip_volume(SMALL_TYPES, BASE_RADIUS_SMALL)
print(f"Estimated mean chip volumes: large={vol_large:.6f}, small={vol_small:.6f}")

FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]

CONFIGS = {}
for fs in FRACS_SMALL:
    fl = 1.0 - fs
    n_large = int(round(fl * TARGET_TOTAL_VOLUME / vol_large)) if fl > 0 else 0
    n_small = int(round(fs * TARGET_TOTAL_VOLUME / vol_small)) if fs > 0 else 0
    n_small = min(n_small, 3000)
    name = f"small{int(fs*100):03d}_large{int(fl*100):03d}"
    CONFIGS[name] = {
        "num_chips":         n_large,
        "base_radius":       BASE_RADIUS_LARGE,
        "num_chips_small":   n_small,
        "base_radius_small": BASE_RADIUS_SMALL,
    }
    print(f"  {name}: {n_large} large + {n_small} small")


def run_wet_simulation(seed, output_path, config):
    """Run yade_woodchip_3D_wet.py with given config and seed."""
    cmd = [
        "yadedaily", "-n", "-x", "yade_woodchip_3D_wet.py", "--",
        "--seed",              str(seed),
        "--output",            str(output_path),
        "--num_chips",         str(config["num_chips"]),
        "--base_radius",       str(config["base_radius"]),
        "--num_chips_small",   str(config["num_chips_small"]),
        "--base_radius_small", str(config["base_radius_small"]),
        "--damping",              str(DAMPING),
        "--friction_angle",       str(FRICTION_ANGLE_LARGE),
        "--friction_angle_small", str(FRICTION_ANGLE_SMALL),
    ]
    print(f"    seed={seed} ...", flush=True)
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().split("\n")[-4:]:
            print(f"      {line}")

    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            d = np.load(output_path)
            if d.ndim == 2 and d.shape[1] == 5 and len(d) > 0:
                return output_path
        except Exception:
            pass

    stderr = [l for l in result.stderr.split("\n")
              if l.strip() and "TCP python prompt" not in l]
    print(f"    FAILED: {' '.join(stderr[:3]) if stderr else f'rc={result.returncode}'}")
    return None


def run_batch(config_name, config, num_seeds, output_dir):
    """Run all seeds for one config; return list of vfrac values."""
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    vfracs = []

    print(f"\n{'='*55}")
    print(f"  Config: {config_name}")
    print(f"  {config}")
    print(f"{'='*55}")

    for seed in range(num_seeds):
        out = config_dir / f"seed{seed}.npy"
        npy = run_wet_simulation(seed, out, config)
        if npy is None:
            continue
        vf, _ = compute_volume_fraction(str(npy), method="monte_carlo",
                                        n_samples=500000, periodic=True)
        if vf is not None:
            vfracs.append(vf)
            print(f"    → VF = {vf:.4f}")
        else:
            print(f"    → VF computation failed")

    return vfracs


# Experimental data (indexed by FRACS_SMALL: 0.0, 0.25, 0.50, 0.75, 1.0)
EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]


def plot_results(all_vfracs, output_dir):
    """Box + strip plot of VF distributions per config."""
    names  = list(all_vfracs.keys())
    data   = [[v for v in all_vfracs[n] if v > 0.02] for n in names]
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours = colours[:len(data)]

    fig, ax = plt.subplots(figsize=(5, 3))

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=2),
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colours):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (vals, color) in enumerate(zip(data, colours), start=1):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=color, edgecolors="black", linewidths=0.5,
                   zorder=5, s=40)

    # Experimental reference points
    x_pos = np.arange(1, len(FRACS_SMALL) + 1)
    ax.scatter(x_pos, EXP_SHAKEN, marker='^', s=60, color='black',
               zorder=6, label='Exp. shaken')
    ax.scatter(x_pos, EXP_UNSHAKEN, marker='v', s=60, color='gray',
               zorder=6, label='Exp. unshaken')
    ax.legend(fontsize=7)

    # X-axis labels: show small/large percentages
    labels = [f"{int(fs*100)}\\%/{int((1-fs)*100)}\\%" for fs in FRACS_SMALL]
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Small / Large volume ratio")
    ax.set_ylabel("Volume fraction")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "vfrac_comparison"
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=150)
    print(f"\nPlot saved to: {out}")
    # plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",      type=int, default=NUM_SEEDS)
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--plot-only",  action="store_true",
                   help="Skip simulations; load results.json and plot")
    args = p.parse_args()
    plot_only = args.plot_only

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "results.json"

    if plot_only:
        with open(results_file) as f:
            all_vfracs = json.load(f)
        print(f"Loaded results from {results_file}")
    else:
        all_vfracs = {}
        for name, cfg in CONFIGS.items():
            vfracs = run_batch(name, cfg, args.seeds, output_dir)
            all_vfracs[name] = vfracs
            if vfracs:
                print(f"\n  {name}: mean={np.mean(vfracs):.4f} std={np.std(vfracs):.4f}  n={len(vfracs)}")

        with open(results_file, "w") as f:
            json.dump(all_vfracs, f, indent=2)

    plot_results(all_vfracs, output_dir)
