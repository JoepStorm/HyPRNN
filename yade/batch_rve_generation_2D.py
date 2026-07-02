"""Batch runner for yade_woodchip_2D.py + 2D volume fraction statistics + plot.

Generates 5 configs varying the small/large chip volume ratio from 0% to 100%.
Chip counts are derived from estimated 3D chip volumes scaled to fill the
gravity-deposition column up to the cut height.
"""

import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rc
from mesh_rve_2D import mesh_batch_friction, plot_meshes_paper
plt.style.use(['science', 'bright'])
rc('text', usetex=True)

# ── Batch settings ───────────────────────────────────────────────────────────
NUM_SEEDS  = 2 # 20
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v2_smaller_chips_10"
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v4_bit_bigger_5"
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v5_bigrve"
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v6_big_aspectratio"
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v8_mix_big_slender_single"
# OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v9_faster"
OUTPUT_DIR = Path(__file__).parent / "generated_meshes_2D_wet/v10"

# ── Chip geometry (must match yade_woodchip_2D.py) ──────────────────────
CHIP_DEFS = {
    1: ([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [-1.5, -0.5, 0.5, 1.5]),
    2: ([-2, -1, 0, 1, 2],                  [-1.5, -0.5, 0.5, 1.5]),
    3: ([-3, -2, -1, 0, 1, 2, 3],           [-2, -1, 0, 1, 2]),
    7:  ([-2.5,-1.5,-0.5,0.5,1.5,2.5],                      [0]),
    11: ([-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5],             [0]),
    12: ([-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5],    [0]),

    # big only, 2 different aspect ratios
    22: ([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),
    41: ([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [0]),

}


# # v1
# BASE_RADIUS_LARGE     = 0.045
# BASE_RADIUS_SMALL     = 0.02
# CUT_HEIGHT            = 0.4
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# TARGET_TOTAL_VOLUME = 0.5 # Target deposited volume to fill the gravity column to just above cut height.
# # v2_smaller_chips
# BASE_RADIUS_LARGE     = 0.0225
# BASE_RADIUS_SMALL     = 0.01
# CUT_HEIGHT            = 0.2
# TARGET_TOTAL_VOLUME = 0.3
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# # v3_big_only; it should have as little variance as the small only one.
# BASE_RADIUS_LARGE     = 0.01
# BASE_RADIUS_SMALL     = 0.01
# CUT_HEIGHT            = 0.2
# TARGET_TOTAL_VOLUME = 0.3
# FRACS_SMALL = [0.0]
# EXP_SHAKEN   = [0.410945]
# EXP_UNSHAKEN = [0.338119]

# # v4_bit_bigger; it should have as little variance as the small only one.
# BASE_RADIUS_LARGE     = 0.015
# BASE_RADIUS_SMALL     = 0.01
# CUT_HEIGHT            = 0.1
# TARGET_TOTAL_VOLUME = 0.15
# FRACS_SMALL = [0.0]
# EXP_SHAKEN   = [0.410945]
# EXP_UNSHAKEN = [0.338119]

# # v5_bigrve. once per seed.
# BASE_RADIUS_LARGE     = 0.015
# BASE_RADIUS_SMALL     = 0.0075
# CUT_HEIGHT            = 0.08
# TARGET_TOTAL_VOLUME = 0.12
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]
# LARGE_TYPES = [1, 2, 3]
# SMALL_TYPES = [7, 11, 12]
# DAMPING               = 0.8
# FRICTION_ANGLE_LARGE  = 1.2  # 1.3; changed to 1.2 for v5_big
# FRICTION_ANGLE_SMALL  = 0.8

# # v6 big aspect only
# LARGE_TYPES     = [22]
# SMALL_TYPES     = [41]
# FRACS_SMALL = [0.0, 1.0]
# BASE_RADIUS_LARGE     = 0.02
# BASE_RADIUS_SMALL     = 0.02
# CUT_HEIGHT            = 0.1
# TARGET_TOTAL_VOLUME = 0.2
# EXP_SHAKEN = None
# EXP_UNSHAKEN = None
# DAMPING               = 0.8
# FRICTION_ANGLE_LARGE  = 1.2
# FRICTION_ANGLE_SMALL  = 1.2

# # v7 mix big slender
# LARGE_TYPES = [1, 2, 7, 41]
# SMALL_TYPES = [7, 11, 12]
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# BASE_RADIUS_LARGE     = 0.015
# BASE_RADIUS_SMALL     = 0.0075
# CUT_HEIGHT            = 0.08
# TARGET_TOTAL_VOLUME = 0.12
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]
# DAMPING               = 0.8
# FRICTION_ANGLE_LARGE  = 1.1
# FRICTION_ANGLE_SMALL  = 0.5

# # v8 reduce small vfrac
# LARGE_TYPES = [1, 2, 7, 41]
# SMALL_TYPES = [7, 11, 12]
# # FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# BASE_RADIUS_LARGE     = 0.015
# BASE_RADIUS_SMALL     = 0.0075
# CUT_HEIGHT            = 0.08
# TARGET_TOTAL_VOLUME = 0.12
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]
# DAMPING               = 0.8
# FRICTION_ANGLE_LARGE  = 1.1
# FRICTION_ANGLE_SMALL  = 1.1

# # v9 faster
# LARGE_TYPES = [1, 2, 7, 41]
# SMALL_TYPES = [7, 11, 12]
# FRACS_SMALL = [0.0, 0.25, 0.50, 0.75, 1.0]
# BASE_RADIUS_LARGE     = 0.03
# BASE_RADIUS_SMALL     = 0.015
# CUT_HEIGHT            = 0.08
# TARGET_TOTAL_VOLUME = 0.2
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]
# DAMPING               = 0.8
# FRICTION_ANGLE_LARGE  = .5
# FRICTION_ANGLE_SMALL  = .5

# v10 small low vfrac
LARGE_TYPES = [1, 2, 7]
SMALL_TYPES = [7, 11, 12]
FRACS_SMALL = [0.0, 0.50, 1.0]
# BASE_RADIUS_LARGE     = 0.03
# BASE_RADIUS_SMALL     = 0.0075
# BASE_RADIUS_LARGE     = 0.05  # v10 good
# BASE_RADIUS_SMALL     = 0.015
BASE_RADIUS_LARGE     = 0.025
BASE_RADIUS_SMALL     = 0.0075
CUT_HEIGHT            = 0.08
TARGET_TOTAL_VOLUME = 0.14
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]
EXP_SHAKEN   = [0.410945,0.463811, 0.288999]
EXP_UNSHAKEN = [0.338119,0.390681, 0.267203]
DAMPING               = 0.8
FRICTION_ANGLE_LARGE  = 1.4     # v10 good had 1.3, 1.1
FRICTION_ANGLE_SMALL  = 1.2

# # Experimental data (indexed by FRACS_SMALL)
# EXP_SHAKEN   = [0.410945, 0.441943, 0.463811, 0.391315, 0.288999]
# EXP_UNSHAKEN = [0.338119, 0.383694, 0.390681, 0.343910, 0.267203]


def estimate_chip_volume(x_offsets, z_offsets, radius, n_samples=200000):
    """Estimate chip volume via Monte Carlo sampling."""
    centers = np.array([(dx * radius, 0, dz * radius)
                        for dz in z_offsets for dx in x_offsets])
    r = radius
    lo = centers.min(axis=0) - r
    hi = centers.max(axis=0) + r
    box_vol = np.prod(hi - lo)
    pts = np.random.uniform(lo, hi, size=(n_samples, 3))
    inside = np.zeros(n_samples, dtype=bool)
    for c in centers:
        inside |= np.sum((pts - c)**2, axis=1) <= r**2
    return box_vol * inside.sum() / n_samples


def mean_chip_volume(types, radius):
    """Average chip volume across types for a given radius."""
    return np.mean([estimate_chip_volume(*CHIP_DEFS[t], radius) for t in types])


# ── Build configs from volume ratios ─────────────────────────────────────────
vol_large = mean_chip_volume(LARGE_TYPES, BASE_RADIUS_LARGE)
vol_small = mean_chip_volume(SMALL_TYPES, BASE_RADIUS_SMALL)
print(f"Estimated mean chip volumes: large={vol_large:.6f}, small={vol_small:.6f}")

CONFIGS = {}
for fs in FRACS_SMALL:
    fl = 1.0 - fs
    n_large = int(round(fl * TARGET_TOTAL_VOLUME / vol_large)) if fl > 0 else 0
    n_small = int(round(fs * TARGET_TOTAL_VOLUME / vol_small)) if fs > 0 else 0
    # n_small = min(n_small, 2000)
    name = f"small{int(fs*100):03d}_large{int(fl*100):03d}"
    CONFIGS[name] = {
        "num_chips":         n_large,
        "base_radius":       BASE_RADIUS_LARGE,
        "num_chips_small":   n_small,
        "base_radius_small": BASE_RADIUS_SMALL,
    }
    print(f"  {name}: {n_large} large + {n_small} small")


def compute_vf_2d(npy_path, n_samples=500000):
    """Compute 2D area fraction via Monte Carlo, accounting for circle overlaps."""
    try:
        d = np.load(npy_path)
        meta = np.load(str(npy_path).replace('.npy', '_meta.npy'))
        lx, ly = meta[0], meta[1]
        pts = np.random.uniform([0, 0], [lx, ly], size=(n_samples, 2))
        inside = np.zeros(n_samples, dtype=bool)
        for cx, cy, r, _ in d:
            dx = np.abs(pts[:, 0] - cx)
            dy = np.abs(pts[:, 1] - cy)
            dx = np.minimum(dx, lx - dx)   # periodic wrap
            dy = np.minimum(dy, ly - dy)
            inside |= dx**2 + dy**2 <= r**2
        return inside.sum() / n_samples
    except Exception:
        return None


def run_2d_simulation(seed, output_path, config):
    """Run yade_woodchip_2D.py with given config and seed."""
    cmd = [
        "yadedaily", "-n", "-x", "yade_woodchip_2D.py", "--",
        "--seed",                str(seed),
        "--output",              str(output_path),
        "--num_chips",           str(config["num_chips"]),
        "--base_radius",         str(config["base_radius"]),
        "--num_chips_small",     str(config["num_chips_small"]),
        "--base_radius_small",   str(config["base_radius_small"]),
        "--damping",             str(DAMPING),
        "--friction_angle",      str(FRICTION_ANGLE_LARGE),
        "--friction_angle_small",str(FRICTION_ANGLE_SMALL),
        "--cut_height",          str(CUT_HEIGHT),
    ]
    print(f"    seed={seed} ...", flush=True)
    print(f"    cmd={cmd}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().split("\n")[-4:]:
            print(f"      {line}")

    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            d = np.load(output_path)
            if d.ndim == 2 and d.shape[1] == 4 and len(d) > 0:
                return output_path
        except Exception:
            pass

    stderr = [l for l in result.stderr.split("\n")
              if l.strip() and "TCP python prompt" not in l]
    print(f"    FAILED: {' '.join(stderr[:3]) if stderr else f'rc={result.returncode}'}")
    return None


def run_batch(config_name, config, seeds, output_dir):
    """Run all seeds for one config; return list of 2D VF values."""
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    vfracs = []

    print(f"\n{'='*55}")
    print(f"  Config: {config_name}")
    print(f"  {config}")
    print(f"{'='*55}")

    for seed in seeds:
        out = config_dir / f"seed{seed}.npy"
        npy = run_2d_simulation(seed, out, config)
        if npy is None:
            continue
        vf = compute_vf_2d(npy)
        if vf is not None:
            vfracs.append(vf)
            print(f"    → VF2D = {vf:.4f}")
        else:
            print(f"    → VF computation failed")

    return vfracs


def collect_mesh_vfracs(output_dir, shrink_factor=0.85):
    """Collect woodchip VF from _vfrac.txt files, grouped by config name."""
    mesh_vfracs = {}
    for config_dir in sorted(output_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        vals = []
        for txt in sorted(config_dir.glob(f"seed*_shrink{shrink_factor}_vfrac.txt")):
            try:
                lines = txt.read_text().strip().splitlines()
                vals.append(float(lines[1].split()[0]))
            except Exception:
                pass
        if vals:
            mesh_vfracs[config_dir.name] = vals
    return mesh_vfracs


def plot_results(all_vfracs, output_dir, mesh_vfracs=None):
    """Box + strip plot of VF distributions per config, with optional mesh VF overlay."""
    names   = list(all_vfracs.keys())
    data_mc = [[v for v in all_vfracs[n] if v > 0.02] for n in names]
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(names)]

    has_mesh = bool(mesh_vfracs)
    offset   = 0.18 if has_mesh else 0.0
    width    = 0.28 if has_mesh else 0.4

    fig, ax = plt.subplots(figsize=(5, 3))

    x_mc = np.arange(1, len(names) + 1) - offset

    # MC (pre-mesh) boxes
    bp_mc = ax.boxplot(data_mc, positions=x_mc, patch_artist=True,
                       widths=width,
                       medianprops=dict(color="black", linewidth=1.5),
                       showfliers=False)
    for patch, color in zip(bp_mc["boxes"], colours):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (vals, color) in enumerate(zip(data_mc, colours)):
        jitter = np.random.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(x_mc[i] + jitter, vals,
                   color=color, edgecolors="black", linewidths=0.5,
                   zorder=5, s=30, marker='o')

    # Mesh (post-shrink) boxes
    if has_mesh:
        data_mesh = [[v for v in mesh_vfracs.get(n, []) if v > 0.02] for n in names]
        x_msh = np.arange(1, len(names) + 1) + offset
        bp_msh = ax.boxplot(data_mesh, positions=x_msh, patch_artist=True,
                            widths=width,
                            medianprops=dict(color="black", linewidth=1.5),
                            showfliers=False)
        for patch, color in zip(bp_msh["boxes"], colours):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_hatch('//')

        for i, (vals, color) in enumerate(zip(data_mesh, colours)):
            jitter = np.random.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(x_msh[i] + jitter, vals,
                       color=color, edgecolors="black", linewidths=0.5,
                       zorder=5, s=30, marker='s')

    x_pos = np.arange(1, len(FRACS_SMALL) + 1)
    if EXP_SHAKEN is not None:
        ax.scatter(x_pos, EXP_SHAKEN,   marker='D', s=60, color='black', zorder=6, label='Exp. shaken')
        ax.scatter(x_pos, EXP_UNSHAKEN, marker='*', s=60, color='gray', zorder=6, label='Exp. unshaken')

    legend_handles = ax.get_legend_handles_labels()[0]
    if has_mesh:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=6, label='MC'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=6, label='Mesh'),
        ]
    ax.legend(handles=legend_handles, fontsize=7)

    labels = [f"{int(fs*100)}\\%/{int((1-fs)*100)}\\%" for fs in FRACS_SMALL]
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Small / Large volume ratio")
    ax.set_ylabel("Volume fraction")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "vfrac_2d_comparison"
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=150)
    print(f"\nPlot saved to: {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",             type=int,   default=NUM_SEEDS,
                   help="Number of seeds to run (starting from 0)")
    p.add_argument("--seed",              type=int,   nargs='+', default=None,
                   help="Run specific seed(s), e.g. --seed 1 2 3 (overrides --seeds)")
    p.add_argument("--output_dir",        type=str,   default=str(OUTPUT_DIR))
    p.add_argument("--plot-only",         action="store_true",
                   help="Skip simulations; load results.json and plot")
    p.add_argument("--mesh",              action="store_true",
                   help="Run meshing after simulations")
    p.add_argument("--mesh-only",         action="store_true",
                   help="Skip simulations; only mesh existing packings")
    p.add_argument("--shrink_factor",     type=float, default=0.85)
    p.add_argument("--mesh_size",         type=float, default=0.02)
    # p.add_argument("--min_radius_factor", type=float, default=0.25)
    p.add_argument("--min_radius_factor", type=float, default=0.1)
    p.add_argument("--plot-mesh-only",    action="store_true",
                   help="Plot paper figure (seed0, fracs 0/50/100) and exit")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "results.json"

    if args.plot_mesh_only:
        plot_meshes_paper(output_dir, shrink_factor=args.shrink_factor)
        exit()

    if args.seed is not None:
        seeds = args.seed
    else:
        seeds = list(range(args.seeds))

    if not args.mesh_only:
        if args.plot_only:
            with open(results_file) as f:
                all_vfracs = json.load(f)
            print(f"Loaded results from {results_file}")
        else:
            all_vfracs = {}
            for name, cfg in CONFIGS.items():
                vfracs = run_batch(name, cfg, seeds, output_dir)
                all_vfracs[name] = vfracs
                if vfracs:
                    print(f"\n  {name}: mean={np.mean(vfracs):.4f} "
                          f"std={np.std(vfracs):.4f}  n={len(vfracs)}")

            with open(results_file, "w") as f:
                json.dump(all_vfracs, f, indent=2)

        mesh_vfracs = collect_mesh_vfracs(output_dir, shrink_factor=args.shrink_factor)
        plot_results(all_vfracs, output_dir, mesh_vfracs=mesh_vfracs or None)

    if args.mesh or args.mesh_only:
        print("\n" + "="*55)
        print("  Meshing packings")
        print("="*55)
        mesh_batch_friction(output_dir,
                            shrink_factor=args.shrink_factor,
                            mesh_size=args.mesh_size,
                            min_radius_factor=args.min_radius_factor)

        mesh_vfracs = collect_mesh_vfracs(output_dir, shrink_factor=args.shrink_factor)
        print(f"mesh_vfracs: {mesh_vfracs}")
        if mesh_vfracs:
            if args.mesh_only and results_file.exists():
                with open(results_file) as f:
                    all_vfracs = json.load(f)
            # if all_vfracs:
                plot_results(all_vfracs, output_dir, mesh_vfracs=mesh_vfracs)

# Invalid meshes:
#   small000_large100/seed8_shrink0.85.msh: Bottom (56) and top (57) boundaries must have equal DOFs
#   small050_large050/seed8_shrink0.85.msh: Left (69) and right (71) boundaries must have equal DOFs
#   small100_large000/seed1_shrink0.85.msh: Left (67) and right (68) boundaries must have equal DOFs
#   small100_large000/seed3_shrink0.85.msh: Left (78) and right (80) boundaries must have equal DOFs

# large: seed 2
# med: seed 4
# small: seed 9