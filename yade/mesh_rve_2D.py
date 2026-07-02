"""Mesh 2D wet RVE packings (output of yade_woodchip_2D.py) into periodic GMSH meshes.

Reuses the convex-hull meshing pipeline from periodic_to_gmsh_convexhull.py.
Expects the batch directory structure: <output_dir>/<config_name>/seed*.npy
with matching seed*_meta.npy files alongside each packing.

Usage:
    python mesh_rve_2D.py --output_dir generated_meshes_2D_wet
    python mesh_rve_2D.py --output_dir generated_meshes_2D_wet --shrink_factor 0.9
    python mesh_rve_2D.py --input path/to/seed0.npy --output path/to/out.msh
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path
from periodic_to_gmsh_convexhull import create_mesh, compute_volume_fractions

# Tag 1 = matrix (fungi/binder), tag 2 = inclusions (woodchips)
MATERIAL_COLORS = {1: '#E8F5E9', 2: '#6D4C41'}


def mesh_file(npy_file, shrink_factor=0.85, mesh_size=0.02,
              hull_resolution=12, min_radius_factor=0.25):
    """Mesh a single packing file; returns path to .msh or None on failure."""
    npy_file = Path(npy_file)
    msh_file = npy_file.with_name(npy_file.stem + f"_shrink{shrink_factor}.msh")
    try:
        create_mesh(str(npy_file), str(msh_file),
                    mesh_size=mesh_size,
                    shrink_factor=shrink_factor,
                    hull_resolution=hull_resolution,
                    min_radius_factor=min_radius_factor)
        compute_volume_fractions(str(msh_file))
        return msh_file
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def mesh_batch_friction(output_dir, shrink_factor=0.85, mesh_size=0.02,
                        hull_resolution=12, min_radius_factor=0.25,
                        skip_existing=True):
    """Mesh all seed*.npy packings found under output_dir/*/."""
    output_dir = Path(output_dir)
    npy_files = sorted(f for f in output_dir.glob("*/seed*.npy")
                       if not f.name.endswith("_meta.npy"))

    if not npy_files:
        print(f"No seed*.npy files found under {output_dir}")
        return

    print(f"Found {len(npy_files)} packing(s) to mesh")
    succeeded, failed, skipped = 0, 0, 0

    for i, npy_file in enumerate(npy_files):
        msh_file = npy_file.with_name(npy_file.stem + f"_shrink{shrink_factor}.msh")
        rel = npy_file.relative_to(output_dir)

        if skip_existing and msh_file.exists():
            print(f"[{i+1}/{len(npy_files)}] Skipping {rel} (already meshed)")
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(npy_files)}] Meshing {rel}")
        result = mesh_file(npy_file, shrink_factor=shrink_factor,
                           mesh_size=mesh_size, hull_resolution=hull_resolution,
                           min_radius_factor=min_radius_factor)
        if result is not None:
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone: {succeeded} meshed, {skipped} skipped, {failed} failed "
          f"out of {len(npy_files)}")

    plot_meshes(output_dir, shrink_factor=shrink_factor)


def _load_mesh(msh_file):
    """Load a GMSH .msh file; return (coords, triangles, phys_tags)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(msh_file))
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    triangles, phys_tags = [], []
    for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=2):
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity)
            for etype, ntags in zip(elem_types, node_tags_list):
                if int(etype) == 2:
                    for e in ntags.reshape(-1, 3):
                        triangles.append([tag_to_idx[int(n)] for n in e])
                        phys_tags.append(phys_tag)
    gmsh.finalize()
    return coords, np.array(triangles), np.array(phys_tags)


def _read_vfrac(msh_file):
    """Read woodchip (inclusion) area fraction from the companion _vfrac.txt; returns float or None."""
    txt = Path(str(msh_file).replace('.msh', '_vfrac.txt'))
    try:
        lines = txt.read_text().strip().splitlines()
        wood_vf = float(lines[1].split()[0])   # first value = wood (inclusions, tag 2)
        return wood_vf
    except Exception:
        return None


def _draw_mesh(ax, msh_file, vf_mc=None):
    """Render a mesh into ax with material colors; returns False on failure."""
    try:
        coords, elems, tags = _load_mesh(msh_file)
        colors = [MATERIAL_COLORS.get(int(t), '#888888') for t in tags]
        pc = PolyCollection(coords[elems], facecolors=colors, edgecolors='none')
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')

        vf_mesh = _read_vfrac(msh_file)
        parts = []
        if vf_mc is not None:
            parts.append(f'MC:{vf_mc:.3f}')
        if vf_mesh is not None:
            parts.append(f'Mesh:{vf_mesh:.3f}')
        if parts:
            ax.set_title('$V_f$=[' + ', '.join(parts) + ']', fontsize=6, pad=2)
        return True
    except Exception:
        ax.text(0.5, 0.5, 'failed', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='red')
        return False


def plot_meshes(output_dir, shrink_factor=0.85):
    """Plot all meshed packings in a grid (rows = configs, cols = seeds) and save."""
    import json
    output_dir = Path(output_dir)

    results_file = output_dir / "results.json"
    all_vfracs = {}
    if results_file.exists():
        with open(results_file) as f:
            all_vfracs = json.load(f)

    # Collect msh files per config, preserving sorted config order
    config_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    mesh_groups = {}
    for config_dir in config_dirs:
        msh_files = sorted(config_dir.glob(f"seed*_shrink{shrink_factor}.msh"))
        if msh_files:
            mesh_groups[config_dir.name] = msh_files

    if not mesh_groups:
        print("No meshes found to plot")
        return

    n_rows = len(mesh_groups)
    n_cols = max(len(v) for v in mesh_groups.values())
    cell_size = 1.6
    label_width = 1.8  # left margin for row labels

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(label_width + n_cols * cell_size,
                                      n_rows * cell_size))
    # Normalise to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_cols):
        axes[0, col].set_title(f"seed {col}", fontsize=8)

    for row, (config_name, msh_files) in enumerate(mesh_groups.items()):
        # Row label: convert "small075_large025" → "75% small\n25% large"
        parts = config_name.split('_')
        label = '\n'.join(p.replace('small', 'sm ').replace('large', 'lg ') + '%'
                          for p in parts)
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0,
                                ha='right', va='center', labelpad=4)

        seed_vfracs = all_vfracs.get(config_name, [])
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(msh_files):
                vf_3d = seed_vfracs[col] if col < len(seed_vfracs) else None
                _draw_mesh(ax, msh_files[col], vf_mc=vf_3d)

    fig.subplots_adjust(left=label_width / fig.get_figwidth(),
                        right=0.98, top=0.93, bottom=0.02,
                        hspace=0.05, wspace=0.05)

    out = output_dir / "meshes_overview"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"Mesh overview saved to {out}.pdf / .png")


def plot_meshes_paper(output_dir, shrink_factor=0.85,
                      fracs=(0.0, 0.5, 1.0), seed=0):
    """Plot a single row of meshes for selected small-fraction configs (paper figure)."""
    output_dir = Path(output_dir)

    config_names = [f"small{int(fs*100):03d}_large{int((1-fs)*100):03d}" for fs in fracs]
    msh_files = []
    for name in config_names:
        msh = output_dir / name / f"seed{seed}_shrink{shrink_factor}.msh"
        if not msh.exists():
            print(f"Missing mesh: {msh}")
            return
        msh_files.append(msh)

    n = len(msh_files)
    cell_size = 2.0
    fig, axes = plt.subplots(1, n, figsize=(n * cell_size, cell_size))
    if n == 1:
        axes = [axes]

    for ax, msh in zip(axes, msh_files):
        ax.axis('off')
        _draw_mesh(ax, msh)
        ax.set_title('')  # no title
        ax.set_xmargin(0)
        ax.set_ymargin(0)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0,
                        wspace=0.1)
    out = output_dir / "meshes_paper"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    plt.close(fig)
    print(f"Paper figure saved to {out}.pdf / .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",        type=str,   default=None,
                   help="Batch directory (config/seed*.npy structure)")
    p.add_argument("--input",             type=str,   default=None,
                   help="Single .npy file to mesh")
    p.add_argument("--output",            type=str,   default=None,
                   help="Output .msh path (single-file mode only)")
    p.add_argument("--shrink_factor",     type=float, default=0.85)
    p.add_argument("--mesh_size",         type=float, default=0.02)
    p.add_argument("--hull_resolution",   type=int,   default=12)
    p.add_argument("--min_radius_factor", type=float, default=0.25)
    p.add_argument("--no-skip",           action="store_true",
                   help="Re-mesh even if .msh already exists")
    p.add_argument("--plot-only",         action="store_true",
                   help="Skip meshing; only plot existing .msh files")
    args = p.parse_args()

    if args.input:
        out = args.output or str(Path(args.input).with_suffix('')) + f"_shrink{args.shrink_factor}.msh"
        create_mesh(args.input, out, mesh_size=args.mesh_size,
                    shrink_factor=args.shrink_factor,
                    hull_resolution=args.hull_resolution,
                    min_radius_factor=args.min_radius_factor)
        compute_volume_fractions(out)
    elif args.output_dir:
        if args.plot_only:
            plot_meshes(args.output_dir, shrink_factor=args.shrink_factor)
        else:
            mesh_batch_friction(args.output_dir,
                                shrink_factor=args.shrink_factor,
                                mesh_size=args.mesh_size,
                                hull_resolution=args.hull_resolution,
                                min_radius_factor=args.min_radius_factor,
                                skip_existing=not args.no_skip)
    else:
        p.error("Provide either --output_dir (batch) or --input (single file)")
