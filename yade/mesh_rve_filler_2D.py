"""Mesh 2D RVE packings that contain filler (output of yade_woodchip_filler.py).

The packing .npy has a 5th column tagging each sphere: 0 = wood, 1 = filler.
Filler occupied space during DEM (holding voids open) but is treated as void
here: only wood spheres become inclusions (tag 2). Everything else — including
the regions the filler held open — falls into the matrix (tag 1) automatically,
because the convex-hull pipeline defines the matrix as "the cell minus the wood".

We simply strip the filler rows, write a wood-only temporary .npy, and reuse the
existing meshing pipeline unchanged.

Usage:
    python mesh_rve_filler_2D.py --output_dir generated_meshes_2D_filler
    python mesh_rve_filler_2D.py --input path/to/seed0.npy --output path/to/out.msh
"""

import argparse
import shutil
import numpy as np
from pathlib import Path

from periodic_to_gmsh_convexhull import create_mesh, compute_volume_fractions
from mesh_rve_2D import plot_meshes   # reuse the overview-grid plotting

WOOD_TAG   = 0
FILLER_TAG = 1


def strip_filler(npy_file):
    """Write a wood-only copy of a filler packing to a temp .npy (+ meta).

    Returns the temp .npy path. Filler spheres (type 1) are dropped, so the
    space they held becomes matrix/void in the resulting mesh.
    """
    npy_file = Path(npy_file)
    data = np.load(npy_file)

    if data.shape[1] >= 5:
        is_filler = data[:, 4] == FILLER_TAG
        wood = data[~is_filler][:, :4]
        print(f"  Dropped {int(is_filler.sum())} filler spheres (-> void), "
              f"kept {len(wood)} wood spheres")
    else:
        print("  No type column found; treating all spheres as wood")
        wood = data[:, :4]

    tmp = npy_file.with_name(npy_file.stem + "_woodonly.npy")
    np.save(tmp, wood)

    # create_mesh derives the meta path by replacing '.npy' -> '_meta.npy'
    meta = npy_file.with_name(npy_file.stem + "_meta.npy")
    if meta.exists():
        shutil.copy(meta, tmp.with_name(tmp.stem + "_meta.npy"))
    return tmp


def mesh_file(npy_file, shrink_factor=0.85, mesh_size=0.02,
              hull_resolution=12, min_radius_factor=0.25):
    """Mesh a single filler packing; returns path to .msh or None on failure."""
    npy_file = Path(npy_file)
    msh_file = npy_file.with_name(npy_file.stem + f"_shrink{shrink_factor}.msh")
    wood_npy = strip_filler(npy_file)
    try:
        create_mesh(str(wood_npy), str(msh_file),
                    mesh_size=mesh_size,
                    shrink_factor=shrink_factor,
                    hull_resolution=hull_resolution,
                    min_radius_factor=min_radius_factor)
        compute_volume_fractions(str(msh_file))
        return msh_file
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    finally:
        wood_npy.unlink(missing_ok=True)
        wood_npy.with_name(wood_npy.stem + "_meta.npy").unlink(missing_ok=True)


def mesh_batch(output_dir, shrink_factor=0.85, mesh_size=0.02,
               hull_resolution=12, min_radius_factor=0.25, skip_existing=True):
    """Mesh all seed*.npy filler packings found under output_dir/*/."""
    output_dir = Path(output_dir)
    npy_files = sorted(f for f in output_dir.glob("*/seed*.npy")
                       if not f.name.endswith("_meta.npy")
                       and "_woodonly" not in f.name)

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
        npy_file = Path(args.input)
        out = args.output or str(npy_file.with_suffix('')) + f"_shrink{args.shrink_factor}.msh"
        wood_npy = strip_filler(npy_file)
        try:
            create_mesh(str(wood_npy), out, mesh_size=args.mesh_size,
                        shrink_factor=args.shrink_factor,
                        hull_resolution=args.hull_resolution,
                        min_radius_factor=args.min_radius_factor)
            compute_volume_fractions(out)
        finally:
            wood_npy.unlink(missing_ok=True)
            wood_npy.with_name(wood_npy.stem + "_meta.npy").unlink(missing_ok=True)
    elif args.output_dir:
        if args.plot_only:
            plot_meshes(args.output_dir, shrink_factor=args.shrink_factor)
        else:
            mesh_batch(args.output_dir,
                       shrink_factor=args.shrink_factor,
                       mesh_size=args.mesh_size,
                       hull_resolution=args.hull_resolution,
                       min_radius_factor=args.min_radius_factor,
                       skip_existing=not args.no_skip)
    else:
        p.error("Provide either --output_dir (batch) or --input (single file)")
