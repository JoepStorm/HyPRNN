"""Mesh 2D RVE packings that contain filler (output of deposit_rve.py).

Filler occupied space during DEM (holding voids open) but is treated as void
here: only wood spheres become inclusions (tag 2). Everything else — including
the regions the filler held open — falls into the matrix (tag 1) automatically,
because the convex-hull pipeline defines the matrix as "the cell minus the wood".

Settings:
    shrink_factor:  default=0.85. How much the convex hulls are shrunk.
    mesh_size:  default=0.02.
    hull_resolution:  default=12.   Influences nodes on circles.
    min_radius_factor:  default=0.25.  Very small spheres below this min_radius_factor are ignored.

Note: gmsh does always produce periodic meshes, even from a periodic geometry.
      This is not handled properly. A different geometry with new seed is used by manually replacing the file.

Usage: set
    python packing_to_mesh.py --input path/to/coords_2D_0_500_0.5.npy
    python packing_to_mesh.py --input path/to/in.npy --output path/to/out.msh
"""

import argparse
import shutil
import numpy as np
from pathlib import Path

from mesh_utils import create_mesh, compute_volume_fractions

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


def mesh_file(npy_file, output=None, shrink_factor=0.85, mesh_size=0.02,
              hull_resolution=12, min_radius_factor=0.25):
    """Strip filler and mesh a single packing; returns the .msh path."""
    npy_file = Path(npy_file)
    msh_file = Path(output) if output else npy_file.with_name(
        npy_file.stem + f"_shrink{shrink_factor}.msh")
    wood_npy = strip_filler(npy_file)
    try:
        create_mesh(str(wood_npy), str(msh_file),
                    mesh_size=mesh_size,
                    shrink_factor=shrink_factor,
                    hull_resolution=hull_resolution,
                    min_radius_factor=min_radius_factor)
        compute_volume_fractions(str(msh_file))
        return msh_file
    finally:
        wood_npy.unlink(missing_ok=True)
        wood_npy.with_name(wood_npy.stem + "_meta.npy").unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",             type=str,   required=True,
                   help=".npy packing file to mesh")
    p.add_argument("--output",            type=str,   default=None,
                   help="Output .msh path (default: alongside input)")
    p.add_argument("--shrink_factor",     type=float, default=0.85)
    p.add_argument("--mesh_size",         type=float, default=0.02)
    p.add_argument("--hull_resolution",   type=int,   default=12)
    p.add_argument("--min_radius_factor", type=float, default=0.25)
    args = p.parse_args()

    mesh_file(args.input, output=args.output,
              shrink_factor=args.shrink_factor,
              mesh_size=args.mesh_size,
              hull_resolution=args.hull_resolution,
              min_radius_factor=args.min_radius_factor)
