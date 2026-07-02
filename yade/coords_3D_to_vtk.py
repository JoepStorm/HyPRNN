"""Convert a saved 3D sphere packing (coords_3D_*.npy) to a ParaView .vtp.

The yade deposition script (yade_woodchip_filler.py) saves the settled 3D
packing as columns [x, y, z, radius, clumpId, type] (type 0 = wood, 1 = filler).
This renders each sphere as actual geometry (a glyphed icosphere scaled by
radius), so the result opens directly in ParaView with no Glyph filter needed.
Colour by 'type' (wood/filler) or 'clumpId' in ParaView.

Usage:
    python coords_3D_to_vtk.py periodic/2D_filler/coords_3D_0.npy
    python coords_3D_to_vtk.py periodic/2D_filler/coords_3D_0.npy out.vtp
"""

import sys
import numpy as np
import pyvista as pv


def coords_to_vtp(npy_file, out_file=None, resolution=12):
    """Glyph each sphere in `npy_file` and write a .vtp."""
    data = np.load(npy_file)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected [x,y,z,radius,...] columns, got {data.shape}")

    cloud = pv.PolyData(data[:, :3])
    cloud['radius'] = data[:, 3]
    cloud['clumpId'] = data[:, 4] if data.shape[1] > 4 else np.zeros(len(data))
    cloud['type'] = data[:, 5] if data.shape[1] > 5 else np.zeros(len(data))

    # factor=2 because Sphere() has radius 0.5, so diameter 1 * radius = 2*radius.
    spheres = cloud.glyph(geom=pv.Sphere(theta_resolution=resolution,
                                         phi_resolution=resolution),
                          scale='radius', factor=2.0, orient=False)

    out_file = out_file or npy_file.replace('.npy', '.vtp')
    spheres.save(out_file)
    print(f"Wrote {len(data)} spheres → {out_file}")
    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    coords_to_vtp(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
