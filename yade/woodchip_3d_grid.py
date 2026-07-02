"""Convert 3D periodic YADE output to DOLFINx mesh using a structured grid.

Instead of boolean operations, creates a regular grid and tags cells based on
whether their centroid is inside an inclusion sphere.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
import basix


def load_periodic_data(npy_file):
    """Load sphere data and cell dimensions."""
    data = np.load(npy_file)  # columns: x, y, z, radius, clumpId

    meta_file = npy_file.replace('.npy', '_meta.npy')
    try:
        cell_size = np.load(meta_file)
        rve_size = cell_size[0], cell_size[1], cell_size[2]
    except FileNotFoundError:
        print(f"Warning: {meta_file} not found, assuming 1.0 x 1.0 x 1.0")
        rve_size = 1.0, 1.0, 1.0

    return data[:, :5], rve_size


def point_in_sphere(points, sphere):
    """Check if points are inside a sphere (with periodic wrapping)."""
    x, y, z, r, _ = sphere
    center = np.array([x, y, z])
    dist_sq = np.sum((points - center)**2, axis=1)
    return dist_sq < r**2


def point_in_any_sphere_periodic(points, spheres, rve_size):
    """Check if points are inside any sphere, accounting for periodicity."""
    Lx, Ly, Lz = rve_size
    inside = np.zeros(len(points), dtype=bool)

    for sphere in spheres:
        x, y, z, r, _ = sphere
        # Check original and periodic images
        for dx in [-Lx, 0, Lx]:
            for dy in [-Ly, 0, Ly]:
                for dz in [-Lz, 0, Lz]:
                    shifted_sphere = [x + dx, y + dy, z + dz, r, 0]
                    inside |= point_in_sphere(points, shifted_sphere)

    return inside


def create_grid_mesh(npy_file, output_xdmf, n_cells=(20, 20, 20), shrink_factor=1.0):
    """Create structured grid mesh with cell tags based on sphere positions."""
    spheres, rve_size = load_periodic_data(npy_file)
    Lx, Ly, Lz = rve_size
    nx, ny, nz = n_cells

    print(f"Loaded {len(spheres)} spheres")

    # Shrink radii
    if shrink_factor != 1.0:
        spheres = spheres.copy()
        spheres[:, 3] *= shrink_factor
        print(f"Shrink factor: {shrink_factor}")
    print(f"RVE size: {Lx:.3f} x {Ly:.3f} x {Lz:.3f}")
    print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells")

    # Create structured mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )

    # Compute cell centroids
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local

    # Get cell midpoints
    cell_midpoints = mesh.compute_midpoints(domain, tdim, np.arange(num_cells))

    # Determine which cells are inside inclusions
    inside_inclusion = point_in_any_sphere_periodic(cell_midpoints, spheres, rve_size)

    # Create cell tags: 1 = matrix, 2 = inclusion
    cell_values = np.where(inside_inclusion, 2, 1).astype(np.int32)
    cell_indices = np.arange(num_cells, dtype=np.int32)

    cell_tags = mesh.meshtags(domain, tdim, cell_indices, cell_values)
    cell_tags.name = "Cell tags"

    # Compute volume fractions
    n_matrix = np.sum(cell_values == 1)
    n_inclusion = np.sum(cell_values == 2)
    vf_inclusion = n_inclusion / num_cells
    vf_matrix = n_matrix / num_cells

    print(f"Matrix cells: {n_matrix} ({vf_matrix*100:.2f}%)")
    print(f"Inclusion cells: {n_inclusion} ({vf_inclusion*100:.2f}%)")

    # Create DG0 function for visualization
    V = fem.functionspace(domain, ("DG", 0))
    material = fem.Function(V, name="material")
    material.x.array[:] = cell_values

    # Save to XDMF
    import os
    os.makedirs(os.path.dirname(output_xdmf), exist_ok=True)

    domain.topology.create_connectivity(tdim, tdim)

    with XDMFFile(domain.comm, output_xdmf, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags, domain.geometry)
        xdmf.write_function(material)

    # Save info file
    info_file = output_xdmf.replace('.xdmf', '_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"RVE size: {Lx:.6f} x {Ly:.6f} x {Lz:.6f}\n")
        f.write(f"Grid: {nx} x {ny} x {nz}\n")
        f.write(f"Total cells: {num_cells}\n")
        f.write(f"Matrix cells: {n_matrix}\n")
        f.write(f"Inclusion cells: {n_inclusion}\n")
        f.write(f"Volume fraction (inclusions): {vf_inclusion:.6f}\n")
        f.write(f"Volume fraction (matrix): {vf_matrix:.6f}\n")

    print(f"Saved to {output_xdmf}")
    print(f"Info saved to {info_file}")


if __name__ == "__main__":
    seed = 0
    small_fraction = 0.0
    num_chips = 20
    small_scale = 0.5
    shrink_factor = 0.85

    # Grid resolution - adjust for accuracy vs speed
    n_cells = (50, 50, 50)

    input_file = f"periodic_3D/depositions/sphere_coordinates_periodic_relax_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.npy"
    output_file = f"periodic_3D/meshes/rve_grid_{small_fraction:.1f}_{num_chips}_{small_scale}_{n_cells[0]}_{seed}.xdmf"

    create_grid_mesh(input_file, output_file, n_cells=n_cells, shrink_factor=shrink_factor)
