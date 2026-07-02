"""Convert settled wood chip data to GMSH mesh."""

import os

import gmsh
import numpy as np
from scipy.spatial import ConvexHull


def circles_to_polygon(xy, radii, num_points_per_circle=16):
    """Approximate a clump of circles as a single polygon using convex hull."""
    angles = np.linspace(0, 2 * np.pi, num_points_per_circle, endpoint=False)
    all_points = []
    for (x, y), r in zip(xy, radii):
        all_points.append(np.column_stack([
            x + r * np.cos(angles),
            y + r * np.sin(angles),
        ]))
    all_points = np.vstack(all_points)
    hull = ConvexHull(all_points)
    return all_points[hull.vertices]


def load_clumps(npy_file):
    """Load sphere data and group by clump ID. Returns list of (xy, radii) per clump."""
    data = np.load(npy_file)  # columns: x, y, radius, clumpId
    clump_ids = np.unique(data[:, 3]).astype(int)
    clumps = []
    for cid in clump_ids:
        mask = data[:, 3].astype(int) == cid
        xy = data[mask, :2]
        radii = data[mask, 2]
        clumps.append((xy, radii))
    return clumps


def create_gmsh_mesh_with_chips(npy_file, output_msh, mesh_size=0.01):
    """Create a 2D GMSH mesh with chips as separate physical regions."""
    clumps = load_clumps(npy_file)

    # Box matches the Yade walls: x in [0, 1], y from 0 to top of settled packing
    box_width = 1.0
    all_data = np.load(npy_file)
    box_height = np.max(all_data[:, 1] + all_data[:, 2])  # max(y + r)

    gmsh.initialize()
    gmsh.model.add("woodchips")

    # Create the outer box
    box = gmsh.model.occ.addRectangle(0, 0, 0, box_width, box_height)

    # Create each wood chip as a polygon
    chip_surfaces = []
    for xy, radii in clumps:
        poly_points = circles_to_polygon(xy, radii)

        point_tags = []
        for px, py in poly_points:
            pt = gmsh.model.occ.addPoint(px, py, 0, mesh_size)
            point_tags.append(pt)

        line_tags = []
        for i in range(len(point_tags)):
            l = gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
            line_tags.append(l)

        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        chip_surfaces.append((2, surf))

    # Fragment (boolean intersection preserving all regions)
    if chip_surfaces:
        all_objects = [(2, box)] + chip_surfaces
        result, result_map = gmsh.model.occ.fragment(all_objects, [])

    gmsh.model.occ.synchronize()

    # Identify surfaces: chips are smaller, matrix is larger
    all_surfaces = gmsh.model.getEntities(2)

    chip_tags = []
    matrix_tags = []

    for dim, tag in all_surfaces:
        mass = gmsh.model.occ.getMass(dim, tag)
        # Chips are small polygons, matrix is the large remaining area
        if mass < 0.01:  # threshold for chip size
            chip_tags.append(tag)
        else:
            matrix_tags.append(tag)

    # Add physical groups
    if matrix_tags:
        gmsh.model.addPhysicalGroup(2, matrix_tags, tag=1, name="fungi")
    if chip_tags:
        gmsh.model.addPhysicalGroup(2, chip_tags, tag=2, name="wood")

    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 3)

    # Write geometry file before meshing
    geo_path = os.path.splitext(output_msh)[0] + ".geo_unrolled"
    gmsh.write(geo_path)
    print(f"Geometry saved to {geo_path}")

    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh)
    print(f"Mesh saved to {output_msh}")

    gmsh.finalize()


if __name__ == "__main__":
    import sys

    npy_file = sys.argv[1] if len(sys.argv) > 1 else "sphere_coordinates.npy"
    output_msh = sys.argv[2] if len(sys.argv) > 2 else "woodchip_mesh.msh"

    create_gmsh_mesh_with_chips(npy_file, output_msh)
