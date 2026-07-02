"""Simple conversion: periodic YADE output to GMSH mesh using Python API.

Creates circles for each sphere, merges clumps via boolean union, and
enforces periodic mesh with gmsh.model.mesh.setPeriodic.
"""

import numpy as np
import gmsh



def load_periodic_data(npy_file):
    """Load sphere data and cell dimensions."""
    data = np.load(npy_file)  # columns: x, y, radius, clumpId

    meta_file = npy_file.replace('.npy', '_meta.npy')
    try:
        cell_size = np.load(meta_file)
        rve_size_x, rve_size_y = cell_size[0], cell_size[1]
    except FileNotFoundError:
        print(f"Warning: {meta_file} not found, assuming 1.0 x 1.0")
        rve_size_x, rve_size_y = 1.0, 1.0

    # Ensure we return x, y, radius, clumpId
    if data.shape[1] >= 4:
        return data[:, :4], rve_size_x, rve_size_y
    else:
        print("Warning: clumpId column not found, treating each sphere as unique clump")
        extended = np.zeros((data.shape[0], 4))
        extended[:, :3] = data[:, :3]
        extended[:, 3] = np.arange(data.shape[0])
        return extended, rve_size_x, rve_size_y


def add_periodic_ghosts(spheres, rve_size_x, rve_size_y):
    """Add ghost copies of spheres that overlap periodic boundaries."""
    ghosts = []

    for x, y, r, cid in spheres:
        overlaps_left = x - r < 0
        overlaps_right = x + r > rve_size_x
        overlaps_bottom = y - r < 0
        overlaps_top = y + r > rve_size_y

        if overlaps_left:
            ghosts.append([x + rve_size_x, y, r, cid])
        if overlaps_right:
            ghosts.append([x - rve_size_x, y, r, cid])
        if overlaps_bottom:
            ghosts.append([x, y + rve_size_y, r, cid])
        if overlaps_top:
            ghosts.append([x, y - rve_size_y, r, cid])

        # Diagonal ghosts for corner overlaps
        if overlaps_left and overlaps_bottom:
            ghosts.append([x + rve_size_x, y + rve_size_y, r, cid])
        if overlaps_left and overlaps_top:
            ghosts.append([x + rve_size_x, y - rve_size_y, r, cid])
        if overlaps_right and overlaps_bottom:
            ghosts.append([x - rve_size_x, y + rve_size_y, r, cid])
        if overlaps_right and overlaps_top:
            ghosts.append([x - rve_size_x, y - rve_size_y, r, cid])

    if ghosts:
        return np.vstack([spheres, np.array(ghosts)])
    return spheres


def get_boundary_curves(rve_size_x, rve_size_y, eps=1e-6):
    """Get curves on each boundary edge, sorted by position."""
    left = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, eps, rve_size_y + eps, eps, dim=1)
    right = gmsh.model.getEntitiesInBoundingBox(
        rve_size_x - eps, -eps, -eps, rve_size_x + eps, rve_size_y + eps, eps, dim=1)
    bottom = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, rve_size_x + eps, eps, eps, dim=1)
    top = gmsh.model.getEntitiesInBoundingBox(
        -eps, rve_size_y - eps, -eps, rve_size_x + eps, rve_size_y + eps, eps, dim=1)

    def get_curve_center(curve_tag):
        """Get the center of mass of a curve."""
        bbox = gmsh.model.getBoundingBox(1, curve_tag)
        # bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        return cx, cy

    # Sort curves by their center position for matching
    left_sorted = sorted([t[1] for t in left], key=lambda c: get_curve_center(c)[1])
    right_sorted = sorted([t[1] for t in right], key=lambda c: get_curve_center(c)[1])
    bottom_sorted = sorted([t[1] for t in bottom], key=lambda c: get_curve_center(c)[0])
    top_sorted = sorted([t[1] for t in top], key=lambda c: get_curve_center(c)[0])

    return left_sorted, right_sorted, bottom_sorted, top_sorted


def create_mesh(npy_file, output_msh, mesh_size=0.02, shrink_factor=1.0):
    """Create periodic GMSH mesh from YADE sphere data."""
    spheres, rve_size_x, rve_size_y = load_periodic_data(npy_file)

    print(f"Loaded {len(spheres)} spheres")
    print(f"RVE size: {rve_size_x:.3f} x {rve_size_y:.3f}")
    print(f"Shrink factor: {shrink_factor}")

    # Shrink radii
    spheres[:, 2] *= shrink_factor

    # Add ghost spheres for periodic boundaries
    n_original = len(spheres)
    spheres = add_periodic_ghosts(spheres, rve_size_x, rve_size_y)
    n_ghosts = len(spheres) - n_original
    print(f"Added {n_ghosts} ghost spheres for periodic boundaries")

    # Compute volume fraction
    total_area = np.sum(np.pi * spheres[:n_original, 2] ** 2)
    vf = total_area / (rve_size_x * rve_size_y)
    print(f"Circle area fraction: {vf:.4f}")

    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("rve")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    # Prevent small elements in tight concave regions
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    occ = gmsh.model.occ

    # Create background rectangle
    rect = occ.addRectangle(0, 0, 0, rve_size_x, rve_size_y)

    # Group spheres by clump ID and create disks
    clumps = {}
    for x, y, r, cid in spheres:
        clump_id = int(cid)
        disk = occ.addDisk(x, y, 0, r, r)
        if clump_id not in clumps:
            clumps[clump_id] = []
        clumps[clump_id].append(disk)

    print(f"Created {len(spheres)} disks in {len(clumps)} clumps")

    # Union disks within each clump
    clump_surfaces = []
    for cid, disks in clumps.items():
        if len(disks) == 1:
            clump_surfaces.append(disks[0])
        else:
            obj = [(2, disks[0])]
            tools = [(2, d) for d in disks[1:]]
            result, _ = occ.fuse(obj, tools, removeObject=True, removeTool=True)
            clump_surfaces.extend([t[1] for t in result if t[0] == 2])

    print(f"After clump unions: {len(clump_surfaces)} surfaces")

    # Union all clumps together
    if len(clump_surfaces) == 1:
        inclusions = clump_surfaces
    else:
        obj = [(2, clump_surfaces[0])]
        tools = [(2, s) for s in clump_surfaces[1:]]
        result, _ = occ.fuse(obj, tools, removeObject=True, removeTool=True)
        inclusions = [t[1] for t in result if t[0] == 2]

    print(f"After global union: {len(inclusions)} inclusion surface(s)")

    # Clip inclusions to RVE boundary
    clip_rect = occ.addRectangle(0, 0, 0, rve_size_x, rve_size_y)
    obj = [(2, s) for s in inclusions]
    tools = [(2, clip_rect)]
    result, _ = occ.intersect(obj, tools, removeObject=True, removeTool=True)
    clipped = [t[1] for t in result if t[0] == 2]

    print(f"After clipping: {len(clipped)} inclusion surface(s)")

    # Subtract inclusions from rectangle to get matrix
    obj = [(2, rect)]
    tools = [(2, s) for s in clipped]
    result, _ = occ.cut(obj, tools, removeObject=True, removeTool=False)
    matrix = [t[1] for t in result if t[0] == 2]

    print(f"Matrix surfaces: {len(matrix)}")

    # Synchronize geometry
    occ.synchronize()

    # Add physical groups
    gmsh.model.addPhysicalGroup(2, matrix, name="matrix")
    gmsh.model.addPhysicalGroup(2, clipped, name="inclusions")

    # Get boundary curves for periodic constraints
    left, right, bottom, top = get_boundary_curves(rve_size_x, rve_size_y)

    print(f"Boundary curves - left: {len(left)}, right: {len(right)}, "
          f"bottom: {len(bottom)}, top: {len(top)}")

    # Set periodic mesh constraints
    translation_x = [1, 0, 0, rve_size_x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    translation_y = [1, 0, 0, 0, 0, 1, 0, rve_size_y, 0, 0, 1, 0, 0, 0, 0, 1]

    # Match curves on opposite boundaries
    # Curves should already be sorted by position, so matching by index
    if len(right) == len(left):
        for r_curve, l_curve in zip(right, left):
            gmsh.model.mesh.setPeriodic(1, [r_curve], [l_curve], translation_x)
        print(f"Set x-periodicity for {len(left)} curve pairs")
    else:
        print(f"Warning: left ({len(left)}) and right ({len(right)}) curve counts differ!")

    if len(top) == len(bottom):
        for t_curve, b_curve in zip(top, bottom):
            gmsh.model.mesh.setPeriodic(1, [t_curve], [b_curve], translation_y)
        print(f"Set y-periodicity for {len(bottom)} curve pairs")
    else:
        print(f"Warning: bottom ({len(bottom)}) and top ({len(top)}) curve counts differ!")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Save mesh
    gmsh.write(output_msh)
    print(f"\nSaved mesh to {output_msh}")

    # Print mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    n_elements = sum(len(tags) for tags in elem_tags)
    print(f"Mesh: {len(node_tags)} nodes, {n_elements} elements")

    gmsh.finalize()


if __name__ == "__main__":
    import os
    shrink_factor = 0.9
    seed = 4
    type = 'vbulky'  # 'single' # 'bulky'  #  'slender'
    num_chips = 400 #600 * 8
    # small_scale = 0.5
    # small_fraction = 0.0
    # num_chips = 400
    # small_fraction = 0.8
    # num_chips = 600
    # small_fraction = 0.0
    # num_chips = 400 #2000
    # small_scale = 0.5

    # clump_shape = [1, 9]
    # name = f'_spheres{clump_shape[0]}x{clump_shape[1]}_{num_chips}_{seed}'

    # input_file = f"aspect_ratio/depositions/coords{name}.npy"
    # output_file = f"aspect_ratio/meshes/mesh{name}_{shrink_factor:.2f}.msh"

    # input_file = f"periodic/depositions/coords_3Dto2D_bulky_600_1.npy"
    # output_file = f"periodic/meshes/rvev3_bulky_600.msh"

    input_file = f"periodic/depositions_3d_to_2d/coords_3Dto2D_{type}_{num_chips}_{seed}_5000.npy"
    output_file = f"periodic/meshes/rvev3_{type}_{num_chips}_{seed}_shrink{shrink_factor}_5000.msh"

    # input_file = f"periodic/depositions/sphere_coordinates_periodic_relax_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.npy"
    # output_file = f"periodic/meshes_shrink{shrink_factor:.1f}/pbc_mesh_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.msh"

    mesh_size = 0.02

    create_mesh(input_file, output_file, mesh_size, shrink_factor)
