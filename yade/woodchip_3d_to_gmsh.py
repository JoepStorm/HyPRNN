"""Convert 3D periodic YADE output to GMSH mesh using Python API.

Creates spheres for each particle, merges clumps via boolean union, and
enforces periodic mesh with gmsh.model.mesh.setPeriodic.
"""

import numpy as np
import gmsh


def load_periodic_data(npy_file):
    """Load sphere data and cell dimensions."""
    data = np.load(npy_file)  # columns: x, y, z, radius, clumpId

    meta_file = npy_file.replace('.npy', '_meta.npy')
    try:
        cell_size = np.load(meta_file)
        rve_size_x, rve_size_y, rve_size_z = cell_size[0], cell_size[1], cell_size[2]
    except FileNotFoundError:
        print(f"Warning: {meta_file} not found, assuming 1.0 x 1.0 x 1.0")
        rve_size_x, rve_size_y, rve_size_z = 1.0, 1.0, 1.0

    if data.shape[1] >= 5:
        return data[:, :5], rve_size_x, rve_size_y, rve_size_z
    else:
        print("Warning: clumpId column not found, treating each sphere as unique clump")
        extended = np.zeros((data.shape[0], 5))
        extended[:, :4] = data[:, :4]
        extended[:, 4] = np.arange(data.shape[0])
        return extended, rve_size_x, rve_size_y, rve_size_z


def add_periodic_ghosts(spheres, rve_size_x, rve_size_y, rve_size_z):
    """Add ghost copies of spheres that overlap periodic boundaries."""
    ghosts = []

    for x, y, z, r, cid in spheres:
        overlaps = {
            'x-': x - r < 0,
            'x+': x + r > rve_size_x,
            'y-': y - r < 0,
            'y+': y + r > rve_size_y,
            'z-': z - r < 0,
            'z+': z + r > rve_size_z,
        }

        # Face ghosts
        if overlaps['x-']:
            ghosts.append([x + rve_size_x, y, z, r, cid])
        if overlaps['x+']:
            ghosts.append([x - rve_size_x, y, z, r, cid])
        if overlaps['y-']:
            ghosts.append([x, y + rve_size_y, z, r, cid])
        if overlaps['y+']:
            ghosts.append([x, y - rve_size_y, z, r, cid])
        if overlaps['z-']:
            ghosts.append([x, y, z + rve_size_z, r, cid])
        if overlaps['z+']:
            ghosts.append([x, y, z - rve_size_z, r, cid])

        # Edge ghosts (12 edges)
        if overlaps['x-'] and overlaps['y-']:
            ghosts.append([x + rve_size_x, y + rve_size_y, z, r, cid])
        if overlaps['x-'] and overlaps['y+']:
            ghosts.append([x + rve_size_x, y - rve_size_y, z, r, cid])
        if overlaps['x+'] and overlaps['y-']:
            ghosts.append([x - rve_size_x, y + rve_size_y, z, r, cid])
        if overlaps['x+'] and overlaps['y+']:
            ghosts.append([x - rve_size_x, y - rve_size_y, z, r, cid])

        if overlaps['x-'] and overlaps['z-']:
            ghosts.append([x + rve_size_x, y, z + rve_size_z, r, cid])
        if overlaps['x-'] and overlaps['z+']:
            ghosts.append([x + rve_size_x, y, z - rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['z-']:
            ghosts.append([x - rve_size_x, y, z + rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['z+']:
            ghosts.append([x - rve_size_x, y, z - rve_size_z, r, cid])

        if overlaps['y-'] and overlaps['z-']:
            ghosts.append([x, y + rve_size_y, z + rve_size_z, r, cid])
        if overlaps['y-'] and overlaps['z+']:
            ghosts.append([x, y + rve_size_y, z - rve_size_z, r, cid])
        if overlaps['y+'] and overlaps['z-']:
            ghosts.append([x, y - rve_size_y, z + rve_size_z, r, cid])
        if overlaps['y+'] and overlaps['z+']:
            ghosts.append([x, y - rve_size_y, z - rve_size_z, r, cid])

        # Corner ghosts (8 corners)
        if overlaps['x-'] and overlaps['y-'] and overlaps['z-']:
            ghosts.append([x + rve_size_x, y + rve_size_y, z + rve_size_z, r, cid])
        if overlaps['x-'] and overlaps['y-'] and overlaps['z+']:
            ghosts.append([x + rve_size_x, y + rve_size_y, z - rve_size_z, r, cid])
        if overlaps['x-'] and overlaps['y+'] and overlaps['z-']:
            ghosts.append([x + rve_size_x, y - rve_size_y, z + rve_size_z, r, cid])
        if overlaps['x-'] and overlaps['y+'] and overlaps['z+']:
            ghosts.append([x + rve_size_x, y - rve_size_y, z - rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['y-'] and overlaps['z-']:
            ghosts.append([x - rve_size_x, y + rve_size_y, z + rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['y-'] and overlaps['z+']:
            ghosts.append([x - rve_size_x, y + rve_size_y, z - rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['y+'] and overlaps['z-']:
            ghosts.append([x - rve_size_x, y - rve_size_y, z + rve_size_z, r, cid])
        if overlaps['x+'] and overlaps['y+'] and overlaps['z+']:
            ghosts.append([x - rve_size_x, y - rve_size_y, z - rve_size_z, r, cid])

    if ghosts:
        return np.vstack([spheres, np.array(ghosts)])
    return spheres


def get_boundary_surfaces(rve_size_x, rve_size_y, rve_size_z, eps=1e-6):
    """Get surfaces on each boundary face, sorted by position."""
    x_neg = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, eps, rve_size_y + eps, rve_size_z + eps, dim=2)
    x_pos = gmsh.model.getEntitiesInBoundingBox(
        rve_size_x - eps, -eps, -eps, rve_size_x + eps, rve_size_y + eps, rve_size_z + eps, dim=2)
    y_neg = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, rve_size_x + eps, eps, rve_size_z + eps, dim=2)
    y_pos = gmsh.model.getEntitiesInBoundingBox(
        -eps, rve_size_y - eps, -eps, rve_size_x + eps, rve_size_y + eps, rve_size_z + eps, dim=2)
    z_neg = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, rve_size_x + eps, rve_size_y + eps, eps, dim=2)
    z_pos = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, rve_size_z - eps, rve_size_x + eps, rve_size_y + eps, rve_size_z + eps, dim=2)

    def get_surface_center(surf_tag):
        """Get the center of mass of a surface."""
        bbox = gmsh.model.getBoundingBox(2, surf_tag)
        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        cz = (bbox[2] + bbox[5]) / 2
        return cx, cy, cz

    # Sort surfaces by their center position for matching
    x_neg_sorted = sorted([t[1] for t in x_neg], key=lambda s: (get_surface_center(s)[1], get_surface_center(s)[2]))
    x_pos_sorted = sorted([t[1] for t in x_pos], key=lambda s: (get_surface_center(s)[1], get_surface_center(s)[2]))
    y_neg_sorted = sorted([t[1] for t in y_neg], key=lambda s: (get_surface_center(s)[0], get_surface_center(s)[2]))
    y_pos_sorted = sorted([t[1] for t in y_pos], key=lambda s: (get_surface_center(s)[0], get_surface_center(s)[2]))
    z_neg_sorted = sorted([t[1] for t in z_neg], key=lambda s: (get_surface_center(s)[0], get_surface_center(s)[1]))
    z_pos_sorted = sorted([t[1] for t in z_pos], key=lambda s: (get_surface_center(s)[0], get_surface_center(s)[1]))

    return x_neg_sorted, x_pos_sorted, y_neg_sorted, y_pos_sorted, z_neg_sorted, z_pos_sorted


def create_mesh(npy_file, output_msh, mesh_size=0.05, shrink_factor=1.0):
    """Create periodic GMSH mesh from YADE sphere data. Simple approach: fuse spheres, subtract from box."""
    spheres, rve_size_x, rve_size_y, rve_size_z = load_periodic_data(npy_file)

    print(f"Loaded {len(spheres)} spheres")
    print(f"RVE size: {rve_size_x:.3f} x {rve_size_y:.3f} x {rve_size_z:.3f}")
    print(f"Shrink factor: {shrink_factor}")

    # Shrink radii
    spheres[:, 3] *= shrink_factor

    # Add ghost spheres for periodic boundaries
    n_original = len(spheres)
    spheres = add_periodic_ghosts(spheres, rve_size_x, rve_size_y, rve_size_z)
    n_ghosts = len(spheres) - n_original
    print(f"Added {n_ghosts} ghost spheres for periodic boundaries")

    # Compute volume fraction
    total_volume = np.sum(4/3 * np.pi * spheres[:n_original, 3] ** 3)
    vf = total_volume / (rve_size_x * rve_size_y * rve_size_z)
    print(f"Sphere volume fraction: {vf:.4f}")

    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("rve")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    occ = gmsh.model.occ

    # Create box
    box = occ.addBox(0, 0, 0, rve_size_x, rve_size_y, rve_size_z)

    # Create all spheres
    all_sphere_tags = []
    for x, y, z, r, cid in spheres:
        all_sphere_tags.append(occ.addSphere(x, y, z, r))

    print(f"Created {len(all_sphere_tags)} spheres, cutting from box...")

    # Cut all spheres from box in one operation
    tools = [(3, s) for s in all_sphere_tags]
    result, _ = occ.cut([(3, box)], tools, removeObject=True, removeTool=False)
    matrix = [t[1] for t in result if t[0] == 3]

    # The spheres (clipped to box) are the inclusions
    # Get what remains of the spheres after clipping
    occ.synchronize()
    all_vols = gmsh.model.getEntities(dim=3)
    inclusions = [t[1] for t in all_vols if t[1] not in matrix]

    print(f"Matrix: {len(matrix)}, Inclusions: {len(inclusions)}")

    occ.synchronize()

    # Synchronize geometry
    occ.synchronize()

    # Add physical groups
    gmsh.model.addPhysicalGroup(3, matrix, name="matrix")
    gmsh.model.addPhysicalGroup(3, inclusions, name="inclusions")

    # Get boundary surfaces for periodic constraints
    x_neg, x_pos, y_neg, y_pos, z_neg, z_pos = get_boundary_surfaces(
        rve_size_x, rve_size_y, rve_size_z)

    print(f"Boundary surfaces - x-: {len(x_neg)}, x+: {len(x_pos)}, "
          f"y-: {len(y_neg)}, y+: {len(y_pos)}, z-: {len(z_neg)}, z+: {len(z_pos)}")

    # Set periodic mesh constraints (4x4 affine transformation matrices)
    translation_x = [1, 0, 0, rve_size_x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    translation_y = [1, 0, 0, 0, 0, 1, 0, rve_size_y, 0, 0, 1, 0, 0, 0, 0, 1]
    translation_z = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, rve_size_z, 0, 0, 0, 1]

    # Match surfaces on opposite boundaries
    if len(x_pos) == len(x_neg):
        for pos_surf, neg_surf in zip(x_pos, x_neg):
            gmsh.model.mesh.setPeriodic(2, [pos_surf], [neg_surf], translation_x)
        print(f"Set x-periodicity for {len(x_neg)} surface pairs")
    else:
        print(f"Warning: x- ({len(x_neg)}) and x+ ({len(x_pos)}) surface counts differ!")

    if len(y_pos) == len(y_neg):
        for pos_surf, neg_surf in zip(y_pos, y_neg):
            gmsh.model.mesh.setPeriodic(2, [pos_surf], [neg_surf], translation_y)
        print(f"Set y-periodicity for {len(y_neg)} surface pairs")
    else:
        print(f"Warning: y- ({len(y_neg)}) and y+ ({len(y_pos)}) surface counts differ!")

    if len(z_pos) == len(z_neg):
        for pos_surf, neg_surf in zip(z_pos, z_neg):
            gmsh.model.mesh.setPeriodic(2, [pos_surf], [neg_surf], translation_z)
        print(f"Set z-periodicity for {len(z_neg)} surface pairs")
    else:
        print(f"Warning: z- ({len(z_neg)}) and z+ ({len(z_pos)}) surface counts differ!")

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Save mesh
    import os
    os.makedirs(os.path.dirname(output_msh), exist_ok=True)
    gmsh.write(output_msh)
    print(f"\nSaved mesh to {output_msh}")

    # Print mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    n_elements = sum(len(tags) for tags in elem_tags)
    print(f"Mesh: {len(node_tags)} nodes, {n_elements} elements")

    gmsh.finalize()


if __name__ == "__main__":
    shrink_factor = 0.8
    seed = 0
    small_fraction = 0.0
    num_chips = 100
    small_scale = 0.5

    input_file = f"periodic_3D/depositions/sphere_coordinates_periodic_relax_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.npy"
    output_file = f"periodic_3D/meshes_shrink{shrink_factor:.1f}/pbc_mesh_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.msh"

    mesh_size = 0.05

    create_mesh(input_file, output_file, mesh_size, shrink_factor)
