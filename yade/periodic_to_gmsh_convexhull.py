"""Conversion: periodic YADE output to GMSH mesh using convex hulls for clumps.

Instead of creating individual circles and boolean-unioning them, this script
computes the convex hull of each clump's spheres (with radius offsets) and
creates polygon-based inclusions. Ghost surfaces for periodic boundaries are
created via occ.copy + occ.translate to guarantee geometric symmetry, ensuring
matching node counts on opposite boundaries.
After meshing, loads into DOLFINx to compute volume fractions.
"""

import os
import numpy as np
import gmsh
from scipy.spatial import ConvexHull
from dataset_yade import generate_grid
from visualize_clumps import config_to_filepath


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

    if data.shape[1] >= 4:
        return data[:, :4], rve_size_x, rve_size_y
    else:
        print("Warning: clumpId column not found, treating each sphere as unique clump")
        extended = np.zeros((data.shape[0], 4))
        extended[:, :3] = data[:, :3]
        extended[:, 3] = np.arange(data.shape[0])
        return extended, rve_size_x, rve_size_y


def find_connected_components(members, proximity_factor=2.0):
    """Split clump members into spatially connected groups.

    Two spheres are connected if their centers are within proximity_factor
    times (r1 + r2). Factor > 1 catches small spheres that belong to the
    clump but don't physically overlap.
    """
    n = len(members)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        xi, yi, ri = members[i]
        for j in range(i + 1, n):
            xj, yj, rj = members[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < (ri + rj) * proximity_factor:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(members[i])
    return list(groups.values())


def clump_convex_hull_points(centers, radii, n_pts=16):
    """Generate boundary points for a clump's convex hull."""
    points = []
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    for (cx, cy), r in zip(centers, radii):
        for a in angles:
            points.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(points)


def make_clump_surface(component, occ, hull_resolution):
    """Create a GMSH surface for a connected component of spheres."""
    if len(component) == 1:
        cx, cy, r = component[0]
        return occ.addDisk(cx, cy, 0, r, r)

    centers = np.array([(cx, cy) for cx, cy, _ in component])
    radii = np.array([r for _, _, r in component])
    points = clump_convex_hull_points(centers, radii, n_pts=hull_resolution)

    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]

    gmsh_points = []
    for px, py in hull_pts:
        gmsh_points.append(occ.addPoint(px, py, 0))

    lines = []
    n = len(gmsh_points)
    for i in range(n):
        lines.append(occ.addLine(gmsh_points[i], gmsh_points[(i + 1) % n]))

    loop = occ.addCurveLoop(lines)
    return occ.addPlaneSurface([loop])


def clump_needs_ghost(component, rve_size_x, rve_size_y):
    """Determine which periodic translations a clump component needs.

    Returns list of (dx, dy) translations for ghost copies.
    """
    translations = []
    for x, y, r in component:
        overlaps_left = x - r < 0
        overlaps_right = x + r > rve_size_x
        overlaps_bottom = y - r < 0
        overlaps_top = y + r > rve_size_y

        if overlaps_left:
            translations.append((rve_size_x, 0))
        if overlaps_right:
            translations.append((-rve_size_x, 0))
        if overlaps_bottom:
            translations.append((0, rve_size_y))
        if overlaps_top:
            translations.append((0, -rve_size_y))
        if overlaps_left and overlaps_bottom:
            translations.append((rve_size_x, rve_size_y))
        if overlaps_left and overlaps_top:
            translations.append((rve_size_x, -rve_size_y))
        if overlaps_right and overlaps_bottom:
            translations.append((-rve_size_x, rve_size_y))
        if overlaps_right and overlaps_top:
            translations.append((-rve_size_x, -rve_size_y))

    # Deduplicate translations
    seen = set()
    unique = []
    for t in translations:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


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
        bbox = gmsh.model.getBoundingBox(1, curve_tag)
        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        return cx, cy

    left_sorted = sorted([t[1] for t in left], key=lambda c: get_curve_center(c)[1])
    right_sorted = sorted([t[1] for t in right], key=lambda c: get_curve_center(c)[1])
    bottom_sorted = sorted([t[1] for t in bottom], key=lambda c: get_curve_center(c)[0])
    top_sorted = sorted([t[1] for t in top], key=lambda c: get_curve_center(c)[0])

    return left_sorted, right_sorted, bottom_sorted, top_sorted


def create_mesh(npy_file, output_msh, mesh_size=0.02, shrink_factor=1.0,
                hull_resolution=12, min_radius_factor=0.5):
    """Create periodic GMSH mesh using convex hulls for clumps.

    Ghost surfaces are created via occ.copy + occ.translate from the original
    clump geometry, guaranteeing identical curves on opposite boundaries.
    """
    spheres, rve_size_x, rve_size_y = load_periodic_data(npy_file)

    print(f"Loaded {len(spheres)} spheres")
    print(f"RVE size: {rve_size_x:.3f} x {rve_size_y:.3f}")
    print(f"Shrink factor: {shrink_factor}")

    # Shrink radii
    spheres[:, 2] *= shrink_factor

    # Filter out tiny spheres that would create sub-mesh-size features
    min_radius = min_radius_factor * mesh_size
    mask = spheres[:, 2] >= min_radius
    n_filtered = np.sum(~mask)
    spheres = spheres[mask]
    if n_filtered > 0:
        print(f"Filtered {n_filtered} spheres with radius < {min_radius:.4f}")

    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("rve")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    occ = gmsh.model.occ

    # Create background rectangle
    rect = occ.addRectangle(0, 0, 0, rve_size_x, rve_size_y)

    # Group spheres by clump ID (original spheres only, no ghosts yet)
    # Spheres with clumpId == -1 are standalone; assign each a unique ID.
    clumps = {}
    next_standalone_id = int(spheres[:, 3].max()) + 1
    for x, y, r, cid in spheres:
        clump_id = int(cid)
        if clump_id == -1:
            clump_id = next_standalone_id
            next_standalone_id += 1
        if clump_id not in clumps:
            clumps[clump_id] = []
        clumps[clump_id].append((x, y, r))

    print(f"{len(spheres)} spheres in {len(clumps)} clumps")

    # Create surfaces for each clump component, then add ghost copies
    # via occ.copy + occ.translate for exact geometric symmetry.
    all_surfaces = []
    n_ghosts = 0

    for cid, members in clumps.items():
        components = find_connected_components(members)
        for component in components:
            surf = make_clump_surface(component, occ, hull_resolution)
            all_surfaces.append(surf)

            # Create ghost copies for boundary-crossing components
            translations = clump_needs_ghost(component, rve_size_x, rve_size_y)
            for dx, dy in translations:
                copied = occ.copy([(2, surf)])
                occ.translate(copied, dx, dy, 0)
                all_surfaces.append(copied[0][1])
                n_ghosts += 1

    print(f"Created {len(all_surfaces)} surfaces ({len(all_surfaces) - n_ghosts} original, {n_ghosts} ghost copies)")

    # Union all clumps (handles overlapping clumps)
    if len(all_surfaces) == 1:
        inclusions = all_surfaces
    else:
        obj = [(2, all_surfaces[0])]
        tools = [(2, s) for s in all_surfaces[1:]]
        result, _ = occ.fuse(obj, tools, removeObject=True, removeTool=True)
        inclusions = [t[1] for t in result if t[0] == 2]

    print(f"After union: {len(inclusions)} inclusion surface(s)")

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

    # Add physical groups (tag 1 = matrix, tag 2 = inclusions)
    gmsh.model.addPhysicalGroup(2, matrix, tag=1, name="matrix")
    gmsh.model.addPhysicalGroup(2, clipped, tag=2, name="inclusions")

    # Get boundary curves for periodic constraints
    left, right, bottom, top = get_boundary_curves(rve_size_x, rve_size_y)

    print(f"Boundary curves - left: {len(left)}, right: {len(right)}, "
          f"bottom: {len(bottom)}, top: {len(top)}")

    # Set periodic mesh constraints
    translation_x = [1, 0, 0, rve_size_x, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    translation_y = [1, 0, 0, 0, 0, 1, 0, rve_size_y, 0, 0, 1, 0, 0, 0, 0, 1]

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


def compute_volume_fractions(msh_file):
    """Compute area fractions from mesh element areas (via gmsh) and save to _vfrac.txt."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(msh_file)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    areas = {}
    for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=2):
        tag_area = 0.0
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity)
            for etype, ntags in zip(elem_types, node_tags_list):
                if int(etype) == 2:   # triangles
                    tri = ntags.reshape(-1, 3)
                    v = coords[[tag_to_idx[int(n)] for n in tri.ravel()]].reshape(-1, 3, 2)
                    tag_area += 0.5 * np.abs(
                        (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                        - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1])
                    ).sum()
        areas[phys_tag] = tag_area

    gmsh.finalize()

    total_area = sum(areas.values())
    for tag, area in areas.items():
        name = "matrix" if tag == 1 else "inclusions" if tag == 2 else f"tag_{tag}"
        print(f"  {name} (tag {tag}): area = {area:.6f}, fraction = {area / total_area:.4f}")
    print(f"  Total area: {total_area:.6f}")

    txt_file = msh_file.replace('.msh', '_vfrac.txt')
    with open(txt_file, 'w') as f:
        f.write("wood fungi\n")
        f.write(f"{areas.get(2, 0.0)} {areas.get(1, 0.0)}\n")
    print(f"Saved area fractions to {txt_file}")


def mesh_batch(r1_values, r2_values, nz_values, seed, shrink_factor,
               data_dir, output_dir, mesh_size=0.02, min_radius_factor=0.25):
    """Mesh all valid (r1, r2, nz) combinations from the dataset grid."""
    configs = generate_grid(r1_values, r2_values, nz_values)

    os.makedirs(output_dir, exist_ok=True)

    succeeded, failed = 0, 0
    for i, (r1, r2, nz) in enumerate(configs):
        input_file = config_to_filepath(r1, r2, nz, seed, data_dir)
        if not os.path.exists(input_file):
            print(f"[{i+1}/{len(configs)}] Skipping r1={r1:.2f} r2={r2:.2f} nz={nz} (file not found)")
            failed += 1
            continue

        output_file = f"{output_dir}/rve_r1_{r1:.2f}_r2_{r2:.2f}_nz_{nz}_{seed}_shrink{shrink_factor}.msh"
        if os.path.exists(output_file):
            print(f"[{i+1}/{len(configs)}] Skipping r1={r1:.2f} r2={r2:.2f} nz={nz} (already meshed)")
            succeeded += 1
            continue

        print(f"\n[{i+1}/{len(configs)}] Meshing r1={r1:.2f} r2={r2:.2f} nz={nz}")
        try:
            create_mesh(input_file, output_file, mesh_size, shrink_factor,
                        min_radius_factor=min_radius_factor)
            compute_volume_fractions(output_file)
            succeeded += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nBatch complete: {succeeded} succeeded, {failed} failed out of {len(configs)}")


if __name__ == "__main__":
    # mode = 'single'
    mode = 'batch'

    mesh_size = 0.02
    shrink_factor = 0.8
    min_radius_factor = 0.25

    if mode == "single":
        seed = 0
        chip_type = 'plate'
        num_chips = 600

        input_file = f"periodic/depositions_3d_to_2d_relaxed/coords_3Dto2D_{chip_type}_{num_chips}_{seed}.npy"
        output_file = f"periodic/meshes_conhull/dep3d2drelaxed/rve_{chip_type}_{num_chips}_{seed}_shrink{shrink_factor}.msh"

        create_mesh(input_file, output_file, mesh_size, shrink_factor, min_radius_factor=min_radius_factor)
        compute_volume_fractions(output_file)
    elif mode == "batch":
        r1 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        r2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        nz = [2, 3, 4, 5, 6, 7, 8]
        seed = 0

        data_dir = 'periodic/depositions_3d_to_2d_relaxed_cont'
        output_dir = 'periodic/meshes_conhull/dep3d2drelaxed_cont'

        mesh_batch(r1, r2, nz, seed, shrink_factor, data_dir, output_dir,
                   mesh_size=mesh_size, min_radius_factor=min_radius_factor)
