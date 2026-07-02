"""Convert periodic wood chip packing to GMSH .geo file with periodic mesh."""

from string import Template
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
    """Load sphere data and group by clump ID."""
    data = np.load(npy_file)
    clump_ids = np.unique(data[:, 3]).astype(int)
    clumps = []
    for cid in clump_ids:
        mask = data[:, 3].astype(int) == cid
        xy = data[mask, :2]
        radii = data[mask, 2]
        clumps.append((xy, radii))
    return clumps


def clump_centroid(xy, radii):
    """Compute centroid weighted by circle area."""
    areas = np.pi * radii**2
    total_area = np.sum(areas)
    cx = np.sum(xy[:, 0] * areas) / total_area
    cy = np.sum(xy[:, 1] * areas) / total_area
    return np.array([cx, cy])


def clump_bounding_box(xy, radii):
    """Get bounding box of clump."""
    x_min = (xy[:, 0] - radii).min()
    x_max = (xy[:, 0] + radii).max()
    y_min = (xy[:, 1] - radii).min()
    y_max = (xy[:, 1] + radii).max()
    return x_min, x_max, y_min, y_max


def shrink_clump(xy, radii, factor):
    """Shrink a clump by a factor around its centroid."""
    centroid = clump_centroid(xy, radii)
    new_xy = centroid + factor * (xy - centroid)
    new_radii = radii * factor
    return new_xy, new_radii


def periodize_clumps(clumps, cell_size, shrink_factor=0.9):
    """Create periodic copies of clumps that cross boundaries.

    For each clump:
    1. Shrink it
    2. Check if it crosses any boundary
    3. If so, create wrapped copies (shifted by ±cell_size)

    This ensures the geometry itself is periodic.
    """
    periodic_clumps = []

    for xy, radii in clumps:
        # Shrink first
        xy, radii = shrink_clump(xy, radii, shrink_factor)

        # Get bounding box
        x_min, x_max, y_min, y_max = clump_bounding_box(xy, radii)

        # Determine which copies we need
        # shifts will be list of (dx, dy) offsets
        shifts = [(0, 0)]  # Always include original

        crosses_left = x_min < 0
        crosses_right = x_max > cell_size
        crosses_bottom = y_min < 0
        crosses_top = y_max > cell_size

        if crosses_left:
            shifts.append((cell_size, 0))
        if crosses_right:
            shifts.append((-cell_size, 0))
        if crosses_bottom:
            shifts.append((0, cell_size))
        if crosses_top:
            shifts.append((0, -cell_size))

        # Corner cases (clump crosses two boundaries)
        if crosses_left and crosses_bottom:
            shifts.append((cell_size, cell_size))
        if crosses_left and crosses_top:
            shifts.append((cell_size, -cell_size))
        if crosses_right and crosses_bottom:
            shifts.append((-cell_size, cell_size))
        if crosses_right and crosses_top:
            shifts.append((-cell_size, -cell_size))

        for dx, dy in shifts:
            shifted_xy = xy.copy()
            shifted_xy[:, 0] += dx
            shifted_xy[:, 1] += dy
            periodic_clumps.append((shifted_xy, radii.copy()))

    print(f"Periodization: {len(clumps)} clumps -> {len(periodic_clumps)} (with periodic copies)")
    return periodic_clumps


def circles_overlap(xy1, r1, xy2, r2):
    """Check if any circle from clump1 overlaps with any circle from clump2."""
    for i in range(len(xy1)):
        for j in range(len(xy2)):
            dist = np.linalg.norm(xy1[i] - xy2[j])
            if dist < r1[i] + r2[j]:
                return True
    return False


def merge_overlapping_clumps(clumps):
    """Merge clumps that overlap into single clumps using Union-Find."""
    if not clumps:
        return clumps

    n = len(clumps)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            xy_i, radii_i = clumps[i]
            xy_j, radii_j = clumps[j]
            if circles_overlap(xy_i, radii_i, xy_j, radii_j):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_clumps = []
    for indices in groups.values():
        all_xy = np.vstack([clumps[i][0] for i in indices])
        all_radii = np.concatenate([clumps[i][1] for i in indices])
        merged_clumps.append((all_xy, all_radii))

    print(f"Merged {n} clumps into {len(merged_clumps)} clumps")
    return merged_clumps


def clip_polygon_to_rect(polygon, x_min, x_max, y_min, y_max):
    """Clip polygon to rectangle using Sutherland-Hodgman algorithm."""
    def clip_edge(poly, edge_func, inside_func):
        if len(poly) == 0:
            return []
        output = []
        for i in range(len(poly)):
            current = poly[i]
            next_pt = poly[(i + 1) % len(poly)]
            current_inside = inside_func(current)
            next_inside = inside_func(next_pt)

            if current_inside:
                output.append(current)
                if not next_inside:
                    output.append(edge_func(current, next_pt))
            elif next_inside:
                output.append(edge_func(current, next_pt))
        return output

    def intersect_left(p1, p2):
        t = (x_min - p1[0]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 0
        return [x_min, p1[1] + t * (p2[1] - p1[1])]

    def intersect_right(p1, p2):
        t = (x_max - p1[0]) / (p2[0] - p1[0]) if p2[0] != p1[0] else 0
        return [x_max, p1[1] + t * (p2[1] - p1[1])]

    def intersect_bottom(p1, p2):
        t = (y_min - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0
        return [p1[0] + t * (p2[0] - p1[0]), y_min]

    def intersect_top(p1, p2):
        t = (y_max - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0
        return [p1[0] + t * (p2[0] - p1[0]), y_max]

    poly = [list(p) for p in polygon]
    poly = clip_edge(poly, intersect_left, lambda p: p[0] >= x_min)
    poly = clip_edge(poly, intersect_right, lambda p: p[0] <= x_max)
    poly = clip_edge(poly, intersect_bottom, lambda p: p[1] >= y_min)
    poly = clip_edge(poly, intersect_top, lambda p: p[1] <= y_max)

    return np.array(poly) if poly else np.array([]).reshape(0, 2)


def compute_volume_fraction(clumps, cell_size):
    """Compute volume fraction using convex hull areas."""
    cell_area = cell_size * cell_size
    total_clump_area = 0.0

    for xy, radii in clumps:
        poly_points = circles_to_polygon(xy, radii)
        poly_points = clip_polygon_to_rect(poly_points, 0, cell_size, 0, cell_size)
        n = len(poly_points)
        if n < 3:
            continue
        area = 0.5 * abs(sum(
            poly_points[i, 0] * poly_points[(i + 1) % n, 1] -
            poly_points[(i + 1) % n, 0] * poly_points[i, 1]
            for i in range(n)
        ))
        total_clump_area += area

    return total_clump_area / cell_area


def create_periodic_geo_file(npy_file, output_geo, cell_size=1.0, mesh_size=0.01, shrink_factor=0.9):
    """Create GMSH .geo file with periodic mesh constraints."""

    clumps = load_clumps(npy_file)
    print(f"Loaded {len(clumps)} clumps")
    print(f"Cell size: {cell_size}")
    print(f"Shrink factor: {shrink_factor}")

    # Create periodic copies and shrink
    clumps = periodize_clumps(clumps, cell_size, shrink_factor)

    # Merge overlapping clumps (including periodic copies that touch)
    clumps = merge_overlapping_clumps(clumps)

    # Compute volume fraction
    vf = compute_volume_fraction(clumps, cell_size)
    print(f"Volume fraction: {vf:.4f}")

    # Build .geo file
    geo_content = f"""SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;
Mesh.CharacteristicLengthMin = {mesh_size};
Mesh.CharacteristicLengthMax = {mesh_size};

// RVE bounding box - we'll use this for periodic constraints
// Points at corners (for periodic matching)
Point(1) = {{0, 0, 0, {mesh_size}}};
Point(2) = {{{cell_size}, 0, 0, {mesh_size}}};
Point(3) = {{{cell_size}, {cell_size}, 0, {mesh_size}}};
Point(4) = {{0, {cell_size}, 0, {mesh_size}}};

// Boundary lines (for periodic constraints)
Line(1) = {{1, 2}};  // bottom
Line(2) = {{2, 3}};  // right
Line(3) = {{3, 4}};  // top
Line(4) = {{4, 1}};  // left

// Curve loop for the outer boundary
Curve Loop(1) = {{1, 2, 3, 4}};

// Clump polygons
"""

    surface_id = 2
    point_id = 100
    line_id = 100
    curve_loop_id = 100
    clump_curve_loops = []

    for xy, radii in clumps:
        poly_points = circles_to_polygon(xy, radii)
        poly_points = clip_polygon_to_rect(poly_points, 0, cell_size, 0, cell_size)

        n_pts = len(poly_points)
        if n_pts < 3:
            continue

        geo_content += f"\n// Clump {surface_id - 1}\n"

        first_point = point_id
        for px, py in poly_points:
            geo_content += f"Point({point_id}) = {{{px}, {py}, 0, {mesh_size}}};\n"
            point_id += 1

        first_line = line_id
        for i in range(n_pts):
            p1 = first_point + i
            p2 = first_point + (i + 1) % n_pts
            geo_content += f"Line({line_id}) = {{{p1}, {p2}}};\n"
            line_id += 1

        line_list = ", ".join(str(first_line + i) for i in range(n_pts))
        geo_content += f"Curve Loop({curve_loop_id}) = {{{line_list}}};\n"
        geo_content += f"Plane Surface({surface_id}) = {{{curve_loop_id}}};\n"

        clump_curve_loops.append(curve_loop_id)
        surface_id += 1
        curve_loop_id += 1

    last_surface = surface_id - 1
    num_clumps = last_surface - 1

    # Create matrix surface (outer boundary minus clump holes)
    if clump_curve_loops:
        hole_list = ", ".join(str(cl) for cl in clump_curve_loops)
        geo_content += f"""
// Matrix surface (RVE boundary minus clumps)
Plane Surface(1000) = {{1, {hole_list}}};
"""
    else:
        geo_content += """
// Matrix surface (no clumps)
Plane Surface(1000) = {1};
"""

    # Physical groups
    if num_clumps > 0:
        geo_content += f"""
// Physical groups
Physical Surface("fungi") = {{1000}};
Physical Surface("wood") = {{2:{last_surface}}};
"""
    else:
        geo_content += """
// Physical groups (no clumps)
Physical Surface("fungi") = {1000};
"""

    # Add periodic constraints for the mesh
    geo_content += f"""
// Periodic mesh constraints
// Right edge = Left edge + translation
Periodic Curve {{2}} = {{-4}} Translate {{{cell_size}, 0, 0}};
// Top edge = Bottom edge + translation
Periodic Curve {{3}} = {{-1}} Translate {{0, {cell_size}, 0}};

// Physical curves for boundary conditions (optional, for FEM)
Physical Curve("left") = {{4}};
Physical Curve("right") = {{2}};
Physical Curve("bottom") = {{1}};
Physical Curve("top") = {{3}};

Mesh 2;
Coherence Mesh;
"""

    with open(output_geo, 'w') as f:
        f.write(geo_content)

    print(f"\nGeometry file saved to {output_geo}")
    print(f"Generated {num_clumps} clump surfaces")
    print(f"\nTo mesh: gmsh {output_geo} -2 -o output.msh")
    print("\nNOTE: The mesh will have matching nodes on opposite boundaries (periodic).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert periodic YADE packing to GMSH")
    parser.add_argument("npy_file", nargs="?", default="sphere_coordinates_periodic.npy")
    parser.add_argument("output_geo", nargs="?", default="woodchip_rve_periodic.geo")
    parser.add_argument("--mesh_size", type=float, default=0.01)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--shrink", type=float, default=0.9, help="Shrink factor for clumps")

    args = parser.parse_args()

    # Try to load cell size from meta file if it exists
    meta_file = args.npy_file.replace('.npy', '_meta.npy')
    try:
        meta = np.load(meta_file)
        cell_size = float(meta[0])  # Assuming square cell, use x dimension
        print(f"Loaded cell size from {meta_file}: {cell_size}")
    except FileNotFoundError:
        cell_size = args.cell_size
        print(f"No meta file found, using cell_size={cell_size}")

    create_periodic_geo_file(
        args.npy_file,
        args.output_geo,
        cell_size=cell_size,
        mesh_size=args.mesh_size,
        shrink_factor=args.shrink
    )
