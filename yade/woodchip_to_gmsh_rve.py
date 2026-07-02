"""Convert settled wood chip data to GMSH .geo file for RVE generation."""

from string import Template

import numpy as np
from scipy.spatial import ConvexHull


# RVE extraction parameters
RVE_LEFT_MARGIN = 0.1      # margin from left boundary (box is 1.2 wide)
RVE_RIGHT_MARGIN = 0.1     # margin from right boundary
RVE_BOTTOM_MARGIN = 0.2    # margin from bottom boundary
RVE_SIZE = 1.0             # square RVE size (1.0 x 1.0)
SHRINK_FACTOR = 0.8 # 0.9       # shrink all clumps by this factor


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


def clump_centroid(xy, radii):
    """Compute the centroid of a clump (weighted by circle area)."""
    areas = np.pi * radii**2
    total_area = np.sum(areas)
    cx = np.sum(xy[:, 0] * areas) / total_area
    cy = np.sum(xy[:, 1] * areas) / total_area
    return np.array([cx, cy])


def shrink_clump(xy, radii, factor):
    """Shrink a clump by a factor around its centroid."""
    centroid = clump_centroid(xy, radii)
    new_xy = centroid + factor * (xy - centroid)
    new_radii = radii * factor
    return new_xy, new_radii


def clump_overlaps_rve(xy, radii, x_min, x_max, y_min, y_max):
    """Check if any circle from the clump overlaps with the RVE bounds."""
    for (cx, cy), r in zip(xy, radii):
        # Check if circle overlaps with rectangle
        # Find closest point on rectangle to circle center
        closest_x = np.clip(cx, x_min, x_max)
        closest_y = np.clip(cy, y_min, y_max)
        dist = np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
        if dist < r or (x_min <= cx <= x_max and y_min <= cy <= y_max):
            return True
    return False


def clip_polygon_to_rect(polygon, x_min, x_max, y_min, y_max):
    """Clip a polygon to a rectangle using Sutherland-Hodgman algorithm."""
    def clip_edge(poly, edge_func, inside_func):
        """Clip polygon against a single edge."""
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


def circles_overlap(xy1, r1, xy2, r2):
    """Check if any circle from clump1 overlaps with any circle from clump2."""
    for i in range(len(xy1)):
        for j in range(len(xy2)):
            dist = np.linalg.norm(xy1[i] - xy2[j])
            if dist < r1[i] + r2[j]:
                return True
    return False


def merge_overlapping_clumps(clumps):
    """Merge clumps that overlap into single clumps."""
    if not clumps:
        return clumps

    # Use Union-Find to group overlapping clumps
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

    # Check all pairs for overlap
    for i in range(n):
        for j in range(i + 1, n):
            xy_i, radii_i = clumps[i]
            xy_j, radii_j = clumps[j]
            if circles_overlap(xy_i, radii_i, xy_j, radii_j):
                union(i, j)

    # Group clumps by their root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge each group into a single clump
    merged_clumps = []
    for indices in groups.values():
        all_xy = np.vstack([clumps[i][0] for i in indices])
        all_radii = np.concatenate([clumps[i][1] for i in indices])
        merged_clumps.append((all_xy, all_radii))

    print(f"Merged {n} clumps into {len(merged_clumps)} clumps")
    return merged_clumps


def extract_rve_clumps(clumps, x_min, x_max, y_min, y_max, shrink_factor):
    """Extract clumps that overlap or fall within RVE bounds and apply shrinking."""
    rve_clumps = []
    for xy, radii in clumps:
        if clump_overlaps_rve(xy, radii, x_min, x_max, y_min, y_max):
            # Shrink the clump
            new_xy, new_radii = shrink_clump(xy, radii, shrink_factor)
            # Translate to RVE coordinates (origin at RVE corner)
            new_xy[:, 0] -= x_min
            new_xy[:, 1] -= y_min
            rve_clumps.append((new_xy, new_radii))

    print(f"Extracted {len(rve_clumps)} clumps overlapping RVE bounds")
    return rve_clumps


def compute_volume_fraction(clumps, rve_size):
    """Compute approximate volume fraction of clumps within RVE using convex hull area."""
    rve_area = rve_size * rve_size
    total_clump_area = 0.0

    for xy, radii in clumps:
        poly_points = circles_to_polygon(xy, radii)
        # Clip to RVE bounds for area calculation
        poly_points[:, 0] = np.clip(poly_points[:, 0], 0, rve_size)
        poly_points[:, 1] = np.clip(poly_points[:, 1], 0, rve_size)
        # Compute polygon area using shoelace formula
        n = len(poly_points)
        area = 0.5 * abs(sum(
            poly_points[i, 0] * poly_points[(i + 1) % n, 1] -
            poly_points[(i + 1) % n, 0] * poly_points[i, 1]
            for i in range(n)
        ))
        total_clump_area += area

    vf = total_clump_area / rve_area
    return vf


def create_geo_file(npy_file, output_geo, mesh_size=0.01, type_name="mix"):
    """Create a GMSH .geo file with RVE clumps using boolean operations."""
    clumps = load_clumps(npy_file)

    # Original box: x in [0, 1.2], y from 0 upward
    box_width = 1.2

    # Define RVE bounds
    rve_x_min = RVE_LEFT_MARGIN
    rve_x_max = box_width - RVE_RIGHT_MARGIN
    rve_y_min = RVE_BOTTOM_MARGIN
    rve_y_max = RVE_BOTTOM_MARGIN + RVE_SIZE

    print(f"RVE bounds: x=[{rve_x_min}, {rve_x_max}], y=[{rve_y_min}, {rve_y_max}]")
    print(f"RVE size: {RVE_SIZE} x {RVE_SIZE}")
    print(f"Shrink factor: {SHRINK_FACTOR}")

    # Extract clumps within RVE and shrink them
    rve_clumps = extract_rve_clumps(clumps, rve_x_min, rve_x_max, rve_y_min, rve_y_max, SHRINK_FACTOR)

    # Merge overlapping clumps
    rve_clumps = merge_overlapping_clumps(rve_clumps)

    # Compute and print volume fraction
    vf = compute_volume_fraction(rve_clumps, RVE_SIZE)
    print(f"Volume fraction: {vf:.4f}")

    # Generate .geo file content
    geo_preamble = Template("""SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;
Mesh.CharacteristicLengthMin = $meshsize;
Mesh.CharacteristicLengthMax = $meshsize;

// RVE bounding box (Surface 1)
Rectangle(1) = {0.0, 0.0, 0, $dx, $dy, 0.0};

// Clump polygons (Surfaces 2 to N+1)
""")

    geo_postamble = Template("""
// Delete the original rectangle (we only needed it for reference)
Delete{ Surface{1}; }

// Create matrix by subtracting clumps from a new rectangle
Rectangle(1000) = {0.0, 0.0, 0, $dx, $dy, 0.0};
BooleanDifference{ Surface{1000}; Delete; }{ Surface{2:$LAST}; }

// Physical groups - matrix is the result of the difference (Surface 1000)
// Clumps remain as surfaces 2 to LAST
Physical Surface("fungi") = {1000};
Physical Surface("wood") = {2:$LAST};

Mesh 2;
Coherence Mesh;
""")

    # Build the .geo file
    geo_content = geo_preamble.substitute(
        meshsize=mesh_size,
        dx=RVE_SIZE,
        dy=RVE_SIZE
    )

    # Add each clump as a polygon surface (starting from surface 2)
    # Rectangle(1) creates Points 1-4 and Lines 1-4, so start after those
    surface_id = 2  # Surface 1 is the rectangle
    point_id = 100  # Start after rectangle's points
    line_id = 100   # Start after rectangle's lines
    curve_loop_id = 100  # Start after rectangle's curve loop

    for xy, radii in rve_clumps:
        poly_points = circles_to_polygon(xy, radii)
        # Clip polygon to RVE bounds
        poly_points = clip_polygon_to_rect(poly_points, 0, RVE_SIZE, 0, RVE_SIZE)

        n_pts = len(poly_points)
        if n_pts < 3:
            continue

        geo_content += f"\n// Clump {surface_id - 1}\n"

        # Add points
        first_point = point_id
        for px, py in poly_points:
            geo_content += f"Point({point_id}) = {{{px}, {py}, 0, {mesh_size}}};\n"
            point_id += 1

        # Add lines
        first_line = line_id
        for i in range(n_pts):
            p1 = first_point + i
            p2 = first_point + (i + 1) % n_pts
            geo_content += f"Line({line_id}) = {{{p1}, {p2}}};\n"
            line_id += 1

        # Add curve loop and surface
        line_list = ", ".join(str(first_line + i) for i in range(n_pts))
        geo_content += f"Curve Loop({curve_loop_id}) = {{{line_list}}};\n"
        geo_content += f"Plane Surface({surface_id}) = {{{curve_loop_id}}};\n"

        surface_id += 1
        curve_loop_id += 1

    # Add postamble with correct number of surfaces
    last_surface = surface_id - 1
    num_clumps = last_surface - 1  # Exclude surface 1 (rectangle)
    geo_content += geo_postamble.substitute(LAST=last_surface, dx=RVE_SIZE, dy=RVE_SIZE)

    # Write .geo file
    with open(output_geo, 'w') as f:
        f.write(geo_content)

    print(f"Geometry file saved to {output_geo}")
    print(f"Generated {num_clumps} clump surfaces (surfaces 2 to {last_surface})")
    print(f"\nTo mesh, run: gmsh2 {output_geo} -2 -o output_{type_name}.msh")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Convert YADE sphere data to GMSH .geo file")
    parser.add_argument("npy_file", nargs="?", default=None, help="Input .npy file")
    parser.add_argument("output_geo", nargs="?", default=None, help="Output .geo file")
    # parser.add_argument("--mesh_size", type=float, default=0.01, help="Mesh element size")
    parser.add_argument("--mesh_size", type=float, default=0.02, help="Mesh element size")
    parser.add_argument("--type_name", type=str, default=None, help="Type name for output naming")

    args = parser.parse_args()

    # Default type_name for manual runs (uncomment one)
    # _default_type = "small"
    # _default_type = "large"
    # _default_type = "large_looser"
    # _default_type = "mix"
    # _default_type = "mix2"
    # _default_type = "vlarge"
    _default_type = "gravity"
    # _default_type = "slender_large_looser"

    # Determine filenames
    if args.npy_file:
        npy_file = args.npy_file
        type_name = args.type_name or "custom"
    else:
        type_name = _default_type
        npy_file = f"meshes/sphere_coordinates_{type_name}.npy"

    # output_geo = args.output_geo or f"meshes/woodchip_rve_{type_name}.geo"
    output_geo = args.output_geo or f"meshes/woodchip_rve_{type_name}_shrink{SHRINK_FACTOR:.1f}.geo"

    create_geo_file(npy_file, output_geo, args.mesh_size, type_name)
