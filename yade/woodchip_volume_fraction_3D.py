"""Compute volume fraction of wood chips in a 3D RVE."""

import os
import numpy as np
from scipy.spatial import ConvexHull

# RVE extraction parameters (matching 2D script conventions)
RVE_LEFT_MARGIN = 0.1      # margin from left boundary (box is 1.2 wide)
RVE_RIGHT_MARGIN = 0.1     # margin from right boundary
RVE_FRONT_MARGIN = 0.1     # margin from front boundary (Z)
RVE_BACK_MARGIN = 0.1      # margin from back boundary (Z)
RVE_BOTTOM_MARGIN = 0.2    # margin from bottom boundary
RVE_SIZE = 1.0             # cubic RVE size (1.0 x 1.0 x 1.0)
SHRINK_FACTOR = 0.95       # shrink all clumps by this factor


def load_clumps_3D(npy_file):
    """Load 3D sphere data and group by clump ID. Returns list of (xyz, radii) per clump."""
    data = np.load(npy_file)  # columns: x, y, z, radius, clumpId
    clump_ids = np.unique(data[:, 4]).astype(int)
    clumps = []
    for cid in clump_ids:
        mask = data[:, 4].astype(int) == cid
        xyz = data[mask, :3]
        radii = data[mask, 3]
        clumps.append((xyz, radii))
    return clumps


def clump_centroid_3D(xyz, radii):
    """Compute the centroid of a 3D clump (weighted by sphere volume)."""
    volumes = (4/3) * np.pi * radii**3
    total_vol = np.sum(volumes)
    cx = np.sum(xyz[:, 0] * volumes) / total_vol
    cy = np.sum(xyz[:, 1] * volumes) / total_vol
    cz = np.sum(xyz[:, 2] * volumes) / total_vol
    return np.array([cx, cy, cz])


def shrink_clump_3D(xyz, radii, factor):
    """Shrink a 3D clump by a factor around its centroid."""
    centroid = clump_centroid_3D(xyz, radii)
    new_xyz = centroid + factor * (xyz - centroid)
    new_radii = radii * factor
    return new_xyz, new_radii


def clump_overlaps_rve_3D(xyz, radii, x_min, x_max, y_min, y_max, z_min, z_max):
    """Check if any sphere from the clump overlaps with the 3D RVE bounds."""
    for (cx, cy, cz), r in zip(xyz, radii):
        # Find closest point on box to sphere center
        closest_x = np.clip(cx, x_min, x_max)
        closest_y = np.clip(cy, y_min, y_max)
        closest_z = np.clip(cz, z_min, z_max)
        dist = np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2 + (cz - closest_z)**2)
        if dist < r or (x_min <= cx <= x_max and y_min <= cy <= y_max and z_min <= cz <= z_max):
            return True
    return False


def spheres_overlap_3D(xyz1, r1, xyz2, r2):
    """Check if any sphere from clump1 overlaps with any sphere from clump2."""
    for i in range(len(xyz1)):
        for j in range(len(xyz2)):
            dist = np.linalg.norm(xyz1[i] - xyz2[j])
            if dist < r1[i] + r2[j]:
                return True
    return False


def merge_overlapping_clumps_3D(clumps):
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
            xyz_i, radii_i = clumps[i]
            xyz_j, radii_j = clumps[j]
            if spheres_overlap_3D(xyz_i, radii_i, xyz_j, radii_j):
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
        all_xyz = np.vstack([clumps[i][0] for i in indices])
        all_radii = np.concatenate([clumps[i][1] for i in indices])
        merged_clumps.append((all_xyz, all_radii))

    print(f"Merged {n} clumps into {len(merged_clumps)} clumps")
    return merged_clumps


def extract_rve_clumps_3D(clumps, x_min, x_max, y_min, y_max, z_min, z_max, shrink_factor):
    """Extract clumps that overlap or fall within 3D RVE bounds and apply shrinking."""
    rve_clumps = []
    for xyz, radii in clumps:
        if clump_overlaps_rve_3D(xyz, radii, x_min, x_max, y_min, y_max, z_min, z_max):
            # Shrink the clump
            new_xyz, new_radii = shrink_clump_3D(xyz, radii, shrink_factor)
            # Translate to RVE coordinates (origin at RVE corner)
            new_xyz[:, 0] -= x_min
            new_xyz[:, 1] -= y_min
            new_xyz[:, 2] -= z_min
            rve_clumps.append((new_xyz, new_radii))

    print(f"Extracted {len(rve_clumps)} clumps overlapping RVE bounds")
    return rve_clumps


def point_in_clump(point, xyz, radii):
    """Check if a point is inside any sphere of the clump."""
    for (cx, cy, cz), r in zip(xyz, radii):
        dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2 + (point[2] - cz)**2)
        if dist <= r:
            return True
    return False


def compute_volume_fraction_monte_carlo(clumps, rve_size, n_samples=500000):
    """Compute volume fraction using Monte Carlo sampling."""
    # Generate random points in the RVE
    np.random.seed(42)  # For reproducibility
    points = np.random.uniform(0, rve_size, size=(n_samples, 3))

    # Count points inside any clump
    inside_count = 0
    for point in points:
        for xyz, radii in clumps:
            if point_in_clump(point, xyz, radii):
                inside_count += 1
                break  # Don't double-count

    vf = inside_count / n_samples
    return vf


def compute_volume_fraction_monte_carlo_fast(clumps, rve_size, n_samples=500000):
    """Compute volume fraction using vectorized Monte Carlo sampling."""
    np.random.seed(42)
    points = np.random.uniform(0, rve_size, size=(n_samples, 3))

    # For each point, check if it's inside any sphere (vectorized per clump)
    inside = np.zeros(n_samples, dtype=bool)

    for i, (xyz, radii) in enumerate(clumps):
        if i % 100 == 0:
            print(f"did {i} clumps")
        for (cx, cy, cz), r in zip(xyz, radii):
            # Vectorized distance computation
            dist_sq = (points[:, 0] - cx)**2 + (points[:, 1] - cy)**2 + (points[:, 2] - cz)**2
            inside |= (dist_sq <= r**2)

    vf = np.sum(inside) / n_samples
    return vf


def spheres_to_surface_points(xyz, radii, num_points_per_sphere=50):
    """Generate points on the surface of spheres for convex hull computation."""
    # Use Fibonacci lattice for uniform sphere sampling
    all_points = []
    for (cx, cy, cz), r in zip(xyz, radii):
        indices = np.arange(num_points_per_sphere)
        phi = np.arccos(1 - 2 * (indices + 0.5) / num_points_per_sphere)
        theta = np.pi * (1 + np.sqrt(5)) * indices

        x = cx + r * np.sin(phi) * np.cos(theta)
        y = cy + r * np.sin(phi) * np.sin(theta)
        z = cz + r * np.cos(phi)

        all_points.append(np.column_stack([x, y, z]))

    return np.vstack(all_points)


def clump_fully_inside_rve(xyz, radii, rve_size):
    """Check if all spheres of a clump are fully inside the RVE."""
    for (cx, cy, cz), r in zip(xyz, radii):
        if cx - r < 0 or cx + r > rve_size:
            return False
        if cy - r < 0 or cy + r > rve_size:
            return False
        if cz - r < 0 or cz + r > rve_size:
            return False
    return True


def compute_volume_fraction_convex_hull(clumps, rve_size):
    """Compute approximate volume fraction using convex hulls.

    Note: Only counts clumps fully inside the RVE. Boundary-crossing clumps
    are skipped because proper 3D boolean intersection is complex.
    This will underestimate the true volume fraction.
    """
    rve_volume = rve_size ** 3
    total_clump_volume = 0.0
    skipped = 0

    for xyz, radii in clumps:
        # Only use clumps fully inside RVE
        if not clump_fully_inside_rve(xyz, radii, rve_size):
            skipped += 1
            continue

        # Generate surface points
        surface_points = spheres_to_surface_points(xyz, radii)

        # Compute convex hull
        try:
            if len(surface_points) >= 4:  # Need at least 4 points for 3D hull
                hull = ConvexHull(surface_points)
                total_clump_volume += hull.volume
        except Exception:
            pass

    if skipped > 0:
        print(f"  (Skipped {skipped} boundary-crossing clumps - will underestimate)")

    vf = total_clump_volume / rve_volume
    return vf


def sphere_cap_volume(r, h):
    """Volume of a spherical cap with height h and sphere radius r."""
    return np.pi * h**2 * (3*r - h) / 3


def sphere_volume_in_box(cx, cy, cz, r, box_min, box_max):
    """Estimate volume of sphere inside axis-aligned box [box_min, box_max]^3.

    Uses sphere cap subtraction for each face the sphere crosses.
    This is approximate when sphere crosses multiple faces (corners/edges).
    """
    full_volume = (4/3) * np.pi * r**3

    # Check if sphere is completely outside
    if (cx + r < box_min or cx - r > box_max or
        cy + r < box_min or cy - r > box_max or
        cz + r < box_min or cz - r > box_max):
        return 0.0

    # Check if sphere is completely inside
    if (cx - r >= box_min and cx + r <= box_max and
        cy - r >= box_min and cy + r <= box_max and
        cz - r >= box_min and cz + r <= box_max):
        return full_volume

    # Sphere crosses boundary - subtract caps for each crossed face
    volume = full_volume

    # X boundaries
    if cx - r < box_min:  # crosses left face
        h = box_min - (cx - r)
        volume -= sphere_cap_volume(r, h)
    if cx + r > box_max:  # crosses right face
        h = (cx + r) - box_max
        volume -= sphere_cap_volume(r, h)

    # Y boundaries
    if cy - r < box_min:  # crosses bottom face
        h = box_min - (cy - r)
        volume -= sphere_cap_volume(r, h)
    if cy + r > box_max:  # crosses top face
        h = (cy + r) - box_max
        volume -= sphere_cap_volume(r, h)

    # Z boundaries
    if cz - r < box_min:  # crosses front face
        h = box_min - (cz - r)
        volume -= sphere_cap_volume(r, h)
    if cz + r > box_max:  # crosses back face
        h = (cz + r) - box_max
        volume -= sphere_cap_volume(r, h)

    # Note: This overcounts subtraction at corners/edges, but is a reasonable approximation
    return max(0.0, volume)


def compute_volume_fraction_spheres(clumps, rve_size):
    """Compute volume fraction by summing sphere volumes clipped to RVE.

    Uses sphere cap subtraction for boundary-crossing spheres.
    Approximate at corners/edges but generally accurate.
    """
    rve_volume = rve_size ** 3
    total_sphere_volume = 0.0

    for xyz, radii in clumps:
        for (cx, cy, cz), r in zip(xyz, radii):
            total_sphere_volume += sphere_volume_in_box(cx, cy, cz, r, 0, rve_size)

    vf = total_sphere_volume / rve_volume
    return vf


def compute_volume_fraction(npy_file, method="monte_carlo", n_samples=500000,
                            periodic=False, rve_size=None, shrink_factor=None):
    """Compute volume fraction of a 3D RVE.

    If periodic=True, assumes coordinates are already in a periodic cell [0, rve_size]^3
    (output from yade_woodchip_3D.py periodic relaxation). Otherwise extracts from
    gravity deposition coordinates.
    """
    clumps = load_clumps_3D(npy_file)
    print(f"Loaded {len(clumps)} clumps from {npy_file}")

    if rve_size is None:
        rve_size = RVE_SIZE
    if shrink_factor is None:
        shrink_factor = SHRINK_FACTOR

    if periodic:
        # Try to load cell size from meta file
        meta_file = npy_file.replace('.npy', '_meta.npy')
        if os.path.exists(meta_file):
            cell = np.load(meta_file)
            rve_size = float(cell[0])  # assume cubic
            print(f"Loaded periodic cell size from meta: {rve_size:.3f}")

        print(f"Periodic mode: RVE size = {rve_size:.3f}")
        # Shrink clumps in place (around their centroids)
        rve_clumps = []
        for xyz, radii in clumps:
            new_xyz, new_radii = shrink_clump_3D(xyz, radii, shrink_factor)
            rve_clumps.append((new_xyz, new_radii))
    else:
        # Original box: x,z in [0, 1.2], y from 0 upward
        box_width = 1.2
        box_depth = 1.2

        # Define RVE bounds
        rve_x_min = RVE_LEFT_MARGIN
        rve_x_max = box_width - RVE_RIGHT_MARGIN
        rve_y_min = RVE_BOTTOM_MARGIN
        rve_y_max = RVE_BOTTOM_MARGIN + RVE_SIZE
        rve_z_min = RVE_FRONT_MARGIN
        rve_z_max = box_depth - RVE_BACK_MARGIN

        print(f"RVE bounds: x=[{rve_x_min}, {rve_x_max}], y=[{rve_y_min}, {rve_y_max}], z=[{rve_z_min}, {rve_z_max}]")
        print(f"RVE size: {rve_size} x {rve_size} x {rve_size}")
        print(f"Shrink factor: {shrink_factor}")

        rve_clumps = extract_rve_clumps_3D(
            clumps, rve_x_min, rve_x_max, rve_y_min, rve_y_max, rve_z_min, rve_z_max, shrink_factor
        )

    # Compute volume fraction
    if method == "monte_carlo":
        print(f"Computing volume fraction using Monte Carlo ({n_samples} samples)...")
        vf = compute_volume_fraction_monte_carlo_fast(rve_clumps, rve_size, n_samples)
    elif method == "convex_hull":
        print("Computing volume fraction using convex hulls...")
        vf = compute_volume_fraction_convex_hull(rve_clumps, rve_size)
    elif method == "spheres":
        print("Computing volume fraction using sphere volumes...")
        vf = compute_volume_fraction_spheres(rve_clumps, rve_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Volume fraction: {vf:.4f} ({vf*100:.2f}%)")
    return vf, rve_clumps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute volume fraction of 3D wood chip RVE")
    parser.add_argument("npy_file", nargs="?", default=None, help="Input .npy file")
    parser.add_argument("--method", type=str, default="monte_carlo",
                        choices=["monte_carlo", "convex_hull", "spheres"],
                        help="Method for volume computation")
    parser.add_argument("--n_samples", type=int, default=500000,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--periodic", action="store_true",
                        help="Periodic mode: coordinates already in [0, rve_size]^3 cell")
    parser.add_argument("--rve_size", type=float, default=None,
                        help="RVE size (auto-detected from meta file in periodic mode)")
    parser.add_argument("--shrink_factor", type=float, default=None,
                        help="Shrink factor for clumps (default: 0.95)")

    args = parser.parse_args()

    # Default type for manual runs
    _default_type = "mix_0.8_scale5"

    if args.npy_file:
        npy_file = args.npy_file
    else:
        npy_file = f"sphere_coordinates_3D_{_default_type}.npy"

    vf, clumps = compute_volume_fraction(
        npy_file, method=args.method, n_samples=args.n_samples,
        periodic=args.periodic, rve_size=args.rve_size,
        shrink_factor=args.shrink_factor,
    )
