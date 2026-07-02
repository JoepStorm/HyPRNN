# -*- encoding=utf-8 -*-
"""3D gravity deposition → horizontal slice → 2D periodic relaxation.

Uses continuous clump shape parametrization via aspect ratios (r1, r2)
with probabilistic rounding for smooth optimization landscapes.

Parameters:
    r1 ∈ (0, 1]: ratio of smallest to largest dimension (nx/nz)
    r2 ∈ [r1, 1]: ratio of middle to largest dimension (ny/nz)
    nz: number of spheres along the largest axis

Corners of the (r1, r2) triangle correspond to named types:
    r1 ≈ 0, r2 ≈ 0  → slender (1x1x8)
    r1 ≈ 0, r2 ≈ 1  → plate   (1x5x5)
    r1 = r2 = 1      → bulky   (2x2x2)
"""

from yade import pack
import random
import numpy as np
import os
import argparse
import sys

# === ARGUMENT PARSING ===
# YADE passes arguments after -- to the script
parser = argparse.ArgumentParser(description="3D→2D woodchip deposition with continuous clump shapes")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--r1", type=float, default=0.15, help="nx/nz ratio ∈ (0, 1]")
parser.add_argument("--r2", type=float, default=0.15, help="ny/nz ratio ∈ [r1, 1]")
parser.add_argument("--nz", type=int, default=6, help="Largest dimension (number of spheres)")
parser.add_argument("--no-gui", action="store_true", help="Disable GUI (for batch runs)")

try:
    idx = sys.argv.index("--")
    script_args = sys.argv[idx + 1:]
except ValueError:
    script_args = [a for a in sys.argv[1:] if a.startswith("--") or
                   (len(sys.argv) > 1 and not a.startswith("-") and
                    sys.argv[max(0, sys.argv.index(a)-1)].startswith("--"))]

args = parser.parse_args(script_args)

# === CONFIGURATION ===
seed = args.seed
base_radius = 0.05
target_size = 1.0
cut_height = 0.25
damping = 0.5
frictionAngle = 0.5

# Continuous clump shape parameters
# Essentially a triangle for r1 and r2
r1 = args.r1
r2 = args.r2
nz = args.nz

num_chips = 400

# Relaxation options
relaxation_damping = 0.95
max_iter_gravity = 20000
max_iter_relaxing = 2000
relaxation_vel_tol = 0.03

# Box dimensions
box_width = 1.5
box_depth = 1.5
box_height = 3.0

show_gui = not args.no_gui


def probabilistic_round(x, rng):
    """Round to integer with probability proportional to fractional part.

    E.g. x=2.3 → 2 with p=0.7, 3 with p=0.3.
    Result is always >= 1.
    """
    lower = int(np.floor(x))
    frac = x - lower
    result = lower + 1 if rng.random() < frac else lower
    return max(1, result)


def ratios_to_shape(r1, r2, nz, rng):
    """Convert continuous ratios to integer clump shape via probabilistic rounding."""
    nx = probabilistic_round(r1 * nz, rng)
    ny = probabilistic_round(r2 * nz, rng)
    return (nx, ny, nz)


# Initialize RNG for shape sampling (separate from position RNG)
shape_rng = random.Random(seed + 1000)
random.seed(seed)

# Output
name = f'_3Dto2D_r1_{r1:.2f}_r2_{r2:.2f}_nz_{nz}_{num_chips}_{seed}'
# output_name = f"periodic/depositions_3d_to_2d_relaxed_cont/coords{name}.npy"
output_name = f"periodic/presentation_demo/coords{name}.npy"

# Expected (mean) shape for spacing calculation
mean_nx = max(1, round(r1 * nz))
mean_ny = max(1, round(r2 * nz))
mean_shape = (mean_nx, mean_ny, nz)

print(f"Config: seed={seed}, chips={num_chips}")
print(f"Continuous params: r1={r1:.3f}, r2={r2:.3f}, nz={nz}")
print(f"Mean shape: {mean_shape}")

# === PHASE 1: 3D GRAVITY DEPOSITION ===
print(f"\nPhase 1: 3D gravity deposition")
print(f"Box: {box_width} x {box_height} x {box_depth}")

mat = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=frictionAngle)
O.materials.append(mat)

# Box walls
O.bodies.append(wall((0, 0, 0), axis=0, sense=1, material=mat))
O.bodies.append(wall((box_width, 0, 0), axis=0, sense=-1, material=mat))
O.bodies.append(wall((0, 0, 0), axis=1, sense=1, material=mat))
O.bodies.append(wall((0, 0, 0), axis=2, sense=1, material=mat))
O.bodies.append(wall((0, 0, box_depth), axis=2, sense=-1, material=mat))


def simple_offsets(n):
    """Return n offsets at unit spacing (no intermediates, sparse clumps)."""
    return [(i - (n - 1) / 2) for i in range(n)]


def create_chip(center, nx, ny, nz):
    """Create a sparse chip: one sphere per grid position."""
    cx, cy, cz = center
    r = base_radius
    spheres = []
    for dz in simple_offsets(nz):
        for dy in simple_offsets(ny):
            for dx in simple_offsets(nx):
                spheres.append(sphere([cx + dx * r, cy + dy * r, cz + dz * r], r, material=mat))
    return O.bodies.appendClumped(spheres)


# Grid-based spawning — use max possible dimension for spacing
max_dim = nz  # nz is always the largest
chip_spacing = (max_dim + 1) * base_radius
min_margin = 0.1
n_cols_x = max(1, int((box_width - 2 * min_margin) / chip_spacing))
n_cols_z = max(1, int((box_depth - 2 * min_margin) / chip_spacing))
margin_x = (box_width - n_cols_x * chip_spacing) / 2
margin_z = (box_depth - n_cols_z * chip_spacing) / 2
chips_per_layer = n_cols_x * n_cols_z

for i in range(num_chips):
    # Sample a shape for this chip
    chip_shape = ratios_to_shape(r1, r2, nz, shape_rng)

    layer = i // chips_per_layer
    pos_in_layer = i % chips_per_layer
    col_x = pos_in_layer % n_cols_x
    col_z = pos_in_layer // n_cols_x
    pos = (
        margin_x + (col_x + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        0.3 + (layer + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        margin_z + (col_z + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
    )

    clump_id, member_ids = create_chip(pos, *chip_shape)

    # Random 3D rotation
    clump = O.bodies[clump_id]
    qx = Quaternion((1, 0, 0), random.uniform(0, 2 * np.pi))
    qy = Quaternion((0, 1, 0), random.uniform(0, 2 * np.pi))
    qz = Quaternion((0, 0, 1), random.uniform(0, 2 * np.pi))
    clump.state.ori = qz * qy * qx

print(f"Created {num_chips} chips with continuous params r1={r1}, r2={r2}, nz={nz}")


# === STATE MACHINE ===
phase = "gravity"
stable_count = 0
phase_start_iter = 0


def get_sphere_data():
    """Get all sphere positions and radii, grouped by clump."""
    clumps = {}
    for b in O.bodies:
        if not hasattr(b.shape, '__class__') or b.shape.__class__.__name__ != 'Sphere':
            continue
        if b.clumpId < 0:
            continue
        pos = b.state.pos
        if b.clumpId not in clumps:
            clumps[b.clumpId] = []
        clumps[b.clumpId].append((pos[0], pos[1], pos[2], b.shape.radius))
    return clumps


def slice_and_extract():
    """Cut a horizontal plane, project to 2D, set up periodic relaxation."""
    global phase, phase_start_iter

    print(f"\n=== Phase 2: Slicing at Y={cut_height:.2f} ===")

    clumps_3d = get_sphere_data()
    if not clumps_3d:
        print("ERROR: No spheres found!")
        O.pause()
        return

    sliced_clumps = {}
    total_circles = 0
    for cid, spheres in clumps_3d.items():
        sliced = []
        for sx, sy, sz, sr in spheres:
            dy = abs(sy - cut_height)
            if dy < sr:
                r_2d = np.sqrt(sr**2 - dy**2)
                sliced.append((sx, sz, r_2d))
        if sliced:
            sliced_clumps[cid] = sliced
            total_circles += len(sliced)

    print(f"Sliced: {len(sliced_clumps)} clumps, {total_circles} circles")

    if total_circles == 0:
        print("ERROR: No spheres intersect the cutting plane! Try adjusting cut_height.")
        O.pause()
        return

    x_min = (box_width - target_size) / 2
    x_max = x_min + target_size
    z_min = (box_depth - target_size) / 2
    z_max = z_min + target_size

    selected = {}
    for cid, circles in sliced_clumps.items():
        cx = np.mean([c[0] for c in circles])
        cz = np.mean([c[1] for c in circles])
        if x_min <= cx <= x_max and z_min <= cz <= z_max:
            selected[cid] = circles

    n_selected_circles = sum(len(c) for c in selected.values())
    print(f"Selected {len(selected)} clumps ({n_selected_circles} circles) in extraction region")

    for b in O.bodies:
        if b.id >= 0:
            O.bodies.erase(b.id)

    O.periodic = True
    periodic_size = target_size
    cell_depth = 2 * base_radius * 1.2
    O.cell.setBox(periodic_size, periodic_size, cell_depth)
    print(f"Periodic cell: {periodic_size:.2f} x {periodic_size:.2f} x {cell_depth:.4f}")

    for cid, circles in selected.items():
        spheres_2d = []
        for cx, cy, cr in circles:
            new_x = (cx - x_min) % periodic_size
            new_y = (cy - z_min) % periodic_size
            spheres_2d.append(sphere([new_x, new_y, base_radius * 1.1], cr, material=mat))

        if len(spheres_2d) >= 2:
            new_cid, _ = O.bodies.appendClumped(spheres_2d)
            O.bodies[new_cid].state.blockedDOFs = 'zXY'
        elif len(spheres_2d) == 1:
            bid = O.bodies.append(spheres_2d[0])
            O.bodies[bid].state.blockedDOFs = 'zXY'

    n_clumps = len([b for b in O.bodies if b.isClump])
    n_standalone = len([b for b in O.bodies if hasattr(b.shape, '__class__')
                        and b.shape.__class__.__name__ == 'Sphere' and b.clumpId < 0])
    print(f"Bodies: {n_clumps} clumps + {n_standalone} standalone spheres")

    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=relaxation_damping),
        PyRunner(command='update()', realPeriod=0.5),
    ]

    O.dt = 0.5 * PWaveTimeStep()

    phase = "relaxing"
    phase_start_iter = O.iter
    print("\n=== Relaxing 2D periodic structure (high damping) ===")


def save_results():
    """Save final 2D particle positions."""
    O.pause()

    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId])

    data = np.array(data)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    np.save(output_name, data)
    np.save(output_name.replace('.npy', '_meta.npy'),
            np.array([O.cell.size[0], O.cell.size[1]]))

    vf = sum(np.pi * r**2 for r in data[:, 2]) / (O.cell.size[0] * O.cell.size[1])
    print(f"\nDone! Saved {len(data)} spheres to {output_name}")
    print(f"Cell: {O.cell.size[0]:.3f} x {O.cell.size[1]:.3f}, volume fraction: {vf:.4f}")


def update():
    """Main simulation control."""
    global phase, stable_count

    unbal = unbalancedForce()
    n_interactions = len(O.interactions)
    phase_iter = O.iter - phase_start_iter

    max_iter = max_iter_gravity if phase == "gravity" else max_iter_relaxing
    if max_iter is not None and phase_iter >= max_iter:
        print(f"{phase}: Max iterations ({max_iter}) reached, forcing convergence")
        stable_count = 3

    if phase == "relaxing":
        max_vel = max((b.state.vel.norm() for b in O.bodies if not isinstance(b.shape, Wall)), default=0)
        is_stable = max_vel < relaxation_vel_tol or n_interactions == 0

        if is_stable:
            stable_count += 1
            print(f"{phase}: stable {stable_count}/1 (max_vel={max_vel:.6f}, interactions={n_interactions})")
        else:
            stable_count = 0
            if O.iter % 2000 == 0:
                print(f"{phase}: max_vel={max_vel:.6f}, interactions={n_interactions}, iter={phase_iter}")

        if stable_count >= 1 or phase_iter >= max_iter:
            stable_count = 0
            save_results()
    else:
        is_stable = unbal < 0.03

        if is_stable:
            stable_count += 1
            print(f"{phase}: stable {stable_count}/3 (unbal={unbal:.4f}, interactions={n_interactions})")
        else:
            stable_count = 0
            if O.iter % 5000 == 0:
                print(f"{phase}: unbal={unbal:.4f}, interactions={n_interactions}, iter={phase_iter}")

        if stable_count >= 3:
            stable_count = 0
            slice_and_extract()


# === ENGINES (Phase 1: 3D non-periodic with walls) ===
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Wall_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    NewtonIntegrator(gravity=(0, -9.81, 0), damping=damping),
    PyRunner(command='update()', realPeriod=0.5),
]

O.dt = 0.5 * PWaveTimeStep()

print("\nStarting 3D gravity deposition...")

if show_gui:
    from yade import qt
    qt.View()
    print("Paused — set up your camera, then press play (▶) or run O.run() in the console")
    O.pause()
else:
    O.run(wait=True)
