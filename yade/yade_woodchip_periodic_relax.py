# -*- encoding=utf-8 -*-
"""Periodic RVE via gravity deposition (non-periodic) + extraction + periodic relaxation."""

from yade import pack
import random
import numpy as np

# === CONFIGURATION ===
seed = 0
small_fraction = 0.0    # 0.8
num_chips = 200     # 600
small_scale = 0.5 #2   # 0.5
clump_shape = [1, 11]

name = f'_spheres{clump_shape[0]}x{clump_shape[1]}_{num_chips}_{seed}'
# name = f'_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}'

output_name = f"aspect_ratio/depositions/coords{name}.npy"
# gravity_output_name = f"periodic/depositions/sphere_coordinates_gravity{name}.npy"  # Save after gravity deposition
base_radius = 0.02
target_size = 1.0
extract_margin = 0.1    # 0.2
damping = 0.5
frictionAngle = 0.5
gui = True

# Convergence options
check_max_vel = False  # If False, only check unbalancedForce for convergence
max_iter_gravity = None  # Max iterations for gravity phase (None = no limit)
max_iter_relaxing = 50000  # Max iterations for relaxation phase (None = no limit)

random.seed(seed)

# === PHASE 1: NON-PERIODIC GRAVITY DEPOSITION ===
# Standard walls, no periodic BC

box_width = 1.2
box_height = 5.0

print(f"Phase 1: Gravity deposition (non-periodic)")
print(f"Box: {box_width} x {box_height}")

# Material
mat = FrictMat(young=1e6, poisson=0.375, density=720, frictionAngle=frictionAngle)
O.materials.append(mat)

# Walls: left, right, bottom
O.bodies.append(wall((0, 0, 0), axis=0, sense=1, material=mat))        # left
O.bodies.append(wall((box_width, 0, 0), axis=0, sense=-1, material=mat))  # right
O.bodies.append(wall((0, 0, 0), axis=1, sense=1, material=mat))        # bottom


def create_chip(center, num_spheres_x, num_rows=1, scale=1.0, angle=0.0):
    """Create a chip with spheres in a grid, rotated by angle.

    num_spheres_x: spheres per row
    num_rows: 1 = single row (slender), 2+ = rectangular
    """
    x, y = center
    r = base_radius * scale
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # X positions (centered)
    if num_spheres_x % 2 == 0:
        x_positions = [(i - (num_spheres_x - 1) / 2) * r for i in range(num_spheres_x)]
    else:
        x_positions = [(i - num_spheres_x // 2) * r for i in range(num_spheres_x)]

    # Y positions (centered)
    if num_rows % 2 == 0:
        y_positions = [(i - (num_rows - 1) / 2) * r for i in range(num_rows)]
    else:
        y_positions = [(i - num_rows // 2) * r for i in range(num_rows)]

    spheres = []
    for py in y_positions:
        for px in x_positions:
            # Rotate around origin, then translate to center
            rot_x = px * cos_a - py * sin_a
            rot_y = px * sin_a + py * cos_a
            spheres.append(sphere([x + rot_x, y + rot_y, 0], r, material=mat))

    return O.bodies.appendClumped(spheres)


# Chip shapes: (num_spheres_x, num_rows)
# Single-row (slender): (5,1), (6,1), (7,1)
# Rectangular: (3,2), (4,2), (5,2), (3,3)
chip_shapes = [
    # (4, 1), (5, 1), (6, 1),  # slender
    # (3, 2), (4, 2), (5, 2),  # 2-row rectangular
    # (3, 3),                   # chunky
    (clump_shape[0], clump_shape[1]),  # custom
]

# Create chips in grid
if small_fraction > 0.5:
    chip_spacing = 12 * base_radius * 0.8  # factor 0.8 for faster falling.
else:   # standard
    chip_spacing = 12 * base_radius
margin = chip_spacing / 2
n_cols = max(1, int((box_width - 2 * margin) / chip_spacing))

for i in range(num_chips):
    col = i % n_cols
    row = i // n_cols
    pos = (
        margin + (col + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        0.5 + (row + 0.5) * chip_spacing + random.uniform(-0.02, 0.02)
    )
    angle = random.uniform(0, 2 * np.pi)

    num_spheres_x, num_rows = random.choice(chip_shapes)

    # Random scale
    if random.random() < small_fraction:
        scale = small_scale
    else:
        scale = 1.0

    clump_id, _ = create_chip(pos, num_spheres_x, num_rows=num_rows, scale=scale, angle=angle)

    O.bodies[clump_id].state.blockedDOFs = 'zXY'

print(f"Created {num_chips} chips")

# === STATE MACHINE ===
phase = "gravity"
stable_count = 0
phase_start_iter = 0


def get_clump_data():
    """Get clump centers and sphere data."""
    clumps = {}
    for b in O.bodies:
        if not hasattr(b.shape, '__class__') or b.shape.__class__.__name__ != 'Sphere':
            continue
        if b.clumpId < 0:
            continue
        pos = b.state.pos
        if b.clumpId not in clumps:
            clumps[b.clumpId] = []
        clumps[b.clumpId].append((pos[0], pos[1], b.shape.radius))

    result = {}
    for cid, spheres in clumps.items():
        cx = np.mean([s[0] for s in spheres])
        cy = np.mean([s[1] for s in spheres])
        result[cid] = {"center": (cx, cy), "spheres": spheres}
    return result


def save_gravity_results():
    """Save particle positions after gravity deposition (before extraction)."""
    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = b.state.pos
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId])

    if data:
        data = np.array(data)
        # np.save(gravity_output_name, data)
        # print(f"Saved gravity deposition: {len(data)} spheres to {gravity_output_name}")


def extract_region():
    """Extract clumps within target region and set up periodic cell."""
    global phase

    # Save state after gravity deposition
    save_gravity_results()

    print("\n=== Phase 2: Extracting region ===")

    clump_data = get_clump_data()
    if not clump_data:
        print("ERROR: No clumps found!")
        O.pause()
        return

    all_y = [d["center"][1] for d in clump_data.values()]
    min_y, max_y = min(all_y), max(all_y)

    # Extraction bounds: target_size x target_size, starting extract_margin above bottom
    y_min = extract_margin
    y_max = y_min + target_size
    x_min = (box_width - target_size) / 2  # Center horizontally
    x_max = x_min + target_size

    print(f"Particle Y range: {min_y:.2f} to {max_y:.2f}")
    print(f"Extracting X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")

    if y_max > max_y:
        print(f"WARNING: Not enough material. Add more material and retry.")
        exit()

    # Select clumps whose centers are inside the region
    selected = []
    for cid, d in clump_data.items():
        cx, cy = d["center"]
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            selected.append((cid, d))

    print(f"Selected {len(selected)} clumps")

    # Clear all bodies
    for b in O.bodies:
        if b.id >= 0:
            O.bodies.erase(b.id)

    # Enable periodic boundaries
    O.periodic = True

    # Cell size must be large enough for clumps
    max_chip_dim = max(max(s[0], s[1]) for s in chip_shapes)
    max_chip_extent = (max_chip_dim + 1) * base_radius * 1.5
    periodic_size = max(target_size, 2.5 * max_chip_extent)
    cell_depth = 2 * base_radius * 1.2  # Thin cell in Z, manually adjusted to visually have no periodicity in z.

    O.cell.setBox(periodic_size, periodic_size, cell_depth)
    print(f"Periodic cell: {periodic_size:.2f} x {periodic_size:.2f} x {cell_depth:.2f}")

    # Recreate clumps with shifted positions
    for _, data in selected:
        spheres = []
        for sx, sy, sr in data["spheres"]:
            new_x = (sx - x_min) % periodic_size
            new_y = (sy - y_min) % periodic_size
            spheres.append(sphere([new_x, new_y, base_radius * 1.1], sr, material=mat))
        if spheres:
            cid, _ = O.bodies.appendClumped(spheres)
            O.bodies[cid].state.blockedDOFs = 'zXY'

    # Update engines for relaxation (no walls, no gravity, periodic)
    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=damping),
        PyRunner(command='update()', realPeriod=0.5),
    ]

    # Recalculate timestep for new configuration
    O.dt = 0.5 * PWaveTimeStep()

    n_bodies = len([b for b in O.bodies if b.isClump])
    print(f"Bodies: {n_bodies} clumps")

    global phase_start_iter
    phase = "relaxing"
    phase_start_iter = O.iter
    print("\n=== Relaxing periodic structure ===")


def save_results():
    """Save particle positions."""
    O.pause()

    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId])

    data = np.array(data)
    np.save(output_name, data)
    np.save(output_name.replace('.npy', '_meta.npy'), np.array([O.cell.size[0], O.cell.size[1]]))

    vf = sum(np.pi * r**2 for r in data[:, 2]) / (O.cell.size[0] * O.cell.size[1])
    print(f"\nDone! Saved {len(data)} spheres to {output_name}")
    print(f"Cell: {O.cell.size[0]:.3f} x {O.cell.size[1]:.3f}, volume fraction: {vf:.4f}")


def update():
    """Main simulation control."""
    global phase, stable_count

    unbal = unbalancedForce()
    n_interactions = len(O.interactions)
    phase_iter = O.iter - phase_start_iter
    max_vel = None

    # Check for max iterations fallback
    max_iter = max_iter_gravity if phase == "gravity" else max_iter_relaxing
    if max_iter is not None and phase_iter >= max_iter:
        print(f"{phase}: Max iterations ({max_iter}) reached, forcing convergence")
        stable_count = 3  # Force transition

    # For relaxation phase, need interactions to actually relax
    if phase == "relaxing" and n_interactions == 0:
        if O.iter % 5000 == 0:
            print(f"{phase}: NO INTERACTIONS - nothing to relax! (unbal={unbal:.4f})")
        # Consider it stable if no interactions (nothing to do)
        is_stable = True
    else:
        # Check unbalanced force first
        unbal_ok = unbal < 0.03

        # Only check velocity if enabled and unbal is satisfied
        if check_max_vel and unbal_ok:
            max_vel = max((b.state.vel.norm() for b in O.bodies if b.isClump), default=0)
            is_stable = max_vel < 0.03
        else:
            max_vel = None
            is_stable = unbal_ok

    if is_stable:
        stable_count += 1
        vel_str = f", vel={max_vel:.4f}" if max_vel is not None else ""
        print(f"{phase}: stable {stable_count}/3 (unbal={unbal:.4f}{vel_str}, interactions={n_interactions})")
    else:
        stable_count = 0
        if O.iter % 5000 == 0:
            vel_str = f", vel={max_vel:.4f}" if max_vel is not None else ""
            print(f"{phase}: unbal={unbal:.4f}{vel_str}, interactions={n_interactions}, iter={phase_iter}")

    if stable_count >= 3:
        stable_count = 0
        if phase == "gravity":
            extract_region()
        elif phase == "relaxing":
            save_results()


# === ENGINES (Phase 1: non-periodic with walls) ===
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

print("\nStarting gravity deposition...")

if gui:
    from yade import qt
    qt.View()
O.run()
