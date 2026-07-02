# -*- encoding=utf-8 -*-
"""3D periodic RVE via gravity deposition + extraction + periodic relaxation."""

from yade import pack
import random
import numpy as np
import os
import argparse

# === PARSE CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="3D periodic RVE via gravity deposition")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_chips", type=int, default=100)
parser.add_argument("--small_fraction", type=float, default=0.0)
parser.add_argument("--small_scale", type=float, default=0.5)
parser.add_argument("--base_radius", type=float, default=0.07)
parser.add_argument("--target_size", type=float, default=1.0)
parser.add_argument("--extract_margin", type=float, default=0.1)
parser.add_argument("--damping", type=float, default=0.5)
parser.add_argument("--frictionAngle", type=float, default=0.5)
parser.add_argument("--position_noise", type=float, default=0.0)
parser.add_argument("--box_width", type=float, default=1.2)
parser.add_argument("--box_depth", type=float, default=1.2)
parser.add_argument("--box_height", type=float, default=5.0)
parser.add_argument("--max_iter_gravity", type=int, default=None)
parser.add_argument("--max_iter_relaxing", type=int, default=50000)
parser.add_argument("--chip_types", type=str, default="1,2,3,4,5,6",
                    help="Comma-separated chip type indices (1-10)")
parser.add_argument("--num_chips_2", type=int, default=0,
                    help="Number of chips for second population (0 = disabled)")
parser.add_argument("--base_radius_2", type=float, default=None,
                    help="Base radius for second population")
parser.add_argument("--chip_types_2", type=str, default=None,
                    help="Comma-separated chip type indices for second population")
parser.add_argument("--output", type=str, default=None,
                    help="Output .npy file path (overrides default naming)")
parser.add_argument("--no-gui", action="store_true")

# YADE passes script args after '--', filter out YADE's own args
import sys
script_args = []
if '--' in sys.argv:
    script_args = sys.argv[sys.argv.index('--') + 1:]
else:
    # Try parsing all args after the script name
    script_args = sys.argv[1:]

args = parser.parse_args(script_args)

# === CONFIGURATION ===
seed = args.seed
small_fraction = args.small_fraction
num_chips = args.num_chips
small_scale = args.small_scale
base_radius = args.base_radius
target_size = args.target_size
extract_margin = args.extract_margin
damping = args.damping
frictionAngle = args.frictionAngle
position_noise = args.position_noise

# Convergence options
max_iter_gravity = args.max_iter_gravity
max_iter_relaxing = args.max_iter_relaxing

# Box dimensions (larger than RVE to avoid boundary effects)
box_width = args.box_width
box_depth = args.box_depth
box_height = args.box_height

show_gui = not args.no_gui

# Output files
if args.output:
    output_name = args.output
    gravity_output_name = args.output.replace('.npy', '_gravity.npy')
else:
    name = f'_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}'
    output_name = f"periodic_3D/depositions/sphere_coordinates_periodic_relax{name}.npy"
    gravity_output_name = f"periodic_3D/depositions/sphere_coordinates_gravity{name}.npy"

random.seed(seed)

print(f"Config: seed={seed}, chips={num_chips}, friction={frictionAngle}, damping={damping}")

# === PHASE 1: NON-PERIODIC GRAVITY DEPOSITION ===
print(f"\nPhase 1: Gravity deposition (non-periodic)")
print(f"Box: {box_width} x {box_height} x {box_depth}")

# Material for all bodies
mat = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=frictionAngle)
O.materials.append(mat)

# Create 3D box using walls (left, right, bottom, front, back walls)
O.bodies.append(wall((0, 0, 0), axis=0, sense=1, material=mat))           # left wall (X=0)
O.bodies.append(wall((box_width, 0, 0), axis=0, sense=-1, material=mat))  # right wall (X=box_width)
O.bodies.append(wall((0, 0, 0), axis=1, sense=1, material=mat))           # bottom wall (Y=0)
O.bodies.append(wall((0, 0, 0), axis=2, sense=1, material=mat))           # front wall (Z=0)
O.bodies.append(wall((0, 0, box_depth), axis=2, sense=-1, material=mat))  # back wall (Z=box_depth)


def noisy_pos(x, y, z, r):
    """Add small random deviation to a 3D position."""
    dx = random.uniform(-position_noise, position_noise) * r
    dy = random.uniform(-position_noise, position_noise) * r
    dz = random.uniform(-position_noise, position_noise) * r
    return [x + dx, y + dy, z + dz]


def make_chip(center, radius, x_offsets, z_offsets):
    """Create a clump from a grid of sphere offsets in the XZ plane (thin in Y)."""
    x, y, z = center
    r = radius
    spheres = []
    for dz in z_offsets:
        for dx in x_offsets:
            noise_y = random.uniform(-0.05, 0.05)
            noise_z = random.uniform(-0.05, 0.05)
            spheres.append(sphere(noisy_pos(
                x + dx*r, y + noise_y*r, z + (dz + noise_z)*r, r), r, material=mat))
    return O.bodies.appendClumped(spheres)


# === CHIP TYPE DEFINITIONS ===
# All chips are thin (1 sphere thick in Y) — realistic woodchip shapes.
# "Flat rectangular" = wide in XZ, "beam" = long in X only.

def create_chip_type1(center, radius):
    """Flat rectangular chip - 4x2 spheres (8 total)."""
    return make_chip(center, radius,
                     [-1.5, -0.5, 0.5, 1.5], [-0.5, 0.5])

def create_chip_type2(center, radius):
    """Flat rectangular chip - 5x2 spheres (10 total)."""
    return make_chip(center, radius,
                     [-2, -1, 0, 1, 2], [-0.5, 0.5])

def create_chip_type3(center, radius):
    """Flat rectangular chip - 3x2 spheres (6 total)."""
    return make_chip(center, radius,
                     [-1, 0, 1], [-0.5, 0.5])

def create_chip_type4(center, radius):
    """Flat rectangular chip - 4x3 spheres (12 total)."""
    return make_chip(center, radius,
                     [-1.5, -0.5, 0.5, 1.5], [-1, 0, 1])

def create_chip_type5(center, radius):
    """Beam chip - 1x4 spheres (4 total)."""
    return make_chip(center, radius,
                     [-1.5, -0.5, 0.5, 1.5], [0])

def create_chip_type6(center, radius):
    """Beam chip - 1x5 spheres (5 total)."""
    return make_chip(center, radius,
                     [-2, -1, 0, 1, 2], [0])

def create_chip_type7(center, radius):
    """Beam chip - 1x6 spheres (6 total)."""
    return make_chip(center, radius,
                     [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [0])

def create_chip_type8(center, radius):
    """Beam chip - 1x3 spheres (3 total)."""
    return make_chip(center, radius,
                     [-1, 0, 1], [0])

def create_chip_type9(center, radius):
    """Flat rectangular chip - 3x3 spheres (9 total)."""
    return make_chip(center, radius,
                     [-1, 0, 1], [-1, 0, 1])

def create_chip_type10(center, radius):
    """Flat rectangular chip - 5x3 spheres (15 total)."""
    return make_chip(center, radius,
                     [-2, -1, 0, 1, 2], [-1, 0, 1])


# Create chips with size distribution using grid-based initialization
all_chip_creators = {
    1: create_chip_type1, 2: create_chip_type2, 3: create_chip_type3,
    4: create_chip_type4, 5: create_chip_type5, 6: create_chip_type6,
    7: create_chip_type7, 8: create_chip_type8, 9: create_chip_type9,
    10: create_chip_type10,
}

# Build single mixed spawn list (like dumping a mixed bag)
selected_types = [int(t) for t in args.chip_types.split(",")]
creators_1 = [all_chip_creators[t] for t in selected_types]
spawn_list = [(random.choice(creators_1), base_radius) for _ in range(num_chips)]

has_pop2 = args.num_chips_2 > 0 and args.chip_types_2 and args.base_radius_2
if has_pop2:
    selected_types_2 = [int(t) for t in args.chip_types_2.split(",")]
    creators_2 = [all_chip_creators[t] for t in selected_types_2]
    for _ in range(args.num_chips_2):
        spawn_list.append((random.choice(creators_2), args.base_radius_2))
    print(f"Population 1: {num_chips} chips, types {selected_types}, radius {base_radius}")
    print(f"Population 2: {args.num_chips_2} chips, types {selected_types_2}, radius {args.base_radius_2}")
else:
    print(f"Using chip types: {selected_types}, radius: {base_radius}")

# Shuffle to mix both populations randomly
random.shuffle(spawn_list)
total_chips = len(spawn_list)

# Spawn chips on a generous grid with random 3D rotation
max_radius = max(r for _, r in spawn_list)
chip_spacing = 6 * max_radius
margin = 0.1
n_cols_x = max(1, int((box_width - 2 * margin) / chip_spacing))
n_cols_z = max(1, int((box_depth - 2 * margin) / chip_spacing))
chips_per_layer = n_cols_x * n_cols_z

for i, (creator, radius) in enumerate(spawn_list):
    layer = i // chips_per_layer
    pos_in_layer = i % chips_per_layer
    col_x = pos_in_layer % n_cols_x
    col_z = pos_in_layer // n_cols_x
    pos = (
        margin + (col_x + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        0.3 + (layer + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        margin + (col_z + 0.5) * chip_spacing + random.uniform(-0.02, 0.02)
    )

    clump_id, member_ids = creator(pos, radius)

    # Apply random 3D rotation via quaternion
    angle = random.uniform(0, 2 * np.pi)
    axis = Vector3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
    axis.normalize()
    q = Quaternion(axis, angle)
    O.bodies[clump_id].state.ori = q

n_layers_used = int(np.ceil(total_chips / chips_per_layer))
total_spheres = sum(1 for b in O.bodies
                    if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere')
print(f"Created {total_chips} chips ({total_spheres} spheres) in {n_layers_used} layers")


# === STATE MACHINE ===
phase = "gravity"
stable_count = 0
phase_start_iter = 0


def get_clump_data():
    """Get clump centers and sphere data (3D)."""
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

    result = {}
    for cid, spheres in clumps.items():
        cx = np.mean([s[0] for s in spheres])
        cy = np.mean([s[1] for s in spheres])
        cz = np.mean([s[2] for s in spheres])
        result[cid] = {"center": (cx, cy, cz), "spheres": spheres}
    return result


def save_gravity_results():
    """Save particle positions after gravity deposition (before extraction)."""
    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = b.state.pos
            data.append([pos[0], pos[1], pos[2], b.shape.radius, b.clumpId])

    if data:
        data = np.array(data)
        os.makedirs(os.path.dirname(gravity_output_name), exist_ok=True)
        np.save(gravity_output_name, data)
        print(f"Saved gravity deposition: {len(data)} spheres to {gravity_output_name}")


def extract_region():
    """Extract clumps within target region and set up periodic cell."""
    global phase, phase_start_iter

    save_gravity_results()

    print("\n=== Phase 2: Extracting region ===")

    clump_data = get_clump_data()
    if not clump_data:
        print("ERROR: No clumps found!")
        O.pause()
        return

    all_y = [d["center"][1] for d in clump_data.values()]
    min_y, max_y = min(all_y), max(all_y)

    # Extraction bounds: target_size cube, starting extract_margin above bottom
    y_min = extract_margin
    y_max = y_min + target_size
    x_min = (box_width - target_size) / 2
    x_max = x_min + target_size
    z_min = (box_depth - target_size) / 2
    z_max = z_min + target_size

    print(f"Particle Y range: {min_y:.2f} to {max_y:.2f}")
    print(f"Extracting X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")

    if y_max > max_y:
        print(f"WARNING: Not enough material. Add more chips and retry.")
        O.pause()
        return

    # Select clumps whose centers are inside the region
    selected = []
    for cid, d in clump_data.items():
        cx, cy, cz = d["center"]
        if x_min <= cx <= x_max and y_min <= cy <= y_max and z_min <= cz <= z_max:
            selected.append((cid, d))

    print(f"Selected {len(selected)} clumps")

    # Clear all bodies
    for b in O.bodies:
        if b.id >= 0:
            O.bodies.erase(b.id)

    # Enable periodic boundaries
    O.periodic = True

    # Cell size for periodic RVE
    periodic_size = target_size
    O.cell.setBox(periodic_size, periodic_size, periodic_size)
    print(f"Periodic cell: {periodic_size:.2f} x {periodic_size:.2f} x {periodic_size:.2f}")

    # Recreate clumps with shifted positions
    for _, data in selected:
        spheres = []
        for sx, sy, sz, sr in data["spheres"]:
            new_x = (sx - x_min) % periodic_size
            new_y = (sy - y_min) % periodic_size
            new_z = (sz - z_min) % periodic_size
            spheres.append(sphere([new_x, new_y, new_z], sr, material=mat))
        if spheres:
            O.bodies.appendClumped(spheres)

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

    O.dt = 0.5 * PWaveTimeStep()

    n_bodies = len([b for b in O.bodies if b.isClump])
    print(f"Bodies: {n_bodies} clumps")

    phase = "relaxing"
    phase_start_iter = O.iter
    print("\n=== Relaxing periodic structure ===")


def save_results():
    """Save final particle positions."""
    O.pause()

    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            data.append([pos[0], pos[1], pos[2], b.shape.radius, b.clumpId])

    data = np.array(data)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    np.save(output_name, data)
    np.save(output_name.replace('.npy', '_meta.npy'),
            np.array([O.cell.size[0], O.cell.size[1], O.cell.size[2]]))

    vf = sum(4/3 * np.pi * r**3 for r in data[:, 3]) / (O.cell.size[0] * O.cell.size[1] * O.cell.size[2])
    print(f"\nDone! Saved {len(data)} spheres to {output_name}")
    print(f"Cell: {O.cell.size[0]:.3f} x {O.cell.size[1]:.3f} x {O.cell.size[2]:.3f}, volume fraction: {vf:.4f}")


def update():
    """Main simulation control."""
    global phase, stable_count

    unbal = unbalancedForce()
    n_interactions = len(O.interactions)
    phase_iter = O.iter - phase_start_iter

    # Check for max iterations fallback
    max_iter = max_iter_gravity if phase == "gravity" else max_iter_relaxing
    if max_iter is not None and phase_iter >= max_iter:
        print(f"{phase}: Max iterations ({max_iter}) reached, forcing convergence")
        stable_count = 3

    if phase == "relaxing" and n_interactions == 0:
        if O.iter % 5000 == 0:
            print(f"{phase}: NO INTERACTIONS - nothing to relax! (unbal={unbal:.4f})")
        is_stable = True
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

# O.dt = 0.5 * PWaveTimeStep()
O.dt = 1.0 * PWaveTimeStep()

print("\nStarting gravity deposition...")

if show_gui:
    from yade import qt
    qt.View()
    O.run()
else:
    O.run(wait=True)
