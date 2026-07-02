# -*- encoding=utf-8 -*-
"""3D periodic RVE with two chip populations: large bulky and small slender.

Phase 1: gravity deposition into a box.
Phase 2: brief periodic relaxation to enforce periodicity.
"""

from yade import pack
import random
import numpy as np
import os
import sys

# CLI overrides — all physics defaults are hardcoded below; batch scripts pass these.
import argparse
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--seed",             type=int,   default=0)
_p.add_argument("--output",           type=str,   default=None)
_p.add_argument("--num_chips",        type=int,   default=200)
_p.add_argument("--base_radius",      type=float, default=0.07)
_p.add_argument("--num_chips_small",  type=int,   default=0)
_p.add_argument("--base_radius_small",type=float, default=0.03)
_p.add_argument("--damping",                type=float, default=0.5)
_p.add_argument("--friction_angle",         type=float, default=0.5)
_p.add_argument("--friction_angle_small",   type=float, default=None)
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
_args, _ = _p.parse_known_args(_argv)

# === CONFIGURATION ===
SEED              = _args.seed
TARGET_SIZE       = 1.0
EXTRACT_MARGIN    = 0.1
BOX_WIDTH         = 1.2
BOX_DEPTH         = 1.2
BOX_HEIGHT        = 5.0
DAMPING           = _args.damping
FRICTION_ANGLE    = _args.friction_angle
FRICTION_ANGLE_SMALL = _args.friction_angle_small if _args.friction_angle_small is not None else FRICTION_ANGLE

# Large (bulky, multi-row) chip population
NUM_CHIPS         = _args.num_chips
BASE_RADIUS       = _args.base_radius
# CHIP_TYPES_LARGE  = [4, 9, 10]
CHIP_TYPES_LARGE  = [1, 2, 3]

# Small (slender, single-row) chip population
NUM_CHIPS_SMALL   = _args.num_chips_small
BASE_RADIUS_SMALL = _args.base_radius_small
CHIP_TYPES_SMALL  = [7, 11, 12]

MAX_ITER_GRAVITY  = 500000  # hard cap for gravity phase
MIN_ITER_GRAVITY  = 5000   # skip convergence check until chips have settled
MAX_ITER_RELAXING = 50000

OUTPUT_NAME        = _args.output or f"periodic_3D/depositions/sphere_coordinates_wet_{SEED}.npy"
GRAVITY_OUTPUT_NAME = OUTPUT_NAME.replace('.npy', '_gravity.npy')

random.seed(SEED)
print(f"Config: seed={SEED}, large={NUM_CHIPS}x{BASE_RADIUS}, "
      f"small={NUM_CHIPS_SMALL}x{BASE_RADIUS_SMALL}")

# === MATERIALS ===
mat_large = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=FRICTION_ANGLE)
mat_small = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=FRICTION_ANGLE_SMALL)

# === PHASE 1: NON-PERIODIC GRAVITY DEPOSITION ===
print(f"\nPhase 1: Gravity deposition — box {BOX_WIDTH} x {BOX_HEIGHT} x {BOX_DEPTH}")

O.bodies.append(wall((0,         0, 0), axis=0, sense= 1, material=mat_large))
O.bodies.append(wall((BOX_WIDTH, 0, 0), axis=0, sense=-1, material=mat_large))
O.bodies.append(wall((0,         0, 0), axis=1, sense= 1, material=mat_large))
O.bodies.append(wall((0,         0, 0), axis=2, sense= 1, material=mat_large))
O.bodies.append(wall((0,  0, BOX_DEPTH), axis=2, sense=-1, material=mat_large))


def make_chip(center, radius, x_offsets, z_offsets, material):
    """Create a clump from a grid of sphere offsets in the XZ plane."""
    x, y, z = center
    r = radius
    spheres = []
    for dz in z_offsets:
        for dx in x_offsets:
            ny = random.uniform(-0.05, 0.05)
            nz = random.uniform(-0.05, 0.05)
            spheres.append(sphere(
                [x + dx*r, y + ny*r, z + (dz + nz)*r], r, material=material))
    return O.bodies.appendClumped(spheres)


# === CHIP TYPE DEFINITIONS ===
CHIP_OFFSETS = {
    # Two-row (plate-like) types
    4:  ([-1.5,-0.5,0.5,1.5],           [-0.5, 0.5]),
    9:  ([-1,0,1],                       [-0.5, 0.5]),
    10: ([-2,-1,0,1,2],                  [-0.5, 0.5]),

    1: ([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [-1.5, -0.5, 0.5, 1.5]),
    2: ([-2, -1, 0, 1, 2], [-1.5, -0.5, 0.5, 1.5]),
    3: ([-3, -2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2]),
    # Single-row (slender) types
    7:  ([-2.5,-1.5,-0.5,0.5,1.5,2.5],  [0]),
    11: ([-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5], [0]),
    12: ([-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5], [0]),
}

def make_creator(x_offsets, z_offsets):
    return lambda c, r, m: make_chip(c, r, x_offsets, z_offsets, m)

all_chip_creators = {t: make_creator(*offs) for t, offs in CHIP_OFFSETS.items()}

# Build single shuffled list mixing both populations
creators_large = [all_chip_creators[t] for t in CHIP_TYPES_LARGE]
creators_small = [all_chip_creators[t] for t in CHIP_TYPES_SMALL]

all_chips = (
    [(random.choice(creators_large), BASE_RADIUS,       mat_large) for _ in range(NUM_CHIPS)] +
    [(random.choice(creators_small), BASE_RADIUS_SMALL,  mat_small) for _ in range(NUM_CHIPS_SMALL)]
)
random.shuffle(all_chips)

# Spacing: based on mean bounding sphere radius of chip types,
# weighted by population. Avoids excessive overlap at spawn while keeping
# the column count manageable for large chip counts.
def _mean_bounding_radius(types, radius):
    """Mean bounding sphere radius across chip types."""
    extents = []
    for t in types:
        x_off, z_off = CHIP_OFFSETS[t]
        max_r = max(np.sqrt(dx**2 + dz**2) * radius + radius
                    for dx in x_off for dz in z_off)
        extents.append(max_r)
    return np.mean(extents)

total_chips   = NUM_CHIPS + NUM_CHIPS_SMALL
extent_large  = _mean_bounding_radius(CHIP_TYPES_LARGE, BASE_RADIUS) if NUM_CHIPS > 0 else 0
extent_small  = _mean_bounding_radius(CHIP_TYPES_SMALL, BASE_RADIUS_SMALL) if NUM_CHIPS_SMALL > 0 else 0
weighted_extent = (extent_large * NUM_CHIPS + extent_small * NUM_CHIPS_SMALL) / total_chips
margin        = 0.1
spacing       = weighted_extent + 0.07
y_start       = 1.0        # start higher so chips clear the floor on spawn
n_cols_x      = max(1, int((BOX_WIDTH  - 2*margin) / spacing))
n_cols_z      = max(1, int((BOX_DEPTH  - 2*margin) / spacing))
per_layer     = n_cols_x * n_cols_z

for i, (creator, radius, material) in enumerate(all_chips):
    layer = i // per_layer
    idx   = i %  per_layer
    col_x = idx % n_cols_x
    col_z = idx // n_cols_x
    pos = (
        margin  + (col_x + 0.5) * spacing + random.uniform(-0.01, 0.01),
        y_start + (layer + 0.5) * spacing + random.uniform(-0.01, 0.01),
        margin  + (col_z + 0.5) * spacing + random.uniform(-0.01, 0.01),
    )
    clump_id, _ = creator(pos, radius, material)
    axis = Vector3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
    axis.normalize()
    O.bodies[clump_id].state.ori = Quaternion(axis, random.uniform(0, 2*np.pi))

n_layers      = int(np.ceil(total_chips / per_layer))
y_top         = y_start + n_layers * spacing
total_spheres = sum(1 for b in O.bodies
                    if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere')
print(f"Created {total_chips} chips ({NUM_CHIPS} large + {NUM_CHIPS_SMALL} small), "
      f"{total_spheres} spheres in {n_layers} layers, top at y={y_top:.2f}")


# === STATE MACHINE ===
phase            = "gravity"
stable_count     = 0
phase_start_iter = 0

MAX_VELOCITY     = 3.0   # m/s  — linear velocity cap
MAX_ANG_VELOCITY = 20.0  # rad/s — angular velocity cap


def cap_velocity():
    """Hard velocity cap applied every few steps during gravity phase."""
    for b in O.bodies:
        if not b.isClump:
            continue
        spd = b.state.vel.norm()
        if spd > MAX_VELOCITY:
            b.state.vel *= MAX_VELOCITY / spd
        ang = b.state.angVel.norm()
        if ang > MAX_ANG_VELOCITY:
            b.state.angVel *= MAX_ANG_VELOCITY / ang


def get_clump_data():
    """Return dict of clump_id → {center, spheres} for all sphere clumps."""
    clumps = {}
    for b in O.bodies:
        if not hasattr(b.shape, '__class__') or b.shape.__class__.__name__ != 'Sphere':
            continue
        if b.clumpId < 0:
            continue
        pos = b.state.pos
        clumps.setdefault(b.clumpId, {"spheres": []})
        clumps[b.clumpId]["spheres"].append((pos[0], pos[1], pos[2], b.shape.radius))

    result = {}
    for cid, d in clumps.items():
        sph = d["spheres"]
        cx = np.mean([s[0] for s in sph])
        cy = np.mean([s[1] for s in sph])
        cz = np.mean([s[2] for s in sph])
        result[cid] = {"center": (cx, cy, cz), "spheres": sph}
    return result


def save_gravity_results():
    data = [[b.state.pos[0], b.state.pos[1], b.state.pos[2], b.shape.radius, b.clumpId]
            for b in O.bodies
            if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere']
    if data:
        os.makedirs(os.path.dirname(GRAVITY_OUTPUT_NAME), exist_ok=True)
        np.save(GRAVITY_OUTPUT_NAME, np.array(data))
        print(f"Saved gravity deposition: {len(data)} spheres → {GRAVITY_OUTPUT_NAME}")


def extract_region():
    """Extract clumps from target cube and set up periodic cell."""
    global phase, phase_start_iter

    save_gravity_results()
    print("\n=== Phase 2: Extracting region ===")

    clump_data = get_clump_data()
    if not clump_data:
        print("ERROR: No clumps found!"); O.pause(); return

    all_y = [d["center"][1] for d in clump_data.values()]
    y_min = EXTRACT_MARGIN
    y_max = y_min + TARGET_SIZE
    x_min = (BOX_WIDTH  - TARGET_SIZE) / 2
    x_max = x_min + TARGET_SIZE
    z_min = (BOX_DEPTH  - TARGET_SIZE) / 2
    z_max = z_min + TARGET_SIZE

    print(f"Particle Y range: {min(all_y):.2f} – {max(all_y):.2f}")
    print(f"Extracting X=[{x_min:.2f},{x_max:.2f}] Y=[{y_min:.2f},{y_max:.2f}] "
          f"Z=[{z_min:.2f},{z_max:.2f}]")

    if y_max > max(all_y):
        print("WARNING: Not enough material. Add more chips."); O.pause(); return

    selected = [(cid, d) for cid, d in clump_data.items()
                if (x_min <= d["center"][0] <= x_max and
                    y_min <= d["center"][1] <= y_max and
                    z_min <= d["center"][2] <= z_max)]
    print(f"Selected {len(selected)} clumps")

    for b in O.bodies:
        if b.id >= 0:
            O.bodies.erase(b.id)

    O.periodic = True
    O.cell.setBox(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
    print(f"Periodic cell: {TARGET_SIZE:.2f}³")

    for cid, data in selected:
        spheres = [sphere(
            [(sx - x_min) % TARGET_SIZE,
             (sy - y_min) % TARGET_SIZE,
             (sz - z_min) % TARGET_SIZE],
            sr, material=mat_large)
            for sx, sy, sz, sr in data["spheres"]]
        if spheres:
            O.bodies.appendClumped(spheres)

    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=DAMPING),
        PyRunner(command='update()', realPeriod=0.5),
    ]
    O.dt = 0.5 * PWaveTimeStep()

    n_clumps = sum(1 for b in O.bodies if b.isClump)
    print(f"Bodies: {n_clumps} clumps")
    phase = "relaxing"
    phase_start_iter = O.iter
    print("\n=== Relaxing periodic structure ===")


def save_results():
    O.pause()
    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            data.append([pos[0], pos[1], pos[2], b.shape.radius, b.clumpId])
    data = np.array(data)
    os.makedirs(os.path.dirname(OUTPUT_NAME), exist_ok=True)
    np.save(OUTPUT_NAME, data)
    np.save(OUTPUT_NAME.replace('.npy', '_meta.npy'),
            np.array([O.cell.size[0], O.cell.size[1], O.cell.size[2]]))
    print(f"\nDone! {len(data)} spheres → {OUTPUT_NAME}")
    print(f"Cell: {O.cell.size[0]:.3f}³  (VF computed separately by woodchip_volume_fraction_3D.py)")


def _check_gravity():
    """Check convergence during gravity deposition."""
    global stable_count
    phase_iter = O.iter - phase_start_iter

    if phase_iter < MIN_ITER_GRAVITY:
        if O.iter % 5000 == 0:
            print(f"gravity: warming up iter={phase_iter}/{MIN_ITER_GRAVITY}")
        return

    if phase_iter >= MAX_ITER_GRAVITY:
        print(f"gravity: max iterations ({MAX_ITER_GRAVITY}) reached, forcing transition")
        extract_region()
        return

    max_vel = max((b.state.vel.norm() for b in O.bodies if b.isClump), default=0)
    unbal = unbalancedForce()

    if max_vel > 1.0:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"gravity: still settling (max_vel={max_vel:.3f}, unbal={unbal:.4f})")
        return

    if unbal < 0.03:
        stable_count += 1
        print(f"gravity: stable {stable_count}/3 (unbal={unbal:.4f})")
    else:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"gravity: unbal={unbal:.4f}, iter={phase_iter}")

    if stable_count >= 3:
        stable_count = 0
        extract_region()


def _check_relaxing():
    """Check convergence during periodic relaxation."""
    global stable_count
    phase_iter = O.iter - phase_start_iter

    if phase_iter >= MAX_ITER_RELAXING:
        print(f"relaxing: max iterations ({MAX_ITER_RELAXING}) reached, saving")
        save_results()
        return

    n_inter = len(O.interactions)
    unbal = unbalancedForce()
    is_stable = n_inter == 0 or unbal < 0.03

    if is_stable:
        stable_count += 1
        print(f"relaxing: stable {stable_count}/3 (unbal={unbal:.4f}, inter={n_inter})")
    else:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"relaxing: unbal={unbal:.4f}, inter={n_inter}, iter={phase_iter}")

    if stable_count >= 3:
        save_results()


def update():
    """Main simulation controller."""
    if phase == "gravity":
        _check_gravity()
    elif phase == "relaxing":
        _check_relaxing()


# === PHASE 1 ENGINES ===
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Wall_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    NewtonIntegrator(gravity=(0, -9.81, 0), damping=DAMPING),
    PyRunner(command='cap_velocity()', iterPeriod=50),
    PyRunner(command='update()', realPeriod=0.5),
]
O.dt = 0.5 * PWaveTimeStep()

print("\nStarting gravity deposition...")
try:
    from yade import qt
    qt.View()
    O.run()
except Exception:
    O.run(wait=True)
