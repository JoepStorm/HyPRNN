# -*- encoding=utf-8 -*-
"""3D gravity deposition → horizontal slice → 2D periodic sheet.

Two chip populations (large bulky, small slender) mixed by volume fraction.
Sphere offsets are denser than the 3D version (step=0.5 vs step=1.0), giving
more solid chips with the same physical extent.

Phase 1: Non-periodic gravity deposition into a box.
Phase 2: Slice at cut_height, extract 2D circles, periodic relaxation.
"""

from yade import pack
import random
import numpy as np
import os
import sys

import argparse
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--seed",                 type=int,   default=0)
_p.add_argument("--output",               type=str,   default=None)
_p.add_argument("--num_chips",            type=int,   default=80)
_p.add_argument("--base_radius",          type=float, default=0.045)
_p.add_argument("--num_chips_small",      type=int,   default=0)
_p.add_argument("--base_radius_small",    type=float, default=0.02)
_p.add_argument("--damping",              type=float, default=0.8)
_p.add_argument("--friction_angle",       type=float, default=1.3)
_p.add_argument("--friction_angle_small", type=float, default=None)
_p.add_argument("--cut_height",           type=float, default=0.5,
                help="Y height at which to slice the 3D packing into 2D circles")
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
_args, _ = _p.parse_known_args(_argv)

# === CONFIGURATION ===
SEED                 = _args.seed
TARGET_SIZE          = 1.0
BOX_WIDTH            = 1.2
BOX_DEPTH            = 1.2
BOX_HEIGHT           = 3.0
DAMPING              = _args.damping
FRICTION_ANGLE       = _args.friction_angle
FRICTION_ANGLE_SMALL = _args.friction_angle_small if _args.friction_angle_small is not None else FRICTION_ANGLE
CUT_HEIGHT           = _args.cut_height

NUM_CHIPS            = _args.num_chips
BASE_RADIUS          = _args.base_radius
NUM_CHIPS_SMALL      = _args.num_chips_small
BASE_RADIUS_SMALL    = _args.base_radius_small

# CHIP_TYPES_LARGE     = [1, 2, 3]
# CHIP_TYPES_SMALL     = [7, 11, 12]

# # v6 big aspect only
# CHIP_TYPES_LARGE     = [22]
# CHIP_TYPES_SMALL     = [41]

# # v9 big aspect only
# CHIP_TYPES_LARGE     = [3]
# CHIP_TYPES_SMALL     = [12]

# v10
CHIP_TYPES_LARGE     = [1, 2, 7]
CHIP_TYPES_SMALL     = [7, 11, 12]

MAX_ITER_GRAVITY     = 300000
MIN_ITER_GRAVITY     = 3000
MAX_ITER_RELAXING    = 30000

OUTPUT_NAME = _args.output or f"periodic/2D_wet/coords_2D_{SEED}.npy"

random.seed(SEED)
print(f"Config: seed={SEED}, large={NUM_CHIPS}×{BASE_RADIUS}, "
      f"small={NUM_CHIPS_SMALL}×{BASE_RADIUS_SMALL}, cut_y={CUT_HEIGHT}")


# === CHIP OFFSET DEFINITIONS ===
def densify(offsets):
    """Insert midpoints between consecutive offsets, halving the sphere spacing."""
    if len(offsets) <= 1:
        return list(offsets)
    result = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        result.append(a)
        result.append((a + b) / 2)
    result.append(offsets[-1])
    return result


_CHIP_OFFSETS_RAW = {
    # Large (bulky, multi-row) — same physical extent as 3D_wet, denser packing
    1: ([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [-1.5, -0.5, 0.5, 1.5]),
    2: ([-2, -1, 0, 1, 2],                  [-1.5, -0.5, 0.5, 1.5]),
    3: ([-3, -2, -1, 0, 1, 2, 3],           [-2, -1, 0, 1, 2]),
    # Small (slender, single-row)
    7:  ([-2.5,-1.5,-0.5,0.5,1.5,2.5],                          [0]),
    11: ([-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5],                 [0]),
    12: ([-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5],        [0]),

    # big only, 2 different aspect ratios
    22: ([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]),           # "large"
    41: ([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [0]),  # "small"
}

# Denser chips
# CHIP_OFFSETS = {t: (densify(x), densify(z)) for t, (x, z) in _CHIP_OFFSETS_RAW.items()}
# Not denser chips
CHIP_OFFSETS = _CHIP_OFFSETS_RAW



# === MATERIALS ===
mat_large = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=FRICTION_ANGLE)
mat_small = FrictMat(young=1e6, poisson=0.3, density=800, frictionAngle=FRICTION_ANGLE_SMALL)


# === PHASE 1: NON-PERIODIC GRAVITY DEPOSITION ===
print(f"\nPhase 1: Gravity deposition — box {BOX_WIDTH} × {BOX_HEIGHT} × {BOX_DEPTH}")

O.bodies.append(wall((0,         0, 0), axis=0, sense= 1, material=mat_large))
O.bodies.append(wall((BOX_WIDTH, 0, 0), axis=0, sense=-1, material=mat_large))
O.bodies.append(wall((0,         0, 0), axis=1, sense= 1, material=mat_large))
O.bodies.append(wall((0,         0, 0), axis=2, sense= 1, material=mat_large))
O.bodies.append(wall((0,  0, BOX_DEPTH), axis=2, sense=-1, material=mat_large))


def make_chip(center, radius, x_offsets, z_offsets, material):
    """Create a clump from a dense grid of spheres in the XZ plane."""
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


def make_creator(x_offsets, z_offsets):
    return lambda c, r, m: make_chip(c, r, x_offsets, z_offsets, m)


all_chip_creators = {t: make_creator(*offs) for t, offs in CHIP_OFFSETS.items()}

creators_large = [all_chip_creators[t] for t in CHIP_TYPES_LARGE]
creators_small = [all_chip_creators[t] for t in CHIP_TYPES_SMALL]

all_chips = (
    [(random.choice(creators_large), BASE_RADIUS,       mat_large) for _ in range(NUM_CHIPS)] +
    [(random.choice(creators_small), BASE_RADIUS_SMALL, mat_small) for _ in range(NUM_CHIPS_SMALL)]
)
random.shuffle(all_chips)


def _mean_bounding_radius(types, radius):
    """Mean bounding sphere radius across chip types (uses raw offsets for spacing)."""
    extents = []
    for t in types:
        x_off, z_off = _CHIP_OFFSETS_RAW[t]
        max_r = max(np.sqrt(dx**2 + dz**2) * radius + radius
                    for dx in x_off for dz in z_off)
        extents.append(max_r)
    return np.mean(extents)


total_chips     = NUM_CHIPS + NUM_CHIPS_SMALL
extent_large    = _mean_bounding_radius(CHIP_TYPES_LARGE, BASE_RADIUS) if NUM_CHIPS > 0 else 0
extent_small    = _mean_bounding_radius(CHIP_TYPES_SMALL, BASE_RADIUS_SMALL) if NUM_CHIPS_SMALL > 0 else 0
weighted_extent = (extent_large * NUM_CHIPS + extent_small * NUM_CHIPS_SMALL) / total_chips
margin          = weighted_extent + 0.01  # just enough to keep chips inside the walls
spacing         = weighted_extent + 0.07
y_start         = 1.0
n_cols_x        = max(1, int((BOX_WIDTH  - 2*margin) / spacing))
n_cols_z        = max(1, int((BOX_DEPTH  - 2*margin) / spacing))
per_layer       = n_cols_x * n_cols_z

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

n_layers = int(np.ceil(total_chips / per_layer))
y_top    = y_start + n_layers * spacing
total_spheres = sum(1 for b in O.bodies
                    if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere')
print(f"Created {total_chips} chips ({NUM_CHIPS} large + {NUM_CHIPS_SMALL} small), "
      f"{total_spheres} spheres, {n_layers} layers, top at y={y_top:.2f}")


# === STATE MACHINE ===
phase            = "gravity"
stable_count     = 0
phase_start_iter = 0

MAX_VELOCITY_GRAVITY  = 3.0
MAX_VELOCITY_RELAXING = 0.3
MAX_ANG_VELOCITY      = 20.0


def cap_velocity(max_vel):
    """Hard velocity cap applied every few steps."""
    for b in O.bodies:
        if not b.isClump:
            continue
        spd = b.state.vel.norm()
        if spd > max_vel:
            b.state.vel *= max_vel / spd
        ang = b.state.angVel.norm()
        if ang > MAX_ANG_VELOCITY:
            b.state.angVel *= MAX_ANG_VELOCITY / ang


def slice_and_relax():
    """Slice at CUT_HEIGHT, extract 2D circles, set up periodic 2D sheet."""
    global phase, phase_start_iter

    print(f"\n=== Phase 2: Slicing at y={CUT_HEIGHT:.2f} ===")

    # Collect all spheres and compute 2D circle intersections
    clumps = {}
    for b in O.bodies:
        if not (hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere'):
            continue
        if b.clumpId < 0:
            continue
        sx, sy, sz = b.state.pos[0], b.state.pos[1], b.state.pos[2]
        sr = b.shape.radius
        dy = abs(sy - CUT_HEIGHT)
        if dy >= sr:
            continue
        r_2d = np.sqrt(sr**2 - dy**2)
        clumps.setdefault(b.clumpId, []).append((sx, sz, r_2d))

    if not clumps:
        print(f"WARNING: No spheres intersect y={CUT_HEIGHT:.2f}. "
              f"Increase num_chips or adjust cut_height.")
        O.pause()
        return

    # Select clumps whose centroid is in the XZ extraction region
    x_min = (BOX_WIDTH  - TARGET_SIZE) / 2
    x_max = x_min + TARGET_SIZE
    z_min = (BOX_DEPTH  - TARGET_SIZE) / 2
    z_max = z_min + TARGET_SIZE

    selected = {}
    for cid, circles in clumps.items():
        cx = np.mean([c[0] for c in circles])
        cz = np.mean([c[1] for c in circles])
        if x_min <= cx <= x_max and z_min <= cz <= z_max:
            selected[cid] = circles

    n_circles = sum(len(c) for c in selected.values())
    print(f"Selected {len(selected)} clumps ({n_circles} circles) from "
          f"X=[{x_min:.2f},{x_max:.2f}] Z=[{z_min:.2f},{z_max:.2f}]")

    if n_circles == 0:
        print("WARNING: No circles in extraction region.")
        O.pause()
        return

    # Clear all bodies
    for b in O.bodies:
        if b.id >= 0:
            O.bodies.erase(b.id)

    # Set up 2D periodic cell: X × Y periodic, Z thin (depth = 2 * max_r)
    max_r      = max(r for _, _, r in (c for clist in selected.values() for c in clist))
    cell_depth = 2 * max_r * 1.1

    O.periodic = True
    O.cell.setBox(TARGET_SIZE, TARGET_SIZE, cell_depth)
    print(f"Periodic cell: {TARGET_SIZE:.2f} × {TARGET_SIZE:.2f} × {cell_depth:.4f}")

    # Place 2D spheres (X from 3D X, Y from 3D Z, thin direction = Z)
    for cid, circles in selected.items():
        spheres_2d = []
        for cx, cz, r_2d in circles:
            new_x = (cx - x_min) % TARGET_SIZE
            new_y = (cz - z_min) % TARGET_SIZE
            spheres_2d.append(sphere([new_x, new_y, cell_depth / 2], r_2d,
                                     material=mat_large))
        if len(spheres_2d) >= 2:
            cid_new, _ = O.bodies.appendClumped(spheres_2d)
            O.bodies[cid_new].state.blockedDOFs = 'zXY'
        elif len(spheres_2d) == 1:
            bid = O.bodies.append(spheres_2d[0])
            O.bodies[bid].state.blockedDOFs = 'zXY'

    n_clumps = sum(1 for b in O.bodies if b.isClump)
    print(f"Bodies: {n_clumps} clumps")

    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=DAMPING),
        # PyRunner(command='cap_velocity(MAX_VELOCITY_RELAXING)', iterPeriod=50),
        PyRunner(command='cap_velocity(MAX_VELOCITY_RELAXING)', iterPeriod=1),
        PyRunner(command='update()', realPeriod=0.5),
    ]
    O.dt = 0.5 * PWaveTimeStep()

    phase = "relaxing"
    phase_start_iter = O.iter
    print("=== Relaxing 2D periodic sheet ===")


def save_results():
    """Save 2D circle positions and cell dimensions."""
    O.pause()
    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId])
    data = np.array(data)

    out_dir = os.path.dirname(OUTPUT_NAME)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(OUTPUT_NAME, data)
    np.save(OUTPUT_NAME.replace('.npy', '_meta.npy'),
            np.array([O.cell.size[0], O.cell.size[1]]))

    print(f"\nDone! {len(data)} circles → {OUTPUT_NAME}")


def _check_gravity():
    global stable_count
    phase_iter = O.iter - phase_start_iter

    if phase_iter < MIN_ITER_GRAVITY:
        if O.iter % 5000 == 0:
            print(f"gravity: warming up iter={phase_iter}/{MIN_ITER_GRAVITY}")
        return

    if phase_iter >= MAX_ITER_GRAVITY:
        print(f"gravity: max iterations ({MAX_ITER_GRAVITY}) reached, forcing transition")
        slice_and_relax()
        return

    max_vel = max((b.state.vel.norm() for b in O.bodies if b.isClump), default=0)
    unbal = unbalancedForce()

    if max_vel > 1.0:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"gravity: still settling (max_vel={max_vel:.3f}, unbal={unbal:.4f})")
        return

    # if unbal < 0.03:
    if unbal < 0.01:
        stable_count += 1
        print(f"gravity: stable {stable_count}/3 (unbal={unbal:.4f})")
    else:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"gravity: unbal={unbal:.4f}, iter={phase_iter}")

    if stable_count >= 3:
        stable_count = 0
        slice_and_relax()


def _check_relaxing():
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
    PyRunner(command='cap_velocity(MAX_VELOCITY_GRAVITY)', iterPeriod=50),
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
