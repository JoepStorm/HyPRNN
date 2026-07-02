# -*- encoding=utf-8 -*-
"""3D gravity deposition → horizontal slice → 2D periodic sheet, WITH filler.

Variant of yade_woodchip_2D.py that adds a second particle population: "filler".
Filler particles interact mechanically with the wood during deposition and
relaxation — they take up space and hold voids open — but are tagged separately
so the mesher (mesh_rve_filler_2D.py) can drop them and treat the space they
occupied as matrix/void.

There is a single wood chip type: a slender single-row chip lying flat in the
horizontal plane. Its in-plane orientation is set by one deposition --angle
(0..1 → 0..180° about the vertical axis); the vertical tilt is forced to 0 for
every chip, so the flat chips can be initialised close together vertically.
Filler blobs are single, slightly oversized spheres.

Apart from material parameters there are two real knobs:
    --angle            deposition angle, 0..1 → 0..180° in the horizontal plane
    --filler_fraction  fraction of deposited particle volume that is filler

Output format adds a 5th column to coords_2D_*.npy:
    [x, y, radius, clumpId, type]   with type 0 = wood, 1 = filler.

Phase 1: Non-periodic gravity deposition into a box (wood + filler mixed).
Phase 2: Slice at cut_height, extract 2D circles, periodic relaxation.
"""

from yade import pack
import random
import numpy as np
import os
import sys

import argparse
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--seed",                  type=int,   default=0)
_p.add_argument("--output",                type=str,   default=None)
_p.add_argument("--num_chips",             type=int,   default=1700, help="Total deposited volume, in wood-chip equivalents (the chip count at 0%% filler)")
_p.add_argument("--base_radius",           type=float, default=0.02)
_p.add_argument("--angle",                 type=float, default=0.0, help="Deposition angle, 0..1 → 0..180° about the vertical (Y) axis")
_p.add_argument("--filler_fraction",       type=float, default=0.0, help="Fraction of deposited particle volume that is filler (0..1)")
_p.add_argument("--position_jitter",       type=float, default=0.0, help="Randomness of initial grid positions, 0 (regular) .. 1 (fills the gap to neighbours)")
_p.add_argument("--init_velocity",         type=float, default=.1, help="Std-dev of random initial velocity kick per chip (capped by MAX_VELOCITY_GRAVITY)")
_p.add_argument("--damping",               type=float, default=0.8)
_p.add_argument("--friction_angle",        type=float, default=0.5)
_p.add_argument("--friction_angle_filler", type=float, default=0.3)
_p.add_argument("--cut_height",            type=float, default=0.066, help="Y height at which to slice the 3D packing into 2D circles")
_p.add_argument("--min_circle_frac",       type=float, default=0.3, help="Drop sliced circles smaller than this fraction of the sphere radius (grazing cuts); wood clumps left with <2 circles are dropped entirely")
_p.add_argument("--pause",                 action="store_true", help="Open the GUI paused at the initial spawn (press play to start)")
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
_args, _ = _p.parse_known_args(_argv)

# === CONFIGURATION ===
SEED                  = _args.seed
TARGET_SIZE           = 1.0
BOX_WIDTH             = 1.2
BOX_DEPTH             = 1.2
BOX_HEIGHT            = 3.0
DAMPING               = _args.damping
FRICTION_ANGLE        = _args.friction_angle
FRICTION_ANGLE_FILLER = _args.friction_angle_filler
CUT_HEIGHT            = _args.cut_height
MIN_CIRCLE_FRAC       = min(max(_args.min_circle_frac, 0.0), 1.0)

NUM_CHIPS             = _args.num_chips
BASE_RADIUS           = _args.base_radius
ANGLE                 = _args.angle
FILLER_FRACTION       = _args.filler_fraction
POSITION_JITTER       = max(0.0, _args.position_jitter)
INIT_VELOCITY         = max(0.0, _args.init_velocity)
PAUSE_AT_INIT         = _args.pause

MAX_ITER_GRAVITY      = 300000
MIN_ITER_GRAVITY      = 3000
MAX_ITER_RELAXING     = 30000

OUTPUT_NAME = _args.output or f"data/datasetv1/coords_2D_{SEED}_{NUM_CHIPS}_{FILLER_FRACTION}.npy"

random.seed(SEED)


# === CHIP GEOMETRY (sphere centre offsets, in units of radius) ===
# Wood: slender single-row chip lying flat in the horizontal (XZ) plane.
WOOD_X = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # = -3.5 to 3.5 = 7. Aspect ratio is 3.5
WOOD_Y = [0.0]  # = -1 to 1 = 2
WOOD_Z = [0.0]  # = -1 to 1 = 2
# Filler: a single, slightly oversized sphere (sacrificial pore template).
FILLER_RADIUS_FACTOR = 1.5  # filler radius in units of the wood sphere radius

def clump_volume(x_off, y_off, z_off, n=64):
    """Union (solid) volume of the overlapping unit-radius spheres in a clump.

    The spheres overlap heavily, so a naive sphere count overestimates the
    volume; this grid estimate captures the true occupied volume. Scale-free
    (unit radius) — only the wood/filler ratio is used.
    """
    centers = np.array([(dx, dy, dz)
                        for dy in y_off for dz in z_off for dx in x_off], float)
    lo, hi = centers.min(0) - 1.0, centers.max(0) + 1.0
    axes = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(grid), bool)
    for c in centers:
        inside |= ((grid - c) ** 2).sum(1) <= 1.0
    cell = np.prod([(hi[k] - lo[k]) / (n - 1) for k in range(3)])
    return inside.sum() * cell


VOL_WOOD   = clump_volume(WOOD_X, WOOD_Y, WOOD_Z)
VOL_FILLER = (4.0 / 3.0) * np.pi * FILLER_RADIUS_FACTOR ** 3

# num_chips sets the TOTAL deposited solid volume (the count at 0% filler).
# Filler replaces wood by volume: a fraction f of that volume becomes filler and
# the rest stays wood, so adding filler reduces the wood count instead of adding
# on top. Counts use each clump's actual overlap-aware volume (filler ≈ 1.04×
# wood here), not a naive sphere count.
_f = min(max(FILLER_FRACTION, 0.0), 1.0)
NUM_WOOD   = int(round((1.0 - _f) * NUM_CHIPS))
NUM_FILLER = int(round(_f * NUM_CHIPS * VOL_WOOD / VOL_FILLER))

print(f"Config: seed={SEED}, total≈{NUM_CHIPS} chip-vol → "
      f"wood={NUM_WOOD} + filler={NUM_FILLER} "
      f"(fraction={FILLER_FRACTION:.2f}, v_f/v_w={VOL_FILLER/VOL_WOOD:.3f}), "
      f"r={BASE_RADIUS}, angle={ANGLE:.2f} ({ANGLE*180:.0f}°), "
      f"jitter={POSITION_JITTER:.2f}, v0={INIT_VELOCITY:.2f}, cut_y={CUT_HEIGHT}")


# === MATERIALS ===
# young here is a numerical contact stiffness
mat_wood   = FrictMat(young=1e6, poisson=0.3, density=500, frictionAngle=FRICTION_ANGLE,        label='wood')
mat_filler = FrictMat(young=1e6, poisson=0.3, density=400, frictionAngle=FRICTION_ANGLE_FILLER, label='filler')

# === PHASE 1: NON-PERIODIC GRAVITY DEPOSITION ===
print(f"\nPhase 1: Gravity deposition — box {BOX_WIDTH} × {BOX_HEIGHT} × {BOX_DEPTH}")

O.bodies.append(wall((0,         0, 0), axis=0, sense= 1, material=mat_wood))
O.bodies.append(wall((BOX_WIDTH, 0, 0), axis=0, sense=-1, material=mat_wood))
O.bodies.append(wall((0,         0, 0), axis=1, sense= 1, material=mat_wood))
O.bodies.append(wall((0,         0, 0), axis=2, sense= 1, material=mat_wood))
O.bodies.append(wall((0,  0, BOX_DEPTH), axis=2, sense=-1, material=mat_wood))


def make_chip(center, radius, x_offsets, y_offsets, z_offsets, material):
    """Create a clump from a grid of spheres around `center` (offsets in radii)."""
    x, y, z = center
    r = radius
    spheres = []
    for dy in y_offsets:
        for dz in z_offsets:
            for dx in x_offsets:
                jy = random.uniform(-0.05, 0.05)
                jz = random.uniform(-0.05, 0.05)
                spheres.append(sphere(
                    [x + dx*r, y + (dy + jy)*r, z + (dz + jz)*r], r, material=material))
    return O.bodies.appendClumped(spheres)


def make_wood(center, radius, material):
    return make_chip(center, radius, WOOD_X, WOOD_Y, WOOD_Z, material)


def make_filler(center, radius, material):
    """Single oversized sphere, wrapped as a one-member clump like the wood."""
    s = sphere(center, FILLER_RADIUS_FACTOR * radius, material=material)
    return O.bodies.appendClumped([s])


all_chips = (
    [(make_wood,   mat_wood,   False) for _ in range(NUM_WOOD)] +
    [(make_filler, mat_filler, True)  for _ in range(NUM_FILLER)]
)
random.shuffle(all_chips)


def chip_extents(x_off, y_off, z_off, r):
    """Horizontal bounding radius (in XZ) and vertical half-extent of a chip."""
    horiz = max(np.hypot(dx, dz) for dx in x_off for dz in z_off) * r + r
    vert  = max(abs(dy) for dy in y_off) * r + r
    return horiz, vert


wood_h, wood_v = chip_extents(WOOD_X, WOOD_Y, WOOD_Z, BASE_RADIUS)
fill_h = fill_v = FILLER_RADIUS_FACTOR * BASE_RADIUS
horiz = max(wood_h, fill_h)
vert  = max(wood_v, fill_v)

# Spacing of initial positions: sufficient gaps to avoid overlap; small vertical space to make dropped space smaller.
# Jitter adds randomness to positions, so accounted for in gaps.
margin     = horiz + 0.01
free_xz    = horiz          # in-layer clearance ≈ one chip half-extent
free_y     = 0.5 * vert     # small gap between stacked layers
spacing_xz = 2 * horiz + free_xz
spacing_y  = 2 * vert  + free_y
y_start    = 1.0
n_cols_x   = max(1, int((BOX_WIDTH - 2*margin) / spacing_xz))
n_cols_z   = max(1, int((BOX_DEPTH - 2*margin) / spacing_xz))
per_layer  = n_cols_x * n_cols_z
# Centre the occupied grid block in the box: n_cols is floored, so without this
# the leftover space lands entirely on the high side and chips sit offset toward
# the origin corner. Offsets are >= margin, so chips still clear the walls.
x_offset   = (BOX_WIDTH - n_cols_x * spacing_xz) / 2
z_offset   = (BOX_DEPTH - n_cols_z * spacing_xz) / 2
total_chips = NUM_WOOD + NUM_FILLER

theta = ANGLE * np.pi  # 0..1 → 0..π rad (0..180°) about the vertical Y axis

for i, (creator, material, is_filler) in enumerate(all_chips):
    layer = i // per_layer
    idx   = i %  per_layer
    col_x = idx %  n_cols_x
    col_z = idx // n_cols_x
    # Jitter is capped at half the free gap, so 0 → perfect grid and 1 → chips at
    # most touch their neighbours, never spawning inside one another.
    pos = (
        x_offset + (col_x + 0.5) * spacing_xz + POSITION_JITTER * 0.5 * free_xz * random.uniform(-1, 1),
        y_start  + (layer + 0.5) * spacing_y  + POSITION_JITTER * 0.5 * free_y  * random.uniform(-1, 1),
        z_offset + (col_z + 0.5) * spacing_xz + POSITION_JITTER * 0.5 * free_xz * random.uniform(-1, 1),
    )
    clump_id, _ = creator(pos, BASE_RADIUS, material)
    if not is_filler:
        # Rotate the flat chip within the horizontal plane; vertical tilt stays 0.
        O.bodies[clump_id].state.ori = Quaternion(Vector3(0, 1, 0), theta)
    # Random initial velocity kick (chips + filler) to break up the grid during
    # settling. Set on the clump body; members follow. Capped by cap_velocity().
    if INIT_VELOCITY > 0:
        v = INIT_VELOCITY
        O.bodies[clump_id].state.vel = Vector3(
            random.gauss(0, v), random.gauss(0, v), random.gauss(0, v))

n_layers = int(np.ceil(total_chips / per_layer))
y_top    = y_start + n_layers * spacing_y
total_spheres = sum(1 for b in O.bodies
                    if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere')
print(f"Created {total_chips} bodies ({NUM_WOOD} wood + {NUM_FILLER} filler), "
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
    """Slice at CUT_HEIGHT, extract 2D circles, set up periodic 2D sheet.

    Each clump's original material is preserved so wood and filler stay
    distinguishable through relaxation and into the saved output.
    """
    global phase, phase_start_iter

    print(f"\n=== Phase 2: Slicing at y={CUT_HEIGHT:.2f} ===")

    # Snapshot the settled 3D packing before it is sliced away, so it can be
    # inspected later (see coords_3D_to_vtk.py).
    save_3D_packing()

    # Collect all spheres and compute 2D circle intersections, keeping the
    # owning clump's material so filler stays tagged.
    clumps = {}
    clump_mat = {}
    n_grazing = 0
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
        # Reject grazing intersections: a cut passing near a sphere's edge yields
        # a tiny cap that, on a flat chip, shows up as a row of non-overlapping
        # loose circles. Because dy varies monotonically along the rigid chip, the
        # retained (large-radius) circles always form one contiguous segment.
        if r_2d < MIN_CIRCLE_FRAC * sr:
            n_grazing += 1
            continue
        clumps.setdefault(b.clumpId, []).append((sx, sz, r_2d))
        clump_mat[b.clumpId] = b.material

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
    n_filler  = sum(1 for cid in selected if clump_mat[cid].label == 'filler')
    print(f"Selected {len(selected)} clumps ({n_circles} circles, "
          f"{n_filler} of them filler) from "
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

    # Place 2D spheres (X from 3D X, Y from 3D Z, thin direction = Z),
    # reusing each clump's original material (wood vs filler).
    n_dropped_wood = 0
    for cid, circles in selected.items():
        mat = clump_mat[cid]
        is_filler = (mat.label == 'filler')
        spheres_2d = []
        for cx, cz, r_2d in circles:
            new_x = (cx - x_min) % TARGET_SIZE
            new_y = (cz - z_min) % TARGET_SIZE
            spheres_2d.append(sphere([new_x, new_y, cell_depth / 2], r_2d,
                                     material=mat))
        if len(spheres_2d) >= 2:
            cid_new, _ = O.bodies.appendClumped(spheres_2d)
            O.bodies[cid_new].state.blockedDOFs = 'zXY'
        elif len(spheres_2d) == 1 and is_filler:
            # A filler blob is a single sphere by design — keep it.
            bid = O.bodies.append(spheres_2d[0])
            O.bodies[bid].state.blockedDOFs = 'zXY'
        else:
            # Wood reduced to a lone circle: the cut merely clipped a chip's tip,
            # so it is not a realistic cross-section — drop it.
            n_dropped_wood += 1

    n_clumps = sum(1 for b in O.bodies if b.isClump)
    print(f"Bodies: {n_clumps} clumps "
          f"(filtered {n_grazing} grazing circles < {MIN_CIRCLE_FRAC:.2f}·r, "
          f"dropped {n_dropped_wood} single-circle wood remnants)")

    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom()],
            [Ip2_FrictMat_FrictMat_FrictPhys()],
            [Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=DAMPING),
        PyRunner(command='cap_velocity(MAX_VELOCITY_RELAXING)', iterPeriod=1),
        PyRunner(command='update()', realPeriod=0.5),
    ]
    O.dt = 0.5 * PWaveTimeStep()

    phase = "relaxing"
    phase_start_iter = O.iter
    print("=== Relaxing 2D periodic sheet (wood + filler) ===")


def save_3D_packing(tag='coords_3D'):
    """Save the current 3D sphere packing (all wood + filler spheres).

    Columns: [x, y, z, radius, clumpId, type]  (type 0 = wood, 1 = filler).
    Written alongside the 2D output with a `tag` name (default coords_3D for the
    settled packing); convert it to a ParaView-viewable .vtp with
    coords_3D_to_vtk.py.
    """
    data = []
    for b in O.bodies:
        if b.shape.__class__.__name__ != 'Sphere' or b.clumpId < 0:
            continue
        x, y, z = b.state.pos[0], b.state.pos[1], b.state.pos[2]
        is_filler = (b.material.label == 'filler')
        data.append([x, y, z, b.shape.radius, b.clumpId,
                     1.0 if is_filler else 0.0])
    data = np.array(data)

    out_3d = (OUTPUT_NAME.replace('coords_2D', tag)
              if 'coords_2D' in OUTPUT_NAME
              else OUTPUT_NAME.replace('.npy', f'_{tag}.npy'))
    out_dir = os.path.dirname(out_3d)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_3d, data)
    print(f"Saved 3D packing: {len(data)} spheres → {out_3d}")


def save_results():
    """Save 2D circle positions, cell dimensions, and a wood/filler type tag.

    Columns: [x, y, radius, clumpId, type]  (type 0 = wood, 1 = filler).
    """
    O.pause()
    data = []
    for b in O.bodies:
        if hasattr(b.shape, '__class__') and b.shape.__class__.__name__ == 'Sphere':
            pos = O.cell.wrap(b.state.pos)
            is_filler = (b.material.label == 'filler')
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId,
                         1.0 if is_filler else 0.0])
    data = np.array(data)

    out_dir = os.path.dirname(OUTPUT_NAME)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(OUTPUT_NAME, data)
    np.save(OUTPUT_NAME.replace('.npy', '_meta.npy'),
            np.array([O.cell.size[0], O.cell.size[1]]))

    n_filler = int((data[:, 4] == 1.0).sum()) if len(data) else 0
    print(f"\nDone! {len(data)} circles ({n_filler} filler) → {OUTPUT_NAME}")


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


# === ENGINES ===
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

try:
    from yade import qt
    qt.View()
    if PAUSE_AT_INIT:
        # One step so clump orientation propagates to member sphere positions
        # (state.ori alone doesn't move members until Newton integrates), then
        # the init dump captures the rotated chips.
        O.step()
        save_3D_packing(tag='coords_3D_init')
        print("\nPaused at initial spawn — press play in the GUI to start.")
    else:
        print("\nStarting gravity deposition...")
        O.run()
except Exception:
    print("\nStarting gravity deposition...")
    O.run(wait=True)
