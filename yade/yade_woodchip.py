# -*- encoding=utf-8 -*-
"""2D gravity deposition of custom wood chip-shaped clumps."""

from yade import pack, plot
import random
import numpy as np
import os
import argparse

# === DEFAULT CONFIGURATION (used for manual runs) ===
# Uncomment one block below to select configuration for manual GUI runs

# # small
# _default_output = "sphere_coordinates_small.npy"
# _default_small_fraction = 1.0
# _default_num_chips = 1000
# _default_base_radius = 0.02
# _default_frictionAngle = 0.5
# _default_damping = 0.4

# # large
# _default_output = "sphere_coordinates_large.npy"
# _default_small_fraction = 0.0
# _default_num_chips = 300
# _default_base_radius = 0.02
# _default_frictionAngle = 0.5
# _default_damping = 0.4

# # large_looser
# _default_output = "sphere_coordinates_large_looser.npy"
# _default_small_fraction = 0.0
# _default_num_chips = 300
# _default_base_radius = 0.02
# _default_frictionAngle = 1.0
# _default_damping = 0.7

# # mix
# _default_output = "sphere_coordinates_mix.npy"
# _default_small_fraction = 0.5
# _default_num_chips = 500
# _default_base_radius = 0.02
# _default_frictionAngle = 0.5
# _default_damping = 0.4

# # mix2
# _default_output = "sphere_coordinates_mix2.npy"
# _default_small_fraction = 0.8
# _default_num_chips = 600
# _default_base_radius = 0.02
# _default_frictionAngle = 0.5
# _default_damping = 0.4

# # vlarge
# _default_output = "sphere_coordinates_vlarge.npy"
# _default_small_fraction = 0.0
# _default_num_chips = 100
# _default_base_radius = 0.04
# _default_frictionAngle = 0.5
# _default_damping = 0.4

# slender
_default_output = "sphere_coordinates_slender_large_looser.npy"
_default_small_fraction = 0.0
_default_num_chips = 300
_default_base_radius = 0.02
_default_frictionAngle = 1.0
_default_damping = 0.7

_default_seed = 0
_default_show_gui = True  # set False for batch runs

# === END DEFAULT CONFIGURATION ===

# Parse command-line arguments (after -- when using yadedaily)
parser = argparse.ArgumentParser(description="YADE woodchip gravity deposition")
parser.add_argument("--output", type=str, default=f"meshes/{_default_output}", help="Output .npy filename")
parser.add_argument("--seed", type=int, default=_default_seed, help="Random seed")
parser.add_argument("--small_fraction", type=float, default=_default_small_fraction)
parser.add_argument("--num_chips", type=int, default=_default_num_chips)
parser.add_argument("--base_radius", type=float, default=_default_base_radius)
parser.add_argument("--frictionAngle", type=float, default=_default_frictionAngle)
parser.add_argument("--damping", type=float, default=_default_damping)
parser.add_argument("--no-gui", action="store_true", help="Disable GUI (for batch runs)")
parser.add_argument("--fix-angles", action="store_true", help="Fix clump angles (no rotation during settling)")
parser.add_argument("--init-angle", type=float, default=None, help="Initial angle in degrees for all clumps (default: random)")

# YADE passes arguments after -- to the script
import sys

# Debug: print sys.argv to understand what YADE passes
# print(f"DEBUG sys.argv: {sys.argv}")

# Find where -- appears and parse everything after it, or try parsing all args
try:
    idx = sys.argv.index("--")
    script_args = sys.argv[idx + 1:]
except ValueError:
    # No -- found, try parsing args that start with --
    script_args = [a for a in sys.argv[1:] if a.startswith("--") or (len(sys.argv) > 1 and sys.argv[sys.argv.index(a)-1].startswith("--"))]
    # Simpler: just try to parse everything after script name
    script_args = sys.argv[1:] if len(sys.argv) > 1 else []

# Filter to only recognized arguments to avoid argparse errors
args, unknown = parser.parse_known_args(script_args)

# Set parameters from args
seed = args.seed
output_name = args.output
small_fraction = args.small_fraction
num_chips = args.num_chips
base_radius = args.base_radius
frictionAngle = args.frictionAngle
damping = args.damping
show_gui = _default_show_gui and not args.no_gui
fix_angles = args.fix_angles
init_angle = args.init_angle  # None means random, otherwise angle in degrees

# Fixed parameters
small_scale = 0.5      # scale factor for small chips relative to big
position_noise = 0.15  # random deviation as fraction of radius (0.15 = 15% of r)
box_width = 1.2        # simulation box width (larger than RVE to avoid boundary effects)

# Damping notes:
#   - Low damping (0.1-0.3): Particles bounce and rearrange more, settling into denser configurations
#   - High damping (0.5-0.8): Particles lose energy quickly, "freezing" in place sooner → looser packings

print(f"Config: output={output_name}, seed={seed}, chips={num_chips}, friction={frictionAngle}, damping={damping}, fix_angles={fix_angles}, init_angle={init_angle}")

# =====================

# Material for all bodies
mat = FrictMat(young=1e6, poisson=0.375, density=720, frictionAngle=frictionAngle)  # frictionAngle=0.5
O.materials.append(mat)

# Create 2D box using walls (left, right, bottom walls only)
O.bodies.append(wall((0, 0, 0), axis=0, sense=1, material=mat))           # left wall
O.bodies.append(wall((box_width, 0, 0), axis=0, sense=-1, material=mat))  # right wall
O.bodies.append(wall((0, 0, 0), axis=1, sense=1, material=mat))           # bottom wall


def noisy_pos(x, y, r):
    """Add small random deviation to a position."""
    dx = random.uniform(-position_noise, position_noise) * r
    dy = random.uniform(-position_noise, position_noise) * r
    return [x + dx, y + dy, 0]


def create_chip_type1(center, scale=1.0):
    """Rectangular chip - 2 rows of 4 spheres with slight offset."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 1.5*r, y - 0.4*r, r), r, material=mat),
        sphere(noisy_pos(x - 0.5*r, y - 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x + 0.5*r, y - 0.45*r, r), r, material=mat),
        sphere(noisy_pos(x + 1.5*r, y - 0.5*r, r), r, material=mat),
        # Top row (slightly offset for organic look)
        sphere(noisy_pos(x - 1.4*r, y + 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x - 0.5*r, y + 0.45*r, r), r, material=mat),
        sphere(noisy_pos(x + 0.5*r, y + 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x + 1.6*r, y + 0.4*r, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)


def create_chip_type2(center, scale=1.0):
    """Elongated rectangular chip - 2 rows of 5 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 2*r, y - 0.45*r, r), r, material=mat),
        sphere(noisy_pos(x - 1*r, y - 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x, y - 0.48*r, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y - 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x + 2*r, y - 0.42*r, r), r, material=mat),
        # Top row
        sphere(noisy_pos(x - 1.9*r, y + 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x - 1*r, y + 0.45*r, r), r, material=mat),
        sphere(noisy_pos(x, y + 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y + 0.48*r, r), r, material=mat),
        sphere(noisy_pos(x + 2.1*r, y + 0.45*r, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)


def create_chip_type3(center, scale=1.0):
    """Shorter rectangular chip - 2 rows of 3 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 1*r, y - 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x, y - 0.45*r, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y - 0.48*r, r), r, material=mat),
        # Top row
        sphere(noisy_pos(x - 0.95*r, y + 0.48*r, r), r, material=mat),
        sphere(noisy_pos(x + 0.05*r, y + 0.5*r, r), r, material=mat),
        sphere(noisy_pos(x + 1.05*r, y + 0.45*r, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)


def create_chip_type4(center, scale=1.0):
    """Thicker rectangular chip - 3 rows of 3 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 1*r, y - 1*r, r), r, material=mat),
        sphere(noisy_pos(x, y - 0.95*r, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y - 1*r, r), r, material=mat),
        # Middle row
        sphere(noisy_pos(x - 0.95*r, y, r), r, material=mat),
        sphere(noisy_pos(x, y, r), r, material=mat),
        sphere(noisy_pos(x + 1.05*r, y, r), r, material=mat),
        # Top row
        sphere(noisy_pos(x - 1*r, y + 0.98*r, r), r, material=mat),
        sphere(noisy_pos(x + 0.02*r, y + 1*r, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y + 0.95*r, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type5(center, scale=1.0):
    """ 1 row of 4 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 1.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 1.5*r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type6(center, scale=1.0):
    """1 row of 5 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 2*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x, y, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 2*r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type7(center, scale=1.0):
    """1 row of 6 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 2.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 1.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 1.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 2.5*r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type8(center, scale=1.0):
    """1 row of 7 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 3*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 2*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x, y, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 2*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 3*r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type9(center, scale=1.0):
    """1 row of 9 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 4*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 3*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 2*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x, y, r), r, material=mat),
        sphere(noisy_pos(x + 1*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 2*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 3*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 4*r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

def create_chip_type10(center, scale=1.0):
    """1 row of 10 spheres."""
    x, y = center
    r = base_radius * scale
    spheres = [
        # Bottom row
        sphere(noisy_pos(x - 4.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 3.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 2.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 1.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x - 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 0.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 1.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 2.5*r, y, r), r, material=mat),
        sphere(noisy_pos(x + 3.5 * r, y, r), r, material=mat),
        sphere(noisy_pos(x + 4.5 * r, y, r), r, material=mat),
    ]
    return O.bodies.appendClumped(spheres)

# Create chips with size distribution using grid-based initialization
random.seed(seed)
# chip_creators = [create_chip_type1, create_chip_type2, create_chip_type3, create_chip_type4, create_chip_type5, create_chip_type6]
# chip_creators = [create_chip_type5, create_chip_type6, create_chip_type7, create_chip_type8]
chip_creators = [create_chip_type6, create_chip_type7, create_chip_type8, create_chip_type9, create_chip_type10]

# Grid-based spawning to avoid initial overlaps
# Spacing based on largest chip size (type2 is ~5 radii wide)
chip_spacing = 6 * base_radius  # minimum spacing between chip centers
margin = 0.1
n_cols = int((box_width - 2 * margin) / chip_spacing)
n_rows = int(np.ceil(num_chips / n_cols))

for i in range(num_chips):
    # Grid position with small random jitter
    col = i % n_cols
    row = i // n_cols
    pos = (
        margin + (col + 0.5) * chip_spacing + random.uniform(-0.02, 0.02),
        0.3 + (row + 0.5) * chip_spacing + random.uniform(-0.02, 0.02)
    )
    # Pick a random chip shape
    creator = random.choice(chip_creators)

    # Determine size based on configured fractions
    if random.random() < small_fraction:
        scale = small_scale
    else:
        scale = 1.0

    clump_id, member_ids = creator(pos, scale=scale)

    # Apply rotation around Z axis only (2D rotation)
    clump = O.bodies[clump_id]
    if init_angle is not None:
        angle = np.radians(init_angle)
    else:
        angle = random.uniform(0, 2 * np.pi)
    clump.state.ori = Quaternion((0, 0, 1), angle)

    # Constrain to 2D: block Z translation and X,Y rotations
    # Optionally also block Z rotation to fix angles
    clump.state.blockedDOFs = 'zXYZ' if fix_angles else 'zXY'

O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Wall_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    NewtonIntegrator(gravity=(0, -9.81, 0), damping=damping),  # 2D: gravity in -Y
    PyRunner(command='checkUnbalanced()', realPeriod=1),
]
O.dt = .5 * PWaveTimeStep()

O.trackEnergy = True


output_dir = os.getcwd()
stable_count = 0  # count consecutive stable checks


def checkUnbalanced():
    global stable_count
    unbal = unbalancedForce()

    # Only compute velocity if unbalanced force is low enough
    if unbal >= 0.03:
        stable_count = 0
        if O.iter % 5000 == 0:
            print(f"Settling... unbal={unbal:.4f}")
        return

    # Compute maximum velocity of all bodies
    max_vel = 0.0
    for b in O.bodies:
        if b.isClump or isinstance(b.shape, Sphere):
            vel = b.state.vel.norm()
            if vel > max_vel:
                max_vel = vel

    # Check if simulation is settled: low unbalanced force AND low velocity
    is_stable = max_vel < 0.03

    if is_stable:
        stable_count += 1
        print(f"Stable check {stable_count}/5 (unbal={unbal:.4f}, max_vel={max_vel:.4f})")
    else:
        stable_count = 0
        print(f"Low unbal but moving... unbal={unbal:.4f}, max_vel={max_vel:.4f}")

    # Require 2 consecutive stable checks before saving
    if stable_count >= 2:
        O.pause()
        # Collect sphere positions and radii
        data = []
        for b in O.bodies:
            if not isinstance(b.shape, Sphere):
                continue
            pos = b.state.pos
            data.append([pos[0], pos[1], b.shape.radius, b.clumpId])
        data = np.array(data)
        out_path = os.path.join(output_dir, output_name)
        np.save(out_path, data)
        print(f"Simulation stabilized. Saved {len(data)} spheres to {out_path}")


# Open GUI viewer if requested
if show_gui:
    from yade import qt
    qt.View()
    O.run()  # non-blocking, GUI stays open
else:
    O.run(wait=True)  # blocking, runs in background

