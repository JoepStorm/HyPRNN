"""
Run this script to generate RVEs to be used for data generation

This is a simple loop over createRVEs_ellipse, allowing for some settings to be passed
"""

import os
import numpy as np
from create_rve import createRVEs_ellipse

# Ellipsoids
# Fix the area of one fiber, such that 50 fibers gives vfrac = 0.4 in a square unit cell RVE
r = np.sqrt( 0.4 / (50 * np.pi))
meshsize = 0.0125

# ellipsoid axes lengths (major, minor)
ratios = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

angle = 0  # angle in radians.
meshes = 1  # Number of random meshes per num_fibers specified
num_fibers = [50]  # [10, 20, 30, 40, 50, 60, 70]   # Updated for bigger domain
seed = 1
delete_geo = False
savefolder_base_meshes = '../meshes/example'
gap_factor = 1  # optional factor to increase distance between ellipses by scaling them with this factor when computing overlap.

# Generate meshes
print(f"MESH GENERATION")
def cal_vfrac(num_fib, a, b):
    base_length_rve = np.sqrt(50 * np.pi * a * b / 0.4)     # 50 fibers, vfrac=0.4
    print(f"base_length_rve: {base_length_rve}")
    fiber_area = num_fib * np.pi * a * b
    return round(fiber_area / base_length_rve**2, 5)

# This generates all the meshes
for i, num_fibs in enumerate(num_fibers):
    for j, ratio in enumerate(ratios):
        a = r * ratio
        b = r * (1 / ratio)
        vfrac = cal_vfrac(num_fibs, a, b)
        savefolder = f"{savefolder_base_meshes}/nfib{num_fibs}_{angle:.4f}_{ratio:.2f}" #_{meshes}"
        os.makedirs(savefolder, exist_ok=True)
        createRVEs_ellipse(meshes, num_fibs, a, b, angle, vfrac, seed, delete_geo, savefolder, gap_factor=gap_factor, meshsize=meshsize)

