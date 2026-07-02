import os
import numpy as np
from create_rve import createRVEs, createRVEs_ellipse


# Ellipsoids

# Fix the area of one fiber, such that 25 fibers gives vfrac = 0.4 in a square unit cell RVE
# UPDATE: we increase the size of the domain. We do this by scaling the fiber sizes down, keeping the area constant. Now 50 fibers = 40%
r = np.sqrt( 0.4 / (50 * np.pi))    # TODO: was 25 * np.pi before
meshsize = 0.0125 # 0.025       # 0.025 = for 25=40, 0.0125 = for 50=40

# ellipsoid axes lengths (major, minor)
ratios = [2.25] # [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5] # [1.25]       # 2 -> 2 & 0.5 = factor 4 difference. 1 -> circle.

angle = 0 #np.pi/4 * 0/3  # angle in radians. 2/3 = 30, 4/3=60, 6/3=90, 8/3=120 10/3=150
meshes = 1  # Number of random meshes per num_fibers specified
# num_fibers = [30] # [35] # [5, 10, 15, 20, 25, 30, 35] #[1, 3, 6]  # Total number of fibers
num_fibers = [50] # [40, 50, 60] #  [10, 20, 30, 40, 50, 60, 70]   # Updated for bigger domain
seed = 1
delete_geo = False
# savefolder_base_meshes = 'meshes/vfrac_ratio_biggerdomain'
savefolder_base_meshes = 'meshes/tmp_periodic_test'
# savefolder_base_meshes = 'meshes/ellipse_ratios_spaced'
gap_factor = 1 #1.015

# Generate meshes
print(f"MESH GENERATION")
def cal_vfrac(num_fib, a, b):
    # base_length_rve = np.sqrt(25 * np.pi * a * b / 0.4)     # 25 fibers, vfrac=0.4
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
        # savefolder = f"{savefolder_base_meshes}/nfib{num_fibs}_{angle:.4f}_{ratio:.2f}_{gap_factor:.3f}" #_{meshes}"
        # Make folder if it doesnt exist yet
        os.makedirs(savefolder, exist_ok=True)
        createRVEs_ellipse(meshes, num_fibs, a, b, angle, vfrac, seed, delete_geo, savefolder, gap_factor=gap_factor, meshsize=meshsize)

