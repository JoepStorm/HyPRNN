import numpy as np
import random
import os
import math
from string import Template

def createRVEs(meshes, num_fib, r, vfrac, seed, delete_geo, savefolder):
    random.seed(seed)
    meshedgetol = 0.00  # Leftover from original file

    # for num_fib in num_fibers:  # Loop over number of fibers
    leng_const_vfrac = np.sqrt(num_fib * np.pi * r ** 2 / vfrac)
    dx = leng_const_vfrac
    dy = leng_const_vfrac

    # Change mesh element size by changing "Mesh.CharacteristicLengthMin" & "..Max"

    geo_template_preamble = Template("""SetFactory("OpenCASCADE");
    Mesh.MshFileVersion = 2.2;
    Mesh.CharacteristicLengthMin = .06;
    Mesh.CharacteristicLengthMax = .06;

    Rectangle(0) = {.0, .0, 0, $dx, $dy, 0.0};
    """)
    geo_template_postamble = Template("""
    Matrix[] = BooleanDifference{ Surface{0}; }{ Surface{1:$NUM}; };
    Fibers[] = BooleanIntersection{ Surface{0}; Delete; }{ Surface{1:$NUM}; Delete; };


    Physical Surface("matrix") = {Matrix[0]:Matrix[#Matrix[]-1]};
    Physical Surface("fiber") = {Fibers[0]:Fibers[#Fibers[]-1]};
    Mesh 2;
    Coherence Mesh;
    """) # Save "rve_$id-fib$fibnum.msh";


    periodicities = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1],
                     [1, -1]]  # does not contain [0, 0], only the outside ones

    for mesh_id in range(meshes):
        """
        Geometry
        """
        x_ar = []
        y_ar = []
        i = 0
        stop = False
        x_pbc_ar = []
        y_pbc_ar = []
        x_pbc_mesh = []
        y_pbc_mesh = []

        while i < num_fib:
            # print(f"start fib, i={i}")
            i += 1
            x_new = random.uniform(0, dx)
            y_new = random.uniform(0, dy)
            attempts = 0

            if len(x_ar) > 0:  # If it is not the first
                # Check if new void overlaps (including periodics of existing voids) or if it does not touch 2 sides (creating two unattached meshes):
                stop = False
                while any(math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < 2 * r for x, y in
                          zip(x_ar + x_pbc_ar, y_ar + y_pbc_ar)) or any(
                        abs(x_new - x) < r + meshedgetol and abs(y_new - y) < r + meshedgetol for x, y in
                        zip([0, dx, dx, 0], [0, 0, dy, dy])):

                    # or any(math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < r for x, y in zip([0, dx, dx, 0], [0, 0, dy, dy])):   # Fiber overlaps corner; replace the 2nd or statement with this when moving to fibers where this is possible
                    x_new = random.uniform(0, dx)
                    y_new = random.uniform(0, dy)
                    attempts += 1
                    if attempts > 1e5:
                        print("Retrying with new voids...")
                        stop = True
                        i = 0
                        x_ar = []
                        y_ar = []
                        x_pbc_ar = []
                        y_pbc_ar = []
                        break
                if stop:
                    continue
            # Check if first attempt overlaps corner (fibers)
            # elif any(math.sqrt((x_new - x)**2 + (y_new - y)**2) < r for x, y in zip([0, 1, 1, 0],[0, 0, 1, 1])):
            #     i = 0
            #     continue

            # Check if first overlaps 2 sides (voids)
            elif any(abs(y_new - y) < r + meshedgetol for x, y in zip([0, dx, dx, 0], [0, 0, dy, dy])):
                i = 0
                continue

            x_ar.append(x_new)
            y_ar.append(y_new)
            for j in range(len(periodicities)):
                x_pbc_ar.append(x_new + periodicities[j][0] * dx)
                y_pbc_ar.append(y_new + periodicities[j][1] * dy)

        # Trim PBC:
        for x_c, y_c in zip(x_pbc_ar, y_pbc_ar):
            # Inside extended box (by definition outside [0 - dx] box
            include = False
            if not (x_c < 0 - r or x_c > dx + r or y_c < 0 - r or y_c > dy + r):
                # Most now fit, however chance that they are in corners and don't overlap; compute those here
                include = True
                outside_region = [0, 0, 0, 0]
                dx_corner = 0
                dy_corner = 0
                if x_c < 0:
                    outside_region[0] = 1
                    dx_corner = 0
                if x_c > dx:
                    outside_region[1] = 1
                    dx_corner = dx
                if y_c < 0:
                    outside_region[2] = 1
                    dy_corner = 0
                if y_c > dy:
                    outside_region[3] = 1
                    dy_corner = dy
                # Outside region = 2 -> might not be inside
                if sum(outside_region) == 2:
                    if math.sqrt((x_c - dx_corner) * (x_c - dx_corner) + (y_c - dy_corner) * (y_c - dy_corner)) > r:
                        include = False

            if include:
                x_pbc_mesh.append(x_c)
                y_pbc_mesh.append(y_c)

        """
        Create .geo file and compute mesh
        """
        geo_file = f"{savefolder}/rve_fib{num_fib}_{mesh_id}.geo"
        f = open(geo_file, "x")
        # Pre-amble
        f.write(geo_template_preamble.substitute(dx=dx, dy=dy))
        # Fibers
        fiberstring = ""
        cur_fib = 1
        for x_c, y_c in zip(x_ar + x_pbc_mesh, y_ar + y_pbc_mesh):
            fiberstring += f"Disk({cur_fib})" + " = { " + f"{x_c}, {y_c}, 0, {r}" + "};\n"
            cur_fib += 1
        f.write(fiberstring)

        # Post-amble
        num_fibers = len(x_ar + x_pbc_mesh)
        f.write(geo_template_postamble.substitute(NUM=num_fibers, id=mesh_id, fibnum=num_fib))
        f.close()

        # Run GMSH to create .msh file
        # os.system(f"gmsh {geo_file}")
        os.system(f"/home/joep/Programs/gmsh-4.13.1-Linux64/bin/gmsh2 {geo_file} -format msh22 -2")     # gmsh2 because of installation problems: python based and 'normal' gmsh mismatch
        # os.remove(f"rve_{mesh_id}-fib{num_fib}.msh")  # Remove duplicate msh file

        # Remove physical groups from .msh file, as the jemjive code expects a different format
        # This is always line 4-8
        with open(f"{savefolder}/rve_fib{num_fib}_{mesh_id}.msh", 'r') as fin:
            data = fin.read().splitlines(True)
        with open(f"{savefolder}/rve_fib{num_fib}_{mesh_id}.msh", 'w') as fout:
            fout.writelines(data[0:3])
            fout.writelines(data[8:])

        # Delete .geo file
        if delete_geo:
            os.remove(f"{geo_file}")


def check_ellipse_overlap(center1, center2, a, b, angle_rad, gap_factor=1):
    """
    Checks if two ellipses with the same size and orientation overlap or touch.

    Args:
        center1 (tuple[float, float]): The (x, y) coordinates of the center of the first ellipse.
        center2 (tuple[float, float]): The (x, y) coordinates of the center of the second ellipse.
        a (float): The length of the semi-major axis.
        b (float): The length of the semi-minor axis.
        angle_rad (float): The rotation angle of the ellipses in radians.

    Returns:
        bool: True if the ellipses overlap or touch, False otherwise.
    """
    # Calculate the vector connecting the centers
    delta_x = center2[0] - center1[0]
    delta_y = center2[1] - center1[1]

    # Rotate this vector by the negative of the ellipse angle
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)

    x_prime_c = delta_x * cos_angle - delta_y * sin_angle
    y_prime_c = delta_x * sin_angle + delta_y * cos_angle

    # Check for overlap
    overlap_value = (x_prime_c ** 2) / ((2 * a * gap_factor) ** 2) + (y_prime_c ** 2) / ((2 * b * gap_factor) ** 2)
    return overlap_value <= 1.0

def check_corner_closure(center, a, b, angle_rad, length_square):
    x_corner = [0, length_square, length_square, 0]
    y_corner = [0, 0, length_square, length_square]

    # Calculate the rotated ellipse's bounding box extents
    length_horizontal = abs(a * np.cos(angle_rad)) + abs(b * np.sin(angle_rad))
    length_vertical = abs(a * np.sin(angle_rad)) + abs(b * np.cos(angle_rad))

    # check if any corner touches two sides
    for x, y in zip(x_corner, y_corner):
        print(f"horizontal check: {abs(center[0] - x)}, {length_horizontal + 1e-6}, vertical check: {abs(center[1] - y)}, {length_vertical + 1e-6}")
        if (abs(center[0] - x) < length_horizontal + 1e-6 and
            abs(center[1] - y) < length_vertical + 1e-6):
            print(f"YES corner closure for center: {center}, a: {a}, b: {b}, angle_rad: {angle_rad}, length_square: {length_square}")
            return True
    print(f"NO corner closure for center: {center}, a: {a}, b: {b}, angle_rad: {angle_rad}, length_square: {length_square}")
    return False




def createRVEs_ellipse(meshes, num_fib, a, b, angle, vfrac, seed, delete_geo, savefolder, gap_factor=1, meshsize=0.025, id_start=0):
    random.seed(seed)

    # for num_fib in num_fibers:  # Loop over number of fibers
    leng_const_vfrac = np.sqrt(num_fib * np.pi * a * b / vfrac)
    dx = leng_const_vfrac
    dy = leng_const_vfrac

    # Change mesh element size by changing "Mesh.CharacteristicLengthMin" & "..Max"

    geo_template_preamble = Template("""SetFactory("OpenCASCADE");
    Mesh.MshFileVersion = 2.2;
    Mesh.CharacteristicLengthMin = $meshsize;
    Mesh.CharacteristicLengthMax = $meshsize;

    Rectangle(0) = {.0, .0, 0, $dx, $dy, 0.0};
    """)
    # Mesh.CharacteristicLengthMin = .025;
    # Mesh.CharacteristicLengthMax = .025;
    geo_template_postamble = Template("""
    Matrix[] = BooleanDifference{ Surface{0}; }{ Surface{1:$NUM}; };
    Fibers[] = BooleanIntersection{ Surface{0}; Delete; }{ Surface{1:$NUM}; Delete; };


    Physical Surface("matrix") = {Matrix[0]:Matrix[#Matrix[]-1]};
    Physical Surface("fiber") = {Fibers[0]:Fibers[#Fibers[]-1]};
    Mesh 2;
    Coherence Mesh;
    """) # Save "rve_$id-fib$fibnum.msh";


    periodicities = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]  # does not contain [0, 0], only the outside ones

    for mesh_id in range(meshes):
        """
        Geometry
        """
        x_ar = []
        y_ar = []
        i = 0
        stop = False
        x_pbc_ar = []
        y_pbc_ar = []
        x_pbc_mesh = []
        y_pbc_mesh = []

        # Calculate the rotated ellipse's bounding box extents. Works because all ellipses have the same orientation and size.
        length_horizontal = abs(a * np.cos(angle)) + abs(b * np.sin(angle))
        length_vertical = abs(a * np.sin(angle)) + abs(b * np.cos(angle))

        # Forbidden sliver-band width. Fiber centers that would produce a periodic
        # sliver thinner than this are rejected, because gmsh tends to break periodic BC matching.
        sliver_threshold = 0.01

        def _in_sliver_band(xc, yc):
            return (
                (dx - length_horizontal < xc < dx - length_horizontal + sliver_threshold) or
                (length_horizontal - sliver_threshold < xc < length_horizontal) or
                (dy - length_vertical < yc < dy - length_vertical + sliver_threshold) or
                (length_vertical - sliver_threshold < yc < length_vertical)
            )

        while i < num_fib:
            i += 1
            x_new = random.uniform(0, dx)
            y_new = random.uniform(0, dy)
            attempts = 0

            if len(x_ar) > 0:  # If it is not the first
                # Check if new void overlaps (including periodics of existing voids)
                # Not yet implented to check if it does not touch 2 sides (creating two unattached meshes):
                stop = False
                while any(check_ellipse_overlap((x_new, y_new), (x, y), a, b, angle, gap_factor=gap_factor) for x, y in zip(x_ar + x_pbc_ar, y_ar + y_pbc_ar)) or check_corner_closure((x_new, y_new), a, b, angle, leng_const_vfrac) or _in_sliver_band(x_new, y_new):
                    x_new = random.uniform(0, dx)
                    y_new = random.uniform(0, dy)
                    # print(f"x_new: {x_new}, y_new: {y_new}")
                    attempts += 1
                    if attempts > 1e5:
                        print("Retrying with new voids...")
                        stop = True
                        i = 0
                        x_ar = []
                        y_ar = []
                        x_pbc_ar = []
                        y_pbc_ar = []
                        break
                if stop:
                    continue
            # First fiber: reject sliver positions
            elif _in_sliver_band(x_new, y_new):
                i = 0
                continue

            x_ar.append(x_new)
            y_ar.append(y_new)
            for j in range(len(periodicities)):
                x_pbc_ar.append(x_new + periodicities[j][0] * dx)
                y_pbc_ar.append(y_new + periodicities[j][1] * dy)

        # Trim PBC:
        for x_c, y_c in zip(x_pbc_ar, y_pbc_ar):
            # Include if inside extended box; where extended box is base box + width and height of ellipse on each side
            extended_box = [-length_horizontal, dx + length_horizontal, -length_vertical, dy + length_vertical]
            include = False
            if not (x_c < extended_box[0] or x_c > extended_box[1] or y_c < extended_box[2] or y_c > extended_box[3]):
            # if not (x_c < 0 - a or x_c > dx + a or y_c < 0 - b or y_c > dy + b):
                include = True
                outside_region = [0, 0, 0, 0]
                dx_corner = 0
                dy_corner = 0
                if x_c < 0:
                    outside_region[0] = 1
                    dx_corner = 0
                if x_c > dx:
                    outside_region[1] = 1
                    dx_corner = dx
                if y_c < 0:
                    outside_region[2] = 1
                    dy_corner = 0
                if y_c > dy:
                    outside_region[3] = 1
                    dy_corner = dy
                # Outside region = 2 -> center is in square diagonally away from a corner -> must touch 2 sides to get inside -> therefore false
                if sum(outside_region) == 2:
                    include = False

            if include:
                x_pbc_mesh.append(x_c)
                y_pbc_mesh.append(y_c)

        """
        Create .geo file and compute mesh
        """
        # geo_file = f"{savefolder}/rve_fib{num_fib}_{id_start + mesh_id}.geo"
        geo_file = f"{savefolder}/rve_{id_start + mesh_id}.geo"
        # geo_file = f"{savefolder}/rve.geo"
        f = open(geo_file, "x")
        # Pre-amble
        f.write(geo_template_preamble.substitute(dx=dx, dy=dy, meshsize=meshsize))

        print(f"x_ar + x_pbc_mesh: {x_ar + x_pbc_mesh}")

        # Fibers
        fiberstring = ""
        cur_fib = 1
        for x_c, y_c in zip(x_ar + x_pbc_mesh, y_ar + y_pbc_mesh):
            # fiberstring += f"Disk({cur_fib})" + " = { " + f"{x_c}, {y_c}, 0, {r}" + "};\n"
            fiberstring += f"Disk({cur_fib})" + " = { " + f"{x_c}, {y_c}, 0, {a}, {b}" + "};\n"
            fiberstring += "Rotate {{0, 0, 1}, {" + f"{x_c}, {y_c}, 0" + "}, " + f"{angle}" + "} { Surface{" + f"{cur_fib}"  "}; }\n"
            cur_fib += 1
        f.write(fiberstring)

        # Post-amble
        num_fibers = len(x_ar + x_pbc_mesh)
        f.write(geo_template_postamble.substitute(NUM=num_fibers, id=mesh_id, fibnum=num_fib))
        f.close()

        # Run GMSH to create .msh file
        # os.system(f"gmsh {geo_file}")
        try:
            os.system(f"/home/joep/Programs/gmsh-4.13.1-Linux64/bin/gmsh2 {geo_file} -format msh22 -2")     # gmsh2 used here is simply a renamed gmsh bin file - you can just use gmsh instead.
        except:
            print(f"gmsh location incorrectly specified in createRVE.py. Change to your version.")
            exit()
        # os.remove(f"rve_{mesh_id}-fib{num_fib}.msh")  # Remove duplicate msh file

        # Remove physical groups from .msh file, as the jemjive code expects a different format
        # This is always line 4-8
        # with open(f"{savefolder}/rve_fib{num_fib}_{mesh_id}.msh", 'r') as fin:
        with open(f"{savefolder}/rve_{id_start + mesh_id}.msh", 'r') as fin:
            data = fin.read().splitlines(True)
        # with open(f"{savefolder}/rve_fib{num_fib}_{mesh_id}.msh", 'w') as fout:
        with open(f"{savefolder}/rve_{id_start + mesh_id}.msh", 'w') as fout:
            fout.writelines(data[0:3])
            fout.writelines(data[8:])

        # Delete .geo file
        if delete_geo:
            os.remove(f"{geo_file}")

# # TEST:
# meshes = 1  # Number of random meshes per num_fibers specified
# num_fibers = [9] #[1, 3, 6]  # Total number of fibers
# r = 0.2  # Fiber radius
# vfrac = 0.4  # Volume fraction. #Will struggle to find solutions above >~0.55
# seed = 0
# delete_geo = False
# createRVEs(meshes, num_fibers, r, vfrac, seed, delete_geo, 'meshes')
