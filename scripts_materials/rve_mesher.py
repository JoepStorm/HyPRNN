"""
RVE Mesh Generation

Creates RVE meshes with elliptical inclusions directly using GMSH,
without writing intermediate files.
"""
import numpy as np
import random
import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import os
from scripts_materials.create_rve import createRVEs_ellipse


def check_ellipse_overlap(center1, center2, a, b, angle_rad, gap_factor=1.0):
    """Check if two ellipses with the same size and orientation overlap."""
    delta_x = center2[0] - center1[0]
    delta_y = center2[1] - center1[1]

    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)

    x_prime_c = delta_x * cos_angle - delta_y * sin_angle
    y_prime_c = delta_x * sin_angle + delta_y * cos_angle

    overlap_value = (x_prime_c ** 2) / ((2 * a * gap_factor) ** 2) + (y_prime_c ** 2) / ((2 * b * gap_factor) ** 2)
    return overlap_value <= 1.0


def check_corner_closure(center, a, b, angle_rad, length_square):
    """Check if an ellipse would close off a corner of the RVE."""
    x_corner = [0, length_square, length_square, 0]
    y_corner = [0, 0, length_square, length_square]

    length_horizontal = abs(a * np.cos(angle_rad)) + abs(b * np.sin(angle_rad))
    length_vertical = abs(a * np.sin(angle_rad)) + abs(b * np.cos(angle_rad))

    for x, y in zip(x_corner, y_corner):
        if (abs(center[0] - x) < length_horizontal + 1e-6 and
            abs(center[1] - y) < length_vertical + 1e-6):
            return True
    return False


def generate_fiber_positions(num_fib, a, b, angle, domain_size, seed=None, gap_factor=1.0, max_attempts=100000):
    """
    Generate non-overlapping fiber positions for an RVE.

    Returns fiber centers within the base domain plus periodic images that intersect the domain.
    """
    if seed is not None:
        random.seed(seed)

    max_d = max(a, b)
    dx = dy = domain_size

    periodicities = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    x_ar = []
    y_ar = []
    x_pbc_ar = []
    y_pbc_ar = []

    i = 0
    while i < num_fib:
        i += 1
        x_new = random.uniform(0, dx)
        y_new = random.uniform(0, dy)
        attempts = 0

        if len(x_ar) > 0:
            stop = False
            while (any(check_ellipse_overlap((x_new, y_new), (x, y), a, b, angle, gap_factor=gap_factor)
                       for x, y in zip(x_ar + x_pbc_ar, y_ar + y_pbc_ar)) or
                   check_corner_closure((x_new, y_new), a, b, angle, domain_size)):
                x_new = random.uniform(0, dx)
                y_new = random.uniform(0, dy)
                attempts += 1
                if attempts > max_attempts:
                    # Restart with new positions
                    stop = True
                    i = 0
                    x_ar = []
                    y_ar = []
                    x_pbc_ar = []
                    y_pbc_ar = []
                    break
            if stop:
                continue
        else:
            # First fiber: check edge proximity
            if any(abs(y_new - y) < max_d for y in [0, dy]):
                i = 0
                continue
            if any(abs(x_new - x) < max_d for x in [0, dx]):
                i = 0
                continue

        x_ar.append(x_new)
        y_ar.append(y_new)
        for per in periodicities:
            x_pbc_ar.append(x_new + per[0] * dx)
            y_pbc_ar.append(y_new + per[1] * dy)

    # Trim periodic images to only those that intersect the domain
    length_horizontal = abs(a * np.cos(angle)) + abs(b * np.sin(angle))
    length_vertical = abs(a * np.sin(angle)) + abs(b * np.cos(angle))

    x_pbc_mesh = []
    y_pbc_mesh = []

    for x_c, y_c in zip(x_pbc_ar, y_pbc_ar):
        extended_box = [-length_horizontal, dx + length_horizontal,
                        -length_vertical, dy + length_vertical]

        if not (x_c < extended_box[0] or x_c > extended_box[1] or
                y_c < extended_box[2] or y_c > extended_box[3]):
            # Check if center is diagonally outside (in a corner region)
            outside_region = [0, 0, 0, 0]
            if x_c < 0:
                outside_region[0] = 1
            if x_c > dx:
                outside_region[1] = 1
            if y_c < 0:
                outside_region[2] = 1
            if y_c > dy:
                outside_region[3] = 1

            # If in corner region, exclude (would need to touch two sides)
            if sum(outside_region) != 2:
                x_pbc_mesh.append(x_c)
                y_pbc_mesh.append(y_c)

    return x_ar + x_pbc_mesh, y_ar + y_pbc_mesh


def create_rve_mesh(
    vfrac: float,
    aspect_ratio: float = 1.0,
    angle: float = 0.0,
    seed: int = None,
    meshsize: float = 0.025,
    gap_factor: float = 1.0,
    domain_size: float = 1.0,
    reference_num_fibers: int = 50,
    reference_vfrac: float = 0.4,
    comm: MPI.Comm = None
):
    """
    Create an RVE mesh with elliptical inclusions.

    The fiber radius is fixed based on reference values (default: 50 fibers at vfrac=0.4).
    The number of fibers is computed from the desired vfrac.
    """
    if comm is None:
        comm = MPI.COMM_SELF

    # Compute number of fibers from vfrac
    num_fibers = round(reference_num_fibers * vfrac / reference_vfrac)

    # Fixed base radius from reference values
    base_radius = np.sqrt(reference_vfrac / (reference_num_fibers * np.pi)) * domain_size

    # Fiber semi-axes for ellipse
    a = base_radius * aspect_ratio
    b = base_radius / aspect_ratio

    # Generate fiber positions
    x_fibers, y_fibers = generate_fiber_positions(
        num_fibers, a, b, angle, domain_size, seed=seed, gap_factor=gap_factor
    )

    # Create mesh using GMSH
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("rve")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize)

    factory = gmsh.model.occ

    # Create domain rectangle (tag 0)
    domain_tag = factory.addRectangle(0, 0, 0, domain_size, domain_size)

    # Create fiber ellipses (tags 1, 2, 3, ...)
    fiber_tags = []
    for x_c, y_c in zip(x_fibers, y_fibers):
        ellipse_tag = factory.addDisk(x_c, y_c, 0, a, b)
        if abs(angle) > 1e-10:
            factory.rotate([(2, ellipse_tag)], x_c, y_c, 0, 0, 0, 1, angle)
        fiber_tags.append(ellipse_tag)

    factory.synchronize()

    if len(fiber_tags) > 0:
        fiber_dimtags = [(2, t) for t in fiber_tags]

        # BooleanDifference: matrix = domain - fibers
        # removeObject=False keeps the domain, removeTool=False keeps fibers
        matrix_result, _ = factory.cut(
            [(2, domain_tag)], fiber_dimtags,
            removeObject=False, removeTool=False)
        factory.synchronize()

        # BooleanIntersection: fibers_clipped = domain ∩ fibers (clips fibers to domain)
        fiber_result, _ = factory.intersect(
            [(2, domain_tag)], fiber_dimtags,
            removeObject=True, removeTool=True)
        factory.synchronize()

        matrix_tags = [dt[1] for dt in matrix_result]
        fiber_final_tags = [dt[1] for dt in fiber_result]

        # Create physical groups
        if matrix_tags:
            gmsh.model.addPhysicalGroup(2, matrix_tags, 1, name="matrix")
        if fiber_final_tags:
            gmsh.model.addPhysicalGroup(2, fiber_final_tags, 2, name="fiber")
    else:
        gmsh.model.addPhysicalGroup(2, [domain_tag], 1, name="matrix")

    factory.synchronize()

    # Set periodic mesh constraints (left-right and bottom-top)
    boundary_entities = gmsh.model.getEntities(1)
    tol = 1e-6
    left, right, bottom, top = [], [], [], []
    for dim, tag in boundary_entities:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        if abs(bbox[0]) < tol and abs(bbox[3]) < tol:
            left.append(tag)
        elif abs(bbox[0] - domain_size) < tol and abs(bbox[3] - domain_size) < tol:
            right.append(tag)
        elif abs(bbox[1]) < tol and abs(bbox[4]) < tol:
            bottom.append(tag)
        elif abs(bbox[1] - domain_size) < tol and abs(bbox[4] - domain_size) < tol:
            top.append(tag)

    for lt, rt in zip(sorted(left, key=lambda t: gmsh.model.getBoundingBox(1, t)[1]),
                       sorted(right, key=lambda t: gmsh.model.getBoundingBox(1, t)[1])):
        gmsh.model.mesh.setPeriodic(1, [rt], [lt], [1, 0, 0, domain_size, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    for b, t in zip(sorted(bottom, key=lambda t: gmsh.model.getBoundingBox(1, t)[0]),
                    sorted(top, key=lambda t: gmsh.model.getBoundingBox(1, t)[0])):
        gmsh.model.mesh.setPeriodic(1, [t], [b], [1, 0, 0, 0, 0, 1, 0, domain_size, 0, 0, 1, 0, 0, 0, 0, 1])

    gmsh.model.mesh.generate(2)

    msh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)

    gmsh.finalize()

    return msh_data.mesh, msh_data.cell_tags, msh_data.facet_tags, domain_size


class RVEMeshConfig:
    """
    Configuration for an RVE mesh.

    The fiber radius is fixed based on reference values (default: 50 fibers at vfrac=0.4
    in a unit domain). The number of fibers is computed from the desired volume fraction.
    """

    def __init__(
        self,
        vfrac: float,
        aspect_ratio: float = 1.0,
        angle: float = 0.0,
        seed: int = None,
        meshsize: float = 0.025,
        gap_factor: float = 1.0,
        domain_size: float = 1.0,
        reference_num_fibers: int = 50,
        reference_vfrac: float = 0.4
    ):
        self.vfrac = vfrac
        self.aspect_ratio = aspect_ratio
        self.angle = angle
        self.seed = seed
        self.meshsize = meshsize
        self.gap_factor = gap_factor
        self.domain_size = domain_size
        self.reference_num_fibers = reference_num_fibers
        self.reference_vfrac = reference_vfrac

        self.num_fibers = round(self.reference_num_fibers * self.vfrac / self.reference_vfrac)
        self.base_radius = np.sqrt(self.reference_vfrac / (self.reference_num_fibers * np.pi)) * self.domain_size

    def create_mesh(self, comm: MPI.Comm = None):
        """Create the mesh in-memory via GMSH Python API."""
        print(f" ---- WARNING: CREATING A MESH VIA GMSH PYTHON API INCONSISTENT WITH DIRECT GMSH - SLOWER CONVERGENCE OBSERVED ---- ")
        return create_rve_mesh(
            vfrac=self.vfrac,
            aspect_ratio=self.aspect_ratio,
            angle=self.angle,
            seed=self.seed,
            meshsize=self.meshsize,
            gap_factor=self.gap_factor,
            domain_size=self.domain_size,
            reference_num_fibers=self.reference_num_fibers,
            reference_vfrac=self.reference_vfrac,
            comm=comm
        )

    def create_mesh_file(self, savefolder: str = '/tmp/rve_meshes', mesh_id: int = 0):
        """Create mesh via .geo/.msh files using createRVEs_ellipse. Returns (msh_path, domain_size)."""

        os.makedirs(savefolder, exist_ok=True)

        a = self.base_radius * self.aspect_ratio
        b = self.base_radius / self.aspect_ratio

        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            createRVEs_ellipse(
                meshes=1,
                num_fib=self.num_fibers,
                a=a, b=b,
                angle=self.angle,
                vfrac=self.vfrac,
                seed=self.seed if self.seed is not None else 0,
                delete_geo=False,
                savefolder=savefolder,
                gap_factor=self.gap_factor,
                meshsize=self.meshsize,
                id_start=mesh_id
            )

        msh_file = f"{savefolder}/rve_{mesh_id}.msh"
        # domain_size is computed by createRVEs_ellipse as sqrt(num_fib * pi * a * b / vfrac)
        domain_size = np.sqrt(self.num_fibers * np.pi * a * b / self.vfrac)
        return msh_file, domain_size

    def __repr__(self):
        return f"RVEMeshConfig(vfrac={self.vfrac}, num_fibers={self.num_fibers}, aspect_ratio={self.aspect_ratio}, angle={self.angle:.4f})"


if __name__ == "__main__":
    # Test mesh generation
    print("Testing RVE mesh generation...")

    config = RVEMeshConfig(
        vfrac=0.3,
        aspect_ratio=1.5,
        angle=np.pi/6,
        seed=42,
        meshsize=0.0125
    )
    print(f"Config: {config}")
    print(f"Base radius: {config.base_radius:.4f}")

    mesh, cell_tags, facet_tags, domain_size = config.create_mesh()

    print(f"Created mesh with {mesh.topology.index_map(2).size_local} cells")
    print(f"Domain size: {domain_size:.4f}")
    print(f"Cell tags unique values: {np.unique(cell_tags.values)}")

    # Compute actual volume fraction
    from dolfinx import fem
    import ufl

    V = fem.functionspace(mesh, ("DG", 0))
    fiber_indicator = fem.Function(V)
    fiber_cells = cell_tags.find(2)
    fiber_indicator.x.array[fiber_cells] = 1.0

    fiber_vol = fem.assemble_scalar(fem.form(fiber_indicator * ufl.dx))
    total_vol = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx))

    print(f"Target vfrac: {config.vfrac:.4f}")
    print(f"Actual vfrac: {fiber_vol/total_vol:.4f}")

    # Plot the mesh
    print(f"Plotting mesh...")
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts_FEM.plotting_utils import color_plot

    # Extract mesh data for plotting
    mesh_node_coords = mesh.geometry.x[:, :2]
    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)
    c_to_v = topology.connectivity(topology.dim, 0)

    num_cells = topology.index_map(topology.dim).size_local
    mesh_elem_nodes = np.array([c_to_v.links(cell)[:3] for cell in range(num_cells)])

    # Color by material tag (1=matrix, 2=fiber)
    colors = cell_tags.values

    color_plot(
        mesh_node_coords=mesh_node_coords,
        mesh_elem_nodes=mesh_elem_nodes,
        colors=colors,
        fname='results/rve_mesh_test',
        bound_values=[1, 2],
        colorscheme='Spectral_rev',
        label='Material'
    )
    print("Saved plot: results/rve_mesh_test.pdf")

    print("Done!")
