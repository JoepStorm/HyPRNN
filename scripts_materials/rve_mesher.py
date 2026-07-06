"""
RVE Mesh Generation

Discrepancies were observed between GMSH python direct mesh creation compared with system gmsh writing to a file.
Therefore, we force to use the same meshing as used in data creation
"""
import os
import numpy as np
from scripts_materials.create_rve import createRVEs_ellipse


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

