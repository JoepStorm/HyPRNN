"""Convert 3D periodic YADE output to DOLFINx mesh and save as XDMF."""

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import functionspace, Function, form, assemble_scalar
import ufl


def load_periodic_data(npy_file):
    """Load sphere data and cell dimensions."""
    data = np.load(npy_file)  # columns: x, y, z, radius, clumpId

    meta_file = npy_file.replace('.npy', '_meta.npy')
    try:
        cell_size = np.load(meta_file)
        rve_size = cell_size[0], cell_size[1], cell_size[2]
    except FileNotFoundError:
        print(f"Warning: {meta_file} not found, assuming 1.0 x 1.0 x 1.0")
        rve_size = 1.0, 1.0, 1.0

    return data[:, :5], rve_size




def add_periodic_ghosts(spheres, rve_size):
    """Add ghost copies of spheres that overlap periodic boundaries."""
    Lx, Ly, Lz = rve_size
    ghosts = []

    for x, y, z, r, cid in spheres:
        overlaps = {
            'x-': x - r < 0, 'x+': x + r > Lx,
            'y-': y - r < 0, 'y+': y + r > Ly,
            'z-': z - r < 0, 'z+': z + r > Lz,
        }

        # Face ghosts
        if overlaps['x-']: ghosts.append([x + Lx, y, z, r, cid])
        if overlaps['x+']: ghosts.append([x - Lx, y, z, r, cid])
        if overlaps['y-']: ghosts.append([x, y + Ly, z, r, cid])
        if overlaps['y+']: ghosts.append([x, y - Ly, z, r, cid])
        if overlaps['z-']: ghosts.append([x, y, z + Lz, r, cid])
        if overlaps['z+']: ghosts.append([x, y, z - Lz, r, cid])

        # Edge ghosts
        if overlaps['x-'] and overlaps['y-']: ghosts.append([x + Lx, y + Ly, z, r, cid])
        if overlaps['x-'] and overlaps['y+']: ghosts.append([x + Lx, y - Ly, z, r, cid])
        if overlaps['x+'] and overlaps['y-']: ghosts.append([x - Lx, y + Ly, z, r, cid])
        if overlaps['x+'] and overlaps['y+']: ghosts.append([x - Lx, y - Ly, z, r, cid])
        if overlaps['x-'] and overlaps['z-']: ghosts.append([x + Lx, y, z + Lz, r, cid])
        if overlaps['x-'] and overlaps['z+']: ghosts.append([x + Lx, y, z - Lz, r, cid])
        if overlaps['x+'] and overlaps['z-']: ghosts.append([x - Lx, y, z + Lz, r, cid])
        if overlaps['x+'] and overlaps['z+']: ghosts.append([x - Lx, y, z - Lz, r, cid])
        if overlaps['y-'] and overlaps['z-']: ghosts.append([x, y + Ly, z + Lz, r, cid])
        if overlaps['y-'] and overlaps['z+']: ghosts.append([x, y + Ly, z - Lz, r, cid])
        if overlaps['y+'] and overlaps['z-']: ghosts.append([x, y - Ly, z + Lz, r, cid])
        if overlaps['y+'] and overlaps['z+']: ghosts.append([x, y - Ly, z - Lz, r, cid])

        # Corner ghosts
        for dx in ([Lx] if overlaps['x-'] else ([-Lx] if overlaps['x+'] else [])):
            for dy in ([Ly] if overlaps['y-'] else ([-Ly] if overlaps['y+'] else [])):
                for dz in ([Lz] if overlaps['z-'] else ([-Lz] if overlaps['z+'] else [])):
                    if dx and dy and dz:
                        ghosts.append([x + dx, y + dy, z + dz, r, cid])

    if ghosts:
        return np.vstack([spheres, np.array(ghosts)])
    return spheres


def create_mesh(npy_file, output_xdmf, mesh_size=0.05, shrink_factor=1.0):
    """Create mesh from YADE sphere data and save as XDMF."""
    spheres, rve_size = load_periodic_data(npy_file)
    Lx, Ly, Lz = rve_size

    print(f"Loaded {len(spheres)} spheres")
    print(f"RVE size: {Lx:.3f} x {Ly:.3f} x {Lz:.3f}")

    # Shrink radii
    spheres[:, 3] *= shrink_factor

    # Add ghost spheres
    n_original = len(spheres)
    spheres = add_periodic_ghosts(spheres, rve_size)
    print(f"Added {len(spheres) - n_original} ghost spheres")

    # Volume fraction
    vf = np.sum(4/3 * np.pi * spheres[:n_original, 3]**3) / (Lx * Ly * Lz)
    print(f"Volume fraction spheres only: {vf:.4f} (Inaccurate approximation!)")

    # Build mesh with gmsh
    gmsh.initialize()
    gmsh.model.add("rve")
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Geometry.OCCParallel", 1)

    occ = gmsh.model.occ

    # Create box
    box = occ.addBox(0, 0, 0, Lx, Ly, Lz)

    # Create spheres
    sphere_tags = [occ.addSphere(x, y, z, r) for x, y, z, r, _ in spheres]
    print(f"Created {len(sphere_tags)} spheres")

    use_fragment = False  # Toggle between fragment (faster) and cut+intersect (generally easier to handle)

    if use_fragment:
        # --- FRAGMENT APPROACH ---
        all_entities = [(3, box)] + [(3, s) for s in sphere_tags]
        print("Fragmenting geometry...")
        result, entity_map = occ.fragment(all_entities, [])
        occ.synchronize()

        # Get all 3D entities that actually exist after fragmentation
        existing_entities = set(t[1] for t in gmsh.model.getEntities(3))

        # entity_map[0] = fragments from the box (matrix where not overlapping with spheres)
        # entity_map[1:] = fragments from spheres (inclusions, may include outside pieces)

        # Collect matrix tags (from box), excluding those also claimed by spheres
        matrix_candidates = set(t[1] for t in entity_map[0] if t[0] == 3)
        inclusion_candidates = set()
        for sphere_frags in entity_map[1:]:
            for dim, tag in sphere_frags:
                if dim == 3:
                    inclusion_candidates.add(tag)

        # Matrix = box fragments that are NOT also sphere fragments
        # (overlapping regions belong to spheres/inclusions)
        matrix = [t for t in matrix_candidates - inclusion_candidates if t in existing_entities]

        # Filter inclusions: keep only those inside the RVE box
        inclusions = []
        to_remove = []
        eps = 1e-6
        for tag in inclusion_candidates:
            if tag not in existing_entities:
                continue
            try:
                bbox = gmsh.model.getBoundingBox(3, tag)
                cx = (bbox[0] + bbox[3]) / 2
                cy = (bbox[1] + bbox[4]) / 2
                cz = (bbox[2] + bbox[5]) / 2
                if -eps < cx < Lx + eps and -eps < cy < Ly + eps and -eps < cz < Lz + eps:
                    inclusions.append(tag)
                else:
                    to_remove.append(tag)
            except Exception:
                # Entity may have been removed/merged
                pass

        # Remove outside fragments
        if to_remove:
            occ.remove([(3, t) for t in to_remove], recursive=True)
            occ.synchronize()

    else:
        # --- CUT + INTERSECT ---
        clip_box = occ.addBox(0, 0, 0, Lx, Ly, Lz)

        # Cut spheres from box
        tools = [(3, s) for s in sphere_tags]
        result, _ = occ.cut([(3, box)], tools, removeObject=True, removeTool=False)
        matrix = [t[1] for t in result if t[0] == 3]

        # Clip spheres to box boundary using intersection
        sphere_entities = [(3, s) for s in sphere_tags]
        result, _ = occ.intersect(sphere_entities, [(3, clip_box)], removeObject=True, removeTool=True)
        inclusions = [t[1] for t in result if t[0] == 3]

        occ.synchronize()

    print(f"Matrix: {len(matrix)}, Inclusions: {len(inclusions)}")

    # Physical groups
    gmsh.model.addPhysicalGroup(3, matrix, tag=1, name="matrix")
    if inclusions:
        gmsh.model.addPhysicalGroup(3, inclusions, tag=2, name="inclusions")

    # Mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # Generate 3D mesh
    print("Generating mesh...")
    gmsh.model.mesh.generate(3)

    # Import to dolfinx
    msh_data = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=3
    )
    mesh = msh_data.mesh
    cell_tags = msh_data.cell_tags

    gmsh.finalize()

    # Create a DG0 function to store material IDs (for ParaView thresholding)
    V = functionspace(mesh, ("DG", 0))
    material = Function(V, name="material")

    # Map cell tags to the DG0 function
    material.x.array[:] = 0  # default
    for tag_value in np.unique(cell_tags.values):
        cells = cell_tags.find(tag_value)
        material.x.array[cells] = tag_value

    # Compute volume fractions by integrating over each domain
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    total_volume = mesh.comm.allreduce(assemble_scalar(form(1 * dx)), op=MPI.SUM)
    matrix_volume = mesh.comm.allreduce(assemble_scalar(form(1 * dx(1))), op=MPI.SUM)
    inclusion_volume = mesh.comm.allreduce(assemble_scalar(form(1 * dx(2))), op=MPI.SUM)

    vf_inclusions = inclusion_volume / total_volume
    vf_matrix = matrix_volume / total_volume

    print(f"Total volume: {total_volume:.6f}")
    print(f"Matrix volume: {matrix_volume:.6f} ({vf_matrix*100:.2f}%)")
    print(f"Inclusion volume: {inclusion_volume:.6f} ({vf_inclusions*100:.2f}%)")

    # Save to XDMF
    import os
    os.makedirs(os.path.dirname(output_xdmf), exist_ok=True)

    # Need to create topology connectivity for meshtags
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

    # Set explicit name for cell_tags so we can read it back
    cell_tags.name = "Cell tags"

    with XDMFFile(mesh.comm, output_xdmf, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)
        xdmf.write_function(material)  # Also write as function for ParaView

    # Save volume fraction to text file
    info_file = output_xdmf.replace('.xdmf', '_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"RVE size: {Lx:.6f} x {Ly:.6f} x {Lz:.6f}\n")
        f.write(f"Total volume: {total_volume:.6f}\n")
        f.write(f"Matrix volume: {matrix_volume:.6f}\n")
        f.write(f"Inclusion volume: {inclusion_volume:.6f}\n")
        f.write(f"Volume fraction (inclusions): {vf_inclusions:.6f}\n")
        f.write(f"Volume fraction (matrix): {vf_matrix:.6f}\n")
        f.write(f"Mesh cells: {mesh.topology.index_map(3).size_global}\n")
        f.write(f"Mesh vertices: {mesh.topology.index_map(0).size_global}\n")

    print(f"Saved to {output_xdmf}")
    print(f"Info saved to {info_file}")
    print(f"Mesh: {mesh.topology.index_map(3).size_global} cells, "
          f"{mesh.topology.index_map(0).size_global} vertices")


if __name__ == "__main__":
    seed = 0
    small_fraction = 0.0
    # num_chips = 100
    num_chips = 20
    small_scale = 0.5
    shrink_factor = 0.85
    mesh_size = 0.1

    input_file = f"periodic_3D/depositions/sphere_coordinates_periodic_relax_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.npy"
    output_file = f"periodic_3D/meshes/rvev2_{small_fraction:.1f}_{num_chips}_{small_scale}_{seed}.xdmf"

    create_mesh(input_file, output_file, mesh_size=mesh_size, shrink_factor=shrink_factor)
