import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection
from matplotlib import rc
rc('text', usetex=True)


def plot_mesh_fast(mesh_node_coords, mesh_elem_nodes, cell_values, fname,
                   cmap='tab10', title=None, show_edges=False):
    """Fast mesh plotting using tripcolor (batch rendering).

    Much faster than plt.fill() for large meshes. Good for discrete cell tags.
    """
    triangulation = Triangulation(mesh_node_coords[:, 0], mesh_node_coords[:, 1], mesh_elem_nodes)

    fig, ax = plt.subplots(figsize=(4, 4))
    tpc = ax.tripcolor(triangulation, cell_values, cmap=cmap, shading='flat')

    if show_edges:
        ax.triplot(triangulation, 'k-', lw=0.1, alpha=0.3)

    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10)

    plt.savefig(fname, bbox_inches='tight', pad_inches=0.02, dpi=150)
    plt.close(fig)


def plot_micro_meshes_grid(mesh_data_list, output_folder, ncols=4):
    """Plot multiple micro meshes in a grid layout.

    Uses fast batch rendering with PolyCollection. Colors indicate material phases.
    """
    os.makedirs(output_folder, exist_ok=True)
    n_meshes = len(mesh_data_list)
    nrows = (n_meshes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).flatten()

    # Define colors for materials (tag 1 = fungi, tag 2 = wood)
    material_colors = {1: '#4CAF50', 2: '#8B4513'}  # green for fungi, brown for wood

    for idx, mesh_data in enumerate(mesh_data_list):
        ax = axes[idx]

        coords = mesh_data['coords'][:, :2]
        cells = mesh_data['cells']
        tag_indices = mesh_data['tag_indices']
        tag_values = mesh_data['tag_values']

        # Map tags to all cells (tag_indices may be sparse)
        cell_colors = np.ones(len(cells))  # default tag 1
        cell_colors[tag_indices] = tag_values

        # Build triangles as vertex coordinates
        triangles = coords[cells]

        # Create face colors array
        facecolors = [material_colors.get(int(t), '#888888') for t in cell_colors]

        # Use PolyCollection for fast rendering
        pc = PolyCollection(triangles, facecolors=facecolors,
                            edgecolors='none', linewidths=0.1)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'RVE {idx}', fontsize=8)

    # Hide unused axes
    for idx in range(n_meshes, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    fname = os.path.join(output_folder, 'micro_meshes_grid.pdf')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.01, dpi=150, format='pdf')

    plt.close(fig)
    print(f"Saved micro mesh grid: {fname}")


def plot_micro_meshes_at_gauss_points(mesh_data_list, gauss_coords, output_folder, scale=0.1):
    """Plot micro meshes at their Gauss point locations in the macro domain.

    Each RVE is scaled and translated to its corresponding Gauss point location.
    """
    os.makedirs(output_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Define colors for materials (tag 1 = fungi, tag 2 = wood)
    material_colors = {1: '#4CAF50', 2: '#8B4513'}

    for idx, mesh_data in enumerate(mesh_data_list):
        coords = mesh_data['coords'][:, :2]
        cells = mesh_data['cells']
        tag_indices = mesh_data['tag_indices']
        tag_values = mesh_data['tag_values']

        # Map tags to all cells
        cell_colors = np.ones(len(cells))
        cell_colors[tag_indices] = tag_values

        # Center RVE at origin, then scale and translate to Gauss point
        rve_center = (coords.max(axis=0) + coords.min(axis=0)) / 2
        coords_centered = coords - rve_center
        coords_scaled = coords_centered * scale + gauss_coords[idx]

        # Build triangles
        triangles = coords_scaled[cells]

        # Create face colors
        facecolors = [material_colors.get(int(t), '#888888') for t in cell_colors]

        pc = PolyCollection(triangles, facecolors=facecolors,
                            edgecolors='none', linewidths=0.1)
        ax.add_collection(pc)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')

    fname = os.path.join(output_folder, 'micro_meshes_at_gauss.pdf')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.01, dpi=150, format='pdf')
    plt.close(fig)
    print(f"Saved micro meshes at Gauss points: {fname}")


def triangle_coords(mesh_node_coords, mesh_elem_nodes, elem):
    new = np.zeros((3,2))
    for k in range(3):
        new[k] = mesh_node_coords[mesh_elem_nodes[elem][k]]
    return new

def color_plot(mesh_node_coords, mesh_elem_nodes, colors, fname, bound_values, colorscheme, label, control_coords=None, plot_png=False, title_text=None):

    cmap_options = {
        'jet': cm.jet,
        'bwr': cm.bwr,
        'coolwarm': cm.coolwarm,
        'Spectral_rev': cm.Spectral_r
    }

    cmap = cmap_options[colorscheme]

    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=bound_values[0], vmax=bound_values[1])

    for i, node in enumerate(mesh_elem_nodes):
        cur_triang = triangle_coords(mesh_node_coords, mesh_elem_nodes, i)
        plt.fill(cur_triang[:, 0], cur_triang[:, 1], c=cmap(norm(colors[i])), edgecolor='none', linewidth=0.0001, zorder=-1)

    if control_coords is not None:
        plt.scatter(control_coords[:,0], control_coords[:,1], c='black', s=14, zorder=1 )

    ax.set_aspect('equal')
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.title(str_title, fontsize=8) #, loc='right')
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), shrink=0.15, pad=-0.01, aspect=10)
    cbar.ax.set_title(label)
    # move title position to right

    # make colorbar horizontal above cbar
    # cbar.ax.xaxis.set_label_position('top')

    if title_text is not None:
        ax.text(0.01, 2.17, title_text, fontsize=10, verticalalignment='top')

    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
    if plot_png:
        plt.savefig(fname + '.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')


def plot_interpolation_field(domain, control_coords, field_dict, output_folder='../results/plots/', title_text=None):
    """
    Plot interpolated micro variable fields as heatmaps

    Args:
        domain: FEniCS mesh domain
        control_coords: [num_control_points, 2] array of control point coordinates
        field_dict: dict of micro variable arrays at quadrature points
        output_folder: folder to save plots
    """
    os.makedirs(output_folder, exist_ok=True)

    # Extract mesh information
    mesh_node_coords = domain.geometry.x[:, :2]  # Node coordinates [num_nodes, 2]
    topology = domain.topology
    topology.create_connectivity(topology.dim, 0)
    c_to_v = topology.connectivity(topology.dim, 0)

    # Build element-to-node connectivity
    num_cells = topology.index_map(topology.dim).size_local
    mesh_elem_nodes = []
    for cell in range(num_cells):
        vertices = c_to_v.links(cell)
        mesh_elem_nodes.append(vertices[:3])  # Assuming triangular elements
    mesh_elem_nodes = np.array(mesh_elem_nodes)

    dict_labels = {
        'vfrac': r'$V_f$',
        'theta': r'$\theta$',
        'ratio': r'$r$',
        'mu': r'$\mu$',
        'mixture': r'mix',
        'fil_frac': 'Pellet\nfraction',
    }

    # Fixed colorbar bounds per variable (independent of the actual data range).
    dict_bounds = {
        'fil_frac': (0.0, 1.0),
        'theta': (-np.pi / 2, np.pi / 2),
    }

    # Plot each micro variable
    for var_name, values in field_dict.items():
        values = np.asarray(values)

        # For quadrature values, take mean per element if multiple IPs per element
        num_ips_per_elem = len(values) // num_cells
        if num_ips_per_elem > 1:
            elem_values = values.reshape(num_cells, num_ips_per_elem).mean(axis=1)
        else:
            elem_values = values

        # Determine bounds: fixed range if specified, else data min/max.
        if var_name in dict_bounds:
            bound_values = list(dict_bounds[var_name])
        else:
            bound_values = [np.min(elem_values), np.max(elem_values)]

        # Create filename
        fname = os.path.join(output_folder, f'interpolated_{var_name}')

        # Plot
        color_plot(
            mesh_node_coords=mesh_node_coords,
            mesh_elem_nodes=mesh_elem_nodes,
            colors=elem_values,
            fname=fname,
            bound_values=bound_values,
            colorscheme='Spectral_rev',
            label=dict_labels[var_name],
            control_coords=control_coords,
            title_text = title_text,
        )
        print(f"Saved interpolation plot: {fname}.pdf")

def color_plot_deformed(mesh_node_coords, mesh_elem_nodes, du, colors, fname, bound_values, colorscheme, label, control_coords=None, plot_png=False, title_text=None, show_undeformed=True):

    cmap_options = {
        'jet': cm.jet,
        'bwr': cm.bwr,
        'coolwarm': cm.coolwarm,
        'Spectral_rev': cm.Spectral_r
    }

    cmap = cmap_options[colorscheme]

    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=bound_values[0], vmax=bound_values[1])

    ## First do background mesh:
    if show_undeformed:
        for i, node in enumerate(mesh_elem_nodes):
            cur_triang = triangle_coords(mesh_node_coords, mesh_elem_nodes, i)
            plt.fill(cur_triang[:, 0], cur_triang[:, 1], c='white', edgecolor='gray', alpha=0.2, linewidth=0.0001, zorder=-2)

    # Then do deformed mesh:
    for i, node in enumerate(mesh_elem_nodes):
        cur_triang = triangle_coords(mesh_node_coords+du, mesh_elem_nodes, i)
        plt.fill(cur_triang[:, 0], cur_triang[:, 1], c=cmap(norm(colors[i])), edgecolor='none', linewidth=0.0001, zorder=-1)

    if control_coords is not None:
        plt.scatter(control_coords[:,0], control_coords[:,1], c='black', s=5, zorder=1 )

    ax.set_aspect('equal')
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), shrink=0.15, pad=-0.01, aspect=10)
    cbar.ax.set_title(label)

    if title_text is not None:
        ax.text(0.01, 2.17, title_text, fontsize=10, verticalalignment='top')

    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
    if plot_png:
        plt.savefig(fname + '.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')


def plot_deformed_stresses(domain, du, colorvals, output_folder='../results/plots/', show_undeformed=True):
    """
    Plot deformed mesh with stresses included. The original background mesh is shown.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Extract mesh information
    mesh_node_coords = domain.geometry.x[:, :2]  # Node coordinates [num_nodes, 2]
    topology = domain.topology
    topology.create_connectivity(topology.dim, 0)
    c_to_v = topology.connectivity(topology.dim, 0)

    # Build element-to-node connectivity
    num_cells = topology.index_map(topology.dim).size_local
    mesh_elem_nodes = []
    for cell in range(num_cells):
        vertices = c_to_v.links(cell)
        mesh_elem_nodes.append(vertices[:3])  # Assuming triangular elements
    mesh_elem_nodes = np.array(mesh_elem_nodes)

    label_names = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$']
    file_names = ['sig_xx', 'sig_yy', 'sig_xy']

    # Loop over stress components
    for i in range(colorvals.shape[1]):
        colorvals_i = colorvals[:, i]

        # Create filename
        fname = f"{output_folder}/deformed_mesh_{file_names[i]}"

        # Determine bounds
        abs_max = np.max(np.abs(colorvals_i))
        bound_values = [-abs_max, abs_max]

        # Plot
        color_plot_deformed(
            mesh_node_coords=mesh_node_coords,
            mesh_elem_nodes=mesh_elem_nodes,
            du=du,
            colors=colorvals_i,
            fname=fname,
            bound_values=bound_values,
            colorscheme='coolwarm',
            label=label_names[i],
            show_undeformed=show_undeformed
        )
    print(f"Saved deformation plot: {fname}")
