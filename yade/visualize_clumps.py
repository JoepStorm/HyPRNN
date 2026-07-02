"""Visualize RVE sphere coordinates as colored circles."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from dataset_yade import generate_grid

def load_periodic_data(npy_file):
    """Load sphere data and cell dimensions."""
    data = np.load(npy_file)

    meta_file = npy_file.replace('.npy', '_meta.npy')
    try:
        cell_size = np.load(meta_file)
        rve_size_x, rve_size_y = cell_size[0], cell_size[1]
    except FileNotFoundError:
        print(f"Warning: {meta_file} not found, assuming 1.0 x 1.0")
        rve_size_x, rve_size_y = 1.0, 1.0

    if data.shape[1] >= 4:
        return data[:, :4], rve_size_x, rve_size_y
    else:
        extended = np.zeros((data.shape[0], 4))
        extended[:, :3] = data[:, :3]
        extended[:, 3] = np.arange(data.shape[0])
        return extended, rve_size_x, rve_size_y


def config_to_filepath(r1, r2, nz, seed, dir):
    """Convert (r1, r2, nz, seed) to the expected .npy filepath."""
    spheres_per_clump = nz * (nz * r1) * (nz * r2)
    num_chips = int(8000 / spheres_per_clump)
    name = f"coords_3Dto2D_r1_{r1:.2f}_r2_{r2:.2f}_nz_{nz}_{num_chips}_{seed}.npy"
    return f"{dir}/{name}"


def plot_rve(npy_file, shrink_factor=1.0, ax=None, title=None):
    """Plot circles colored by clump ID."""
    spheres, rve_size_x, rve_size_y = load_periodic_data(npy_file)
    spheres[:, 2] *= shrink_factor

    clump_ids = spheres[:, 3].astype(int)
    unique_clumps = np.unique(clump_ids)
    cmap = cm.get_cmap('tab20', len(unique_clumps))
    clump_to_color = {cid: i for i, cid in enumerate(unique_clumps)}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8 * rve_size_y / rve_size_x))

    patches = []
    colors = []
    for x, y, r, cid in spheres:
        patches.append(Circle((x, y), r))
        colors.append(clump_to_color[int(cid)])

    collection = PatchCollection(patches, cmap=cmap, alpha=0.7, edgecolors='k', linewidths=0.3)
    collection.set_array(np.array(colors, dtype=float))
    collection.set_clim(-0.5, len(unique_clumps) - 0.5)
    ax.add_collection(collection)

    ax.set_xlim(0, rve_size_x)
    ax.set_ylim(0, rve_size_y)
    ax.set_aspect('equal')

    if title:
        ax.set_title(title, fontsize=8)
    else:
        ax.set_title(f'{len(spheres)} spheres, {len(unique_clumps)} clumps, shrink={shrink_factor}')

    if standalone:
        plt.tight_layout()
        plt.savefig(f"{npy_file[:-4]}.pdf", bbox_inches='tight', format='pdf')


def plot_batch(r1_values, r2_values, nz_values, seed, shrink_factor, data_dir):
    """Plot a grid of RVEs for all valid (r1, r2, nz) combinations."""
    configs = generate_grid(r1_values, r2_values, nz_values)

    # Check which files exist
    valid = []
    for r1, r2, nz in configs:
        fp = config_to_filepath(r1, r2, nz, seed, data_dir)
        try:
            valid.append((r1, r2, nz, fp))
        except:
            print(f"Missing: r1={r1:.2f} r2={r2:.2f} nz={nz} ({fp})")

    if not valid:
        print("No files found!")
        return

    # Group by nz for separate figures
    nz_groups = {}
    for r1, r2, nz, fp in valid:
        nz_groups.setdefault(nz, []).append((r1, r2, fp))

    for nz, entries in sorted(nz_groups.items()):
        r1_unique = sorted(set(r1 for r1, _, _ in entries))
        r2_unique = sorted(set(r2 for _, r2, _ in entries))
        lookup = {(r1, r2): fp for r1, r2, fp in entries}

        ncols = len(r2_unique)
        nrows = len(r1_unique)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

        for i, r1 in enumerate(r1_unique):
            for j, r2 in enumerate(r2_unique):
                ax = axes[i, j]
                if (r1, r2) in lookup:
                    plot_rve(lookup[(r1, r2)], shrink_factor, ax=ax,
                             title=f"r1={r1:.2f} r2={r2:.2f}")
                else:
                    ax.set_visible(False)

        fig.suptitle(f"nz={nz}", fontsize=14)
        plt.tight_layout()
        out = f"{data_dir}/batch_nz_{nz}.pdf"
        plt.savefig(out, bbox_inches='tight', format='pdf')
        print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    mode = 'single'
    # mode = 'batch'

    if mode == "single":
        shrink_factor = 0.9
        seed = 0
        type = 'slender' # 'vbulky'
        num_chips = 600

        input_file = f"periodic/depositions_3d_to_2d/coords_3Dto2D_{type}_{num_chips}_{seed}.npy"
        plot_rve(input_file, shrink_factor)
    elif mode == "batch":
        r1 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        r2 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        nz = [2, 3, 4, 5, 6, 7, 8]
        seed = 0
        shrink_factor = 0.9

        data_dir = 'periodic/depositions_3d_to_2d_relaxed_cont'

        plot_batch(r1, r2, nz, seed, shrink_factor, data_dir)
