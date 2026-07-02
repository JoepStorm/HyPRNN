"""Visualize uniaxial stress-strain paths from uniaxialDeposition.py output.

Plots F_xx -> PK1_xx, F_yy -> PK1_yy, F_xy -> PK1_xy.
Lines are colored by mix fraction (fraction of small chips).
Tension and compression appear on the same axes (different x ranges).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

plt.style.use(['science', 'bright'])
from matplotlib import rc
rc('text', usetex=True)

MIX_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]


def config_name(mix):
    return f"small{int(mix*100):03d}_large{int((1-mix)*100):03d}"


def load_vfracs(matparam, mesh_dir, shrink_factor):
    """Read woodchip vfrac from _vfrac.txt for each sample; returns list of float or None."""
    mesh_dir = Path(mesh_dir)
    vfracs = []
    for mp in matparam:
        txt = (mesh_dir / config_name(mp['mix'])
               / f"seed{mp['seed']}_shrink{shrink_factor}_vfrac.txt")
        try:
            lines = txt.read_text().strip().splitlines()
            vfracs.append(float(lines[1].split()[0]))   # wood = tag 2 = first value
        except Exception:
            vfracs.append(None)
    return vfracs


def load_data(data_dir):
    prefix = Path(data_dir) / "uniaxial_deposition"
    F   = np.load(f"{prefix}_F.npy")       # (N, T, 2, 2)
    PK1 = np.load(f"{prefix}_PK1.npy")     # (N, T, 2, 2)

    matparam = []
    with open(f"{prefix}_matparam.data") as f:
        next(f)   # skip header
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            vals = line.split()
            matparam.append({'mix': float(vals[0]),
                             'seed': int(vals[1]),
                             'load_sign': int(vals[2])})
    return F, PK1, matparam


def deformed_slice(F_sample):
    """Return the contiguous deformed region of a path, handling the shift for failed samples.

    Failed paths have identity at the start and converged steps shifted to the end.
    Returns None if the path has no deformation at all (failed at t=0).
    """
    mask = np.abs(F_sample[:, 0, 0] - 1.0) > 1e-4
    if not np.any(mask):
        return None, None
    idx = np.where(mask)[0]
    return idx[0], idx[-1] + 1


def main(data_dir, output_dir, mesh_dir, shrink_factor):
    F, PK1, matparam = load_data(data_dir)
    n_samples = F.shape[0]
    vfracs = load_vfracs(matparam, mesh_dir, shrink_factor)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ri, ci = 0, 0   # F_xx → PK1_xx

    # ── Colormap 1: discrete mix values ──────────────────────────────────────
    mix_cmap = cm.get_cmap('viridis', len(MIX_VALUES))
    mix_to_color = {m: mix_cmap(i) for i, m in enumerate(MIX_VALUES)}

    # ── Colormap 2: continuous vfrac ─────────────────────────────────────────
    valid_vf = [v for v in vfracs if v is not None]
    vf_min, vf_max = (min(valid_vf), max(valid_vf)) if valid_vf else (0, 1)
    vf_norm = plt.Normalize(vmin=vf_min, vmax=vf_max)
    vf_cmap = cm.get_cmap('plasma_r')

    fig, (ax_mix, ax_vf) = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)

    plotted_mix = set()
    n_plotted = 0

    for i in range(n_samples):
        start, end = deformed_slice(F[i])
        if start is None:
            continue
        n_plotted += 1

        x = F[i, start:end, ri, ci]
        y = PK1[i, start:end, ri, ci]

        mix = matparam[i]['mix']
        plotted_mix.add(mix)
        ax_mix.plot(x, y, color=mix_to_color[mix], linewidth=0.7, alpha=0.8)

        vf = vfracs[i]
        color_vf = vf_cmap(vf_norm(vf)) if vf is not None else '#aaaaaa'
        ax_vf.plot(x, y, color=color_vf, linewidth=0.7, alpha=0.8)

    print(f"Plotted {n_plotted}/{n_samples} samples with deformation")

    for ax in (ax_mix, ax_vf):
        ax.set_xlabel(r'$F_{xx}$', labelpad=1)
        ax.axhline(0, color='k', linewidth=0.4, linestyle=':', alpha=0.3)
        ax.axvline(1, color='k', linewidth=0.4, linestyle=':', alpha=0.3)
        ax.minorticks_off()

    ax_mix.set_ylabel(r'$P_{xx}$', labelpad=1)

    # Mix legend
    legend_handles = [
        Line2D([0], [0], color=mix_to_color[m], linewidth=1.5,
               label=f'{int(m * 100)}\\% small')
        for m in sorted(plotted_mix)
    ]
    ax_mix.legend(handles=legend_handles, fontsize=6, loc='upper left')
    ax_mix.set_title('colored by mixture', fontsize=7)

    # Vfrac colorbar
    sm = cm.ScalarMappable(cmap=vf_cmap, norm=vf_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_vf, pad=0.02)
    cbar.set_label(r'$V_f$ (mesh)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax_vf.set_title('colored by $V_f$', fontsize=7)

    fig.tight_layout()
    out = output_dir / "uniaxial_deposition_curves"
    fig.savefig(f"{out}.pdf", bbox_inches='tight')
    fig.savefig(f"{out}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved to {out}.pdf / .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      type=str,   default="../data/uniaxial_deposition")
    p.add_argument("--output_dir",    type=str,   default="../data/uniaxial_deposition")
    p.add_argument("--mesh_dir",      type=str,   default="../yade/generated_meshes_2D_wet")
    p.add_argument("--shrink_factor", type=float, default=0.85)
    args = p.parse_args()
    main(args.data_dir, args.output_dir, args.mesh_dir, args.shrink_factor)
