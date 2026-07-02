"""Build a filler-fraction sweep: deposit + mesh one RVE per filler fraction.

Runs yade_woodchip_filler.py for filler_fraction = 0.0, 0.1, ... 0.9 and meshes
each resulting packing with mesh_rve_filler_2D.py, producing 10 .msh files in
OUT_DIR (one per fraction). Afterwards it writes two PDFs:
    vfrac_vs_filler.pdf  - inclusion volume fraction vs filler fraction
    meshes_row.pdf       - the 10 meshes in a single row (matrix vs inclusions)
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mesh_utils import draw_mesh, read_inclusion_vfrac, MATERIAL_COLORS

HERE    = Path(__file__).resolve().parent
# OUT_DIR = HERE / "data" / "datasetv6_r0.02"
OUT_DIR = HERE / "data" / "dataset_combi_v4v6"

YADE    = "yadedaily"          # DEM runner
SEED    = 2
FRACTIONS = np.round(np.arange(0.0, 1.0, 0.1), 1)   # 0.0 .. 0.9
vel = 0.1 #0.05 # 0.1


OUT_DIR.mkdir(parents=True, exist_ok=True)


meshes = []   # (frac, msh_path)
for frac in FRACTIONS:
    # tag = f"coords_2D_{SEED}_500_{frac:.1f}"
    tag = f"coords_2D_500_{frac:.1f}"
    npy = OUT_DIR / f"{tag}.npy"
    msh = OUT_DIR / f"{tag}.msh"
    print(f"\n{'='*60}\nfiller_fraction = {frac:.1f}  ->  {msh.name}\n{'='*60}")

    # # 1) DEM deposition + slice -> 2D packing .npy (+ _meta.npy)
    # subprocess.run(
    #     [YADE, "-x", "-n", "yade_woodchip_filler.py", "--",
    #      "--seed", str(SEED),
    #      "--filler_fraction", f"{frac:.1f}",
    #      "--init_velocity", f"{vel:.2f}",
    #      "--output", str(npy)],
    #     cwd=HERE, check=True)

    # 2) Strip filler + mesh -> .msh
    subprocess.run(
        [sys.executable, "mesh_rve_filler_2D.py",
         "--input", str(npy),
         "--output", str(msh)],
        cwd=HERE, check=True)
    meshes.append((frac, msh))

print(f"\nDone. {len(meshes)} meshes in {OUT_DIR}")


# === Plot 1: inclusion volume fraction vs filler fraction ===
vfracs = [read_inclusion_vfrac(msh) for _, msh in meshes]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot([f for f, _ in meshes], vfracs, 'o-')
ax.set_xlabel("Filler fraction")
ax.set_ylabel("Inclusion volume fraction")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "vfrac_vs_filler.pdf")
plt.close(fig)
print(f"Saved {OUT_DIR / 'vfrac_vs_filler.pdf'}")


# === Plot 2: the 10 meshes in a single row ===
n = len(meshes)
fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
for ax, (frac, msh) in zip(np.atleast_1d(axes), meshes):
    draw_mesh(ax, msh)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"filler={frac:.1f}", fontsize=8)
legend = [Patch(facecolor=MATERIAL_COLORS[2], label='inclusions'),
          Patch(facecolor=MATERIAL_COLORS[1], label='matrix')]
fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT_DIR / "meshes_row.pdf")
plt.close(fig)
print(f"Saved {OUT_DIR / 'meshes_row.pdf'}")
