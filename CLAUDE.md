# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-scale computational mechanics project implementing **FE² (FE-squared) analysis** with **Physics-informed Recurrent Neural Networks (PRNNs)** as material surrogates. The project combines finite element analysis with machine learning to accelerate expensive Representative Volume Element (RVE) computations in heterogeneous woodchip composite materials (wood chips in a fungal mycelium matrix), in a 2D large-deformation setting.

## Repository Cleanup (in progress)

This repo was copied from another project and is being cleaned up for publication: redundant files, stale results, and dead links to old experiment versions are being removed. Git history only starts at the cleanup (heavy data dirs are gitignored) — treat deletions of data files as final and check with the user before removing anything non-obviously redundant.

When helping with cleanup:
- **Spotting unused files**: grep the whole repo for a file's name/module (import statements, string paths, CLI invocations) before flagging it as unused. A script with no incoming references and no `if __name__ == "__main__"` usage documented anywhere is a deletion candidate; report it rather than deleting outright.
- **Stale version references**: watch for hardcoded paths like `.../v5_bigrve`, `.../v5`, `.../v7_combine_v4_v5` — these point at old experiment runs that no longer exist or matter. Commented-out lines referencing old versions (`# OUTPUT_DIR = ...v5_bigrve`) should usually just be deleted, not kept as history.
- **Consistency**: when editing or reviewing multiple files in one pass, keep docstring style, header comments, and naming identical across them — don't let one file drift to a different convention than its siblings.
- **Redundant logic**: flag code sections/branches/functions that are dead or logically superseded (e.g. an old implementation left in place after a rewrite, duplicated logic copy-pasted across files). Do NOT flag blocks where commented-out lines are intentional manual toggles between alternative settings/configs (e.g. `# base_folder = ".../v4"` next to the active `base_folder = ".../v5"` line, switched by hand between runs) — that's a deliberate pattern here, not dead code.

## Key Commands

### Training PRNN Models
```bash
cd scripts_surrogates
python train_surrogate.py                 # Train a single (Hy)PRNN surrogate
python train_multiconfig_batch.py      # Train multiple architectures for comparison
python train_surrogate_batch.py           # Timed batch runs (PRNN / HyPRNN / NN)
python train_surrogate_deposit_batch.py   # Batch on the deposition filler dataset
```

### FE² Validation and Design Studies
```bash
cd scripts_FEM
python validate_bending.py         # 3-point bending: FE² ground truth vs surrogates
python validate_fe2.py             # RVE vs PRNN stress response on identical load paths
python graded_ring_pressure.py     # Pressurized annulus with graded PRNN material
python optimize_ring.py            # Optimize radial grading of the annulus
python squeeze_hole.py             # Hole-squeeze design with filler PRNN surrogate
python validate_ring.py            # Validate optimized ring grading against FE²
```

### Data Generation
```bash
cd scripts_data_creation
python create_uniaxial_data.py                # Uniaxial training data (synthetic ellipse RVEs)
python create_mixed_data.py                   # Mixed-loading training data (synthetic ellipse RVEs)
python create_deposition_data.py              # Mixed-loading data from YADE-deposited meshes
python create_uniaxial_deposition_data.py     # Uniaxial runs on YADE-deposited meshes
```

### RVE Microstructure Generation (YADE)
```bash
cd yade
# Single deposition, run manually with args to inspect behavior (GUI)
yadedaily deposit_rve.py -- --seed 0 --filler_fraction 0.5

# Filler-fraction sweep: deposit + mesh one RVE per fraction, plus overview plots
python generate_rve_dataset.py

# Visualize a settled 3D packing in ParaView
python packing_to_vtk.py data/datasetv1/coords_3D_0_500_0.5.npy

# Mesh a single packing (strip filler, convex-hull periodic mesh)
python packing_to_mesh.py --input data/datasetv1/coords_2D_0_500_0.5.npy
```

### Material Testing
```bash
cd scripts_materials
python rve_material.py         # Test 2D RVE homogenization standalone
```

## Architecture Overview

### Multi-scale FE² Framework
```
Macro FEM (dolfinx) → Material Models → Micro RVE (dolfinx) / PRNN Surrogate
                           ↓                                      ↓
                    Quadrature Points                    Fast ML Prediction
```

### End-to-End Data Flow
```
1. Particle Deposition (YADE)
   └─> deposit_rve.py: periodic 2D RVE packings
2. Mesh Generation
   └─> packing_to_mesh.py: packing → periodic GMSH mesh
3. RVE Data Generation
   └─> create_*_data.py: loading sequences → {F, PK1, cauchy}.npy
4. PRNN Training
   └─> train_*.py: JAX/Flax training → trained_models/
5. Macro FE² Inference
   └─> validate_*.py / design scripts: PRNN replaces RVE at each Gauss point
```

### Core Components

**Particle Deposition & Meshing** (`yade/`):
- `deposit_rve.py`: 3D gravity deposition (wood chips + filler) → horizontal slice → relaxed periodic 2D sheet
- `generate_rve_dataset.py`: filler-fraction sweep — deposit + mesh one RVE per fraction, with overview plots
- `packing_to_mesh.py`: strip filler from a packing and mesh it (wood = inclusions, rest = matrix)
- `mesh_utils.py`: supporting functions — convex-hull periodic meshing pipeline and mesh plotting
- `packing_to_vtk.py`: settled 3D packing → ParaView `.vtp`
- Output stored in `yade/data/`

**Materials System** (`scripts_materials/`):
- `rve_material.py`: Full 2D RVE homogenization with periodic BC (stress + tangent)
- `prnn_material.py`: PRNN surrogate material (fast ML prediction)
- `fe2_material.py`: FE² material — parallel RVE solves with per-quadrature-point microstructure
- `neohooke.py`: Neo-Hookean stress functions (UFL + JAX variants)
- `rve_mesher.py` / `create_meshes.py` / `create_rve.py`: RVE mesh generation utilities

**Training Data Generation** (`scripts_data_creation/`):
- `create_uniaxial_data.py` / `create_mixed_data.py`: RVE simulations on synthetic ellipse meshes
- `create_deposition_data.py` / `create_uniaxial_deposition_data.py`: RVE simulations on YADE-deposited meshes
- `plot_uniaxial_deposition.py`: visualize create_uniaxial_deposition_data.py output

**Neural Networks** (`scripts_surrogates/`):
- `HyPRNN.py`: HyPRNN — physics-informed shared-hypernetwork PRNN (core ML module)
- `StandardNN.py`: Plain feed-forward NN baseline
- `trainer.py`: JAX/Flax training framework
- `data_utils.py`: Data processing, normalization, and dataset loading
- `plot_LD_data.py` / `plot_uniaxial_data.py` / `plot_learncurve.py` / `plot_median_curves.py`: dataset and training-result plots

**FEM Drivers** (`scripts_FEM/`):
- `validate_bending.py`: 3-point bending — FE² ground truth vs surrogate models (deformation, stress maps, timing)
- `validate_fe2.py`: RVE vs PRNN stress response on identical load paths
- `graded_ring_pressure.py` / `optimize_ring.py` / `validate_ring.py`: graded annulus study — simulate, optimize, validate
- `squeeze_hole.py` / `optimization.py`: design optimization with PRNN surrogates
- `evaluate_fe2_surrogates.py`: surrogate losses on saved FE² strain trajectories
- `plotting_utils.py`: shared plotting helpers

### Physics-Informed Architecture

The PRNN models use custom JAX layers enforcing physical constraints:
- **Deformation gradient decomposition layers**: F = R·U factorization
- **Material constraint layers**: Positive definiteness, frame invariance
- **Hypernetworks**: Parameter variation for material heterogeneity (HyperLU layers)

## Coding Style

### Documentation
- **Concise docstrings**: Describe what a function/class does in 1-2 sentences. Don't list all parameters and return values - the code should be self-explanatory.
- Good: `"""Solve the RVE and return homogenized stress and tangent."""`
- Bad: Long parameter lists with types and descriptions for obvious arguments.

### Code Quality
- **Robust fixes over shortcuts**: Take time to understand root causes. Quick hacks accumulate technical debt.
- **Favor simplicity**: Simpler code is easier to debug and maintain. Before adding complexity, ask if there's a simpler way.
- **Proactive simplification**: When reading code, consider if it can be simplified. If significant simplification is possible, ask the user before refactoring.

### General Principles
- Imports at the top of files (standard Python convention)
- Prefer editing existing files over creating new ones
- Keep functions focused - if a function does too much, split it

### Line Length
- The editor soft-wraps long lines automatically, so don't split a line purely because it's "a bit too long". A slightly long `ax.plot(...)`, function call, or expression can stay as a one-liner.
- Do split when it genuinely aids clarity — e.g. a call with many (say 5+) parameters, or nested arguments where one-per-line makes the structure readable. Readability is the criterion, not a strict column limit.

## Development Patterns

### Code Organization
- **Material Interface**: All materials inherit from common `Material` base class
- **JAX Integration**: Neural networks use JAX for automatic differentiation and JIT compilation
- **MPI Parallelization**: FEM computations distributed via mpi4py/PETSc

### Dependencies
- **FEniCSx/dolfinx**: Modern finite element framework
- **dolfinx_mpc**: Multi-point constraints for periodic boundary conditions
- **dolfinx_materials**: QuadratureMap / custom material integration at Gauss points
- **JAX/Flax**: Neural network framework with automatic differentiation
- **MPI4Py/PETSc**: Parallel computing and linear algebra
- **YADE**: Discrete element method for particle deposition simulations
- **gmsh**: Mesh generation (Python API)

### File Naming Conventions
- `LD*`: Large deformation formulations
- `*_material.py`: Material model implementations
- `create_*.py`: Data generation scripts
- `train_*.py`: Training scripts
- `plot_*.py`: Plotting / visualization scripts

## Testing Approach

- Use `validate_fe2.py` / `validate_bending.py` for FE² vs surrogate validation
- Material models have standalone test capabilities via `if __name__ == "__main__"`
- No formal pytest structure; testing is integration-focused

## Mesh and Data Structure

- **RVE meshes**: `meshes/` (GMSH-generated)
- **RVE packings + meshes (filler workflow)**: `yade/data/` (per dataset version)
- **Training data**: `data/` (NumPy arrays: `F.npy`, `PK1.npy`, `cauchy.npy`)
- **Model storage**: `trained_models/` at repo root (JAX/Flax checkpoints)
- **Shared material constants**: `material_params.py` at repo root (wood/fungi E, nu, Lamé constants) — import these instead of hardcoding values
- **Simulation results**: `results/` (VTX `.bp` format for ParaView)

## Performance Considerations

- PRNN inference replaces expensive RVE computations (1000x+ speedup)
- JAX JIT compilation for neural networks
- MPI parallelization for macro-scale FEM
- Batched material point evaluation at quadrature points
