# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-scale computational mechanics project implementing **FE² (FE-squared) analysis** with **Physics-informed Recurrent Neural Networks (PRNNs)** as material surrogates. The project combines finite element analysis with machine learning to accelerate expensive Representative Volume Element (RVE) computations in heterogeneous woodchip composite materials. Both 2D and 3D workflows are supported.

## Repository Cleanup (in progress)

This repo was copied from another project and is being cleaned up: redundant files, stale results, and dead links to old experiment versions (e.g. `v5`, `v6`, `v7` folders in `results/`, `yade/generated_meshes_2D_wet/`, `data/deposition/`) are being removed. Git history only starts at the cleanup (heavy data dirs are gitignored) — treat deletions of data files as final and check with the user before removing anything non-obviously redundant.

When helping with cleanup:
- **Spotting unused files**: grep the whole repo for a file's name/module (import statements, string paths, CLI invocations) before flagging it as unused. A script with no incoming references and no `if __name__ == "__main__"` usage documented anywhere is a deletion candidate; report it rather than deleting outright.
- **Stale version references**: watch for hardcoded paths like `.../v5_bigrve`, `.../v5`, `.../v7_combine_v4_v5` — these point at old experiment runs that no longer exist or matter. Commented-out lines referencing old versions (`# OUTPUT_DIR = ...v5_bigrve`) should usually just be deleted, not kept as history.
- **Consistency**: when editing or reviewing multiple files in one pass, keep docstring style, header comments, and naming identical across them — don't let one file drift to a different convention than its siblings.
- **Redundant logic**: flag code sections/branches/functions that are dead or logically superseded (e.g. an old implementation left in place after a rewrite, duplicated logic copy-pasted across files). Do NOT flag blocks where commented-out lines are intentional manual toggles between alternative settings/configs (e.g. `# base_folder = ".../v4"` next to the active `base_folder = ".../v5"` line, switched by hand between runs) — that's a deliberate pattern here, not dead code.

## Key Commands

### Training PRNN Models
```bash
cd scripts_surrogates
python train_LDprnn.py              # Train a single PRNN surrogate
python train_multiconfig_batch.py   # Train multiple configurations for comparison
```

### Running FE² Simulations
```bash
cd scripts_FEM
python LDFE2.py         # Main FE² simulation with PRNN materials
python testFE2.py       # Simple FE² test with RVE materials
```

### Data Generation
```bash
cd scripts_surrogates
python createUniaxialData.py       # Generate uniaxial training data (2D)
python createMixedData.py          # Generate mixed loading training data (2D)
python createWoodchipData3D.py     # Generate 3D woodchip RVE training data
```

### RVE Microstructure Generation (YADE)
```bash
cd yade
# Single deposition, run manually with args to inspect behavior (GUI)
yadedaily yade_woodchip_filler.py -- --seed 0 --filler_fraction 0.5

# Filler-fraction sweep: deposit + mesh one RVE per fraction, plus overview plots
python batch_filler_dataset.py

# Visualize a settled 3D packing in ParaView
python coords_3D_to_vtk.py data/datasetv1/coords_3D_0_500_0.5.npy

# Mesh a single packing (strip filler, convex-hull periodic mesh)
python mesh_rve_filler_2D.py --input data/datasetv1/coords_2D_0_500_0.5.npy
```

### Material Testing
```bash
cd scripts_materials
python RVE_material.py         # Test 2D RVE homogenization
python RVE_material_3D.py      # Test 3D RVE homogenization
python PRNN_mat.py             # Test PRNN material interface
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
   └─> yade_woodchip_filler.py: periodic 2D RVE packings
2. Mesh Generation
   └─> mesh_rve_filler_2D.py: packing → periodic GMSH mesh
3. RVE Data Generation
   └─> createWoodchipData*.py: loading sequences → {F, PK1, cauchy}.npy
4. PRNN Training
   └─> train_*.py: JAX/Flax training → trained_models/
5. Macro FE² Inference
   └─> LDFE2.py: PRNN replaces RVE at each Gauss point
```

### Core Components

**Particle Deposition & Meshing** (`yade/`):
- `yade_woodchip_filler.py`: 3D gravity deposition (wood chips + filler) → horizontal slice → relaxed periodic 2D sheet
- `batch_filler_dataset.py`: filler-fraction sweep — deposit + mesh one RVE per fraction, with overview plots
- `mesh_rve_filler_2D.py`: strip filler from a packing and mesh it (wood = inclusions, rest = matrix)
- `mesh_utils.py`: supporting functions — convex-hull periodic meshing pipeline and mesh plotting
- `coords_3D_to_vtk.py`: settled 3D packing → ParaView `.vtp`
- Output stored in `yade/data/`

**Materials System** (`scripts_materials/`):
- `RVE_material.py`: Full 2D RVE homogenization with periodic BC (stress + tangent)
- `RVE_material_3D.py`: Simplified 3D RVE homogenization (stress only, no tangent)
- `PRNN_mat.py`: PRNN surrogate material (fast ML prediction)
- `fe2mat.py`: Abstract FE² material interface with RVEStateManager
- `fe2mat_heterogeneous_v2.py`: Parallel FE² with different RVEs per quadrature point
- Material models: `neohooke.py`, `bonet.py`, `orthotropic.py`
- `rve_mesher.py`: RVE mesh generation utilities

**Neural Networks** (`scripts_surrogates/`):
- `LDprnn.py`: Physics-informed RNN architectures (core ML module)
- `trainer.py`: JAX/Flax training framework
- `data_utils.py`: Data processing, normalization, and dataset loading
- `train_multiconfig_batch.py`: Multi-architecture comparison training

**FEM Drivers** (`scripts_FEM/`):
- `LDFE2.py`: Production FE² driver with PRNN materials
- `testFE2.py`: Simple test case for validation
- `graded_hole_compression.py`: FE² with stress concentrations
- `optimization.py` / `optimize_auxetic.py`: Design optimization workflows
- `validate_fe2_solution.py`: PRNN vs RVE comparison plots

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
- **MWE (Minimal Working Examples)**: `MWE_RVE/` contains standalone test files for component testing
- **Material Interface**: All materials inherit from common `Material` base class
- **JAX Integration**: Neural networks use JAX for automatic differentiation and JIT compilation
- **MPI Parallelization**: FEM computations distributed via mpi4py/PETSc

### Dependencies
- **FEniCSx/dolfinx**: Modern finite element framework
- **dolfinx_mpc**: Multi-point constraints for periodic boundary conditions
- **dolfinx_materials**: Advanced material modeling (external dependency in `MWE_RVE/`)
- **JAX/Flax**: Neural network framework with automatic differentiation
- **MPI4Py/PETSc**: Parallel computing and linear algebra
- **YADE**: Discrete element method for particle deposition simulations
- **gmsh**: Mesh generation (Python API)

### File Naming Conventions
- `LD*`: Large deformation formulations
- `*_mat.py`: Material model implementations
- `create*.py`: Data generation scripts
- `train_*.py`: Training scripts
- `MWE_*`: Minimal working examples
- `*_3D.*`: 3D variants of existing 2D workflows

## Testing Approach

- Use `testFE2.py` for basic FE² validation
- MWE files in `MWE_RVE/` for component testing
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
- 3D RVE solver is simplified (stress only) to manage computational cost
