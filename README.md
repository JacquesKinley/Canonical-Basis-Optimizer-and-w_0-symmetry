# Canonical Basis Optimizer and \(w_0\)-Symmetry

This repository contains SageMath code developed as part of my thesis on \(w_0\)-symmetric bases of the Specht module.

## Main computations

The repository includes:

- **Evacuation-adapted weak Bruhat graphs:** Construction and analysis of evacuation-adapted connected components of weak Bruhat graphs associated with standard Young tableaux.
- **Canonical basis optimization:** Computation of the extremal base-change matrices \(A_{\min}\) and \(A_{\max}\), including evacuation intertwining constraints.

## Requirements

- SageMath
- Gurobi and `gurobipy` (for optimization)
- NumPy
- `tqdm`
- A C compiler and Nguyen's `wgraph.c` (for KL left-cell computations)

## Attribution

The optimization code builds upon work from Tom Goertzen's [canonical-basis-optimizer](https://github.com/TomGoertzen/canonical-basis-optimizer) repository, associated with the paper *Kazhdan–Lusztig Basis and Optimization* by Tom Goertzen and Geordie Williamson.

The code in this repository includes modifications for evacuation constraints, last-letter tableau ordering, and direct optimization over upper-unitriangular base-change matrices.

See `THIRD_PARTY.md` for further attribution.R
