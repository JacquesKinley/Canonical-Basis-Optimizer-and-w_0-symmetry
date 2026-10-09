# Canonical Basis Optimizer and \(w_0\)-Symmetry

This repository contains SageMath code developed as part of my thesis on \(w_0\)-symmetric bases of the Specht module.

#Main Contributions

This repository contains two principal computational contributions developed for my thesis on (w_0)-symmetric bases of Specht modules.

1. Evacuation-Adapted Weak Bruhat Graphs

A SageMath implementation for constructing evacuation-adapted subgraphs of weak Bruhat graphs and determining their connected components for partitions (\lambda\vdash n).

2. Evacuation-Constrained KL Left-Cell Optimization

An extension of Goertzen and Williamson's canonical-basis optimization implementation, incorporating the additional longest-element intertwining constraint

$$
A^{-1}PA=E.
$$

This allows the investigation of extremal invariant simplicial cones subject to a prescribed action of (w_0).

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
