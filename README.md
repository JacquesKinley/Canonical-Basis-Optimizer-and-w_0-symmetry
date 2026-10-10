# Canonical Basis Optimizer and $$w_0$$-Symmetry

This repository contains SageMath code developed as part of my thesis on $$w_0$$-symmetric bases of the Specht module.

## Main Contributions

This repository contains three principal computational contributions developed for my thesis on $$w_0$$-symmetric bases of Specht modules.

**1. Young–Kazhdan–Lusztig Base-Change Matrix Computations**

(File: YoungSeminormalToKLBaseChange.sage) Computes exact base-change matrices between the standard polytabloid basis, Young's monic seminormal basis, and the Kazhdan–Lusztig left-cell basis.

In particular, it computes

$$
A_{\mathbb{KL}}^{-1}A_{\mathbb{Y}},
$$

expressing Young's seminormal basis vectors in Kazhdan–Lusztig coordinates.

The implementation uses rational arithmetic, checks the generator-intertwining equations exactly, and measures the distance of the transition matrix from the identity.

This provides an independent computational tool for comparing the two bases in one base-change matrix.

**2. Evacuation-Adapted Weak Bruhat Graphs**

Consider the weak Bruhat graph with nodes as standard Young tableaux. We say a labelled edge $$T$$ -- $$s_i$$ -- $$T'$$ in $$\mathcal{B}^\lambda$$ is evacuation-adapted if $$ev(T)$$ -- $$s_{n-i}$$ -- $$ev(T')$$ is also an edge in $\mathcal{B}^\lambda$.
We include a SageMath implementation for constructing evacuation-adapted subgraphs of weak Bruhat graphs and determining their connected components for partitions $$\lambda\vdash n$$.

**3. Evacuation-Constrained KL Left-Cell Optimization**

An extension of Goertzen and Williamson's canonical-basis optimization implementation, incorporating the additional longest-element intertwining constraint

$$
A^{-1}PA=E.
$$

This allows the investigation of extremal invariant simplicial cones over $$w_0$$-symmetric bases of the Specht module.


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
