# Third-Party Software and Attribution

## Canonical Basis Optimizer

This repository contains code adapted from:

**Tom Goertzen**, [canonical-basis-optimizer](https://github.com/TomGoertzen/canonical-basis-optimizer).

Associated mathematical publication:

Tom Goertzen and Geordie Williamson, *Kazhdan–Lusztig Basis and Optimization*, arXiv:2604.18894 (2026).

The original repository is distributed under the MIT License. Its copyright and licensing notices should be preserved for any redistributed derivative code.

### Modifications

The optimization implementation of Goertzen and Williamson has been adapted to incorporate the additional longest-element intertwining constraint

\[
A^{-1}PA=E,
\]

where \(P\) represents the action of the longest element \(w_0\) in the initial basis and \(E\) is the corresponding signed evacuation permutation matrix.

The implementation also includes adjustments to maintain consistent last-letter ordering of standard Young tableaux and additional computational diagnostics.

The original optimization framework, including upper-unitriangular base-change matrices, invariant-cone constraints, and the computation of extremal solutions, is due to Goertzen and Williamson.

## Additional dependencies

KL left-cell computations use Nguyen's `wgraph.c`. Its original source, attribution, and redistribution terms should be documented separately.
