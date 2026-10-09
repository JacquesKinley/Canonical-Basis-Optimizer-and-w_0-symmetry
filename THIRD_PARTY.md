# Third-Party Software and Attribution

## Canonical Basis Optimizer

This repository contains code adapted from:

**Tom Goertzen**, [canonical-basis-optimizer](https://github.com/TomGoertzen/canonical-basis-optimizer).

Associated mathematical publication:

Tom Goertzen and Geordie Williamson, *Kazhdan–Lusztig Basis and Optimization*, arXiv:2604.18894 (2026).

The original repository is distributed under the MIT License. Its copyright and licensing notices should be preserved for any redistributed derivative code.

### Modifications

The optimization implementation has been extended to incorporate:

- Evacuation and longest-element intertwining constraints.
- Last-letter ordering of standard Young tableaux.
- Direct optimization over upper-unitriangular base-change matrices.
- Computation of extremal solutions and corresponding transition matrices.

## Additional dependencies

KL left-cell computations use Nguyen's `wgraph.c`. Its original source, attribution, and redistribution terms should be documented separately.
