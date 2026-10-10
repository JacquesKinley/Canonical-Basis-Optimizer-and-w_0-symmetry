# Third-Party Software and Attribution

## Canonical Basis Optimizer

This repository contains code adapted from:

**Tom Goertzen**, [canonical-basis-optimizer](https://github.com/TomGoertzen/canonical-basis-optimizer).

Associated mathematical publication:

Tom Goertzen and Geordie Williamson, *Kazhdan–Lusztig Basis and Optimization*, arXiv:2604.18894 (2026).

The original repository is distributed under the MIT License. Its copyright and licensing notices should be preserved for any redistributed derivative code.

### Modifications

The optimization implementation of Goertzen and Williamson has been extended to incorporate the additional longest-element intertwining constraint

$$
A^{-1}PA=E,
$$

where $$P$$ represents the action of $$w_0$$ on the polytabloid basis and $$E$$ is the corresponding signed evacuation permutation matrix.

The original optimization framework, including the last-letter ordering of standard Young tableaux, upper-unitriangular base-change matrices, and computation of extremal solutions $$A_{\min}$$ and $$A_{\max}$$, is due to Goertzen and Williamson.

## Additional dependencies

KL left-cell computations use Nguyen's `wgraph.c`. Its original source, attribution, and redistribution terms should be documented separately.

The use of AI [ChatGPT 5.6 sol (Plus)] was implemented to help find the Evacuation-adapted connected components of the weak Bruhat graph, the Y-KL base-Change matrices and extend Goertzen's code to include the long-element intertwining constraint.
