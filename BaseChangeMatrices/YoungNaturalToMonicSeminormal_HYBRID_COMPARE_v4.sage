# YoungNaturalToMonicSeminormal.sage
#
# Exact SageMath implementation of the Armon--Halverson transition-matrix
# algorithm, together with the diagonal renormalisation and unitriangular
# inversion that produces the seminormal convention
#
#       [s_i] = [[a, 1-a^2],
#                [1,   -a ]]
#
# on every standard two-tableau block ordered from lower to higher weak order.
#
# Matrix conventions
# ------------------
# Let
#       N = (n_T)  be Young's natural / standard-polytabloid basis,
#       V = (v_T)  be the Armon--Halverson seminormal basis,
#       F = (f_T)  be the monic seminormal basis used by the user.
#
# The paper's transition matrix H is defined by
#
#       N = V H,             n_T = sum_S H[S,T] v_S.
#
# Let D = diag(H[T,T]) and f_T = H[T,T] v_T.  Then
#
#       F = V D = N H^(-1) D.
#
# Hence the desired upper-unitriangular matrix is
#
#       A = H^(-1) D = (D^(-1) H)^(-1).
#
# Its columns express the monic seminormal vectors in the
# natural / polytabloid basis:
#
#       f_T = sum_S A[S,T] n_S.
#
# Run with:
#       sage YoungNaturalToMonicSeminormal.sage
#
# Or from another Sage file:
#       load("YoungNaturalToMonicSeminormal.sage")
#       result = young_natural_to_monic_seminormal([3,2])
#       result["A"]

from functools import lru_cache
from fractions import Fraction
import math
from sage.all import QQ, Integer, matrix, diagonal_matrix, identity_matrix

YOUNG_SEMINORMAL_FILE_VERSION = "2026-07-27-hybrid-compare-v4"


# ---------------------------------------------------------------------------
# Partitions and standard tableaux
# ---------------------------------------------------------------------------

def normalise_partition(partition):
    """
    Convert partition to a decreasing tuple of positive Python integers.
    """
    lam = tuple(int(x) for x in partition)

    if not lam:
        raise ValueError("The partition must be nonempty.")

    if any(x <= 0 for x in lam):
        raise ValueError("All parts of the partition must be positive.")

    if any(lam[r] < lam[r + 1] for r in range(len(lam) - 1)):
        raise ValueError("The parts must be weakly decreasing.")

    return lam


@lru_cache(maxsize=None)
def _standard_tableaux_cached(shape):
    """
    Return all standard tableaux of straight shape 'shape'.

    A tableau is stored as a tuple of row-tuples, for example

        ((1,3,5), (2,4))

    for the tableau 1 3 5 / 2 4.

    The recursion removes a corner containing n and then reinserts n.
    """
    shape = tuple(shape)
    n = sum(shape)

    if n == 0:
        return ((),)

    tableaux = []

    for r in range(len(shape)):
        next_row_length = shape[r + 1] if r + 1 < len(shape) else 0

        # The last box of row r is removable precisely when the next row
        # is strictly shorter.
        if shape[r] <= next_row_length:
            continue

        smaller = list(shape)
        smaller[r] -= 1

        if smaller[r] == 0:
            # This can only remove the final row of a straight partition.
            smaller.pop(r)

        smaller = tuple(smaller)

        for smaller_tableau in _standard_tableaux_cached(smaller):
            rows = [list(row) for row in smaller_tableau]

            if r == len(rows):
                rows.append([n])
            else:
                rows[r].append(n)

            tableaux.append(tuple(tuple(row) for row in rows))

    return tuple(tableaux)


def tableau_positions(tableau):
    """
    Return the dictionary entry -> (row, column), using zero-based coordinates.
    """
    positions = {}

    for r, row in enumerate(tableau):
        for c, entry in enumerate(row):
            positions[int(entry)] = (r, c)

    return positions


def tableau_size(tableau):
    return sum(len(row) for row in tableau)


def column_reading_tableau(partition):
    """
    Fill columns from left to right, and each column from top to bottom.

    For (3,2), this returns ((1,3,5),(2,4)).
    """
    lam = normalise_partition(partition)
    rows = [[None] * row_length for row_length in lam]

    entry = 1
    for c in range(lam[0]):
        for r in range(len(lam)):
            if c < lam[r]:
                rows[r][c] = entry
                entry += 1

    return tuple(tuple(row) for row in rows)


def is_standard(tableau):
    """
    Test strict increase along rows and down columns.
    """
    for row in tableau:
        if any(row[c] >= row[c + 1] for c in range(len(row) - 1)):
            return False

    for r in range(len(tableau) - 1):
        common_length = min(len(tableau[r]), len(tableau[r + 1]))

        for c in range(common_length):
            if tableau[r][c] >= tableau[r + 1][c]:
                return False

    return True


def swap_adjacent(tableau, i, positions=None):
    """
    Swap i and i+1.  Return the resulting tableau when it is standard,
    and return None otherwise.
    """
    if positions is None:
        positions = tableau_positions(tableau)

    rows = [list(row) for row in tableau]
    r1, c1 = positions[i]
    r2, c2 = positions[i + 1]

    rows[r1][c1], rows[r2][c2] = rows[r2][c2], rows[r1][c1]
    swapped = tuple(tuple(row) for row in rows)

    return swapped if is_standard(swapped) else None


def weak_length(tableau, positions=None):
    """
    Length of the word taking the column-reading tableau to T.

    This is the number of tableau inversions (i,j) with i>j and i strictly
    south-west of j.
    """
    if positions is None:
        positions = tableau_positions(tableau)

    n = tableau_size(tableau)
    length = 0

    for i in range(2, n + 1):
        ri, ci = positions[i]

        for j in range(1, i):
            rj, cj = positions[j]

            if ri > rj and ci < cj:
                length += 1

    return length


def last_letter_key(tableau, positions=None):
    """
    Recursive last-letter ordering key.

    Compare first the row occupied by n, then the row occupied by n-1,
    and so on.  Rows are numbered from top to bottom, beginning at 0.

    For shape (3,2), this gives the paper's order

        135/24, 125/34, 134/25, 124/35, 123/45.
    """
    if positions is None:
        positions = tableau_positions(tableau)

    n = tableau_size(tableau)
    return tuple(positions[k][0] for k in range(n, 0, -1))


def standard_tableaux(partition, order="last_letter"):
    """
    Return all SYT(partition) in a specified Bruhat-compatible total order.

    Supported orders:
        "last_letter"  -- the ordering used in the user's computations;
        "depth"        -- weak length, then last-letter key.

    The routine later verifies that the chosen order is a linear extension
    of weak order.
    """
    lam = normalise_partition(partition)
    tableaux = list(_standard_tableaux_cached(lam))
    positions = {T: tableau_positions(T) for T in tableaux}

    if order == "last_letter":
        tableaux.sort(key=lambda T: last_letter_key(T))
    elif order == "depth":
        tableaux.sort(
            key=lambda T: (
                weak_length(T, positions[T]),
                last_letter_key(T),
            )
        )
    else:
        raise ValueError("order must be 'last_letter' or 'depth'.")

    return tableaux


# ---------------------------------------------------------------------------
# Armon--Halverson data
# ---------------------------------------------------------------------------

def axial_coefficient(tableau, i, positions=None):
    r"""
    Return

        a_i(T) = 1 / (ct_T(i+1) - ct_T(i)),

    where ct(row,column) = column-row.
    """
    if positions is None:
        positions = tableau_positions(tableau)

    r1, c1 = positions[i]
    r2, c2 = positions[i + 1]

    content_i = c1 - r1
    content_ip1 = c2 - r2
    denominator = content_ip1 - content_i

    if denominator == 0:
        raise ZeroDivisionError(
            "Consecutive entries unexpectedly have equal content."
        )

    return QQ(1) / QQ(denominator)


def _precompute_tableau_data(tableaux):
    """
    Precompute positions, weak lengths, adjacent swaps, and axial coefficients.
    """
    n = tableau_size(tableaux[0])
    positions = {T: tableau_positions(T) for T in tableaux}
    lengths = {T: weak_length(T, positions[T]) for T in tableaux}
    swaps = {}
    axial = {}

    for T in tableaux:
        for i in range(1, n):
            swaps[(T, i)] = swap_adjacent(T, i, positions[T])
            axial[(T, i)] = axial_coefficient(T, i, positions[T])

    return n, positions, lengths, swaps, axial


def _validate_weak_order_extension(tableaux, lengths, swaps):
    """
    Verify that every downward weak cover occurs earlier in the list.
    """
    index = {T: j for j, T in enumerate(tableaux)}
    n = tableau_size(tableaux[0])

    for T in tableaux:
        for i in range(1, n):
            Q = swaps[(T, i)]

            if Q is None:
                continue

            if lengths[Q] == lengths[T] - 1 and index[Q] >= index[T]:
                raise ValueError(
                    "The selected tableau order is not a linear extension "
                    "of weak order."
                )


def _choose_weak_predecessor(T, index, lengths, swaps, n):
    """
    Choose one T' = s_i(T) immediately below T in weak order.

    Corollary 4.6 is independent of which predecessor is chosen.
    We choose the smallest valid generator label i for determinism.
    """
    candidates = []

    for i in range(1, n):
        predecessor = swaps[(T, i)]

        if predecessor is None:
            continue

        if (
            lengths[predecessor] == lengths[T] - 1
            and index[predecessor] < index[T]
        ):
            candidates.append((i, predecessor))

    if not candidates:
        raise RuntimeError(
            "No downward weak-order cover was found for a nonminimal tableau."
        )

    return min(candidates, key=lambda pair: pair[0])


def armon_halverson_matrix(partition, order="last_letter"):
    r"""
    Compute the Armon--Halverson matrix H exactly over QQ.

    The convention is

        n_T = sum_S H[S,T] v_S,

    so N = V H.

    The columns are computed with Corollary 4.6.  If T' = s_l(T) is an
    immediate weak predecessor, then

        H[S,T] =
            a_l(S) H[S,T']
            + (1-a_l(S)) H[s_l(S),T'],

    when s_l(S) is standard, and only the first term otherwise.
    """
    lam = normalise_partition(partition)
    tableaux = standard_tableaux(lam, order=order)
    m = len(tableaux)
    index = {T: j for j, T in enumerate(tableaux)}

    n, positions, lengths, swaps, axial = _precompute_tableau_data(tableaux)
    _validate_weak_order_extension(tableaux, lengths, swaps)

    C = column_reading_tableau(lam)

    if C not in index:
        raise RuntimeError("The column-reading tableau was not generated.")

    if index[C] != 0:
        raise RuntimeError(
            "The column-reading tableau must be first in the selected order."
        )

    H = matrix(QQ, m, m, 0)
    H[0, 0] = QQ(1)

    predecessor_data = [None] * m

    for column in range(1, m):
        T = tableaux[column]
        ell, T_prime = _choose_weak_predecessor(
            T, index, lengths, swaps, n
        )
        previous_column = index[T_prime]
        predecessor_data[column] = {
            "generator": ell,
            "predecessor_index": previous_column,
            "predecessor": T_prime,
        }

        for row, S in enumerate(tableaux):
            a = axial[(S, ell)]
            value = a * H[row, previous_column]
            S_prime = swaps[(S, ell)]

            if S_prime is not None:
                value += (
                    QQ(1) - a
                ) * H[index[S_prime], previous_column]

            H[row, column] = value

    if any(H[r, c] != 0 for r in range(m) for c in range(r)):
        raise ArithmeticError(
            "H is not upper triangular; check the tableau ordering."
        )

    if any(H[j, j] == 0 for j in range(m)):
        raise ArithmeticError("H has a zero diagonal entry.")

    return {
        "partition": lam,
        "tableaux": tableaux,
        "index": index,
        "H": H,
        "predecessors": predecessor_data,
        "positions": positions,
        "lengths": lengths,
        "swaps": swaps,
        "axial": axial,
    }


# ---------------------------------------------------------------------------
# Conversion to the monic upper-unitriangular seminormal basis
# ---------------------------------------------------------------------------

def upper_unitriangular_conversion(H):
    r"""
    Given the paper's upper-triangular matrix H, construct

        D = diag(H[T,T]),
        U = D^(-1) H,
        A = U^(-1) = H^(-1)D.

    A is computed by the requested recursion, not by calling matrix inverse:

        A[T,T] = 1,

        A[S,T] = -sum_{S<R<=T} U[S,R] A[R,T].

    In matrix-index notation, for i<j,

        A[i,j] = -sum_{k=i+1}^j U[i,k] A[k,j].
    """
    if H.nrows() != H.ncols():
        raise ValueError("H must be square.")

    m = H.nrows()
    diagonal = [H[j, j] for j in range(m)]

    if any(d == 0 for d in diagonal):
        raise ValueError("H must have nonzero diagonal.")

    D = diagonal_matrix(QQ, diagonal)

    # Left multiplication by D^(-1) divides row i by H[i,i].
    U = matrix(
        QQ,
        m,
        m,
        lambda i, j: H[i, j] / diagonal[i],
    )

    A = matrix(QQ, m, m, 0)

    for column in range(m):
        A[column, column] = QQ(1)

        # Compute the column from bottom to top.
        for row in range(column - 1, -1, -1):
            A[row, column] = -sum(
                U[row, k] * A[k, column]
                for k in range(row + 1, column + 1)
            )

    I = identity_matrix(QQ, m)

    if U * A != I:
        raise ArithmeticError("The recursive inversion check U*A=I failed.")

    if H * A != D:
        raise ArithmeticError("The exact change-of-basis check H*A=D failed.")

    return {
        "D": D,
        "U": U,
        "A": A,
        "diagonal": diagonal,
    }




def rationalize_numeric_matrix(
    M,
    max_denominator=100,
    tolerance=1e-4,
    force=False,
):
    r"""
    Convert a numerical matrix to QQ by reconstructing nearby small fractions.

    If the input base ring is inexact (for example RDF), reconstruction is
    automatic.  Exact input matrices are preserved unless ``force=True``.

    Returns ``(M_QQ, max_error, reconstruction_used)``.
    """
    M0 = matrix(M)

    try:
        input_is_exact = bool(M0.base_ring().is_exact())
    except Exception:
        input_is_exact = False

    reconstruct = force or not input_is_exact

    if not reconstruct:
        return matrix(QQ, M0), 0.0, False

    if max_denominator <= 0:
        raise ValueError("max_denominator must be positive.")

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    Q = matrix(QQ, M0.nrows(), M0.ncols(), 0)
    max_error = 0.0

    for i in range(M0.nrows()):
        for j in range(M0.ncols()):
            xf = float(M0[i, j])

            if not math.isfinite(xf):
                raise ValueError(
                    "Cannot rationalise a non-finite matrix entry."
                )

            if abs(xf) <= tolerance:
                q = QQ(0)
            elif abs(xf - round(xf)) <= tolerance:
                q = QQ(int(round(xf)))
            else:
                fraction = Fraction(xf).limit_denominator(
                    int(max_denominator)
                )
                q = QQ(fraction.numerator) / QQ(fraction.denominator)

            error = abs(xf - float(q))
            max_error = max(max_error, error)

            if error > tolerance:
                raise ArithmeticError(
                    "Could not certify rational reconstruction of entry "
                    "({},{}): numerical value {}, candidate {}, error {}, "
                    "tolerance {}."
                    .format(i, j, xf, q, error, tolerance)
                )

            Q[i, j] = q

    return Q, max_error, True


def invert_upper_unitriangular(M):
    r"""
    Invert an upper-unitriangular matrix exactly over QQ by recursion.

    If X = M^(-1), then

        X[j,j] = 1,
        X[i,j] = -sum_{i<k<=j} M[i,k] X[k,j].
    """
    M = matrix(QQ, M)

    if M.nrows() != M.ncols():
        raise ValueError("The matrix must be square.")

    m = M.nrows()

    if any(M[i, j] != 0 for i in range(m) for j in range(i)):
        raise ValueError("The matrix must be upper triangular.")

    if any(M[i, i] != 1 for i in range(m)):
        raise ValueError("The matrix must have diagonal entries equal to 1.")

    X = matrix(QQ, m, m, 0)

    for column in range(m):
        X[column, column] = QQ(1)
        for row in range(column - 1, -1, -1):
            X[row, column] = -sum(
                M[row, k] * X[k, column]
                for k in range(row + 1, column + 1)
            )

    if M * X != identity_matrix(QQ, m):
        raise ArithmeticError("The recursive inverse check failed.")

    return X



def basis_A_to_armon_halverson(
    A,
    partition,
    order="last_letter",
    rationalize_input=True,
    max_denominator=100,
    rational_tolerance=1e-4,
    force_rational_reconstruction=False,
):
    r"""
    Compare an arbitrary basis matrix A with the Armon--Halverson and monic
    Young seminormal bases.

    This is the correct function for a hybrid optimiser basis.

    Conventions
    -----------
    Let

        N = (n_T)       be the natural / standard-polytabloid basis,
        V = (v_T)       be the Armon--Halverson seminormal basis,
        F = (f_T)       be the monic Young seminormal basis,
        B = (b_T)       be the arbitrary input basis.

    The input matrix A is defined by

        B = N A.

    The paper's matrix H_AH satisfies

        N = V H_AH.

    Therefore

        B = V (H_AH A).

    Thus

        M_AH = H_AH A

    is the matrix whose columns express the input basis vectors b_T in the
    Armon--Halverson basis V.  Equivalently, it converts coordinate columns by

        [x]_AH = M_AH [x]_B.

    Its inverse converts in the opposite direction:

        [x]_B = M_AH^(-1) [x]_AH.

    If D = diag(H_AH[T,T]), then the monic seminormal basis is

        F = V D.

    Hence

        B = F C,
        C = D^(-1) H_AH A.

    The matrix C is usually the cleanest comparison between a hybrid basis and
    Young's monic seminormal basis.  It also equals

        C = A_Young^(-1) A,

    where A_Young is the upper-unitriangular polytabloid-to-monic-seminormal
    transition matrix.

    Returns
    -------
    A:
        Exact input basis matrix over QQ.

    H_AH:
        The genuine Armon--Halverson transition matrix, N = V H_AH.

    D:
        Diagonal scaling from the Armon--Halverson basis to the monic
        seminormal basis, F = V D.

    A_Young:
        Genuine upper-unitriangular transition matrix, F = N A_Young.

    hybrid_in_AH:
        H_AH*A.  Columns express the hybrid/input basis in the
        Armon--Halverson basis.  Also converts hybrid coordinates to
        Armon--Halverson coordinates.

    AH_in_hybrid:
        (H_AH*A)^(-1).  Converts Armon--Halverson coordinates to hybrid
        coordinates.

    hybrid_in_monic:
        D^(-1)*H_AH*A = A_Young^(-1)*A.  Columns express the hybrid basis in
        the monic Young seminormal basis.

    monic_in_hybrid:
        inverse of hybrid_in_monic.
    """
    A_input = matrix(A)

    if rationalize_input:
        A_exact, input_error, rationalisation_used = rationalize_numeric_matrix(
            A_input,
            max_denominator=max_denominator,
            tolerance=rational_tolerance,
            force=force_rational_reconstruction,
        )
    else:
        A_exact = matrix(QQ, A_input)
        input_error = 0.0
        rationalisation_used = False

    if A_exact.nrows() != A_exact.ncols():
        raise ValueError("A must be square.")

    ah = armon_halverson_matrix(partition, order=order)
    H_AH = ah["H"]
    m = H_AH.nrows()

    if A_exact.nrows() != m:
        raise ValueError(
            "The size of A does not equal the number of standard tableaux "
            "of the supplied partition."
        )

    try:
        A_inverse = A_exact.inverse()
    except ZeroDivisionError:
        raise ValueError("A must be invertible.")
    except ArithmeticError:
        raise ValueError("A must be invertible.")

    conversion = upper_unitriangular_conversion(H_AH)
    D = conversion["D"]
    A_Young = conversion["A"]

    hybrid_in_AH = H_AH * A_exact
    AH_in_hybrid = A_inverse * H_AH.inverse()

    hybrid_in_monic = D.inverse() * hybrid_in_AH
    monic_in_hybrid = hybrid_in_monic.inverse()

    # Exact consistency checks.
    if H_AH * A_exact != hybrid_in_AH:
        raise ArithmeticError("Failed the identity hybrid_in_AH = H_AH*A.")

    if hybrid_in_AH * AH_in_hybrid != identity_matrix(QQ, m):
        raise ArithmeticError(
            "Failed the coordinate-direction inverse check for the "
            "Armon--Halverson comparison."
        )

    if D * hybrid_in_monic != hybrid_in_AH:
        raise ArithmeticError(
            "Failed the identity D*hybrid_in_monic = hybrid_in_AH."
        )

    if A_Young * hybrid_in_monic != A_exact:
        raise ArithmeticError(
            "Failed the identity A_Young*hybrid_in_monic = A."
        )

    if hybrid_in_monic * monic_in_hybrid != identity_matrix(QQ, m):
        raise ArithmeticError(
            "Failed the coordinate-direction inverse check for the monic "
            "seminormal comparison."
        )

    result = dict(ah)
    result.update({
        "A_input": A_input,
        "A": A_exact,
        "A_inverse": A_inverse,
        "H_AH": H_AH,
        "D": D,
        "A_Young": A_Young,
        "hybrid_in_AH": hybrid_in_AH,
        "AH_coordinates_from_hybrid_coordinates": hybrid_in_AH,
        "AH_in_hybrid": AH_in_hybrid,
        "hybrid_coordinates_from_AH_coordinates": AH_in_hybrid,
        "hybrid_in_monic": hybrid_in_monic,
        "monic_coordinates_from_hybrid_coordinates": hybrid_in_monic,
        "monic_in_hybrid": monic_in_hybrid,
        "hybrid_coordinates_from_monic_coordinates": monic_in_hybrid,
        "input_rationalisation_used": rationalisation_used,
        "input_rationalisation_max_error": input_error,
        "max_denominator": max_denominator,
        "rational_tolerance": rational_tolerance,
    })

    return result


# A shorter alias with the same behaviour.
compare_hybrid_with_seminormal = basis_A_to_armon_halverson


def monic_A_to_armon_halverson(
    A,
    partition=None,
    diagonal=None,
    order="last_letter",
    verify_against_algorithm=True,
    rationalize_input=True,
    max_denominator=100,
    rational_tolerance=1e-4,
    force_rational_reconstruction=False,
):
    r"""
    Convert a monic upper-unitriangular transition matrix A into the
    Armon--Halverson transition matrix H.

    Relations:

        A = H^(-1) D = U^(-1),
        U = D^(-1) H,
        U = A^(-1),
        H = D A^(-1).

    A determines U uniquely but does not determine D.  Supply either
    partition=... so the Armon--Halverson diagonal is computed from their
    algorithm, or diagonal=... explicitly.
    """
    A_input = matrix(A)

    if rationalize_input:
        A, input_rationalisation_error, rationalisation_used = (
            rationalize_numeric_matrix(
                A_input,
                max_denominator=max_denominator,
                tolerance=rational_tolerance,
                force=force_rational_reconstruction,
            )
        )
    else:
        A = matrix(QQ, A_input)
        input_rationalisation_error = 0.0
        rationalisation_used = False

    if A.nrows() != A.ncols():
        raise ValueError("A must be square.")

    m = A.nrows()

    if any(A[i, j] != 0 for i in range(m) for j in range(i)):
        raise ValueError("A must be upper triangular.")

    if any(A[i, i] != 1 for i in range(m)):
        raise ValueError("A must be upper unitriangular.")

    if partition is not None and diagonal is not None:
        raise ValueError("Supply either partition or diagonal, not both.")

    if partition is None and diagonal is None:
        raise ValueError(
            "A determines U=A^(-1), but not D. Supply partition=... "
            "or diagonal=...."
        )

    U = invert_upper_unitriangular(A)
    result = {
        "A_input": A_input,
        "A": A,
        "U": U,
        "input_rationalisation_used": rationalisation_used,
        "input_rationalisation_max_error": input_rationalisation_error,
        "max_denominator": max_denominator,
        "rational_tolerance": rational_tolerance,
    }

    if partition is not None:
        ah = armon_halverson_matrix(partition, order=order)
        H_algorithm = ah["H"]

        if H_algorithm.nrows() != m:
            raise ValueError(
                "The size of A does not equal the number of standard "
                "tableaux of the supplied partition."
            )

        diagonal_values = [H_algorithm[j, j] for j in range(m)]
        result.update({
            "partition": ah["partition"],
            "tableaux": ah["tableaux"],
            "index": ah["index"],
            "positions": ah["positions"],
            "lengths": ah["lengths"],
            "swaps": ah["swaps"],
            "axial": ah["axial"],
            "predecessors": ah["predecessors"],
            "H_algorithm": H_algorithm,
        })
    else:
        diagonal_values = [QQ(x) for x in diagonal]
        if len(diagonal_values) != m:
            raise ValueError(
                "The explicit diagonal must have one entry for each row of A."
            )
        if any(d == 0 for d in diagonal_values):
            raise ValueError("All diagonal entries must be nonzero.")

    D = diagonal_matrix(QQ, diagonal_values)
    H = D * U

    if H * A != D:
        raise ArithmeticError("The exact reconstruction check H*A=D failed.")

    result.update({
        "D": D,
        "H": H,
        "diagonal": diagonal_values,
    })

    if partition is not None:
        matches = (H == result["H_algorithm"])
        result["matches_algorithm"] = matches
        if verify_against_algorithm and not matches:
            raise ArithmeticError(
                "The reconstructed H does not match the direct "
                "Armon--Halverson matrix."
            )

    return result


def armon_halverson_entry_from_A(
    A,
    partition,
    S,
    T,
    order="last_letter",
    rationalize_input=True,
    max_denominator=100,
    rational_tolerance=1e-4,
):
    r"""
    Track a pair (S,T) through the correct arbitrary-basis comparison.

    S and T may be zero-based matrix indices or tableau tuples such as

        ((1,3,5),(2,4)).

    The returned entries include:

        A[S,T],
        H_AH[S,T],
        (H_AH*A)[S,T],
        (D^(-1)*H_AH*A)[S,T].

    The final two entries describe the hybrid basis in the paper's
    seminormal basis and in the monic seminormal basis, respectively.
    """
    result = basis_A_to_armon_halverson(
        A,
        partition=partition,
        order=order,
        rationalize_input=rationalize_input,
        max_denominator=max_denominator,
        rational_tolerance=rational_tolerance,
    )

    tableaux = result["tableaux"]
    index = result["index"]
    m = len(tableaux)

    def resolve(X, name):
        if isinstance(X, (int, Integer)):
            j = int(X)
            if not (0 <= j < m):
                raise IndexError(
                    "{} index must lie between 0 and {}.".format(name, m - 1)
                )
            return j, tableaux[j]

        X = tuple(tuple(int(x) for x in row) for row in X)
        if X not in index:
            raise ValueError(
                "{} is not a standard tableau of the supplied shape."
                .format(name)
            )
        return index[X], X

    row, S_tableau = resolve(S, "S")
    column, T_tableau = resolve(T, "T")

    return {
        "S_index": row,
        "T_index": column,
        "S": S_tableau,
        "T": T_tableau,
        "A_entry": result["A"][row, column],
        "H_AH_entry": result["H_AH"][row, column],
        "hybrid_in_AH_entry": result["hybrid_in_AH"][row, column],
        "AH_in_hybrid_entry": result["AH_in_hybrid"][row, column],
        "hybrid_in_monic_entry": result["hybrid_in_monic"][row, column],
        "monic_in_hybrid_entry": result["monic_in_hybrid"][row, column],
        "conversion": result,
    }


def young_natural_to_monic_seminormal(partition, order="last_letter"):
    r"""
    Main routine.

    Return all relevant exact matrices for a given partition:

        H : Armon--Halverson matrix, N = V H;
        D : diagonal renormalisation, F = V D;
        U : D^(-1)H;
        A : desired upper-unitriangular matrix, F = N A.

    Thus the T-th column of A gives f_T in the standard
    polytabloid/natural basis.
    """
    result = armon_halverson_matrix(partition, order=order)
    conversion = upper_unitriangular_conversion(result["H"])
    result.update(conversion)
    return result


# ---------------------------------------------------------------------------
# Seminormal generator matrices and checks
# ---------------------------------------------------------------------------

def paper_seminormal_generator(result, i):
    r"""
    Matrix of s_i on the Armon--Halverson seminormal basis V:

        s_i v_T = a_i(T)v_T + (1+a_i(T))v_{s_iT}.
    """
    tableaux = result["tableaux"]
    index = result["index"]
    swaps = result["swaps"]
    axial = result["axial"]
    m = len(tableaux)
    n = tableau_size(tableaux[0])

    if not (1 <= i < n):
        raise ValueError("Generator index i must satisfy 1 <= i < n.")

    M = matrix(QQ, m, m, 0)

    for column, T in enumerate(tableaux):
        a = axial[(T, i)]
        M[column, column] = a
        T_prime = swaps[(T, i)]

        if T_prime is not None:
            M[index[T_prime], column] = QQ(1) + a

    return M


def monic_seminormal_generator(result, i):
    r"""
    Matrix of s_i on F = V D.

    This is D^(-1) [s_i]_V D and has, on each ordered standard pair,
    the block

        [[a, 1-a^2],
         [1,   -a ]].
    """
    M = paper_seminormal_generator(result, i)
    diagonal = result["diagonal"]
    m = M.nrows()

    return matrix(
        QQ,
        m,
        m,
        lambda r, c: M[r, c] * diagonal[c] / diagonal[r],
    )


def all_monic_seminormal_generators(result):
    """
    Return the list [ [s_1]_F, ..., [s_{n-1}]_F ].
    """
    n = tableau_size(result["tableaux"][0])
    return [monic_seminormal_generator(result, i) for i in range(1, n)]



def check_loaded_young_seminormal_file():
    """Print identifying information for the currently loaded file."""
    print("Young seminormal file version:", YOUNG_SEMINORMAL_FILE_VERSION)
    print("monic_A_to_armon_halverson is defined:",
          callable(monic_A_to_armon_halverson))


# ---------------------------------------------------------------------------
# Display helpers and exact self-test
# ---------------------------------------------------------------------------

def tableau_string(tableau):
    return " / ".join(" ".join(str(x) for x in row) for row in tableau)


def print_tableau_order(result):
    """
    Print the matrix index, weak length, and tableau.
    """
    for j, T in enumerate(result["tableaux"], start=1):
        print(
            "{:>4}: length {:>3}   {}".format(
                j,
                result["lengths"][T],
                tableau_string(T),
            )
        )


def self_test_32(verbose=True):
    """
    Check the exact (3,2) matrices against Example 4.5 and the user's A_min.
    """
    result = young_natural_to_monic_seminormal([3, 2])

    expected_H = matrix(
        QQ,
        [
            [1, QQ(1)/2, QQ(1)/2, QQ(1)/4, -QQ(1)/4],
            [0, QQ(3)/2, 0, QQ(3)/4, QQ(3)/4],
            [0, 0, QQ(3)/2, QQ(3)/4, QQ(3)/4],
            [0, 0, 0, QQ(9)/4, QQ(3)/4],
            [0, 0, 0, 0, 3],
        ],
    )

    expected_A = matrix(
        QQ,
        [
            [1, -QQ(1)/2, -QQ(1)/2, QQ(1)/4, QQ(2)/3],
            [0, 1, 0, -QQ(1)/2, -QQ(1)/3],
            [0, 0, 1, -QQ(1)/2, -QQ(1)/3],
            [0, 0, 0, 1, -QQ(1)/3],
            [0, 0, 0, 0, 1],
        ],
    )

    assert result["H"] == expected_H
    assert result["A"] == expected_A

    reverse = monic_A_to_armon_halverson(
        expected_A,
        partition=[3, 2],
    )
    assert reverse["U"] == result["U"]
    assert reverse["D"] == result["D"]
    assert reverse["H"] == expected_H
    assert reverse["matches_algorithm"] is True

    # The same conversion must work when A comes directly from a numerical
    # optimiser as an RDF matrix.
    reverse_float = monic_A_to_armon_halverson(
        matrix(RDF, expected_A),
        partition=[3, 2],
    )
    assert reverse_float["A"] == expected_A
    assert reverse_float["H"] == expected_H
    assert reverse_float["input_rationalisation_used"] is True

    hybrid_A = matrix(
        QQ,
        [
            [1, -QQ(1)/2, -1, 1, QQ(1)/2],
            [0, 1, 0, -1, -QQ(1)/2],
            [0, 0, 1, -1, -QQ(1)/2],
            [0, 0, 0, 1, -QQ(1)/2],
            [0, 0, 0, 0, 1],
        ],
    )

    hybrid_comparison = basis_A_to_armon_halverson(
        hybrid_A,
        partition=[3, 2],
    )

    expected_hybrid_in_AH = matrix(
        QQ,
        [
            [1, 0, -QQ(1)/2, QQ(1)/4, -QQ(3)/8],
            [0, QQ(3)/2, 0, -QQ(3)/4, -QQ(3)/8],
            [0, 0, QQ(3)/2, -QQ(3)/4, -QQ(3)/8],
            [0, 0, 0, QQ(9)/4, -QQ(3)/8],
            [0, 0, 0, 0, 3],
        ],
    )

    expected_hybrid_in_monic = matrix(
        QQ,
        [
            [1, 0, -QQ(1)/2, QQ(1)/4, -QQ(3)/8],
            [0, 1, 0, -QQ(1)/2, -QQ(1)/4],
            [0, 0, 1, -QQ(1)/2, -QQ(1)/4],
            [0, 0, 0, 1, -QQ(1)/6],
            [0, 0, 0, 0, 1],
        ],
    )

    assert hybrid_comparison["hybrid_in_AH"] == expected_hybrid_in_AH
    assert hybrid_comparison["hybrid_in_monic"] == expected_hybrid_in_monic

    if verbose:
        print("Tableau order:")
        print_tableau_order(result)
        print("\nArmon--Halverson H:")
        print(result["H"])
        print("\nD:")
        print(result["D"])
        print("\nU = D^(-1)H:")
        print(result["U"])
        print("\nDesired A = U^(-1):")
        print(result["A"])
        print("\nExact (3,2) self-test passed.")

    return result


if __name__ == "__main__":
    self_test_32(verbose=True)
