# YoungSeminormalToKLBaseChange.sage
#
# Exact base-change matrices among three bases of the Specht module S^lambda:
#
#     N  = standard polytabloid / Young natural basis,
#     Y  = monic Young seminormal basis,
#     KL = Kazhdan--Lusztig left-cell basis.
#
# Column convention:
#
#     Y  = N A_Young,
#     KL = N A_KL.
#
# Consequently
#
#     Y = KL (A_KL^(-1) A_Young),
#
# so A_KL^(-1) A_Young is the requested base-change matrix from the
# Young seminormal basis to the KL basis.
#
# Put this file in the same directory as
#
#     YoungNatural.sage
#     YoungNaturalToMonicSeminormal_HYBRID_COMPARE_v4.sage
#     wgraph.c
#
# Then run, for example,
#
#     load("YoungSeminormalToKLBaseChange.sage")
#     result = print_young_to_KL_base_change([3, 2])
#
# For another source location, pass c_source_filename explicitly:
#
#     result = print_young_to_KL_base_change(
#         [4, 3],
#         c_source_filename="/full/path/to/wgraph.c",
#     )

from sage.all import *

import os
import math
import struct
import subprocess
import tempfile

import numpy as np


YOUNG_TO_KL_FILE_VERSION = "2026-08-21-exact-last-letter-v2-distance"


# Load the two earlier Young-basis files when their functions have not already
# been loaded in the current Sage session.  These are deliberately relative
# filenames, so the project directory remains portable.
if "GetSpechtOperators" not in globals():
    load("YoungNatural.sage")

if "young_natural_to_monic_seminormal" not in globals():
    load("YoungNaturalToMonicSeminormal_HYBRID_COMPARE_v4.sage")


def check_loaded_young_to_KL_file():
    """Print identifying information for this file and its dependencies."""
    print("Young-to-KL file version:", YOUNG_TO_KL_FILE_VERSION)
    print(
        "GetSpechtOperators is defined:",
        callable(globals().get("GetSpechtOperators")),
    )
    print(
        "young_natural_to_monic_seminormal is defined:",
        callable(globals().get("young_natural_to_monic_seminormal")),
    )


def upper_unitriangular_distance(A):
    r"""Return the Euclidean norm of the strict-upper entries of ``A``.

    For an upper-unitriangular matrix this is exactly ``||A-I||_F``.  This is
    the same distance convention used in
    ``KLLeftCellOptimizer_EvacAdapted_directA_switch_v6.sage``; it is not the
    spectral/operator matrix 2-norm.
    """
    if A is None:
        return None
    if A.nrows() != A.ncols():
        raise ValueError("upper_unitriangular_distance requires a square matrix.")

    squared_entries = (
        float(A[row, column])**2
        for row in range(A.nrows())
        for column in range(row + 1, A.ncols())
    )
    return math.sqrt(math.fsum(squared_entries))


def _strict_upper_squared_distance_exact(A):
    """Return ``sum_{i<j} A[i,j]^2`` exactly over the matrix base ring."""
    if A.nrows() != A.ncols():
        raise ValueError("The strict-upper distance requires a square matrix.")
    return sum(
        (A[row, column]**2
         for row in range(A.nrows())
         for column in range(row + 1, A.ncols())),
        A.base_ring()(0),
    )


def _normalise_partition_young_KL(partition):
    lam = tuple(Integer(x) for x in partition)
    if not lam:
        raise ValueError("The partition must be nonempty.")
    if any(x <= 0 for x in lam):
        raise ValueError("All parts of the partition must be positive.")
    if any(lam[i] < lam[i + 1] for i in range(len(lam) - 1)):
        raise ValueError("The parts must be weakly decreasing.")
    return lam


def _tuple_tableau_young_KL(T):
    """Convert a Sage/list tableau to a hashable tuple of row-tuples."""
    try:
        rows = T.to_list()
    except AttributeError:
        rows = T
    return tuple(tuple(Integer(x) for x in row) for row in rows)


def _last_letter_key_young_KL(T):
    """Last-letter / total-index order used by the preceding project files."""
    T = _tuple_tableau_young_KL(T)
    positions = {}
    for row, entries in enumerate(T):
        for column, value in enumerate(entries):
            positions[Integer(value)] = (row, column)
    n = sum(len(row) for row in T)
    return tuple(positions[k][0] for k in range(n, 0, -1))


def _standard_tableaux_last_letter_young_KL(partition):
    tableaux = [
        _tuple_tableau_young_KL(T)
        for T in StandardTableaux(partition)
    ]
    return sorted(tableaux, key=_last_letter_key_young_KL)


def _polytabloid_operators_last_letter(partition, expected_tableaux):
    r"""Return the exact (1+s_i) polytabloid operators in last-letter order.

    ``GetSpechtOperators`` builds its matrices in Sage's
    ``StandardTableaux(partition).list()`` order.  This function explicitly
    conjugates those matrices into the last-letter order used by both the
    Armon--Halverson file and the KL C generator.
    """
    raw_tableaux = [
        _tuple_tableau_young_KL(T)
        for T in StandardTableaux(partition).list()
    ]
    last_tableaux = [
        _tuple_tableau_young_KL(T)
        for T in expected_tableaux
    ]

    if set(raw_tableaux) != set(last_tableaux):
        raise RuntimeError(
            "The polytabloid and Armon--Halverson tableau sets disagree."
        )

    raw_index = {T: i for i, T in enumerate(raw_tableaux)}
    permutation = [raw_index[T] for T in last_tableaux]

    raw_operators = [Matrix(QQ, M) for M in GetSpechtOperators(partition)]
    operators = [
        M.matrix_from_rows_and_columns(permutation, permutation)
        for M in raw_operators
    ]

    return operators, permutation, raw_tableaux


def _KL_left_cell_operators_from_C(
    partition,
    c_source_filename="wgraph.c",
    verbose=False,
):
    r"""Compute exact KL left-cell (1+s_i) matrices using ``wgraph.c``.

    This is the Nguyen/W-graph routine used by the earlier optimiser.  A
    temporary copy of the C source is edited and compiled, so the user's
    original ``wgraph.c`` is not modified and no ``operators.bin`` is left in
    the project directory.

    The C generator enumerates tableaux in the last-letter/total-index order
    used by the earlier strict-order optimiser files.
    """
    lam = _normalise_partition_young_KL(partition)
    c_source_filename = os.path.abspath(str(c_source_filename))
    if not os.path.isfile(c_source_filename):
        raise OSError(
            "Could not find the KL C source file %r. "
            "Pass c_source_filename with its full path."
            % c_source_filename
        )

    with open(c_source_filename, "r") as source_file:
        source_lines = source_file.readlines()

    part_string = "{" + ",".join(str(x) for x in lam) + "}"
    new_lines = []
    found_partition_length = False
    found_partition = False

    for line in source_lines:
        if "#define PARTITION_LEN" in line:
            new_lines.append(
                "#define PARTITION_LEN %s\n" % len(lam)
            )
            found_partition_length = True
        elif "const int PARTITION" in line:
            new_lines.append(
                "const int PARTITION[%s] = %s;\n"
                % (len(lam), part_string)
            )
            found_partition = True
        else:
            new_lines.append(line)

    if not found_partition_length or not found_partition:
        raise RuntimeError(
            "The expected PARTITION_LEN and PARTITION declarations were not "
            "found in %s." % c_source_filename
        )

    with tempfile.TemporaryDirectory(prefix="young_KL_") as temporary_directory:
        generated_source = os.path.join(temporary_directory, "wgraph_generated.c")
        executable = os.path.join(temporary_directory, "wgraph_generator")
        binary_output = os.path.join(temporary_directory, "operators.bin")

        with open(generated_source, "w") as generated_file:
            generated_file.writelines(new_lines)

        compile_process = subprocess.run(
            ["gcc", "-O3", generated_source, "-o", executable],
            cwd=temporary_directory,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            text=True,
        )
        if compile_process.returncode != 0:
            message = compile_process.stderr if not verbose else "see output above"
            raise RuntimeError("Compilation of wgraph.c failed: %s" % message)

        run_process = subprocess.run(
            [executable],
            cwd=temporary_directory,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            text=True,
        )
        if run_process.returncode != 0:
            message = run_process.stderr if not verbose else "see output above"
            raise RuntimeError("The KL C generator failed: %s" % message)

        if not os.path.isfile(binary_output):
            raise RuntimeError("The KL C generator did not create operators.bin.")

        matrices = []
        with open(binary_output, "rb") as binary_file:
            header = binary_file.read(8)
            if len(header) != 8:
                raise RuntimeError("operators.bin has an incomplete header.")
            number_of_generators, dimension = struct.unpack("ii", header)

            for generator in range(number_of_generators):
                nnz_bytes = binary_file.read(4)
                if len(nnz_bytes) != 4:
                    raise RuntimeError(
                        "operators.bin ended before generator %s." % generator
                    )
                number_of_nonzero_entries = struct.unpack("i", nnz_bytes)[0]
                chunk_size = number_of_nonzero_entries * 3 * 4
                raw_data = binary_file.read(chunk_size)
                if len(raw_data) != chunk_size:
                    raise RuntimeError(
                        "operators.bin has incomplete data for generator %s."
                        % generator
                    )

                if number_of_nonzero_entries:
                    array = np.frombuffer(raw_data, dtype=np.int32).reshape(
                        (number_of_nonzero_entries, 3)
                    )
                    data = {
                        (int(row), int(column)): int(value)
                        for row, column, value in array
                    }
                    matrix_KL = Matrix(
                        ZZ,
                        dimension,
                        dimension,
                        data,
                        sparse=True,
                    )
                else:
                    matrix_KL = Matrix(
                        ZZ,
                        dimension,
                        dimension,
                        sparse=True,
                    )
                matrices.append(matrix_KL)

    return matrices


def _recover_upper_unitriangular_transition_young_KL(
    operators_old,
    operators_new,
    verbose=True,
):
    r"""Solve S_i A = A K_i exactly with A upper unitriangular over QQ."""
    operators_old = [Matrix(QQ, M) for M in operators_old]
    operators_new = [Matrix(QQ, M) for M in operators_new]

    if len(operators_old) != len(operators_new):
        raise ValueError("The two operator lists have different lengths.")
    if not operators_old:
        raise ValueError("At least one generator operator is required.")

    dimension = operators_old[0].nrows()
    all_operators = operators_old + operators_new
    if any(
        M.nrows() != dimension or M.ncols() != dimension
        for M in all_operators
    ):
        raise ValueError("All operators must be square matrices of equal size.")

    pairs = [
        (row, column)
        for row in range(dimension)
        for column in range(row + 1, dimension)
    ]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    sparse_entries = {}
    right_hand_side = []
    equation_number = 0

    # If A=I+sum x_ij E_ij, then the (r,c) coefficient of x_ij in
    # S*A-A*K is S[r,i] delta_(j,c) - delta_(r,i) K[j,c].
    for S, K in zip(operators_old, operators_new):
        constant_part = S - K
        for row in range(dimension):
            for column in range(dimension):
                equation = {}

                for i in range(column):
                    coefficient = S[row, i]
                    if coefficient != 0:
                        variable = pair_index[(i, column)]
                        equation[variable] = (
                            equation.get(variable, QQ(0)) + coefficient
                        )

                for j in range(row + 1, dimension):
                    coefficient = -K[j, column]
                    if coefficient != 0:
                        variable = pair_index[(row, j)]
                        equation[variable] = (
                            equation.get(variable, QQ(0)) + coefficient
                        )

                equation = {
                    variable: coefficient
                    for variable, coefficient in equation.items()
                    if coefficient != 0
                }
                value = -constant_part[row, column]

                if equation or value != 0:
                    for variable, coefficient in equation.items():
                        sparse_entries[(equation_number, variable)] = coefficient
                    right_hand_side.append(value)
                    equation_number += 1

    if pairs:
        coefficient_matrix = Matrix(
            QQ,
            equation_number,
            len(pairs),
            sparse_entries,
            sparse=True,
        )
        rhs_vector = vector(QQ, right_hand_side)
        try:
            solution = coefficient_matrix.solve_right(rhs_vector)
        except ValueError as error:
            raise RuntimeError(
                "No upper-unitriangular transition satisfies "
                "(1+s_i)_N A = A (1+s_i)_KL.  This normally indicates a "
                "tableau-order mismatch."
            ) from error
        kernel_dimension = len(pairs) - coefficient_matrix.rank()
    else:
        solution = vector(QQ, [])
        kernel_dimension = 0

    if kernel_dimension != 0:
        raise RuntimeError(
            "The normalized intertwining equations did not determine A_KL "
            "uniquely; solution-family dimension = %s." % kernel_dimension
        )

    transition = identity_matrix(QQ, dimension)
    for value, (row, column) in zip(solution, pairs):
        transition[row, column] = value

    residual = max(
        [
            abs(entry)
            for S, K in zip(operators_old, operators_new)
            for entry in (S * transition - transition * K).list()
        ]
        or [QQ(0)]
    )
    if residual != 0:
        raise ArithmeticError(
            "The recovered A_KL has nonzero exact residual %s." % residual
        )

    if verbose:
        print("Recovered A_KL exactly in polytabloid coordinates.")
        print("strict-upper variables =", len(pairs))
        print("linear equations retained =", equation_number)
        print("solution-family dimension =", kernel_dimension)
        print("maximum exact intertwining residual =", residual)

    return transition


def young_and_KL_base_change(
    partition,
    c_source_filename="wgraph.c",
    verbose=True,
):
    r"""Compute A_Young, A_KL, and A_KL^(-1) A_Young exactly.

    All matrices and tableau lists are in last-letter/total-index order.

    Returns a dictionary containing

        A_Young:
            Y = N A_Young, obtained from Armon--Halverson.

        A_KL:
            KL = N A_KL, recovered from exact generator intertwining.

        A_Young_to_KL:
            A_KL^(-1) A_Young.  Its columns express the Young seminormal
            basis vectors in the KL basis.  It also sends Young coordinate
            columns to KL coordinate columns.
    """
    lam = _normalise_partition_young_KL(partition)

    young_data = young_natural_to_monic_seminormal(
        list(lam),
        order="last_letter",
    )
    tableaux_young = [
        _tuple_tableau_young_KL(T)
        for T in young_data["tableaux"]
    ]
    tableaux_expected = _standard_tableaux_last_letter_young_KL(lam)

    if tableaux_young != tableaux_expected:
        raise RuntimeError(
            "The Armon--Halverson file did not return the expected "
            "last-letter tableau order."
        )

    A_Young = Matrix(QQ, young_data["A"])

    operators_polytabloid, permutation, tableaux_raw = (
        _polytabloid_operators_last_letter(lam, tableaux_young)
    )
    operators_KL = _KL_left_cell_operators_from_C(
        lam,
        c_source_filename=c_source_filename,
        verbose=verbose,
    )

    dimension = len(tableaux_young)
    if len(operators_KL) != sum(lam) - 1:
        raise RuntimeError(
            "The KL generator returned %s operators; expected %s."
            % (len(operators_KL), sum(lam) - 1)
        )
    if any(
        M.nrows() != dimension or M.ncols() != dimension
        for M in operators_KL
    ):
        raise RuntimeError(
            "The KL operator dimension does not equal the number of tableaux."
        )

    A_KL = _recover_upper_unitriangular_transition_young_KL(
        operators_polytabloid,
        operators_KL,
        verbose=verbose,
    )

    if A_Young.nrows() != A_KL.nrows():
        raise RuntimeError("A_Young and A_KL have different dimensions.")

    A_Young_to_KL = A_KL.inverse() * A_Young
    A_KL_to_Young = A_Young.inverse() * A_KL
    strict_upper_distance_squared = _strict_upper_squared_distance_exact(
        A_Young_to_KL
    )
    strict_upper_distance = upper_unitriangular_distance(A_Young_to_KL)

    # Exact convention checks.
    identity = identity_matrix(QQ, dimension)
    if A_KL * A_Young_to_KL != A_Young:
        raise ArithmeticError(
            "Failed the exact identity A_KL*A_Young_to_KL=A_Young."
        )
    if A_Young_to_KL * A_KL_to_Young != identity:
        raise ArithmeticError("The two basis-change matrices are not inverses.")

    return {
        "partition": lam,
        "tableaux": tableaux_young,
        "dimension": dimension,
        "A_Young": A_Young,
        "A_KL": A_KL,
        "A_Young_to_KL": A_Young_to_KL,
        "A_KL_inverse_A_Young": A_Young_to_KL,
        "A_KL_to_Young": A_KL_to_Young,
        "strict_upper_distance_Young_to_KL": strict_upper_distance,
        "strict_upper_distance_squared_Young_to_KL": (
            strict_upper_distance_squared
        ),
        "H_AH": Matrix(QQ, young_data["H"]),
        "D_AH": Matrix(QQ, young_data["D"]),
        "U_AH": Matrix(QQ, young_data["U"]),
        "operators_polytabloid": operators_polytabloid,
        "operators_KL": operators_KL,
        "sage_to_last_letter_permutation": permutation,
        "sage_tableaux": tableaux_raw,
    }


def print_young_to_KL_base_change(
    partition,
    c_source_filename="wgraph.c",
    verbose=True,
    print_tableaux=True,
    round_digits=3,
):
    r"""Compute and print the two basis matrices and the requested product."""
    result = young_and_KL_base_change(
        partition,
        c_source_filename=c_source_filename,
        verbose=verbose,
    )

    print("\n============================================================")
    print("partition =", result["partition"])
    print("dimension =", result["dimension"])
    print("order = last-letter / total-index")
    print("============================================================")

    if print_tableaux:
        print("\nTableau order:")
        for index, tableau in enumerate(result["tableaux"], start=1):
            print("%4s: %s" % (index, tableau))

    print("\nA_Young, defined by Young = Polytabloid * A_Young:")
    print(result["A_Young"])

    print("\nA_KL, defined by KL = Polytabloid * A_KL:")
    print(result["A_KL"])

    print("\nA_KL^(-1) * A_Young (Young -> KL):")
    print(result["A_Young_to_KL"])

    distance = result["strict_upper_distance_Young_to_KL"]
    distance_squared = result[
        "strict_upper_distance_squared_Young_to_KL"
    ]
    print(
        "\nstrict-upper distance of A_KL^(-1) A_Young "
        "= sqrt(sum_{i<j} |(A_KL^(-1) A_Young)_ij|^2)"
    )
    print("exact squared distance =", distance_squared)
    print("distance =", round(distance, round_digits))

    return result


# Example (uncomment to run directly after loading the file):
# result = print_young_to_KL_base_change([3, 2])
