# KLLeftCellOptimizer_EvacAdapted_directA_switch_v4.sage
#
# Load after:
#   load("/mnt/c/Users/Jacques/Desktop/KLLeftCellOptimizer.sage")
#   load("/mnt/c/Users/Jacques/Desktop/KLLeftCellOptimizer_EvacuationPatch_fixed.sage")
# Keep Nguyen's wgraph.c in the Sage working directory (wgraph(1).c is also
# detected automatically), or pass nguyen_wgraph_file="/full/path/wgraph.c".
#
# This version is strict about ordering:
#   - P_KL is built in last-letter/total-index order.
#   - Specht/seminormal operators and Gram matrix are reordered to the same
#     last-letter order before optimisation.
#   - A is upper unitriangular in last-letter order.
#   - W0_old is computed from the reordered operators, hence also last-letter.
#
# It optimises tr(A^t G A) subject to
#   A upper unitriangular in last-letter order,
#   W0_old*A = A*P_KL,
#   A^{-1}(1+s_i)A >= 0.

import logging
import math
import os
import gurobipy as gp
from gurobipy import GRB
from sage.all import *

logging.getLogger("gurobipy").setLevel(logging.WARNING)


OPTIMIZER_FILE_VERSION = "2026-08-03-direct-A-switch-ATGA-mipgap-nguyen-v15"


def check_loaded_optimizer_file():
    """Identify the optimiser implementation currently loaded in Sage."""
    print("Optimizer file version:", OPTIMIZER_FILE_VERSION)
    print("direct-A formulation: True")
    print(
        "distance/transition output available:",
        "upper_unitriangular_distance" in globals(),
    )


def _quiet_model(name, solver_verbose=False):
    if solver_verbose:
        return gp.Model(name)
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        return gp.Model(name, env=env)
    except Exception:
        model = gp.Model(name)
        model.Params.OutputFlag = 0
        return model


def _format_scalar_for_print(x, digits=3, zero_tol=None):
    """Format numeric values compactly for matrix output.

    Display only; returned matrices are unchanged.
    """
    if zero_tol is None:
        zero_tol = 10.0**(-digits) / 2.0
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf) < zero_tol:
        xf = 0.0
    nearest = round(xf)
    if abs(xf - nearest) < zero_tol:
        return str(int(nearest))
    return ("%%.%df" % digits) % xf


def print_matrix_rounded(M, digits=3, zero_tol=None):
    """Pretty-print a matrix to fixed decimal places.

    Use this for Gurobi/RDF output, where exact Sage printing is unreadable.
    """
    try:
        nr = M.nrows()
        nc = M.ncols()
    except Exception:
        print(M)
        return
    entries = [[_format_scalar_for_print(M[i, j], digits, zero_tol) for j in range(nc)] for i in range(nr)]
    width = max([len(e) for row in entries for e in row] + [1])
    for row in entries:
        print("[" + " ".join(e.rjust(width) for e in row) + "]")

def _tableau_from_basis_key(k):
    """Try to convert a Sage Specht basis key/tableau to tuple-of-tuples."""
    # The helper _tuple_tableau is supplied by KLLeftCellOptimizer_EvacuationPatch_fixed.sage.
    try:
        return _tuple_tableau(k)
    except Exception:
        pass
    try:
        return _tuple_tableau(k.to_tableau())
    except Exception:
        pass
    try:
        return _tuple_tableau(k.tableau())
    except Exception:
        pass
    try:
        return _tuple_tableau(k[0])
    except Exception:
        pass
    raise TypeError("Could not convert basis key %r to a tableau." % (k,))


def specht_basis_tableaux_in_tom_order(partition_rep):
    """
    Recover the tableau order used by Sage's Specht module basis.

    Tom's Specht Gram uses SM.basis().list(); the keys of this basis should be
    the corresponding tableaux in the same order.  We make this explicit so we
    can reorder everything to last-letter order.
    """
    rank_w = sum(partition_rep) - 1
    SGA = SymmetricGroupAlgebra(QQ, rank_w + 1)
    SM = SGA.specht_module(partition_rep)
    B = SM.basis()

    candidates = []
    for attr in ["keys", "indices"]:
        try:
            obj = getattr(B, attr)()
            candidates.append(list(obj))
        except Exception:
            pass
    try:
        candidates.append(list(SM.indices()))
    except Exception:
        pass

    last_tabs = standard_tableaux_last_letter(partition_rep)
    last_set = set(last_tabs)

    for cand in candidates:
        if len(cand) != len(last_tabs):
            continue
        try:
            tabs = [_tableau_from_basis_key(k) for k in cand]
        except Exception:
            continue
        if set(tabs) == last_set:
            return tabs

    raise RuntimeError(
        "Could not infer Tom/Sage Specht basis tableau order.\n"
        "Without this, we cannot safely impose last-letter upper-unitriangularity.\n"
        "Print SM.basis().keys() in Sage and adapt specht_basis_tableaux_in_tom_order."
    )


def permutation_to_last_letter_order(partition_rep):
    """
    Return perm such that if M is in Tom/Sage basis order, then
        M_last = M[perm, perm]
    is in last-letter order.
    """
    tom_tabs = specht_basis_tableaux_in_tom_order(partition_rep)
    last_tabs = standard_tableaux_last_letter(partition_rep)
    tom_index = {T: i for i, T in enumerate(tom_tabs)}
    perm = [tom_index[T] for T in last_tabs]
    return perm, tom_tabs, last_tabs


def reorder_matrix_to_last_letter(M, perm):
    M = Matrix(QQ, M)
    return M.matrix_from_rows_and_columns(perm, perm)


def print_last_letter_order_diagnostic(partition_rep):
    perm, tom_tabs, last_tabs = permutation_to_last_letter_order(partition_rep)
    print("partition =", partition_rep)
    print("Tom/Sage order:")
    for i, T in enumerate(tom_tabs):
        print("%2d:" % i, T, "last-key=", last_letter_key(T))
    print("\nlast-letter order:")
    for i, T in enumerate(last_tabs):
        print("%2d:" % i, T, "last-key=", last_letter_key(T))
    print("\nperm Tom/Sage -> last-letter =", perm)
    return perm, tom_tabs, last_tabs


def _get_starting_ops_and_gram_last_letter(partition_rep, start_seminormal):
    """
    Return (ops, G, basis_name, perm, tom_tabs, last_tabs), with ops and G in
    last-letter order.  This is the key correction.
    """
    perm, tom_tabs, last_tabs = permutation_to_last_letter_order(partition_rep)

    rank_w = sum(partition_rep) - 1
    SGA = SymmetricGroupAlgebra(QQ, rank_w + 1)
    SM = SGA.specht_module(partition_rep)

    if start_seminormal:
        raw_ops = [Matrix(QQ, M) for M in GetSeminormalFormOperators(partition_rep)]
        raw_G = Matrix(QQ, SeminormalGram(partition_rep))
        basis_name = "seminormal, reordered to last-letter"
    else:
        raw_ops = [Matrix(QQ, M) for M in GetSpechtOperators(partition_rep)]
        raw_basis = SM.basis().list()
        d = raw_ops[0].nrows()
        raw_G = Matrix(QQ, d, d, lambda i, j: SM.bilinear_form(raw_basis[i], raw_basis[j]))
        basis_name = "Specht/polytabloid, reordered to last-letter"

    ops = [reorder_matrix_to_last_letter(M, perm) for M in raw_ops]
    G = reorder_matrix_to_last_letter(raw_G, perm)
    return ops, G, basis_name, perm, tom_tabs, last_tabs


def _reduced_words_for_w0(rank_w):
    word_a = []
    for i in range(1, rank_w + 1):
        for j in range(i, 0, -1):
            word_a.append(j)
    word_b = []
    for i in range(rank_w, 0, -1):
        for j in range(1, i + 1):
            word_b.append(j)
    words = []
    for w in [word_a, list(reversed(word_a)), word_b, list(reversed(word_b))]:
        if w not in words:
            words.append(w)
    return words


def long_element_from_s_plus_1_last_letter(act_s_plus_1, verbose=False):
    """Compute W0_old from reordered last-letter matrices 1+s_i."""
    rank_w = len(act_s_plus_1)
    d = act_s_plus_1[0].nrows()
    I = identity_matrix(QQ, d)
    S = [Matrix(QQ, M) - I for M in act_s_plus_1]
    T = [Matrix(QQ, M) for M in act_s_plus_1]

    def ok(W):
        return W*W == I and all(W*T[k] == T[rank_w - 1 - k]*W for k in range(rank_w))

    candidates = []
    for word in _reduced_words_for_w0(rank_w):
        M = I
        for a in word:
            M = S[a-1] * M
        candidates.append(("left", word, M))
        M = I
        for a in word:
            M = M * S[a-1]
        candidates.append(("right", word, M))

    for side, word, W in candidates:
        if ok(W):
            if verbose:
                print("W0_old found by", side, "multiplication with word", word)
            return W

    raise RuntimeError(
        "Could not construct W0_old satisfying W0(1+s_i)=(1+s_{n-i})W0.\n"
        "This usually means the generator convention/order is not what we think."
    )


def build_P_KL_last_letter_strict(partition_rep, P_matrix=None, verbose=False):
    """
    Build P_KL in last-letter order.  Do not silently fall back to an unsigned P
    if signs/order cannot be checked from KL operators.
    """
    if P_matrix is not None:
        return Matrix(QQ, P_matrix)

    kl_ops = KLLeftCellOperators(partition_rep)
    P, tabs, perm, qval = signed_evacuation_matrix_last_letter(
        partition_rep,
        act_s_plus_1=kl_ops,
        use_operator_relations=True,
        verbose=verbose
    )
    return Matrix(QQ, P)


def upper_unitriangular_intertwiner_family_last_letter(W_old, P_target, verbose=False):
    """
    Solve A=I+strict-upper and W_old*A=A*P_target in last-letter order.
    Returns A0, C_basis with all solutions A=A0+sum x_t C_t.
    """
    W = Matrix(QQ, W_old)
    P = Matrix(QQ, P_target)
    d = W.nrows()
    if P.nrows() != d or P.ncols() != d:
        raise ValueError("P_target has incompatible size.")

    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    rows = []
    rhs = []

    # Equation is W*A = A*P in Tom's original convention.
    # Tom's constraints are S_k*A = A*B_k, so A is the usual column
    # change-of-basis matrix and the target w0 matrix is A^{-1} W A = P.
    # A = I + N, N strictly upper triangular.
    for r in range(d):
        for c in range(d):
            const = W[r, c] - P[r, c]
            coeffs = []
            for (i, j) in pairs:
                # (W E_ij)_{r,c} - (E_ij P)_{r,c}
                coeffs.append(W[r, i]*(1 if j == c else 0) - (1 if r == i else 0)*P[j, c])
            if const != 0 or any(u != 0 for u in coeffs):
                rows.append(coeffs)
                rhs.append(-const)

    A0 = identity_matrix(QQ, d)
    if len(rows) == 0:
        K = Matrix(QQ, 0, len(pairs)).right_kernel()
    else:
        M = Matrix(QQ, rows)
        b = vector(QQ, rhs)
        if M.rank() != M.augment(b).rank():
            raise RuntimeError(
                "No upper-unitriangular A in last-letter order solves W0_old*A=A*P_KL.\n"
                "This means either the KL signs/order are wrong, or such an A does not exist."
            )
        x0 = M.solve_right(b)
        K = M.right_kernel()
        for coeff, (i, j) in zip(x0, pairs):
            A0[i, j] = coeff

    C_basis = []
    for v in K.basis():
        C = zero_matrix(QQ, d, d)
        for coeff, (i, j) in zip(v, pairs):
            C[i, j] = coeff
        C_basis.append(C)

    if verbose:
        print("original strict-upper A variables =", len(pairs))
        print("A variables after W0_old*A=A*P_KL =", len(C_basis))
        print("particular A0 =")
        print(A0)
        print("check W0_old*A0=A0*P_KL:", W*A0 == A0*P)
    return A0, C_basis


def symbolic_intertwiner_matrix_last_letter(A0, C_basis):
    d = A0.nrows()
    m = len(C_basis)
    if m == 0:
        return Matrix(QQ, A0)
    R = PolynomialRing(QQ, ["a%d" % t for t in range(m)])
    gens = R.gens()
    A = Matrix(R, d, d)
    for i in range(d):
        for j in range(d):
            A[i, j] = R(A0[i, j])
            for t in range(m):
                if C_basis[t][i, j] != 0:
                    A[i, j] += R(C_basis[t][i, j]) * gens[t]
    return A


def choose_global_sign_for_upper_intertwiner_last_letter(W_old, P_target):
    """Choose between P and -P so that an upper-unitriangular intertwiner exists.

    Stembridge q=-1 only fixes the number of fixed columns.  When there are no
    fixed columns, as for partition [2,1], the global sign of the signed
    evacuation matrix is not fixed by the trace.  The convention compatible
    with Tom's upper-unitriangular change-of-basis is the one for which
        W0_old * A = A * P
    has an upper-unitriangular solution.

    Returns (P_chosen, A0, C_basis, sign), where sign is +1 or -1.
    """
    P = Matrix(QQ, P_target)
    errors = []
    for sgn in [Integer(1), Integer(-1)]:
        try:
            A0, C_basis = upper_unitriangular_intertwiner_family_last_letter(W_old, sgn * P, verbose=False)
            return sgn * P, A0, C_basis, sgn
        except RuntimeError as e:
            errors.append((sgn, str(e)))
    raise RuntimeError(
        "Neither P nor -P admits an upper-unitriangular solution of A*W0_old=P*A.\n"
        "This means there is still an order/sign convention mismatch beyond the global sign.\n"
        "Errors: %s" % errors
    )


def solve_evac_adapted_A_last_letter(partition_rep,
                                      minimal=False,
                                      P_matrix=None,
                                      start_seminormal=False,
                                      impose_intertwiner=True,
                                      continuous=True,
                                      x_bound=20.0,
                                      p_bound=None,
                                      entry_bound=None,
                                      mip_gap=0.05,
                                      solver_verbose=False,
                                      diagnostics=False):
    """Robust direct-A formulation.

    IMPORTANT: older versions bounded coordinates in an arbitrary rational
    nullspace basis C_basis.  In higher dimension those coordinates can be
    huge/badly scaled even when the entries of A are small.  This version uses
    the actual strict-upper entries A[i,j] as variables, imposes W*A=A*P as
    linear constraints, and therefore bounds the mathematically meaningful
    quantities.
    """
    ops, G, basis_name, perm, tom_tabs, last_tabs = _get_starting_ops_and_gram_last_letter(
        partition_rep, start_seminormal)
    rank_w = len(ops)
    d = ops[0].nrows()
    # Long-element data are only constructed when the optional intertwiner
    # constraint is requested.  With it OFF, A ranges over all upper
    # unitriangular matrices subject only to transformed-operator nonnegativity.
    if impose_intertwiner:
        P_raw = build_P_KL_last_letter_strict(partition_rep, P_matrix=P_matrix, verbose=False)
        W_old = long_element_from_s_plus_1_last_letter(ops, verbose=False)
        P, A0, C_basis, P_global_sign = choose_global_sign_for_upper_intertwiner_last_letter(W_old, P_raw)
    else:
        P = None
        W_old = None
        A0 = identity_matrix(QQ, d)
        C_basis = []
        P_global_sign = None

    model = _quiet_model("evacuation_adapted_last_letter_optimizer_direct_A",
                         solver_verbose=solver_verbose)

    # Backwards compatibility: x_bound used to bound nullspace coordinates.
    # It now bounds actual strict-upper entries unless entry_bound is supplied.
    A_bound = entry_bound if entry_bound is not None else x_bound
    if A_bound is None:
        alb, aub = -GRB.INFINITY, GRB.INFINITY
    else:
        alb, aub = -float(A_bound), float(A_bound)

    avtype = GRB.CONTINUOUS if continuous else GRB.INTEGER
    a_pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    a_vars = model.addVars(a_pairs, vtype=avtype, lb=alb, ub=aub, name="A")

    def A_expr(i, j):
        if i == j:
            return 1.0
        if i > j:
            return 0.0
        return a_vars[i, j]

    # Optional exact structural constraint
    #     [w0]_old A = A [w0]_KL.
    # Here [w0]_old is the long element in the starting polytabloid/seminormal
    # basis and [w0]_KL is the signed evacuation matrix in the KL basis.
    if impose_intertwiner:
        for r in range(d):
            for c in range(d):
                lhs = gp.quicksum(float(W_old[r, l]) * A_expr(l, c)
                                  for l in range(d) if W_old[r, l] != 0)
                rhs = gp.quicksum(A_expr(r, l) * float(P[l, c])
                                  for l in range(d) if P[l, c] != 0)
                model.addConstr(lhs == rhs, name="WAP_%d_%d" % (r, c))

    p_index = [(i, j, k) for i in range(d) for j in range(d) for k in range(rank_w)]
    if p_bound is None:
        B_vars = model.addVars(p_index, vtype=GRB.CONTINUOUS, lb=0.0, name="B")
    else:
        B_vars = model.addVars(p_index, vtype=GRB.CONTINUOUS, lb=0.0,
                               ub=float(p_bound), name="B")

    # S_k*A = A*B_k, B_k >= 0.
    for k in range(rank_w):
        Sk = ops[k]
        for i in range(d):
            for j in range(d):
                lhs = gp.quicksum(float(Sk[i, l]) * A_expr(l, j)
                                  for l in range(d) if Sk[i, l] != 0)
                rhs = gp.quicksum(A_expr(i, l) * B_vars[l, j, k]
                                  for l in range(d))
                model.addQConstr(lhs == rhs, name="SAB_%d_%d_%d" % (k, i, j))

    obj = gp.quicksum(float(G[i, j]) * A_expr(i, l) * A_expr(j, l)
                      for l in range(d) for i in range(d) for j in range(d)
                      if G[i, j] != 0)
    model.setObjective(obj, GRB.MINIMIZE if minimal else GRB.MAXIMIZE)

    # A0 is a genuine W0-intertwiner and is a much better start than arbitrary
    # zero kernel coordinates.  B starts are supplied when A0 is invertible.
    for i, j in a_pairs:
        a_vars[i, j].Start = float(A0[i, j])
    try:
        A0R = Matrix(RDF, A0)
        if abs(float(A0R.det())) > 1e-12:
            for k, Sk in enumerate(ops):
                B0 = A0R.inverse() * Matrix(RDF, Sk) * A0R
                for i in range(d):
                    for j in range(d):
                        B_vars[i, j, k].Start = max(0.0, float(B0[i, j]))
    except Exception:
        pass

    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1
    model.Params.NumericFocus = 3
    model.Params.ScaleFlag = 2
    model.Params.FeasibilityTol = 1e-7
    model.Params.OptimalityTol = 1e-7
    model.Params.MIPGap = float(mip_gap)
    model.Params.Heuristics = 0.2
    if not solver_verbose:
        model.Params.OutputFlag = 0

    # Graceful Ctrl+C handling:
    # one Ctrl+C asks Gurobi to stop and keeps the best incumbent found so far.
    interrupted_by_keyboard = False
    try:
        model.optimize()
    except KeyboardInterrupt:
        interrupted_by_keyboard = True
        try:
            model.terminate()
        except Exception:
            pass
        print("\nCtrl+C received: keeping the best incumbent found so far.")

    status_map = {
        GRB.OPTIMAL: "OPTIMAL", GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INTERRUPTED: "INTERRUPTED",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED",
    }
    interrupted = interrupted_by_keyboard or (model.Status == GRB.INTERRUPTED)
    status = "INTERRUPTED" if interrupted else status_map.get(model.Status, str(model.Status))
    meta = {
        "status": status, "gurobi_status": model.Status,
        "basis_name": basis_name, "dimension": d,
        "original_upper_variables": d*(d-1)//2,
        "variables_after_w0": (len(C_basis) if impose_intertwiner else d*(d-1)//2),
        "impose_intertwiner": impose_intertwiner,
        "P": P, "P_global_sign": P_global_sign, "W_old": W_old,
        "A0": A0, "C_basis": C_basis, "perm": perm,
        "tom_tabs": tom_tabs, "last_tabs": last_tabs,
        "last_letter_order": True, "direct_A_formulation": True,
        "A_entry_bound": A_bound,
        "interrupted": interrupted,
    }

    if model.Status == GRB.INFEASIBLE:
        if diagnostics:
            print("Model is INFEASIBLE; writing IIS to model_iis.ilp")
            model.computeIIS()
            model.write("model_iis.ilp")
        return None, None, meta
    if model.SolCount == 0:
        return None, None, meta

    def clean_float(v):
        vf = float(v)
        if abs(vf) < 1e-10: return 0.0
        if abs(vf-round(vf)) < 1e-8: return float(round(vf))
        return round(vf, 12)

    A = identity_matrix(RDF, d)
    for i, j in a_pairs:
        A[i, j] = clean_float(a_vars[i, j].X)

    ops_R = [Matrix(RDF, S) for S in ops]
    B_ops = [A.inverse() * S * A for S in ops_R]
    min_B = min(float(z) for B in B_ops for z in B.list()) if B_ops else None
    if impose_intertwiner:
        P_R, W_R = Matrix(RDF, P), Matrix(RDF, W_old)
        wab_res = max(abs(float(z)) for z in (W_R*A-A*P_R).list())
    else:
        wab_res = None
    sab_res = max(abs(float(z)) for k, S in enumerate(ops_R)
                  for z in (S*A-A*B_ops[k]).list())
    # Save incumbent-quality information.  For an interrupted run,
    # objective is the current best feasible value, not a certified optimum.
    try:
        best_bound = float(model.ObjBound)
    except Exception:
        best_bound = None
    try:
        current_gap = float(model.MIPGap)
    except Exception:
        current_gap = None

    meta.update({
        "objective": model.ObjVal,
        "best_bound": best_bound,
        "current_mip_gap": current_gap,
        "w0_adapted_check": (wab_res < 1e-6) if impose_intertwiner else None,
        "upper_unitriangular_check": True,
        "min_B_entry": min_B,
        "WAP_residual": wab_res,
        "SAB_residual": sab_res,
        "x_values": [clean_float(a_vars[i,j].X) for i,j in a_pairs],
    })
    return A, B_ops, meta



def _matrices_close_for_print(A, B, tol=1e-6):
    """Numerically test whether two Sage matrices agree entrywise up to tol."""
    if A is None or B is None:
        return False
    if A.nrows() != B.nrows() or A.ncols() != B.ncols():
        return False
    max_diff = max([abs(float(A[i, j]) - float(B[i, j]))
                    for i in range(A.nrows()) for j in range(A.ncols())])
    return max_diff <= tol


def _matrix_max_abs_difference(A, B):
    """Return max |A_ij-B_ij| as a float, or None if dimensions do not match."""
    if A is None or B is None:
        return None
    if A.nrows() != B.nrows() or A.ncols() != B.ncols():
        return None
    return max([abs(float(A[i, j]) - float(B[i, j]))
                for i in range(A.nrows()) for j in range(A.ncols())])


def upper_unitriangular_distance(A):
    r"""Return the Euclidean norm of the strict-upper entries of ``A``.

    For an upper-unitriangular matrix this equals ``||A-I||_F``.  It is not
    the spectral/operator matrix 2-norm.
    """
    if A is None:
        return None
    if A.nrows() != A.ncols():
        raise ValueError("upper_unitriangular_distance requires a square matrix.")

    squared_entries = (
        float(A[i, j])**2
        for i in range(A.nrows())
        for j in range(i + 1, A.ncols())
    )
    return math.sqrt(math.fsum(squared_entries))


def _print_upper_unitriangular_distance(label, A, round_digits=3):
    """Print the requested strict-upper Euclidean distance."""
    distance = upper_unitriangular_distance(A)
    if distance is None:
        print("strict-upper distance of %s = unavailable" % label)
        return
    print(
        "strict-upper distance of %s = "
        "sqrt(sum_{i<j} |%s_ij|^2) = %s"
        % (label, label, _format_scalar_for_print(distance, round_digits))
    )


def _print_A_distance_summary(A_min, A_max, round_digits=3):
    """Print the strict-upper Euclidean distances of both optima."""
    print("\n--- STRICT-UPPER EUCLIDEAN DISTANCES ---")
    _print_upper_unitriangular_distance("A_min", A_min, round_digits)
    _print_upper_unitriangular_distance("A_max", A_max, round_digits)


def _print_Amax_inverse_Amin_block(
    A_max,
    A_min,
    round_digits=3,
):
    """Print ``A_max^(-1) A_min`` and its strict-upper distance."""
    print("\n--- TRANSITION MATRIX A_max^-1 A_min ---")
    if A_max is None or A_min is None:
        print("A_max^-1 A_min is unavailable because an optimum was not found.")
        return None

    transition = A_max^(-1) * A_min
    print("A_max^-1 A_min =")
    print_matrix_rounded(transition, round_digits)
    _print_upper_unitriangular_distance(
        "A_max^-1 A_min",
        transition,
        round_digits,
    )

    return transition


def _print_transformed_gram(label, A, G, round_digits=3):
    r"""Print ``A^T G A`` for one optimum."""
    if A is None:
        print("%s^T G %s is unavailable because an optimum was not found." % (label, label))
        return None

    A_R = Matrix(RDF, A)
    G_R = Matrix(RDF, G)
    transformed_gram = A_R.transpose() * G_R * A_R
    print("%s^T G %s =" % (label, label))
    print_matrix_rounded(transformed_gram, round_digits)
    return transformed_gram


def _print_solution_block(
    label,
    A,
    B_ops,
    meta,
    G=None,
    round_digits=3,
    print_B=True,
    print_ATGA=False,
):
    """Print one solved optimiser block in the same style as before."""
    print("status =", meta["status"])
    if A is None:
        print("No solution found.")
        print(label, "=")
        print(None)
        return

    print("objective =", _format_scalar_for_print(meta["objective"], round_digits))
    if meta.get("interrupted", False):
        print("NOTE: this is the best feasible incumbent found before interruption; optimality is NOT proved.")
        if meta.get("best_bound") is not None:
            print("best bound =", _format_scalar_for_print(meta["best_bound"], round_digits))
        if meta.get("current_mip_gap") is not None:
            print("current relative MIP gap =", _format_scalar_for_print(meta["current_mip_gap"], round_digits))
    print("upper unitriangular in last-letter order:", meta["upper_unitriangular_check"])
    if meta.get("impose_intertwiner", True):
        print("W0_old*%s=%s*P_KL:" % (label, label), meta["w0_adapted_check"])
    else:
        print("long-element intertwiner constraint: OFF")
    print("min entry among all B_%s_i:" % label, _format_scalar_for_print(meta["min_B_entry"], round_digits))
    print(label, "=")
    print_matrix_rounded(A, round_digits)

    if print_ATGA:
        print("")
        _print_transformed_gram(label, A, G, round_digits)

    if print_B:
        for i, B in enumerate(B_ops, start=1):
            print("\nB_%s_%d = %s^-1 A_%d %s =" % (label, i, label, i, label))
            print_matrix_rounded(B, round_digits)


def solve_min_max_evac_adapted_last_letter(partition_rep,
                                            P_matrix=None,
                                            start_seminormal=False,
                                            impose_intertwiner=True,
                                            continuous=True,
                                            x_bound=20.0,
                                            p_bound=None,
                                            entry_bound=None,
                                            mip_gap=0.05,
                                            solver_verbose=False,
                                            print_W0=True,
                                            print_P=True,
                                            print_family=True,
                                            print_B=True,
                                            print_ATGA=False,
                                            print_order=False,
                                            diagnostics=False,
                                            round_digits=3,
                                            equality_tol=1e-6,
                                            stop_if_min_equals_max=True,
                                            nguyen_wgraph_file=None,
                                            nguyen_compare_tol=1e-6,
                                            nguyen_verbose=False):
    r"""Solve and print both optima using the direct-``A`` formulation.

    It always prints the requested Gurobi ``MIPGap`` before doing any work and
    always checks ``A_max`` against Nguyen's W-graph operators.  Set
    ``print_ATGA=True`` to print ``A_min^T G A_min`` and ``A_max^T G A_max``,
    where ``G`` is the Gram matrix used by the objective (the polytabloid Gram
    matrix when ``start_seminormal=False``).

    It always prints the strict-upper Euclidean distances of ``A_min`` and
    ``A_max``.  If the matrices differ by more than ``equality_tol``, it also
    prints ``A_max^(-1) A_min`` and its strict-upper distance.
    """
    print("requested Gurobi MIPGap =", mip_gap)
    ops, G, basis_name, perm, tom_tabs, last_tabs = _get_starting_ops_and_gram_last_letter(partition_rep, start_seminormal)
    d = ops[0].nrows()
    if impose_intertwiner:
        P_raw = build_P_KL_last_letter_strict(partition_rep, P_matrix=P_matrix, verbose=True)
        W_old = long_element_from_s_plus_1_last_letter(ops, verbose=False)
        P, A0, C_basis, P_global_sign = choose_global_sign_for_upper_intertwiner_last_letter(W_old, P_raw)
    else:
        P = None
        W_old = None
        A0 = identity_matrix(QQ, d)
        C_basis = []
        P_global_sign = None

    print("partition =", partition_rep)
    print("starting basis =", basis_name)
    print("ORDER = last-letter / total-index order")
    print("dimension =", d)
    print("original strict-upper A variables =", d*(d-1)//2)
    print("long-element intertwiner constraint =", "ON" if impose_intertwiner else "OFF")
    print("print A^T G A =", "ON" if print_ATGA else "OFF")
    if print_ATGA:
        print("G is the Gram matrix of the starting basis shown above.")
    if impose_intertwiner:
        print("A variables after W0_old*A=A*P_KL =", len(C_basis))
        print("particular A0 satisfies W0_old*A0=A0*P_KL:", W_old*A0 == A0*P)
        print("global sign chosen for P_KL =", P_global_sign)
    else:
        print("independent strict-upper A variables =", d*(d-1)//2)
    print("A0 is upper unitriangular:", all(A0[i, i] == 1 for i in range(d)) and all(A0[i, j] == 0 for i in range(d) for j in range(i)))

    if print_order:
        print("\nTom/Sage order -> last-letter permutation:")
        print(perm)
        print("\nLast-letter tableaux order:")
        for i, T in enumerate(last_tabs):
            print("%2d:" % i, T, "last-key=", last_letter_key(T))

    if print_W0 and impose_intertwiner:
        basis_label = "polytabloid/Specht" if not start_seminormal else "seminormal"
        print("\nW0_old = [w0] in the starting %s basis, reordered to last-letter order:" % basis_label)
        print(W_old)
        print("W0_old^2=I:", W_old*W_old == identity_matrix(QQ, d))

    if print_P and impose_intertwiner:
        print("\nP_KL = [w0]_KL in last-letter order:")
        print(P)
        print("P^2=I:", P*P == identity_matrix(QQ, d))

    if print_family and impose_intertwiner:
        print("\nGeneral upper-unitriangular evacuation-adapted A, in last-letter order:")
        print(symbolic_intertwiner_matrix_last_letter(A0, C_basis))

    # Solve both optimisation problems before printing the detailed matrices.
    # This lets us avoid printing duplicate output when A_min and A_max coincide.
    A_max, B_max, meta_max = solve_evac_adapted_A_last_letter(
        partition_rep, minimal=False, P_matrix=P, start_seminormal=start_seminormal,
        impose_intertwiner=impose_intertwiner, continuous=continuous, x_bound=x_bound, p_bound=p_bound, entry_bound=entry_bound,
        mip_gap=mip_gap, solver_verbose=solver_verbose, diagnostics=diagnostics)

    # If Ctrl+C interrupted MAX, keep and show the current incumbent,
    # but continue on to the MIN problem so that A_min and
    # A_max^(-1) A_min can still be computed later.
    if meta_max.get("interrupted", False):
        print("\n--- MAXIMISE tr(A^t G A): INTERRUPTED ---")
        _print_solution_block(
            "A_max",
            A_max,
            B_max,
            meta_max,
            G=G,
            round_digits=round_digits,
            print_B=print_B,
            print_ATGA=print_ATGA,
        )
        if A_max is None:
            print("\nMAX was interrupted before any feasible incumbent was found.")
            print("Continuing to MIN, but A_max^-1 A_min will be unavailable.")
        else:
            print("\nUsing this interrupted MAX incumbent as A_max for the later comparison.")
            print("It is feasible, but it is NOT a certified global maximum.")
        print("\nContinuing to MINIMISE tr(A^t G A) ...")

    A_min, B_min, meta_min = solve_evac_adapted_A_last_letter(
        partition_rep, minimal=True, P_matrix=P, start_seminormal=start_seminormal,
        impose_intertwiner=impose_intertwiner, continuous=continuous, x_bound=x_bound, p_bound=p_bound, entry_bound=entry_bound,
        mip_gap=mip_gap, solver_verbose=solver_verbose, diagnostics=diagnostics)

    # If Ctrl+C interrupted MIN, keep the current incumbent and continue
    # into the normal comparison/output block below.
    if meta_min.get("interrupted", False):
        print("\nMIN was interrupted by the user.")
        if A_min is None:
            print("No feasible MIN incumbent was found before interruption.")
            print("A_max^-1 A_min will therefore be unavailable.")
        else:
            print("Keeping the current feasible MIN incumbent as A_min.")
            print("It is NOT a certified global minimum.")

    same_A = _matrices_close_for_print(A_min, A_max, equality_tol)
    max_diff = _matrix_max_abs_difference(A_min, A_max)

    if meta_max.get("interrupted", False) or meta_min.get("interrupted", False):
        print("\nNOTE: at least one optimisation was interrupted.")
        print("The matrices below are feasible incumbents where available;")
        print("interrupted objectives are not certified global optima.")

    print("\n--- MINIMISE tr(A^t G A) ---")
    _print_solution_block(
        "A_min",
        A_min,
        B_min,
        meta_min,
        G=G,
        round_digits=round_digits,
        print_B=print_B,
        print_ATGA=print_ATGA,
    )

    print("\nA_min = A_max:", same_A)
    if max_diff is not None:
        print("max |A_min - A_max| =", _format_scalar_for_print(max_diff, round_digits))
        print("equality tolerance =", equality_tol)

    if (same_A and stop_if_min_equals_max
            and not meta_max.get("interrupted", False)
            and not meta_min.get("interrupted", False)):
        print("\nA_min and A_max agree within tolerance; omitting duplicate A_max output.")
        _print_A_distance_summary(A_min, A_max, round_digits)
        _print_Amax_Nguyen_check(
            partition_rep,
            A_max,
            B_max,
            wgraph_file=nguyen_wgraph_file,
            tol=nguyen_compare_tol,
            nguyen_verbose=nguyen_verbose,
        )
        return A_max, B_max, A_min, B_min

    print("\n--- MAXIMISE tr(A^t G A) ---")
    _print_solution_block(
        "A_max",
        A_max,
        B_max,
        meta_max,
        G=G,
        round_digits=round_digits,
        print_B=print_B,
        print_ATGA=print_ATGA,
    )

    _print_A_distance_summary(A_min, A_max, round_digits)

    if not same_A:
        _print_Amax_inverse_Amin_block(
            A_max,
            A_min,
            round_digits=round_digits,
        )

    _print_Amax_Nguyen_check(
        partition_rep,
        A_max,
        B_max,
        wgraph_file=nguyen_wgraph_file,
        tol=nguyen_compare_tol,
        nguyen_verbose=nguyen_verbose,
    )

    return A_max, B_max, A_min, B_min


# ---------------------------------------------------------------------------
# Nguyen/KL comparison helpers, v9
# ---------------------------------------------------------------------------

def _round_matrix_to_integer_matrix(M, tol=1e-6):
    """Round a numerical matrix to a Sage integer matrix for KL comparison."""
    d = M.nrows()
    e = M.ncols()
    R = Matrix(ZZ, d, e)
    max_rounding_error = 0.0
    for i in range(d):
        for j in range(e):
            x = float(M[i, j])
            r = int(round(x))
            max_rounding_error = max(max_rounding_error, abs(x - r))
            if abs(x - r) > tol:
                # Still round for display/comparison, but the caller can inspect the error.
                pass
            R[i, j] = r
    return R, max_rounding_error


def _resolve_nguyen_wgraph_file(wgraph_file=None):
    """Resolve Nguyen's C source, accepting common duplicate-download names."""
    if wgraph_file is not None:
        candidate = os.path.expanduser(str(wgraph_file))
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError("Nguyen wgraph source was not found: %s" % candidate)

    for candidate in ("wgraph.c", "wgraph(1).c"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Nguyen wgraph source was not found. Put wgraph.c in the current "
        "working directory or pass nguyen_wgraph_file='/full/path/wgraph.c'."
    )


def compare_B_ops_to_Nguyen(partition_rep, B_ops, tol=1e-6, verbose=True,
                            wgraph_file=None, nguyen_verbose=False):
    """Compare transformed operators B_i=A^{-1}(1+s_i)A with Nguyen KL operators.

    Assumes both are in last-letter / total-index order.  This is the same
    convention used by signed_evacuation_matrix_last_letter(... act_s_plus_1=KLLeftCellOperators(...)).

    Requires KLLeftCellOperators from Tom's KLLeftCellOptimizer.sage, plus wgraph.c.
    """
    if 'KLLeftCellOperators' not in globals():
        raise RuntimeError(
            "KLLeftCellOperators is not loaded. Load Tom's KLLeftCellOptimizer.sage "
            "if you want compare_Nguyen=True."
        )

    resolved_wgraph_file = _resolve_nguyen_wgraph_file(wgraph_file)
    kl_matrices = [Matrix(ZZ, M) for M in KLLeftCellOperators(
        partition_rep,
        c_source_filename=resolved_wgraph_file,
        verbose=nguyen_verbose,
    )]
    if not kl_matrices:
        raise RuntimeError("Nguyen's W-graph generator returned no operators.")

    rounded_B = []
    rounding_errors = []
    for B in B_ops:
        R, err = _round_matrix_to_integer_matrix(B, tol=tol)
        rounded_B.append(R)
        rounding_errors.append(err)

    same = (len(kl_matrices) == len(rounded_B))
    if same:
        for K, B in zip(kl_matrices, rounded_B):
            same = same and (K.nrows() == B.nrows())
            same = same and (K.ncols() == B.ncols())
            same = same and (K == B)

    max_rounding_error = max(rounding_errors) if rounding_errors else 0.0
    # Rounded equality alone is not enough: e.g. 1.49 would round to 1.
    same = same and (max_rounding_error <= tol)

    if verbose:
        print("Nguyen wgraph source =", resolved_wgraph_file)
        print("The optimum is the Kazhdan-Lusztig basis:", same)
        print("max rounding error before integer comparison =", _format_scalar_for_print(max_rounding_error, 6))
        print("comparison tolerance =", tol)

    return same, rounded_B, kl_matrices, max_rounding_error


def _print_Amax_Nguyen_check(partition_rep, A_max, B_max, wgraph_file=None,
                             tol=1e-6, nguyen_verbose=False):
    """Always attempt and report the Nguyen/KL check for ``A_max``."""
    print("\n--- NGUYEN W-GRAPH CHECK FOR A_max ---")
    if A_max is None or B_max is None:
        print("A_max gives the Kazhdan-Lusztig basis: UNAVAILABLE")
        print("reason = no A_max solution was found")
        return None

    try:
        resolved_wgraph_file = _resolve_nguyen_wgraph_file(wgraph_file)
        same, rounded_B, kl_matrices, max_rounding_error = compare_B_ops_to_Nguyen(
            partition_rep,
            B_max,
            tol=tol,
            verbose=False,
            wgraph_file=resolved_wgraph_file,
            nguyen_verbose=nguyen_verbose,
        )
        print("Nguyen wgraph source =", resolved_wgraph_file)
        print("A_max gives the Kazhdan-Lusztig basis:", same)
        print("max rounding error before integer comparison =", _format_scalar_for_print(max_rounding_error, 6))
        print("comparison tolerance =", tol)
        return same
    except Exception as error:
        print("A_max gives the Kazhdan-Lusztig basis: UNAVAILABLE")
        print("Nguyen check failed:", error)
        return None


def KLLeftcellOptimizer_w0_constrained(partition_rep,
                                        start_seminormal=True,
                                        compare_Nguyen=True,
                                        continuous=True,
                                        x_bound=50.0,
                                        p_bound=None,
                                        entry_bound=None,
                                        mip_gap=0.05,
                                        minimal=False,
                                        verbose=False,
                                        diagnostics=False,
                                        compare_tol=1e-6):
    """Tom-style wrapper with the extra w0-adapted constraint.

    Optimises tr(A^t G A) subject to:
        A upper unitriangular in last-letter order,
        [w0]_old A = A [w0]_KL,
        A^{-1}(1+s_i)A >= 0.

    If compare_Nguyen=True, prints whether the resulting transformed matrices
    agree with Nguyen's KL left-cell operators and returns the rounded integer
    transformed matrices, matching Tom's KLLeftcellOptimizer style.
    """
    A, B_ops, meta = solve_evac_adapted_A_last_letter(
        partition_rep,
        minimal=minimal,
        P_matrix=None,
        start_seminormal=start_seminormal,
        continuous=continuous,
        x_bound=x_bound,
        p_bound=p_bound,
        entry_bound=entry_bound,
        mip_gap=mip_gap,
        solver_verbose=verbose,
        diagnostics=diagnostics
    )

    if A is None:
        print("No feasible w0-adapted solution for", partition_rep)
        print("status =", meta["status"])
        return None

    if compare_Nguyen:
        same, rounded_B, kl_matrices, max_rounding_error = compare_B_ops_to_Nguyen(
            partition_rep,
            B_ops,
            tol=compare_tol,
            verbose=True
        )
        return A, rounded_B

    return A
