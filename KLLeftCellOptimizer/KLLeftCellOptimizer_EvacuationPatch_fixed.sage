# KLLeftCellOptimizer_EvacuationPatch.sage
#
# Add this file next to Tom's KLLeftCellOptimizer.sage.
# Usage:
#   load("KLLeftCellOptimizer(1).sage")    # or your local Tom file
#   load("KLLeftCellOptimizer_EvacuationPatch.sage")
#   P, tabs, perm, qval = signed_evacuation_matrix_last_letter([4,4,1], verbose=True)
#   A_max, B_max, A_min, B_min = print_min_max_evac([4,4,1], P, start_seminormal=True)
#
# The centralizer optimizer parametrises A as A = I + sum_t x_t C_t, where
# C_t is a QQ-basis of {strictly upper triangular N : NP=PN}. Hence A is
# automatically upper unitriangular and AP=PA.

import gurobipy as gp
from gurobipy import GRB
from sage.all import *
from fractions import Fraction

# ---------------------------------------------------------------------
# 1. Standard tableaux in last-letter / total-index order
# ---------------------------------------------------------------------

def _tuple_tableau(T):
    """Convert a Sage tableau/list tableau to an immutable tuple-of-tuples."""
    try:
        L = T.to_list()
    except AttributeError:
        L = T
    return tuple(tuple(Integer(x) for x in row) for row in L)


def _positions(T):
    pos = {}
    for r, row in enumerate(T):
        for c, value in enumerate(row):
            pos[Integer(value)] = (r, c)
    return pos


def last_letter_key(T):
    """
    Last-letter / total-index order key.

    We sort by the row containing n, then n-1, ..., then 1.
    Row 0 is the top row. This is the convention used in the user's
    computations for the KL/SYT indexing.
    """
    T = _tuple_tableau(T)
    pos = _positions(T)
    n = sum(len(row) for row in T)
    return tuple(pos[k][0] for k in range(n, 0, -1))


def standard_tableaux_last_letter(partition_rep):
    """All SYT of shape partition_rep in last-letter order."""
    tabs = [_tuple_tableau(T) for T in StandardTableaux(partition_rep)]
    return sorted(tabs, key=last_letter_key)


# ---------------------------------------------------------------------
# 2. Schuetzenberger evacuation and Stembridge q=-1 check
# ---------------------------------------------------------------------

def evacuate_tableau(T):
    """
    Schuetzenberger evacuation by repeated jeu-de-taquin.

    T is represented as tuple-of-tuples.  The output has the same shape.
    """
    T = _tuple_tableau(T)
    n = sum(len(row) for row in T)

    work = {(r, c): Integer(value)
            for r, row in enumerate(T)
            for c, value in enumerate(row)}
    evacuated = {}

    for m in range(n, 0, -1):
        inv = {value: cell for cell, value in work.items()}
        hole = inv[1]
        del work[hole]

        while True:
            r, c = hole
            candidates = []
            for nb in ((r, c + 1), (r + 1, c)):
                if nb in work:
                    candidates.append((work[nb], nb))
            if not candidates:
                break
            _, nb = min(candidates)
            work[hole] = work[nb]
            del work[nb]
            hole = nb

        for cell in list(work.keys()):
            work[cell] -= 1

        evacuated[hole] = Integer(m)

    return tuple(
        tuple(evacuated[(r, c)] for c in range(len(row)))
        for r, row in enumerate(T)
    )


def _hook_lengths(partition_rep):
    lam = tuple(Integer(x) for x in partition_rep)
    if len(lam) == 0:
        return []
    max_col = max(lam)
    col_heights = [sum(row_len > c for row_len in lam) for c in range(max_col)]
    hooks = []
    for r, row_len in enumerate(lam):
        for c in range(row_len):
            hooks.append((row_len - c) + (col_heights[c] - r) - 1)
    return hooks


def q_hook_at_minus_one(partition_rep):
    """
    Evaluate the major-index/fake-degree q-hook formula at q=-1.

    Stembridge q=-1 phenomenon says |value| is the number of
    self-evacuating SYT.  The sign is the trace sign convention for w0.

    Important Sage/Python compatibility note:
    use Sage rationals QQ, not Python fractions.Fraction.  In Sage,
    numerator/denominator are methods, which can confuse Fraction.
    """
    lam = tuple(Integer(x) for x in partition_rep)
    n = Integer(sum(lam))
    b_lam = sum(Integer(r) * row_len for r, row_len in enumerate(lam))

    numerator_even = [Integer(i) for i in range(1, int(n) + 1) if i % 2 == 0]
    denominator_even = [Integer(h) for h in _hook_lengths(lam) if h % 2 == 0]

    if len(numerator_even) > len(denominator_even):
        return Integer(0)
    if len(numerator_even) < len(denominator_even):
        raise RuntimeError("Unexpected pole in q-hook formula at q=-1.")

    value = QQ((-1) ** int(b_lam))
    for a in numerator_even:
        value *= QQ(a)
    for b in denominator_even:
        value /= QQ(b)

    if value.denominator() != 1:
        raise RuntimeError("q-hook value was not integral: %s" % value)
    return Integer(value.numerator())


def evacuation_permutation_last_letter(partition_rep, verbose=True):
    """
    Return (tableaux, permutation, q_minus_one).

    The permutation is column convention:
        e_j maps to e_{permutation[j]} under unsigned evacuation.
    """
    tabs = standard_tableaux_last_letter(partition_rep)
    index = {T: i for i, T in enumerate(tabs)}
    perm = [index[evacuate_tableau(T)] for T in tabs]

    qval = q_hook_at_minus_one(partition_rep)
    fixed = sum(1 for j, pj in enumerate(perm) if j == pj)

    if fixed != abs(qval):
        raise RuntimeError(
            "Stembridge q=-1 check failed for %s: fixed=%s but |q(-1)|=%s"
            % (partition_rep, fixed, abs(qval))
        )

    if verbose:
        print("partition =", partition_rep)
        print("dimension =", len(tabs))
        print("q-hook at -1 =", qval)
        print("self-evacuating tableaux / fixed columns =", fixed)

    return tabs, perm, qval


# ---------------------------------------------------------------------
# 3. Build signed P=[w0]_KL in last-letter order
# ---------------------------------------------------------------------

def _relation_sign_bfs(num_vertices, relations):
    """
    Solve sign relations u_a = rel*u_b, rel in {+1,-1}.
    Returns one assignment and the connected components.
    """
    graph = [[] for _ in range(num_vertices)]
    for a, b, rel in relations:
        rel = Integer(rel)
        graph[a].append((b, rel))       # u_a = rel u_b
        graph[b].append((a, rel))       # u_b = rel u_a, since rel^{-1}=rel

    signs = [None] * num_vertices
    components = []

    for start in range(num_vertices):
        if signs[start] is not None:
            continue
        signs[start] = Integer(1)
        comp = [start]
        queue = [start]
        while queue:
            a = queue.pop()
            for b, rel in graph[a]:
                # u_a = rel*u_b, hence u_b = rel*u_a
                wanted = rel * signs[a]
                if signs[b] is None:
                    signs[b] = wanted
                    comp.append(b)
                    queue.append(b)
                elif signs[b] != wanted:
                    raise RuntimeError("Inconsistent sign relations for P.")
        components.append(comp)

    return signs, components


def _choose_component_flips_from_trace(signs, components, fixed_cols, qval):
    """
    Flip signs on connected components so that trace(P)=qval when possible.
    """
    qval = Integer(qval)
    fixed_set = set(fixed_cols)
    contributions = []
    for comp in components:
        c = sum(signs[j] for j in comp if j in fixed_set)
        contributions.append(Integer(c))

    current_trace = sum(contributions)
    if current_trace == qval:
        return [Integer(1)] * len(components)
    if -current_trace == qval:
        return [Integer(-1)] * len(components)

    # Dynamic programming for sum epsilon_t * contributions[t] = qval.
    # This is usually tiny; it is here only to handle multiple components.
    dp = {Integer(0): []}
    for t, c in enumerate(contributions):
        new = {}
        for s, choices in dp.items():
            new[s + c] = choices + [Integer(1)]
            new[s - c] = choices + [Integer(-1)]
        dp = new
    if qval not in dp:
        raise RuntimeError(
            "Could not choose signs with trace q(-1). q(-1)=%s, possible traces=%s"
            % (qval, sorted(dp.keys()))
        )
    return dp[qval]


def signed_evacuation_matrix_last_letter(partition_rep,
                                         act_s_plus_1=None,
                                         use_operator_relations=True,
                                         verbose=True):
    """
    Construct a signed evacuation matrix P in last-letter order.

    If act_s_plus_1 is supplied, the signs are solved from the w0 relation

        P (1+s_i) = (1+s_{n-i}) P.

    This is safer than guessing signs.  Stembridge q=-1 is used to verify
    the number of fixed columns and to choose the trace sign.

    If act_s_plus_1 is omitted, we use the global sign convention determined
    by q(-1).  That is enough for many centralizer-dimension computations,
    but the operator-relation version is the one to trust before imposing
    AP=PA in the optimizer.
    """
    tabs, perm, qval = evacuation_permutation_last_letter(partition_rep, verbose=verbose)
    d = len(tabs)
    fixed_cols = [j for j, pj in enumerate(perm) if j == pj]

    if act_s_plus_1 is not None and use_operator_relations:
        rank_w = sum(partition_rep) - 1
        inv_perm = [None] * d
        for j, pj in enumerate(perm):
            inv_perm[pj] = j

        relations = []

        # P^2=I implies sign_j = sign_{perm[j]}.
        for j, pj in enumerate(perm):
            relations.append((j, pj, Integer(1)))

        # P S_i = S_{n-i} P, where matrices are for 1+s_i.
        for k in range(rank_w):
            S_left = Matrix(QQ, act_s_plus_1[k])
            S_right = Matrix(QQ, act_s_plus_1[rank_w - 1 - k])
            for r in range(d):
                a = inv_perm[r]
                for c in range(d):
                    left_coeff = S_left[a, c]
                    right_coeff = S_right[r, perm[c]]

                    if left_coeff == 0 and right_coeff == 0:
                        continue
                    if left_coeff == 0 or right_coeff == 0:
                        raise RuntimeError(
                            "Evacuation support/order does not match operators at k=%s, r=%s, c=%s."
                            % (k, r, c)
                        )
                    ratio = QQ(right_coeff) / QQ(left_coeff)
                    if ratio not in [QQ(1), QQ(-1)]:
                        raise RuntimeError(
                            "Non-sign ratio in w0 relation: %s at k=%s, r=%s, c=%s."
                            % (ratio, k, r, c)
                        )
                    # left_coeff*u_a = right_coeff*u_c, so u_a = ratio*u_c.
                    relations.append((a, c, Integer(ratio)))

        signs, components = _relation_sign_bfs(d, relations)
        flips = _choose_component_flips_from_trace(signs, components, fixed_cols, qval)
        for t, comp in enumerate(components):
            for j in comp:
                signs[j] *= flips[t]
    else:
        # Fallback: global sign chosen to make trace agree when fixed columns exist.
        if qval > 0:
            global_sign = Integer(1)
        elif qval < 0:
            global_sign = Integer(-1)
        else:
            global_sign = Integer(1)  # trace gives no information; centralizer unchanged by -P
        signs = [global_sign] * d

    P = zero_matrix(QQ, d, d)
    for j, pj in enumerate(perm):
        P[pj, j] = signs[j]

    if verbose:
        print("trace(P) =", P.trace())
        print("P^2 = I:", P * P == identity_matrix(QQ, d))
        print("last-letter order verified by q=-1:", P.trace() == qval)

    return P, tabs, perm, qval


def check_w0_relation(partition_rep, P, act_s_plus_1=None, start_seminormal=True):
    """
    Check P(1+s_i)=(1+s_{n-i})P in the same basis/order as the operators.
    """
    rank_w = sum(partition_rep) - 1
    if act_s_plus_1 is None:
        if start_seminormal:
            act_s_plus_1 = GetSeminormalFormOperators(partition_rep)
        else:
            act_s_plus_1 = GetSpechtOperators(partition_rep)
    P = Matrix(QQ, P)
    checks = []
    for k in range(rank_w):
        checks.append(P * Matrix(QQ, act_s_plus_1[k]) == Matrix(QQ, act_s_plus_1[rank_w - 1 - k]) * P)
    print("w0 conjugation/order check:", all(checks), checks)
    return all(checks)


# ---------------------------------------------------------------------
# 4. Strict-upper centralizer basis: A = I + sum x_t C_t
# ---------------------------------------------------------------------

def strict_upper_commuting_basis(P_matrix):
    """
    Return a QQ-basis C_1,...,C_m for

        {N strictly upper triangular : NP=PN}.

    Then A=I+sum_t x_t C_t is automatically upper unitriangular and AP=PA.
    """
    P = Matrix(QQ, P_matrix)
    n = P.nrows()
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    rows = []
    for r in range(n):
        for c in range(n):
            coeffs = []
            for (i, j) in pairs:
                # Entry (r,c) of E_ij P - P E_ij.
                coeffs.append((1 if r == i else 0) * P[j, c]
                              - P[r, i] * (1 if c == j else 0))
            if any(a != 0 for a in coeffs):
                rows.append(coeffs)

    if len(rows) == 0:
        K = Matrix(QQ, 0, len(pairs)).right_kernel()
    else:
        K = Matrix(QQ, rows).right_kernel()

    basis = []
    for v in K.basis():
        C = zero_matrix(QQ, n, n)
        for coeff, (i, j) in zip(v, pairs):
            C[i, j] = coeff
        basis.append(C)
    return basis


# ---------------------------------------------------------------------
# 5. Optimizer with AP=PA built in
# ---------------------------------------------------------------------

def KLLeftcellOptimizer_EvacCentralizer(partition_rep,
                                        P_matrix,
                                        start_seminormal=True,
                                        continuous=True,
                                        optimisation=True,
                                        minimal=False,
                                        verbose=False,
                                        init=True,
                                        entry_bound=None,
                                        p_bound=3.1,
                                        x_bound=10.0,
                                        mip_gap=0.05,
                                        return_model=False):
    """
    Tom's optimisation problem, with the extra restriction AP=PA.

    Instead of creating all A_ij variables, it creates only x_t variables in

        A = I + sum_t x_t C_t,

    where C_t spans the strict-upper centralizer of P.

    The nonnegative matrices B_k are still represented by p_vars and satisfy

        (1+s_k) A = A B_k,

    so B_k = A^{-1}(1+s_k)A and B_k is entrywise nonnegative.
    """
    rank_w = sum(partition_rep) - 1

    if optimisation:
        SGA = SymmetricGroupAlgebra(QQ, rank_w + 1)
        SM = SGA.specht_module(partition_rep)
        n = SM.dimension()

    if start_seminormal:
        act_s_plus_1 = GetSeminormalFormOperators(partition_rep)
        n = act_s_plus_1[0].ncols()
    else:
        act_s_plus_1 = GetSpechtOperators(partition_rep)
        n = act_s_plus_1[0].ncols()

    P = Matrix(QQ, P_matrix)
    if P.nrows() != n or P.ncols() != n:
        raise ValueError("P_matrix has size %s but module dimension is %s." % (P.nrows(), n))

    C_basis = strict_upper_commuting_basis(P)
    m = len(C_basis)

    if verbose:
        print("partition =", partition_rep)
        print("Specht dimension =", n)
        print("original strict-upper A variables =", n * (n - 1) // 2)
        print("A variables after AP=PA =", m)

    model = gp.Model("KL_left_cell_with_evacuation_centralizer")
    if not verbose:
        model.Params.OutputFlag = 0

    if continuous:
        x_vars = model.addVars(range(m), vtype=GRB.CONTINUOUS,
                               lb=-float(x_bound), ub=float(x_bound), name="x")
        p_vars = model.addVars([(i, j, k) for i in range(n) for j in range(n) for k in range(rank_w)],
                               vtype=GRB.CONTINUOUS, lb=0.0, ub=float(p_bound), name="B")
    else:
        x_vars = model.addVars(range(m), vtype=GRB.INTEGER,
                               lb=-int(x_bound), ub=int(x_bound), name="x")
        p_vars = model.addVars([(i, j, k) for i in range(n) for j in range(n) for k in range(rank_w)],
                               vtype=GRB.CONTINUOUS, lb=0.0, ub=float(max(p_bound, 10.0)), name="B")

    def A_expr(i, j):
        # C_basis is strictly upper triangular, so A has diagonal 1 and lower part 0 automatically.
        const = 1.0 if i == j else 0.0
        terms = [float(C_basis[t][i, j]) * x_vars[t]
                 for t in range(m) if C_basis[t][i, j] != 0]
        return const + gp.quicksum(terms) if terms else const

    # Optional compactness bounds on entries of A.  Tom's original code used
    # about 1.5 in seminormal and 5.1 in Specht mode; pass entry_bound to mimic that.
    if entry_bound is not None:
        for i in range(n):
            for j in range(i + 1, n):
                aij = A_expr(i, j)
                model.addConstr(aij <= float(entry_bound), name="A_entry_ub_%d_%d" % (i, j))
                model.addConstr(aij >= -float(entry_bound), name="A_entry_lb_%d_%d" % (i, j))

    # Matrix multiplication constraints: S_k A = A B_k.
    support_list = [set(M.support()) for M in act_s_plus_1]
    for k in range(rank_w):
        Sk = Matrix(QQ, act_s_plus_1[k])
        supp = support_list[k]
        if verbose:
            print("adding constraints for generator", k + 1, "of", rank_w)
        for i in range(n):
            for j in range(n):
                lhs_terms = [float(Sk[i, l]) * A_expr(l, j)
                             for l in range(0, j + 1)
                             if (i, l) in supp]
                lhs = gp.quicksum(lhs_terms) if lhs_terms else 0.0

                rhs_terms = [A_expr(i, l) * p_vars[l, j, k]
                             for l in range(i, n)]
                rhs = gp.quicksum(rhs_terms) if rhs_terms else 0.0

                model.addQConstr(lhs == rhs, name="SA_equals_AB_%d_%d_%d" % (i, j, k))

    # Objective: tr(A^t G A), same as Tom's code.
    if optimisation:
        terms = []
        if start_seminormal:
            G = SeminormalGram(partition_rep)
            for l in range(n):
                for i in range(l + 1):
                    for j in range(l + 1):
                        if G[i, j] != 0:
                            terms.append(float(G[i, j]) * A_expr(i, l) * A_expr(j, l))
        else:
            basis = SM.basis().list()
            for l in range(n):
                for i in range(l + 1):
                    for j in range(l + 1):
                        gij = SM.bilinear_form(basis[i], basis[j])
                        if gij != 0:
                            terms.append(float(gij) * A_expr(i, l) * A_expr(j, l))
        model.setObjective(gp.quicksum(terms), GRB.MINIMIZE if minimal else GRB.MAXIMIZE)
    else:
        model.setObjective(0.0, GRB.MINIMIZE)

    if init:
        # A=I means all centralizer coordinates are zero.
        for t in range(m):
            x_vars[t].Start = 0.0
        # With A=I, B_k starts at the original operator matrix.
        for k in range(rank_w):
            for i in range(n):
                for j in range(n):
                    p_vars[i, j, k].Start = max(0.0, float(act_s_plus_1[k][i, j]))

    model.Params.NonConvex = 2
    model.Params.MIPFocus = 3
    model.Params.MIPGap = float(mip_gap)
    model.Params.Heuristics = 0.2
    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("Model is INFEASIBLE.")
        model.computeIIS()
        model.write("model_iis.ilp")
        return (None, None, model) if return_model else (None, None)
    if model.Status in [GRB.INF_OR_UNBD, GRB.UNBOUNDED]:
        print("Model is infeasible or unbounded. Status =", model.Status)
        return (None, None, model) if return_model else (None, None)
    if model.SolCount == 0:
        print("No solution found. Status =", model.Status)
        return (None, None, model) if return_model else (None, None)

    A = Matrix(QQ, n, n)
    for i in range(n):
        for j in range(n):
            val = 1.0 if i == j else 0.0
            for t in range(m):
                if C_basis[t][i, j] != 0:
                    val += float(C_basis[t][i, j]) * x_vars[t].X
            A[i, j] = QQ(str(round(val, 10)))

    B_ops = [A^(-1) * Matrix(QQ, S) * A for S in act_s_plus_1]

    if verbose:
        print("objective =", model.ObjVal)
        print("AP=PA check:", A * P == P * A)
        print("min entry among transformed B_k =",
              min([min(B.list()) for B in B_ops]))

    if return_model:
        return A, B_ops, model
    return A, B_ops


def print_min_max_evac(partition_rep,
                       P_matrix=None,
                       start_seminormal=True,
                       continuous=True,
                       entry_bound=None,
                       p_bound=3.1,
                       x_bound=10.0,
                       mip_gap=0.05,
                       verbose=False):
    """
    Convenience wrapper: compute and print A_max/B_max and A_min/B_min.

    If P_matrix is None, construct P from last-letter evacuation.  If you need
    signs matched to a particular operator basis, pass that basis to
    signed_evacuation_matrix_last_letter first and then pass P_matrix here.
    """
    if P_matrix is None:
        P_matrix, _, _, _ = signed_evacuation_matrix_last_letter(partition_rep, verbose=True)
    P = Matrix(QQ, P_matrix)

    C_basis = strict_upper_commuting_basis(P)
    print("partition =", partition_rep)
    print("Specht dimension =", P.nrows())
    print("original strict-upper A variables =", P.nrows() * (P.nrows() - 1) // 2)
    print("A variables after upper-unitriangularity + AP=PA =", len(C_basis))
    print("")

    A_max, B_max = KLLeftcellOptimizer_EvacCentralizer(
        partition_rep, P,
        start_seminormal=start_seminormal,
        continuous=continuous,
        minimal=False,
        verbose=verbose,
        entry_bound=entry_bound,
        p_bound=p_bound,
        x_bound=x_bound,
        mip_gap=mip_gap)

    print("A_max =")
    print(A_max)
    print("")
    if B_max is not None:
        for r, B in enumerate(B_max, start=1):
            print("B_max_%d = A_max^-1 A_%d A_max =" % (r, r))
            print(B)
            print("")

    A_min, B_min = KLLeftcellOptimizer_EvacCentralizer(
        partition_rep, P,
        start_seminormal=start_seminormal,
        continuous=continuous,
        minimal=True,
        verbose=verbose,
        entry_bound=entry_bound,
        p_bound=p_bound,
        x_bound=x_bound,
        mip_gap=mip_gap)

    print("A_min =")
    print(A_min)
    print("")
    if B_min is not None:
        for r, B in enumerate(B_min, start=1):
            print("B_min_%d = A_min^-1 A_%d A_min =" % (r, r))
            print(B)
            print("")

    return A_max, B_max, A_min, B_min
