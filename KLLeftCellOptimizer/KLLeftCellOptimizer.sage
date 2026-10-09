import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm
import time
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import itertools
import subprocess
import re
import os
import struct
import subprocess
from sage.all import Matrix, ZZ
load("/mnt/c/Users/Jacques/Desktop/YoungNatural.sage")

# Imports Kazhdan-Lusztig left cell representation operators computed by Nguyen's algorithm (Type A admissible cells are Kazhdan–Lusztig, 2020)
# Operators are of the form 1+s in the KL left cell basis 
def KLLeftCellOperators(partition, c_source_filename="wgraph.c", executable_name="./wgraph_gen", verbose=False):
    """
    Generates W-graph operators for a given partition.
    
    Args:
        partition (list): The partition (e.g., [3, 2, 1]).
        c_source_filename (str): Path to the C source file.
        executable_name (str): Name for the compiled executable.
        verbose (bool): If True, prints progress and C output. Default is False.
        
    Returns:
        list: A list of SageMath sparse matrices representing the operators (1+s).
    """
    
    # Helper for conditional printing
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Determine subprocess output target
    # If not verbose, send stdout/stderr to DEVNULL (black hole)
    # If verbose, use None (inherits stdout/stderr from parent, showing output in console)
    io_target = None if verbose else subprocess.DEVNULL

    # --- 1. Update Partition in C Code ---
    part_str = "{" + ",".join(map(str, partition)) + "}"
    part_len = len(partition)
    
    # Read/Modify C file (silently)
    with open(c_source_filename, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if "#define PARTITION_LEN" in line:
            new_lines.append(f"#define PARTITION_LEN {part_len}\n")
        elif "const int PARTITION" in line:
            new_lines.append(f"const int PARTITION[{part_len}] = {part_str}; \n")
        else:
            new_lines.append(line)
            
    with open(c_source_filename, 'w') as f:
        f.writelines(new_lines)
        
    # --- 2. Compile and Run ---
    vprint(f"Compiling {c_source_filename}...")
    try:
        subprocess.run(
            ["gcc", "-O3", "-march=native", c_source_filename, "-o", executable_name], 
            check=True, 
            stdout=io_target, 
            stderr=io_target
        )
    except subprocess.CalledProcessError:
        print("Error: Compilation failed.")
        return []
    
    vprint("Running C generator...")
    try:
        subprocess.run(
            [executable_name], 
            check=True, 
            stdout=io_target, 
            stderr=io_target
        )
    except subprocess.CalledProcessError:
        print("Error: C execution failed.")
        return []
    
    # --- 3. Load Binary Operators ---
    vprint("Loading operators into Sage...")
    matrices = []
    
    filename = "operators.bin"
    if not os.path.exists(filename):
        raise FileNotFoundError("operators.bin was not generated.")

    with open(filename, "rb") as f:
        # Read Header: [Num Gens, Dimension]
        header_bytes = f.read(8) # 2 * 4 bytes
        num_gens, dim = struct.unpack("ii", header_bytes)
        
        vprint(f"Detected {num_gens} generators, dimension {dim}")
        
        for s in range(num_gens):
            # Read NNZ for this generator
            nnz_bytes = f.read(4)
            nnz = struct.unpack("i", nnz_bytes)[0]
            
            # Read the raw data chunk
            chunk_size = nnz * 3 * 4
            if chunk_size > 0:
                raw_data = f.read(chunk_size)
                # Reshape to (NNZ, 3) -> columns: [row, col, val]
                data_arr = np.frombuffer(raw_data, dtype=np.int32).reshape((nnz, 3))
                
                rows = data_arr[:, 0]
                cols = data_arr[:, 1]
                vals = data_arr[:, 2]
                
                # Zip is faster than iterating rows manually in python
                data_dict = {(int(r), int(c)): int(v) for r, c, v in zip(rows, cols, vals)}
                
                M = Matrix(ZZ, dim, dim, data_dict, sparse=True)
            else:
                M = Matrix(ZZ, dim, dim, sparse=True)
                
            matrices.append(M)

    return matrices

# This is an extremely unefficient way to compute the Gram Matrix in the Seminormal basis
# One can also obtain it directly by using the base change matrix between Seminormal and orthogonal basis
# (see James Kerber Chapter 3) 
def SeminormalGram(partition_rep):
    g=SpechtGram(partition_rep)
    return diagonal_matrix(g.LU()[2].diagonal())

# Returns Gram Matrix for Specht Module
# Since the Spechtmodule is irreducible it has a unique invariant dot product (up to scalar)
# This defines a unique Gram Matrix (up to scalar)
def SpechtGram(partition_rep):
    rank_w=sum(partition_rep)-1
    SGA = SymmetricGroupAlgebra(QQ, rank_w+1)
    SM = SGA.specht_module(partition_rep)
    W = SGA.group()
    simple_reflections=W.simple_reflections()
    n=SM.dimension()
    basis = (SM.basis()).list()
    return matrix([[SM.bilinear_form(basis[i],basis[j]) for i in range(n)] for j in range(n)])

# Compute Kazhdan-Lusztig left cell representation using optimisation
# tested for all S_n modules for n<=9 and some more of dimension <=200
def KLLeftcellOptimizer(partition_rep, start_seminormal: bool = True, 
    compare_Nguyen: bool = True, 
    continuous: bool = True, optimisation: bool = True,verbose=False,minimal: bool =False,look_for_two_solutions:bool = False,init=True):
    if verbose:
        print("Optimiser for partition: ",partition_rep)
        print("Build Specht module and correspondting matrices, s+1, for simple reflections s")
    t=time.time()
    rank_w=sum(partition_rep)-1
    if optimisation:
        SGA = SymmetricGroupAlgebra(QQ, rank_w+1)
        SM = SGA.specht_module(partition_rep)
        W = SGA.group()
        simple_reflections=W.simple_reflections()
        n=SM.dimension()
    if start_seminormal:
        act_s_plus_1=GetSeminormalFormOperators(partition_rep)
        n=act_s_plus_1[0].ncols()
    else:
        act_s_plus_1=GetSpechtOperators(partition_rep)
        n=act_s_plus_1[0].ncols()
    if verbose:
        print("Built Specht module with dimension ",n)
        print("This took", int(round(time.time()-t,0)), "seconds.")
        print("Build Variables for Model")
    # Prepare lists of indices
    a_indices = [(i, j) for i in range(n) for j in range(n)]
    p_indices = [(i, j, k) for i in range(n) for j in range(n) for k in range(rank_w)]
    #Creating variables for model.# Initialize the Gurobi model
    model = gp.Model("minimal_bases_model")
    if not verbose:
        model.Params.OutputFlag = 0
    # Add variables in bulk
    if continuous:
        if start_seminormal:
            # these values seem to be small
            a_vars = model.addVars(a_indices, vtype=GRB.CONTINUOUS, lb=-1.5, ub=1.5, name="a_vars")
        else:
            a_vars = model.addVars(a_indices, vtype=GRB.CONTINUOUS, lb=-5.1, ub=5.1, name="a_vars")
        for v in a_vars.values():
            v.BranchPriority = int(10)  # High priority
        p_vars = model.addVars(p_indices, vtype=GRB.CONTINUOUS, lb=0, ub=3.1, name="p_vars")
    else:
        a_vars = model.addVars(a_indices, vtype=GRB.INTEGER, lb=-10, ub=10, name="a_vars")
        p_vars = model.addVars(p_indices, vtype=GRB.CONTINUOUS, lb=0, ub=10, name="p_vars")
    if verbose:
        print("Enforce unit upper triangularity")
    quad_constraints=[]
    # Enforce that A is unit upper triangular: a_{i,j} = 0 for j > i, a_{i,i} = 1
    for i in range(n):
        for j in range(n):
            if j < i:
                lhs=a_vars[i, j]
                rhs=0
                quad_constraints.append((lhs, GRB.EQUAL, rhs, f"upper_zero_{i}_{j}"))
            elif i == j:
                lhs=a_vars[i, j]
                rhs=1
                quad_constraints.append((lhs, GRB.EQUAL, rhs, f"diag_one_{i}")) 

    # Creating constraints for model: matrix multiplication A*A_k=P_k*A
    if verbose:
        print("Enforce non-negativity condition")
    t=time.time()
    if verbose:
        for k in tqdm(range(rank_w), desc="Non-negativity for each reflection"):
            for i in range(n):
                for j in range(n):
                    terms = [ act_s_plus_1[k][i,l]*a_vars[l,j] for l in range(0,j+1) if (i,l) in act_s_plus_1[k].support()]
                    lhs = gp.quicksum(terms)
                    rhs_terms = [a_vars[i,l]*p_vars[l,j,k] for l in range(i,n)]
                    rhs = gp.quicksum(rhs_terms)
                    quad_constraints.append((lhs, GRB.EQUAL, rhs, f"matrix_multiplication_constraint_{i}_{j}_{k}")) 
    else:
        for k in range(rank_w):
            for i in range(n):
                for j in range(n):
                    terms = [ act_s_plus_1[k][i,l]*a_vars[l,j] for l in range(0,j+1) if (i,l) in act_s_plus_1[k].support()]
                    lhs = gp.quicksum(terms)
                    rhs_terms = [a_vars[i,l]*p_vars[l,j,k] for l in range(i,n)]
                    rhs = gp.quicksum(rhs_terms)
                    quad_constraints.append((lhs, GRB.EQUAL, rhs, f"matrix_multiplication_constraint_{i}_{j}_{k}"))

    epsilon = 1e-4
    if verbose:
        iterator = tqdm(quad_constraints, desc="Adding matrix multiplication constraints")
    else:
        iterator = quad_constraints
    # Add quadratic constraints with a status bar
    for lhs, sense, rhs, name in iterator:
        #if continuous:
        #	# CRITICAL FIX: Turn strict equality into a range (sandwich)
        #	# This prevents "Root Relaxation Infeasible" caused by floating point math
        #	model.addQConstr(lhs <= rhs + epsilon, name=name + "_UB")
        #	model.addQConstr(lhs >= rhs - epsilon, name=name + "_LB")
        #else:
            # Keep strict equality for Integer runs (as they are exact)
            model.addQConstr(lhs, sense, rhs, name=name)
    # Create objective function
    if optimisation:
            basis = (SM.basis()).list()
            terms = []
            if start_seminormal:
                G=SeminormalGram(partition_rep)
            for l in range(n):
                for i in range(l+1):
                    term = []
                    # the j loop is not really necessary and leads to the block form of the "Gram matrix"
                    for j in range(l+1):
                        if start_seminormal:
                            terms.append(G[i,j]*a_vars[i,l]*a_vars[j,l])
                        else:
                            terms.append(SM.bilinear_form(basis[i],basis[j])*a_vars[i,l]*a_vars[j,l])
            objective = gp.quicksum(terms)
            if not minimal:
                model.setObjective(objective, GRB.MAXIMIZE)
            else:
                model.setObjective(objective, GRB.MINIMIZE)
    else:
        model.setObjective(0, GRB.MINIMIZE)
    if verbose:
        print("Solve model")
    t=time.time()
    # Warm start logic
    if start_seminormal and init:
        if verbose:
            print("Setting warm start: A = Identity, P = act_s_plus_1")
        
        # Set Start values for a_vars (Identity Matrix)
        for i in range(n):
            for j in range(n):
                if i == j:
                    a_vars[i, j].Start = 1.0
                else:
                    a_vars[i, j].Start = 0.0
        
        # Set Start values for p_vars (act_s_plus_1 values)
        for k in range(rank_w):
            for i in range(n):
                for j in range(n):
                    # Extract the numerical value from the Sage/Matrix object
                    p_val = float(act_s_plus_1[k][i, j])
                    p_vars[i, j, k].Start = p_val	
    model.Params.MIPFocus = 3
    model.Params.MIPGap = 0.05
    model.Params.Heuristics = 0.2
    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        print("\nModel is INFEASIBLE.")
        # This tells Gurobi to calculate WHICH constraints are fighting each other
        model.computeIIS() 
        model.write("model_iis.ilp")
        print("Irreducible Inconsistent Subsystem (IIS) written to 'model_iis.ilp'")
        print("Use the file to see which constraints are contradictory.")
        return None
    
    elif model.Status == GRB.INF_OR_UNBD:
        print("\nModel is Infeasible or Unbounded.")
        return None
    if verbose:
        print("This took", int(round(time.time()-t,0)), "seconds.")

    a_matrix = Matrix(QQ, n, n)
    for i in range(n):
        for j in range(n):
                a_value = a_vars[(i,j)].X
                a_matrix[i, j] = round(a_value,3)
                if minimal==False and start_seminormal==False:
                    a_matrix[i, j] = round(a_value,0)


    #print([a_matrix*s*a_matrix^-1 for s in act_s_plus_1])
    # in case we have more than one solution we want to compare them without rounding to integers
    if look_for_two_solutions and model.SolCount >= 2:
        a_matrix = Matrix(QQ, n, n)
        for i in range(n):
            for j in range(n):
                    a_value = a_vars[(i,j)].X
                    a_matrix[i, j] = a_value
        model.setParam(GRB.Param.SolutionNumber, 1)
        print("\nSwitch to second solution.")
        # Now we compare with evaluated Kazhdan-Lusztig polynomials at 1
        a_matrix2 = Matrix(QQ, n, n)
        for i in range(n):
            for j in range(n):
                a_value = a_vars[(i,j)].Xn
                a_matrix2[i, j] = a_value
        return [a_matrix,a_matrix2]
    if compare_Nguyen:
        kl_matrices=KLLeftCellOperators(partition_rep)
        our_matrices=[a_matrix^-1*s*a_matrix for s in act_s_plus_1]
        if start_seminormal:
            our_matrices=[a_matrix^-1*s*a_matrix for s in act_s_plus_1]
            our_matrices=[matrix([[int(round(s[i,j],2)) for j in range(n)] for i in range(n)]) for s in our_matrices]
        same=True
        for i in range(0,len(kl_matrices)):
            same = same & (kl_matrices[i]==our_matrices[i])
        if verbose or compare_Nguyen or not same:
            print(f"The optimum for S^{partition_rep} is the Kazhdan-Lusztig basis: ", same)
        return a_matrix, our_matrices
    else:
        return a_matrix
