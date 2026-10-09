from sage.combinat.tableau import StandardTableaux
from sage.combinat.permutation import Permutations,Permutation
from sage.groups.perm_gps.permgroup_named import SymmetricGroup
from sage.all import matrix, Permutation, SymmetricGroup


# A naive implementation of Young's natural basis following Garsia&McLarnan '88
# See also Curtis '81 for the same method 

def tableau_intersection(T1, T2):
    """
    Compute the intersection of tableaux T1 and T2.
    Returns a tableau if valid, else None.
    """
    shape = T1.shape()
    n = sum(shape)
    intersection = [[None for _ in range(shape[i])] for i in range(len(shape))]
    
    # Build intersection: i,j entry is intersection of T1's ith row and T2's jth column
    for i in range(len(shape)):
        row_vals = set(T1[i])  # Values in T1's ith row
        for j in range(shape[0]):  # Columns up to the longest row
            col_vals = set((T2.conjugate())[j])  # Values in T2's jth column
            common = row_vals.intersection(col_vals)
            if len(common) > 1:
                return 0  # Not a valid tableau
            if len(common) == 1:
                intersection[i][j] = list(common)[0]

    perm = [0 for i in range(1,n+1)]
    for i in range(len(shape)):
        for j in range(shape[i]):
            perm[T2[i][j]-1]=intersection[i][j]
    return (Permutation(perm)).sign()

def c_matrix(partition, sigma):
    """
    Compute the matrix C^lambda(sigma) for a partition and permutation sigma.
    Entries are c(T_i, sigma T_j) where c(T_i, T_j) = 1 if intersection is a tableau, 0 otherwise.
    """
    # Get all standard tableaux of the given shape
    ordered_tableaux = (StandardTableaux(partition)).list()
    #ordered_tableaux.reverse()
    # Sort tableaux in last letter order
    n = len(ordered_tableaux)
    
    # Compute matrix entries
    S_n = SymmetricGroup(sum(partition))
    sigma_perm = S_n(sigma)  # Convert sigma to a permutation group element
    
    return matrix([[tableau_intersection(ordered_tableaux[i], (ordered_tableaux[j]).symmetric_group_action_on_entries(Permutation((sigma_perm)))) for j in range(n)] for i in range(n)])

# note in order to have the same conventions as in the Garsia-McLarnan '88 (Thm 1.1), we need to change the order and transpose
# see also Curtis '81 for the same construction
# Boerner used the first letter order instead
def GetYoungNatural(partition,sigma):
    return c_matrix(partition,sigma)*c_matrix(partition,())^-1

# Corresponds to the matrix from acting on the polytabloid basis (James-Kerber)
def GetSpecht(partition,sigma):
    return c_matrix(partition,())^-1*c_matrix(partition,sigma)

# note in order to have the same conventions as in the Garsia McLarnan '88 (Thm 1.1), we need to change the order and transpose
# see also Curtis '81 for the same construction
# we observe that the matrix m below effectively describes the base change between the Specht basis and Young natural basis
def GetNaturalYoungOperators(partition):
    m=c_matrix(partition,())
    n=m.ncols()
    simple_reflections=SymmetricGroup(sum(partition)).simple_reflections()
    act_s_plus_1=[matrix.identity(n)+c_matrix(partition,s)*m^-1 for s in simple_reflections]
    #return [x.transpose() for x in act_s_plus_1]
    return  act_s_plus_1


# Thm 1.4 in GM
# it follows that the matrix m, corresponding to the trivial element, gives the base change between the Specht and Young natural rep. over Z
def GetSpechtOperators(partition):
    m=c_matrix(partition,())
    n=m.ncols()
    simple_reflections=SymmetricGroup(sum(partition)).simple_reflections()
    act_s_plus_1=[matrix.identity(n)+m^-1*c_matrix(partition,s) for s in simple_reflections]
    #return [x.transpose() for x in act_s_plus_1]
    return  act_s_plus_1

def GetSeminormalFormOperators(partition):
    """
    Computes the matrices for (1 + s_i) in Young's Seminormal form (Rational) 
    for a given partition, from first principles.
    The basis is reversed (conjugated by the anti-diagonal identity matrix).
    """
    lam = Partition(partition)
    n = sum(lam)
    syt_list = StandardTableaux(lam).list()
    t_to_idx = {tuple(tuple(r) for r in t): i for i, t in enumerate(syt_list)}
    dim = len(syt_list)
    
    matrices = []

    for k in range(1, n):
        # M will store the matrix for s_k
        M = matrix(QQ, dim, dim)
        
        for i, t in enumerate(syt_list):
            r_k, c_k = -1, -1
            r_kp1, c_kp1 = -1, -1
            
            for r, row in enumerate(t):
                if k in row:
                    r_k, c_k = r, row.index(k)
                if k+1 in row:
                    r_kp1, c_kp1 = r, row.index(k+1)
            
            # Compute Axial Distance d (content diff)
            # content = col - row
            cont_k = c_k - r_k
            cont_kp1 = c_kp1 - r_kp1
            d = cont_kp1 - cont_k
            if d == 1:
                M[i, i] = 1
            elif d == -1:
                M[i, i] = -1
            else:
                t_prime_data = [list(row) for row in t]
                t_prime_data[r_k][c_k] = k+1
                t_prime_data[r_kp1][c_kp1] = k
                t_prime_tuple = tuple(tuple(row) for row in t_prime_data)
                
                j = t_to_idx[t_prime_tuple] 
                if i < j:
                    M[i, i] = 1/d
                    M[j, i] = 1
                else:
                    M[j, i] = 1 - 1/(d**2)
                    M[i, i] = 1/d

        # 4. Compute 1 + s_k
        M_one_plus_s = matrix.identity(dim) + M
        M = M_one_plus_s
        
        matrices.append(M)
    return matrices