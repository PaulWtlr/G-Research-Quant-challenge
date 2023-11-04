import numpy as np
from scipy.optimize import linprog
from scipy.stats import bernoulli
from scipy.sparse import lil_matrix

def find_optimal_transport(C, G, P):
    
    N = G.shape[0]
    EXP = 28.4
    # Vectorized Bernoulli sampling
    upper_triangle_indices = np.triu_indices(N, 1)
    bernoulli_matrix = np.zeros((N, N))
    bernoulli_matrix[upper_triangle_indices] = bernoulli.rvs(P[upper_triangle_indices])
    bernoulli_matrix += bernoulli_matrix.T
    np.fill_diagonal(bernoulli_matrix, bernoulli.rvs(np.diag(P)))
    # Profit calculation optimized
    pi = (C[:, [0]].T - C[:, [1]] - G)
    for i in range(N):
        for j in range(N):
            pi[i][j] *= (1 - P[i][j])**(2 + 5*G[i][j])

    print(pi)
    # Objective function
    objective_vector = -pi.flatten()
    
    # Initialize the upper-bound matrices as LIL-format sparse matrices
    A_ub_supply = lil_matrix((N, N**2))
    A_ub_demand = lil_matrix((N, N**2))

    # Construct the sparse matrices as before
    for i in range(N):
        A_ub_supply[i, i*N:(i+1)*N] = 1

    for j in range(N):
        A_ub_demand[:, j*N:(j+1)*N] = np.eye(N)

    # Convert to CSR format before vstack, which is more efficient for arithmetic operations
    A_ub_supply = A_ub_supply.tocsr()
    A_ub_demand = A_ub_demand.tocsr()

    # Stack the two constraints (scipy v1.3.0 and above support vstack for csr_matrices)
    from scipy.sparse import vstack
    A_ub = vstack([A_ub_supply, A_ub_demand])
    print(A_ub)

    
    # Base vector
    b_ub = np.concatenate((C[:, 3], C[:, 2]))
    # Linear programming optimization
    result = linprog(objective_vector, A_ub=A_ub, b_ub=b_ub, method='highs')
    # Instructions from the result
    x = result.x
    threshold = 1e-5  # Threshold to consider a float essentially 'zero'
    instructions = [[i // N, i % N, int(x[i])] for i in range(len(x)) if x[i] > threshold]
    
    return np.array(instructions)
