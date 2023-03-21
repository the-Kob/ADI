import numpy as np

# X -> States
# A -> Actions
# P -> Transition probabilities

# M = (X, A, P, c, 0.99)

def value_iteration(M, eps):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]
    
    # Initialize J with the size of state-space
    J = np.zeros((len(X), 1))
    err = 1.0
    niter = 0
    
    while err > eps:
        # Auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))
        
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)
        
        # Compute minimum row-wise
        Jnew = np.min(Q, axis=1, keepdims=True)
        
        # Compute error
        err = np.linalg.norm(J - Jnew)
        
        # Update
        J = Jnew
        niter += 1
        
    print(f'Done after {niter} iterations.')
    return np.round(J, 3)

# J = value_iteration(M, le-8)
# print(J)
            