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
    
    # Initialize pi with the uniform policy
    pol = np.ones((len(X), len(A))) / len(A)
    quit = False
    niter = 0
    
    while not quit:
        # Auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))

        # Policy evaluation
        cpi = np.sum(c * pol, axis=1, keepdims=True)
        Ppi = pol[:, 0, None] * P[a]

        J = np.linalg.inv(np.eye(len(X))- gamma * Ppi.dot(cpi))

        # Compute Q values
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)
        
        # Compute greedy policy
        Qmin = np.min(Q, axis=1, keepdims=True)

        pnew = np.isclose(0, Qmin, atol=1e-8, rtol=1e-8).astype(int)
        pnew = pnew / pnew.sum(axis=1, keepdims=True)

        # Compute stopping condition
        quit = (pol == pnew).all()

        # Update
        pol = pnew
        niter += 1
        
        
    print(f'Done after {niter} iterations.')
    return np.round(pol, 3)

# pol = policy_iteration(M)
# print(pol)