import numpy as np
import numpy.random as rand
import time

# Activity 1
def load_mdp(filename, g):

    file = np.load(filename)

    X = file["X"]
    A = file["A"]
    P = file["P"]
    c = file["c"]

    # Define the MDP
    M = (X, A, P, c, g)

    return M

# Activity 2
def noisy_policy(M, a, eps):
    nStates = len(M[0])
    nActions = len(M[1])

    pol = np.ones((nStates, nActions)) * eps / (nActions - 1)
    
    for i in range(nStates):
        pol[i, a] = 1 - eps
        

    return pol

# Activity 3
def evaluate_pol(M, pol):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]

    nStates = len(X)
    nActions = len(A)

    J = np.zeros((nStates, 1))
    
    cpi = np.sum(c * pol, axis=1, keepdims=True)
    Ppi = pol[:, 0, None] * P[0]

    for a in range(1, nActions):
        Ppi += pol[:, a, None] * P[a]

    J = np.linalg.inv(np.eye(nStates) - gamma * Ppi).dot(cpi)

    return J

# Activity 4
def value_iteration(M):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]

    eps = 10e-8
    
    # Initialize J with the size of state-space
    J = np.zeros((len(X), 1))
    err = 1.0
    niter = 0
    start = time.time()

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
    
    elapsedTime = time.time() - start

    print(f'Execution time: {round(elapsedTime, 3)} seconds')
    print(f'N. iterations: {niter}')
    return J

# Activity 5
def policy_iteration(M):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]
    
    # Initialize pi with the uniform policy
    pol = np.ones((len(X), len(A))) / len(A)
    quit = False
    niter = 0
    start = time.time() 
    
    while not quit:
        # Auxiliary array to store intermediate values
        Q = np.zeros((len(X), len(A)))

        # Policy evaluation
        cpi = np.sum(c * pol, axis=1, keepdims=True)
        Ppi = pol[:, 0, None] * P[0]

        for a in range(1, len(A)):
            Ppi += pol[:, a, None] * P[a]

        J = np.linalg.inv(np.eye(len(X)) - gamma * Ppi).dot(cpi)

        # Compute Q values
        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)
        
        # Compute greedy policy
        Qmin = np.min(Q, axis=1, keepdims=True)

        pnew = np.isclose(Q, Qmin, atol=1e-8, rtol=1e-8).astype(int)
        pnew = pnew / pnew.sum(axis=1, keepdims=True)

        # Compute stopping condition
        quit = (pol == pnew).all()

        # Update
        pol = pnew
        niter += 1
    
    elapsedTime = time.time() - start
        
    print(f'Execution time: {round(elapsedTime, 3)} seconds')
    print(f'N. iterations: {niter}')
    return pol

# Activity 6
NRUNS = 100 # Do not delete this

def simulate(M, pol, x0, length):
    X = M[0]
    A = M[1]
    P = M[2]
    c = M[3]
    gamma = M[4]
    
    trajectories = np.zeros(NRUNS)
    
    # For each run
    for i in range(len(trajectories)):
        x = x0

        # Setup first action, cost to go and next state
        action = rand.choice(len(A), 1, p=pol[x, :])
        costToGo = c[x, action]
        x = rand.choice(len(X), 1, p=P[action][0][x])

        # For each following step in the run
        for step in range(1, length):
            action = rand.choice(len(A), 1, p=pol[x, :][0])
            cost = c[x, action]
            x = rand.choice(len(X), 1, p=P[action][0][x][0])
    
            costToGo += np.power(gamma, step) * cost

        trajectories[i] = costToGo
 
    return np.mean(trajectories)
