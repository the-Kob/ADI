# #### Activity 1.        

import numpy as np
import numpy.random as rand

# Insert your code here.
def load_pomdp(filename, gamma):
    
    file = np.load(filename)

    X = tuple(file["X"]) # States
    A = tuple(file["A"]) # Actions
    Z = tuple(file["Z"]) # Observations
    P = tuple(file["P"]) # Transition probability matrix
    O = tuple(file["O"]) # Observation probability matrix
    c = file["c"] # Cost function
    
    # Define the MDP
    M = (X, A, Z, P, O, c, gamma)

    return M

# #### Activity 2.

def gen_trajectory(M, x0, n):

    X = M[0]
    A = M[1]
    Z = M[2]
    P = M[3] 
    O = M[4]
    
    # Setup output arrays
    traj = np.zeros((n + 1), dtype=int)
    actions = np.zeros((n), dtype=int)
    observations = np.zeros((n), dtype=int)

    x = x0
    traj[0] = x

    for i in range(0, n):
        # Choose action
        actions[i] = np.random.choice(len(A))

        # Get resulting state 
        x = np.random.choice(len(X), p=P[actions[i]][x, :])
        traj[i + 1] = x

        # Get resulting observation
        observations[i] = np.random.choice(len(Z), p=O[actions[i]][x, :])
        
    output = (traj, actions, observations)

    return output

# #### Activity 3.

def belief_update(P, O, belief, action, obs):
    belief = belief.dot(P[action].dot(np.diag(O[action][:, obs])))
    updated_belief = belief / np.linalg.norm(belief)

    return updated_belief

def sample_beliefs(M, n):
    X = M[0]
    P = M[3] 
    O = M[4]

    # Choose random initial state
    x0 = rand.randint(0, len(X))

    # Generate trajectory with previous function
    T = gen_trajectory(M, x0, n)

    actions = T[1]
    z = T[2]

    # Uniform initial belief distribution
    init_distr = np.ones((1, len(z))) / len(z)
    mu = init_distr

    beliefs = ()

    for i in range(0, len(z)):
       
       chosen_action = actions[i]
       current_obs = z[i]
       mu = belief_update(P, O, mu, chosen_action, current_obs)

       # If there aren't any duplicates, append belief state
       for j in beliefs:
           if(np.linalg.norm(mu - j) > 1e-3):
            beliefs.append(mu)

    return beliefs

# #### Activity 4
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
        
    return Q

def solve_mdp(M):

    X = M[0]
    A = M[1]
    Z = M[2]
    P = M[3] 
    O = M[4]
    c = M[5]
    gamma = M[6]

    underlying_M = (X, A, P, c, gamma)

    Q = value_iteration(underlying_M, 1e-8)

    return Q
 
# #### Activity 5
import numpy as np

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

    return pol


# Insert your code here.
def get_heuristic_action(M, belief, Q, heuristic):

    if(heuristic == "mls"):
        X = M[0]
        A = M[1]
        Z = M[2]
        P = M[3] 
        O = M[4]
        c = M[5]
        gamma = M[6]

        underlying_M = (X, A, P, c, gamma)

        pol = policy_iteration(underlying_M)
        best_action = pol[np.argmax(belief)]
        return best_action

    elif(heuristic == "av"):
        X = M[0]
        A = M[1]
        Z = M[2]
        P = M[3] 
        O = M[4]
        c = M[5]
        gamma = M[6]

        underlying_M = (X, A, P, c, gamma)
        pol = policy_iteration(underlying_M)
        I = np.diag(pol)

        best_action = np.argmax(np.sum(belief).dot(I))
        return best_action

    elif(heuristic == "q-mdp"):
        best_action = np.argmin(np.sum(belief.dot(Q)))
        return best_action


# #### Activity 6

def solve_fib(M):
    X = M[0]
    A = M[1]
    Z = M[2]
    P = M[3] 
    O = M[4]
    c = M[5]
    gamma = M[6]

    eps = 10e-1
    
    # Initialize Q
    Q = np.zeros((len(X), len(A)))
    newQ
    err = 1.0

    while err > eps:
        
        for a in range(len(A)):
            newQ = np.zeros((len(X), len(A)))
            inner_sum = np.sum(P[a].dot(O[a]).dot(Q[:, a, None]))
            newQ[:, a, None] = c[:, a, None] + gamma * np.sum(np.argmin(np.sum(inner_sum)))
        
        # Compute error
        err = np.linalg.norm(Q - newQ)
        
        # Update
        Q = newQ

    return Q

# Answer : The FIB Q-function is much more "conservative" in terms of actions, in relation to all the other heuristics. In other words, it makes sure to take the most "secure path" to secure its goal, like we saw with the Tiger problem in theoretical classes

