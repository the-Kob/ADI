# #### Activity 1.        

import numpy as np
import numpy.random as rnd

def sample_transition(M, s, a):
    X = M[0] 
    P = M[2]
    C = M[3]
    
    c = C[s, a]

    s_prime = rnd.choice(len(X), p=P[a][s, :])

    output = (s, a, c, s_prime)

    return output

# #### Activity 2.        

def egreedy(Q, eps=0.1):
    if(rnd.random() < eps):
        N = len(Q)
        chosenIndex = rnd.choice(N)
    else:
        # To get all the indices of the min value
        mins = np.flatnonzero(np.isclose(Q, Q.min()))
        chosenIndex = rnd.choice(mins)
        
    return chosenIndex

# #### Activity 3. 

def mb_learning(mdp, n, qinit, Pinit, cinit):
    X = mdp[0] 
    A = mdp[1]
    gamma = mdp[4]

    P = Pinit
    C = cinit
    Q = qinit

    visits = np.zeros((len(X), len(A))) # tracker of n visits for each pair (s, a)

    s = rnd.choice(len(X)) # initial random state

    for step in range(n):
        a = egreedy(Q[s, :], eps=0.15)
        s, a, cnew, snew = sample_transition(mdp, s, a)
        
        # Increment number of visits
        visits[s, a] += 1
        stepSize = 1 / (visits[s, a] + 1)

        C[s, a] = C[s, a] + stepSize * (cnew - C[s, a])

        P[a][s, :] = P[a][s, :] + stepSize * (- P[a][s, :]) # Update all the states
        P[a][s, snew] = P[a][s, snew] + stepSize # Update the specific state
        
        Q[s, a] = C[s, a] + gamma * np.sum(P[a][s, :] * Q.min(axis=1))
        
        s = snew

    output = (Q, P, C)

    return output

# #### Activity 4. 

def qlearning(mdp, n, qinit):
    X = mdp[0] 
    gamma = mdp[4]

    stepSize = 0.3
    Q = qinit

    # Arbitrary state chosen at random
    s = rnd.choice(len(X))

    for step in range(n):
        a = egreedy(Q[s, :], eps=0.15)
        s, a, cnew, snew = sample_transition(mdp, s, a)

        # Q-learning update
        Q[s, a] = Q[s, a] + stepSize * (cnew + gamma * Q[snew].min() - Q[s, a]) 

        s = snew

    return Q

# #### Activity 5. 

def sarsa(mdp, n, qinit):
    X = mdp[0] 
    gamma = mdp[4]

    stepSize = 0.3
    Q = qinit

    # Arbitrary state chosen at random
    s = rnd.choice(len(X))
    a = egreedy(Q[s, :], eps=0.15)

    for step in range(n):
        s, a, cnew, snew = sample_transition(mdp, s, a)

        # SARSA update
        anew = egreedy(Q[snew, :], eps=0.15) # calculate next action aswell
        Q[s, a] = Q[s, a] + stepSize * (cnew + gamma * Q[snew, anew] - Q[s, a]) 

        s = snew
        a = anew

    return Q

# #### Activity 6.

# Answer: With an increase in the number of iterations, we can verify a general decrease in the error associated with the Q-function for all the algorithms above.
# However, one can notice a clearly better performance in the Model-based algorithm - the Q-function error decreases much faster compared to the other algorithms.
# The SARSA and Q-learning plots are similar, only being noticeable a small difference when the number of iterations reaches a considerable value (around 0.5e6 iterations).
# Considering we are applying a greedy exploration mechanism, we can state that both the SARSA (on-policy) and Q-learning (off-policy) algorithms are learning the optimal policy, the former having more stable updates.
# That being said, if we were to adopt a model-free learning process, we should opt for SARSA, in this specific case. On the other hand, if model-based learning is an available option, we should instead opt for it.

