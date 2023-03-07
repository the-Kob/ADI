
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plot

# Activity 1      

def load_chain(filename):
    trans_matrix = np.load(filename)
    states = []
    counter = 0
    for i in trans_matrix[0]:
        states.append(str(counter))
        counter += 1
    
    output = (states, trans_matrix)

    return output

# Activity 2

def prob_trajectory(markov_chain, trajectory):
    trans_matrix = markov_chain[1]
    
    resulting_prob = 1

    for i in range(len(trajectory)):
        if i == len(trajectory) - 1: continue # skips last element in trajectory

        resulting_prob *= trans_matrix[int(trajectory[i])][int(trajectory[i + 1])]
        

    return resulting_prob

# Activity 3

def stationary_dist(markov_chain):
  P = markov_chain[1]

  # We want to solve (I - P).T * mu.T = 0 <-> A * mu = b
  A = (np.identity(len(markov_chain[0])) - P).T

  # We need to add a row of ones to A and a last one 1 to b to make mu1 + mu2 + ... = 1
  A[-1,:] = np.ones(len(markov_chain[0]))
  b = np.zeros(len(markov_chain[0]))
  b[-1] = 1

  pi = np.linalg.solve(A, b)

  return pi

# Answer
# The stationary distribution tells us, over a long period of time, the probability of the truck being in a given stop. 
# That being said, it can also be used to predict where the truck will spend most of its time - the bigger the probability in the stationary distribution, the more time the truck spends in the corresponding stop. 
# This is similar to how the Google's algorithm, PageRank, works.

# Activity 4

def compute_dist(markov_chain, init_dist, num_steps):
    exp_markov_chain = np.linalg.matrix_power(markov_chain[1], num_steps)

    return np.matmul(init_dist, exp_markov_chain)

# Answer
# A chain is ergodic if, being irreducible and aperiodic, reaches its stationary distribution after enough time steps and for any given initial distribution. 
# Like the markov chain described in the homework, we can easily infer this chain is also irreducible and aperiodic (for the reasons stated in the homework) - it is, therefore, a positive chain. 
# This exercises allows to use random initial distributions to prove that, after enough time steps (in this case, around 2000 steps seem to be enough), the chain will always reach a stationary distribution, confirmed by verifying that u * P^2000 = u* is always true. 

# Activity 5

def simulate(markov_chain, init_dist, num_steps):
    trajectory = []
    next_dist = init_dist[0]
    
    # Go through number of steps, choose each one and add them to trajectory
    for i in range(num_steps):
        curr_state = rand.choice(len(markov_chain[0]), 1, p=next_dist)
        trajectory.append(str(curr_state[0]))
        next_dist = markov_chain[1][curr_state[0]] # update distribution
        
    return trajectory

# Activity 6

M = load_chain('garbage-big.npy')

# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

np.random.seed(42)

# Simulate trajectory
traj = simulate(M, u, 50000)
traj = list(map(int,traj))

# Get stationary distribution
u_star = stationary_dist(M)


plot.hist(traj, range(0, nS + 1), density=True, edgecolor = 'black', align='left') # empirical distribution
plot.scatter(range(0, nS), u_star, color='red') # theoretical distribution

plot.show()