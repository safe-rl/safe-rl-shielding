# I use an e-greedy policy to select actions based on estimate state action values.
# e-greedy takes a random action with probability 'epsilson' so as to encourage the
# agent to explore the state action space and otherwise it takes the greedy action

import numpy as np
import random as rnd

class eGreedy:
    
    def __init__(self, eExplore = 0.1, eExploit = 0.05, decay=0.0001):
        self.epsilon = eExplore
        self.eExploit = eExploit
        self.decay = decay
        
    def enact(self, actions, values):
        # initialise the action to the greedy choice
        action = actions[np.argmax(values)]
        
        # but if we happen to want to explore, return a random action
        if rnd.random() < self.epsilon:
            action = np.random.choice(actions)
        
        # this let's the policy move from exploration to exploitation as the agent better understands the environment
        self.epsilon -= (self.epsilon - self.eExploit) * self.decay
        
        return action

