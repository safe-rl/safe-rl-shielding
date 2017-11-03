from pybrain3.rl.explorers.explorer import Explorer
from copy import deepcopy
from scipy import where
from random import choice, random
from math import sqrt, log
import numpy as np

class MyUCBExplorer(Explorer):
    def __init__(self, shield_options, exploration):
        Explorer.__init__(self, 5, 5)
        self.exploration = exploration
        self.shield_options = max(1, shield_options)
    
    def _setModule(self, module):
        """ Tell the explorer the module. """
        self._module = module
        # copy the original module for exploration
        self.n_values = deepcopy(module)
        self.n_values._params[:] = 0
        
    def activate(self, state, action):
        """ Save the current state for state-dependent exploration. """
        self.state = state
        return Explorer.activate(self, state, action)
    
    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module
                
        values = self.module.getActionValues(self.state)        
        n_values = self.n_values.getActionValues(self.state)
        values = map(lambda x, y: x + self.exploration * (sqrt(2 * log(self.experiment.stepid, 2) / y) if y > 0 else 1000), values, n_values);
        
        actions = []
        for i in range(self.shield_options):
            new_action = where(values == max(values))[0]
            new_action = choice(new_action) 
            values[new_action] = -10000
            actions.append(new_action)
        
        while len(actions) < self.outdim:
            actions.append(-1)
            
        outbuf[:] = actions
        
class MyGreedyExplorer(Explorer):
    def __init__(self, shield_options, exploration):
        Explorer.__init__(self, 5, 5)
        self.exploration = exploration
        self.shield_options = max(1, shield_options)
    
    def _setModule(self, module):
        """ Tell the explorer the module. """
        self._module = module

    def activate(self, state, action):
        """ Save the current state for state-dependent exploration. """
        self.state = state
        return Explorer.activate(self, state, action)
    
    def _forwardImplementation(self, inbuf, outbuf):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module
                
        values = self.module.getActionValues(self.state)        
        
        actions = []
        if random() <= self.exploration:
            for i in range(self.shield_options):
                new_action = choice(range(len(values))) 
                np.delete(values, new_action)
                actions.append(new_action)
        else:
            for i in range(self.shield_options):
                new_action = where(values == max(values))[0]
                new_action = choice(new_action) 
                np.delete(values, new_action)
                actions.append(new_action)
        
        while len(actions) < self.outdim:
            actions.append(-1)
            
        outbuf[:] = actions
        
