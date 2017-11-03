from pybrain3.rl.learners.valuebased.interface import ActionValueInterface
from pybrain3.structure.modules import Table, Module
from pybrain3.structure.parametercontainer import ParameterContainer
from scipy import where
from random import choice
import numpy as np

class MyActionValueTable(Table, ActionValueInterface):
    def __init__(self, numStates, numActions, name=None):
        Module.__init__(self, 1, 5, name)
        ParameterContainer.__init__(self, numStates * numActions)
        self.numRows = numStates
        self.numColumns = numActions

    @property
    def numActions(self):
        return self.numColumns

    def _forwardImplementation(self, inbuf, outbuf):
        """ Take a vector of length 1 (the state coordinate) and return
            the action with the maximum value over all actions for this state.
        """
        outbuf[:] = self.getMaxAction(inbuf[0])

    def getMaxAction(self, state):
        """ Return the action with the maximal value for the given state. """
        values = self.params.reshape(self.numRows, self.numColumns)[int(state), :].flatten()
       
        actions = []
        for i in range(self.outdim):
            action = where(values == max(values))[0]
            action = choice(action)
            np.delete(values, action)
            actions.append(action)
        return actions

    def getActionValues(self, state):
        if isinstance(state,list):
            return self.params.reshape(self.numRows, self.numColumns)[list(map(int,state)), :].flatten()
        else:
            return self.params.reshape(self.numRows, self.numColumns)[state, :].flatten()


    def initialize(self, value=0.0):
        """ Initialize the whole table with the given value. """
        self._params[:] = value
