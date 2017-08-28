# I use a densely connected neural network with two hidden layers to model the Q-function. 
# I've set up the network with as many output nodes as there are actions availabile to the agent. That way
# we only need one pass through the network to calculate the maximum state action value which is required for Q-learning

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

class deepQNetwork:
    
    def __init__(self, learningRate, noOfStateVariables, noOfActions):
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=noOfStateVariables))
        self.model.add(Activation('relu'))
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(noOfActions,activation='linear')) 
        self.model.compile(lr=learningRate, optimizer='rmsprop', loss='mse')

    def predict(self, states):        
        return self.model.predict(states,batch_size=1)

    def predict_all(self, states):        
        return self.model.predict(states, batch_size=32)
    
    def fit(self, states, targets):
        self.model.fit(states, targets, verbose=False)
