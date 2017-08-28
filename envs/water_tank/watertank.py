#!/usr/bin/python
#
# Simulates an MDP-Strategy
# ---> Water tank example.

import math
import os
import sys, code
import resource
import copy
import itertools
import random
from PIL import Image
import os, pygame, pygame.locals
from pybrain.rl.environments import Environment
from pybrain.rl.environments import Task
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from my_pybrain.my_explorer import MyUCBExplorer
from my_pybrain.my_explorer import MyGreedyExplorer
from my_pybrain.my_table import MyActionValueTable
from my_pybrain.my_learner import MyQ, SARSA
from pybrain.utilities import abstractMethod
import numpy as np
from itertools import product
from copy import deepcopy
from scipy import argmax
from scipy import where
from random import choice
import importlib
from time import sleep
np.set_printoptions(threshold=np.inf)

import argparse

parser = argparse.ArgumentParser(description='Simulator')
parser.add_argument("-c", "--collect-data", dest="collect_data_file", help="Provide a file for collecting convergence data")
parser.add_argument('-g', "--gen-spec", dest='gen_spec', help='Generate shield files', action='store_true', default=False)
parser.add_argument('-l', "--load", dest='load_file', help='Load Q-Table from file')
parser.add_argument('-s', "--save", dest='save_file', help='Save Q-Table to file')
parser.add_argument('-t', "--train", dest='train', help='Training activated', type=float, default=.2)
parser.add_argument('-o', "--shield_options", dest='shield_options', help='Number of actions the shield can choose of. 0 disables the shield', type=int, default=0)
parser.add_argument('-n', "--negative-reward", dest='neg_reward', help='Indicated whether negative reward should be used for unsafe actions', action='store_true', default=False)
parser.add_argument('-p', "--huge-negative-reward", dest='huge_neg_reward', help='Indicated whether a huge negative reward should be used for unsafe actions', action='store_true', default=False)
parser.add_argument('-r', "--sarsa", dest='sarsa', help='Indicated whether to use SARSA or default Q-learning', action='store_true', default=False)
parser.add_argument("--num-steps", dest='num_steps', help='Number of interactions', type=int, default=1000000)

args = parser.parse_args()

collect_data_file = args.collect_data_file
gen_spec = args.gen_spec
shield_options = args.shield_options
load_file = args.load_file
save_file = args.save_file
exploration = args.train
neg_reward = args.neg_reward
huge_neg_reward = args.huge_neg_reward
MAX_STEPS = args.num_steps
interval = 100

if shield_options > 0:
    try:
        mod_name = "watertank_shield"
        Shield = importlib.import_module(mod_name.replace(os.path.sep, ".")).Shield
    except ImportError as e:
        print ("Could not find file " + pngFileBasis + "_" + str(shield_options) + ".py")
        print (e)
        exit()

else:
    from no_shield import Shield

# ==================================
# Construct MDP --> States
# ==================================
stateMapper = {}
for xA in range(1,100):
    for switchState in [-3,-2,-1,0,1,2,3]:
        stateNum = len(stateMapper)
        stateMapper[(xA,switchState)] = stateNum

# Add error state
errorState = len(stateMapper)
errorStateKey = (-1,0)
stateMapper[errorStateKey] = errorState

# Simplify lookup for building the transition list,
# but build reverse state mapper first
reverseStateMapper = {}
for (a,b) in stateMapper.items():
    reverseStateMapper[b] = a
for xA in [0,-1,-2,-3,100,101,102,103]:
    for switchState in [-3,-2,-1,0,1,2,3]:
        stateMapper[(xA,switchState)] = errorState

# ==================================
# Construct MDP --> Transition list
# ==================================

# Error State -- No actions possible
transitions = []
transitions.append([errorState,0,errorState,1.0])
transitions.append([errorState,1,errorState,1.0])

for xA in range(1,100):
    for switchState in [-3,-2,-1,1,2,3]:
        currentState = stateMapper[(xA,switchState)]
        outFlow0Probability = (math.sin(xA/12.345)+1.0)*0.35
        inFlow1Probability = (math.sin(xA/18+1.2345)+1.0)*0.45

        total = (outFlow0Probability*(1.0-inFlow1Probability)) \
        + (outFlow0Probability*(1.0-inFlow1Probability)+(1.0-outFlow0Probability)*inFlow1Probability)\
        + ((1.0-outFlow0Probability)*inFlow1Probability)

        normfac = 1.2/total
       
        # Closed Valve transitions
        if switchState > 1:
            transitions.append([currentState,0,errorState,1.0])
        elif switchState==1:
            transitions.append([currentState,0,stateMapper[(xA,-3)],outFlow0Probability])
            transitions.append([currentState,0,stateMapper[(xA-1,-3)],1.0-outFlow0Probability])
        else:
            transitions.append([currentState,0,stateMapper[(xA,min(-1,switchState+1))],outFlow0Probability])
            transitions.append([currentState,0,stateMapper[(xA-1,min(-1,switchState+1))],1.0-outFlow0Probability])
        
        # Open Valve transitions
        if switchState < -1:
            transitions.append([currentState,True,errorState,1.0])
        elif switchState==-1:
            transitions.append([currentState,True,stateMapper[(xA+2,3)],outFlow0Probability*(1.0-inFlow1Probability)])
            transitions.append([currentState,True,stateMapper[(xA+1,3)],outFlow0Probability*(inFlow1Probability)+(1.0-outFlow0Probability)*(1.0-inFlow1Probability)])
            transitions.append([currentState,True,stateMapper[(xA+0,3)],(1.0-outFlow0Probability)*inFlow1Probability])
        else:
            transitions.append([currentState,True,stateMapper[(xA+2,max(1,switchState-1))],outFlow0Probability*(1.0-inFlow1Probability)])
            transitions.append([currentState,True,stateMapper[(xA+1,max(1,switchState-1))],outFlow0Probability*(inFlow1Probability)+(1.0-outFlow0Probability)*(1.0-inFlow1Probability)])
            transitions.append([currentState,True,stateMapper[(xA+0,max(1,switchState-1))],(1.0-outFlow0Probability)*inFlow1Probability])


# Now define the reward -- it is only dependent on the state in this scenario (whether it is the state before or 
# after a transition doesn't really matter)
normlist = []
for xA in range(0,101):
    normlist.append(-1*xA*(1+math.sin(xA*0.4+0.5)*0.95))

norm_max = max(normlist)
norm_min = min(normlist)


stateToRewardMapper = {errorState:0.0}
for xA in range(0,101):
    for switchState in [-3,-2,-1,1,2,3]:
        stateToRewardMapper[stateMapper[(xA,switchState)]] = (2 * ((-1*xA*(1+math.sin(xA*0.4+0.5)*0.95)) - norm_min)/(norm_max - norm_min)) - 1

transitionLists = {}
for (a,b,c,d) in transitions:
    if not (a,b) in transitionLists:
        transitionLists[(a,b)] = [(c,d)]
    else:
        transitionLists[(a,b)].append((c,d))

class Map(Environment):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = stateMapper[(50,1)]
        self.penalty = 0
        
    def performAction(self, action):  
        error = len(reverseStateMapper) - 1       
        actions = action[action != -1]
        actions = list(map(int, actions))
        encoded_actions = []

        for a in actions:
            encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(3, '0')))))
        (xA,switchState) = reverseStateMapper[self.state]
        payoff = stateToRewardMapper[stateMapper[(xA,switchState)]]
        
        
        corr_action = shield.tick(reverseStateMapper[self.state][0],reverseStateMapper[self.state][1],actions[0])
        used_actions = []
        for a in actions:
            if a == corr_action: break
            used_actions.append(a)
                
        agent.lastaction = corr_action
        transitionList = transitionLists[(self.state, corr_action)]
        
        dest = None
        randomNumber = random.random()
        for (a,b) in transitionList:
            if randomNumber<=b:
                dest = a
                break
            else:
                randomNumber -= b

        if (dest==None):
            print ("Rounding error?")
            dest = transitionList[0][0]
        # print reverseStateMapper[dest]
        if dest == errorState: 
            experiment.acc_reward -= 1
            # self.penalty += 1
            if shield_options > 0 and not args.huge_neg_reward:
                print ("Shields are not allowed to make errors!")
                exit()

        
        self.state = dest
            
    def getSensors(self):
        return [self.state]
        
class MaintainLevel(Task):
    def __init__(self, env):
        Task.__init__(self, env)
        self.last_reward = 0
        
    def getReward(self):
        ret = self.last_reward
        self.last_reward = stateToRewardMapper[self.env.state] - self.env.penalty
        self.env.penalty = 0
        return self.last_reward

class MyExperiment(Experiment):
    def __init__(self, task, agent):
        Experiment.__init__(self, task, agent)
        
        agent.learner.explorer.experiment = self
    
        self.isPaused = False
        self.isCrashed = False
        self.speed = 10
        self.num = 0
        self.XA = 50
        self.switch_state = 1
        
        self.count = 0
        self.acc_reward = 0
        self.collect_data = False
        if collect_data_file != None:
            self.collect_data = True
            self.collect_episode_data_file = open(collect_data_file + "_episodelen.data", "w")
            self.collect_reward_data_file = open(collect_data_file + "_avg_reward.data", "w")
    
    def _oneInteraction(self):
        
        resetInThisRound = False
                
        old = (self.XA, self.switch_state)
        (self.XA,self.switch_state) = reverseStateMapper[level.state]
        payoff = stateToRewardMapper[level.state]

        self.acc_reward += payoff * 10
        if self.collect_data:
            self.count += 1
            if payoff > 0:
                self.collect_episode_data_file.write(str(self.count) + "\n")
                self.count = 0
            if self.stepid % interval == 0:
                self.collect_reward_data_file.write(str(self.acc_reward / float(interval)) + "\n")
                self.acc_reward = 0
            if self.stepid % 100000 == 0:
                pass
        
        if self.stepid % interval == 0:
            sys.stdout.write("\033[K")
            sys.stdout.write("[{2}{3}] ({0}/{1}) | alpha = {4} | epsilon = {5}\n".format(self.stepid, MAX_STEPS, '#'*int(math.floor(self.stepid/float(MAX_STEPS)*20)), ' '*int((20 - math.floor(self.stepid/float(MAX_STEPS)*20))), learner.alpha, learner.explorer.exploration))
            sys.stdout.write("\033[F")
            
        if self.stepid >= MAX_STEPS:
            print ("\nSimulation done!")
            
            sys.exit()          
            
        if payoff > 0:
            # episode done
            if save_file != None:
                controller.params.reshape(controller.numRows, controller.numColumns).tofile(save_file)
            learner.alpha *= 0.999999
            learner.explorer.exploration *= 0.999999
        if level.state == errorState:
            level.reset()

        self.isCrashed = False
        if not self.isPaused:
            return Experiment._oneInteraction(self)
        else: return self.stepid


shield = Shield()
level = Map()
task = MaintainLevel(level)
controller = MyActionValueTable(len(reverseStateMapper), 2)
if load_file != None:
    controller.initialize(np.fromfile(load_file))
else:
    controller.initialize(0.)
alpha = .5
gamma = .99
if not args.sarsa:
    learner = MyQ(alpha, gamma, neg_reward)
    learner.explorer = MyGreedyExplorer(shield_options, exploration)
elif args.sarsa:
    learner = SARSA(alpha, gamma)
learner.explorer = MyGreedyExplorer(shield_options, exploration)
learner.explorer._setModule(controller)
agent = LearningAgent(controller, learner)

draw = False
EXPLORATION_FACTOR = exploration

experiment = MyExperiment(task, agent)  
while 1:
    experiment.doInteractions(interval)
    
    agent.learn()
    agent.reset()