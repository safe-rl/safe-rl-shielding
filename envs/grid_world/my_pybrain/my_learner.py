from pybrain3.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain3.rl.learners.valuebased import SARSA
class MyQ(ValueBasedLearner):

    offPolicy = True
    batchMode = True

    def __init__(self, alpha=0.5, gamma=0.99, neg_reward=False):
        ValueBasedLearner.__init__(self)

        self.alpha = alpha
        self.gamma = gamma
        self.neg_reward = neg_reward

        self.laststate = None
        self.lastactions = None
        
    def learn(self):
        """ Learn on the current dataset, either for many timesteps and
            even episodes (batchMode = True) or for a single timestep
            (batchMode = False). Batch mode is possible, because Q-Learning
            is an off-policy method.

            In batchMode, the algorithm goes through all the samples in the
            history and performs an update on each of them. if batchMode is
            False, only the last data sample is considered. The user himself
            has to make sure to keep the dataset consistent with the agent's
            history.
        """
        if self.batchMode:
            samples = self.dataset
        else:
            samples = [[self.dataset.getSample()]]

        for seq in samples:
            # information from the previous episode (sequence)
            # should not influence the training on this episode
            self.laststate = None
            self.lastactions = None
            self.lastreward = None

            for state, actions, reward in seq:
                actions = list(map(int, actions[actions != -1]))
                state = int(state)

                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastactions = actions
                    self.laststate = state
                    self.lastreward = reward
                    continue

                maxnext = self.module.getValue(state, self.module.getMaxAction(state)[0])
                for action in self.lastactions[:-1]:
                    qvalue = self.module.getValue(self.laststate, action)
                    self.module.updateValue(self.laststate, action, qvalue + self.alpha * ((-1 if self.neg_reward else self.lastreward) + self.gamma * maxnext - qvalue))
                action = self.lastactions[-1]
                qvalue = self.module.getValue(self.laststate, action)
                self.module.updateValue(self.laststate, action, qvalue + self.alpha * (self.lastreward + self.gamma * maxnext - qvalue))

                # move state to oldstate
                self.laststate = state
                self.lastactions = actions
                self.lastreward = reward
