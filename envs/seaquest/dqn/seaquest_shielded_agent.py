from .agent import Agent
from .seaquest import Shield
from numpy import where, append
import math

ACTION_MAP = { 0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8 }
REVERSE_ACTION_MAP = { 0: {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9 },
                        1: {0: 1, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17} }
def FIRE(x): return x in [1] + range(10, 18)
 
class ShieldedAgent(Agent):
    def __init__(self, config, environment, sess):
        Agent.__init__(self, config, environment, sess)
        self.shield = Shield()
        self.last_frame = None
        self.punish = config.negative_reward
        self.modified = False
        
    def observe(self, screen, reward, action, terminal):
        reward = -1 if (self.punish and self.modified) else reward
        Agent.observe(self, screen, reward, action, terminal)
        
    def predict(self, s_t, test_ep=None):
        agent_action = Agent.predict(self, s_t, test_ep)
            
            # 0 - standing still
            # 1 - standing still & fire
            # 2 - move up
            # 3 - move right
            # 4 - move left
            # 5 - move down
            # 6 - move right up
            # 7 - move left up
            # 8 - move right down
            # 9 - move left down
            #10 - move up & fire
            #11 - move right & fire
            #12 - move left & fire
            #13 - move down & fire
            #14 - move right up & fire
            #15 - move left up & fire
            #16 - move right down & fire
            #17 - move left down & fire
                
        binary_signals = self.process_input(self.env._observation)
 
        fire = FIRE(agent_action)
        action = ACTION_MAP[agent_action]
        binary_signals.extend(map(int, list(bin(action)[2:].rjust(4, '0'))))

        binary_signals = self.shield.tick(binary_signals)

        action = int("".join(map(str, binary_signals[:len(binary_signals) -1])), 2)
        action = REVERSE_ACTION_MAP[fire][action]
        
        self.modified = agent_action != action
        
        return action    
    
    # extract needed information from RAM. Could also be done from image, but that's more complicated/complex/work
    def process_input(self, observation):
        if self.last_frame != None:
            k = self.env.env.ale.getFrameNumber() - self.last_frame
        else:
            k = 1
        
        self.last_frame = self.env.env.ale.getFrameNumber()
                    
        k_enc = map(int, list(bin(k - 1)[2:].rjust(2, '0')))
        
        ram = self.env.env._get_ram()

        
        depth_enc = map(int, list(bin(ram[97] - 13)[2:].rjust(7, '0')))
        
        oxy_full = 1 if ram[102] == 64 else 0
        oxy_low = 1 if ram[102] <= 4 else 0
        diver_found = 1 if ram[62] > 0 else 0
        
        binary = []
        binary.extend(depth_enc)
        binary.extend(k_enc)
        binary.append(oxy_low)
        binary.append(oxy_full)
        binary.append(diver_found)
        return binary

        
