import numpy as np
from env_road3 import Env 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from time import time
import itertools
import json

def create_model(env, h_layers, nodes, target_model_update=1e-2, lr = 1e-3, optimizer = Adam):
    model = Sequential()
    nb_actions = env.action_space.n
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    for i in range(h_layers):
        model.add(Dense(nodes))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    
    memory = SequentialMemory(limit=500000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=target_model_update, policy=policy)
    dqn.compile(optimizer(lr=lr), metrics=['mse'])
    return dqn

def fit_test(dqn, env, weight_name):
    score = 0
    fscore = 0
    counter = 0
    steps = 5000
    start=time()

    # .2 reward per step
    while time() - start < 500:
        counter += 1
        train_history = dqn.fit(env, nb_steps=steps, visualize=False, verbose=0)
        end = time()
        env.render()
        test_history = dqn.test(env, nb_episodes=1, visualize=True, verbose=0)
        score = np.mean(test_history.history['episode_reward'])
        print score
        if score > 28000:
            total_time = end - start
            dqn.save_weights('{}_weights_{}.h5f'.format(weight_name,counter), overwrite=True)
            break
        env.render(mode=False)
    else:
        total_time = 999

    return total_time


def main():
    np.random.seed(123)
    env = Env()
    a = [[3],[8,12,16,20],[1e-2],[1e-3]]
    #a = [[3],[16],[1e-3],[1e-2]]
    #a = [[3],[16],[1e-2],[1e-3]]
    parameters = list(itertools.product(*a))
    model_description = {}
    for param in parameters:
        h_layers, nodes, target_model_update, lr = param
        dqn = create_model(env, h_layers, nodes,target_model_update, lr)
        weight_name = 'h_layers: {} nodes: {} target model update: {} lr: {}'.format(h_layers, nodes, target_model_update, lr)
        print "trying {}".format(weight_name) 
        total_time = fit_test(dqn, env, weight_name)
        model_description[weight_name] = total_time
        print json.dumps(model_description, indent=1)    
    return model_description


if __name__ == '__main__':
    model_desc = main()
    print json.dumps(model_desc, indent=1)




