import numpy as np
from env_road3 import Env 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from datetime import datetime
from safelearn2 import Shield
import pickle
import argparse



np.random.seed(123)
parser = argparse.ArgumentParser(description='Car')
parser.add_argument("-c", "--collect-data", dest="collect_data_file", help="Provide a file for collecting convergence data")
parser.add_argument('-s', "--shield_options", dest='shield', help='Indicate whether a post-posed shield should be used or not', action='store_true', default=False)
parser.add_argument('-v', "--viz_options", dest='viz', help='Indicate whether to visualize or not', action='store_true', default=False)
parser.add_argument('-vv', "--vizviz_options", dest='all', help='Indicate whether to visualize training or not', action='store_true', default=False)
parser.add_argument('-n', "--negative-reward", dest='neg_reward', help='Indicated whether negative reward should be used for unsafe actions', action='store_true', default=False)
parser.add_argument('-m', "--manual", dest='manual', help='Indicated whether input is from user or agent', action='store_true', default=False)
parser.add_argument('-r', "--sarsa", dest='sarsa', help='Indicated whether to use SARSA or default Q-learning', action='store_true', default=False)
parser.add_argument('-bn', "--bigneg", dest='big_neg', help='Indicated whether to use big negative', action='store_true', default=False)
parser.add_argument("--num-steps", dest='num_steps', help='Number of interactions', type=int, default=20)
parser.add_argument('-p', "--pre", dest='preemptive', help='Indicated whether to use a preemptive shield or not', action='store_true', default=False)

nb_actions = 8
observation_space = (4,)
model = Sequential()
model.add(Flatten(input_shape=(1,) + observation_space))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print ("model initiated")
memory = SequentialMemory(limit=500000, window_length=1)
policy = BoltzmannQPolicy()

args = parser.parse_args()
huge_neg = False

if args.shield and args.preemptive:
    exit()

if args.shield:
    shield = Shield()
    ENV_NAME = 'Car_shield'
    filename = 'car_1_avg_reward.data'
    # pkl_name = 'car-shielded.pkl'

if args.preemptive:
    shield = Shield()
    ENV_NAME = 'Car_shield_preemptive'
    filename = 'car_1_preemptive_avg_reward.data'
    # pkl_name = 'car-shielded.pkl'

elif not args.shield and not args.big_neg:
    shield = None
    ENV_NAME = 'Car_noshield'
    filename = 'car_0_avg_reward.data'
    # pkl_name = 'car-noshield.pkl'

elif not args.shield and args.big_neg:
    shield = Shield()
    ENV_NAME = 'Car_noshield_huge_neg'
    filename = '_0_huge_neg_avg_reward.data'
    huge_neg = True
    # pkl_name = 'car-noshield.pkl'

if args.sarsa:
    filename = 'sarsa_' + filename
    # pkl_name = 'sarsa_' + pkl_name
    ENV_NAME = 'sarsa_' + ENV_NAME
    dqn = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10,
                   policy=policy,shield=shield, preemtive=args.preemtive)

elif not args.sarsa:
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy,shield=shield, huge_neg=huge_neg, preemptive=args.preemptive)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.big_neg:
    filename = 'bigneg_' + filename
    # pkl_name = 'bigneg_' + pkl_name
    ENV_NAME = 'bigneg_' + ENV_NAME

env = Env(env_label=ENV_NAME,big_neg=args.big_neg,viz=args.viz)
score = 0
counter = 0
start=datetime.now()
#env.render()
score_log = []
steps = 200
max_count = 300
if not args.viz:
    target = open(filename, 'w')
while counter <= max_count:
    counter += 1
    train_history = dqn.fit(env, nb_steps=steps, visualize=False, verbose=0)
    env.render()
    test_history = dqn.test(env, nb_episodes=1, visualize=True, verbose=0)
    score = np.mean(test_history.history['episode_reward'])
    print (score, counter)
    if not args.viz:
        target.write(str(score))
        target.write("\n")
    end = datetime.now()
    score_log.append(score)
    env.render(mode=False)

# with open(pkl_name, 'wb') as f:
#     pickle.dump(score_log, f)

print('It took {} steps and {} seconds and {} accidents'.format(counter*steps,end-start,env.accidents))
