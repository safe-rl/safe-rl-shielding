# This algorith is based around DQN, parameterised like so:

# Q-function is a dense network with 2x100 node hidden layers
# experience replay contained the most recent 10000 state, action, reward triplets.
# learning took place after every episode using a minibatch size of 100
# learning rate = 0.01
# gamma = 0.99 
# eGreediness = 0.05


from collections import deque
import env_road as env_m
#from gym import wrappers
import numpy as np
from agent_pr import agent
import random
import time
from hyperopt import STATUS_OK
import gym
def send_email(body):
    import smtplib

    gmail_user = ''
    gmail_pwd = ''
    FROM = ''
    TO = ''
    SUBJECT = 'Found One!'
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print ('successfully sent the mail')
    except:
        print ("failed to send mail")

def objective(args):
    LR, MS, BS, WE, target = args

    LEFT = 0
    RIGHT = 1
    MAX_TIMESTEPS = 500

    blob = agent(4,[i for i in range(0,8)], epsilon=1, learningRate=LR, memorySize=MS, batch_size=BS, WE=WE)
    env = env_m.Env()
    #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-v1',force=True)
    notify_value = -1
    t = 0
    avgreward = deque([],100)
    avgQ = deque([],100)
    trials = 100000
    x = deque([],500)
    x.append(0)
    y = deque([],500)
    y.append(-1)
    xQ = deque([],500)
    xQ.append(0)
    yQ = deque([],500)
    yQ.append(-1)
    maxsofar = 0
    maxQsofar = 0
    viz_flag = False
    S_list = []
    q_est_trials = 1000
    for i_episode in range(q_est_trials):
        #print('{}/{}'.format(i_episode,q_est_trials))
        S = env.reset(blob.Q_est, t, viz_flag)
        done = False   
        t = 0
        tot_R = 0  
        while not done:
            t += 1
            S_list.append(S)
            A = random.choice([0,1,2,3,4,5,6,7])#blob.act(S)
            S_dash, R, done = env.step(A)
            blob.observe(S,A,R,S_dash)
            #self.Q.predict(state[np.newaxis,:])
            tot_R += R
            S = np.copy(S_dash)    

    for i_episode in range(trials):
        
        S = env.reset(blob.Q_est, t, viz_flag)
        done = False   
        t = 0
        tot_R = 0  
        while not done:
            t += 1
            A = blob.act(S)
            S_dash, R, done = env.step(A)
            blob.observe(S,A,R,S_dash)
            tot_R += R
            S = np.copy(S_dash)
            
        # every now and then stop, and think things through:
        if i_episode > 55:
            blob.reflect(i_episode)
            
        # when the episode ends the agent will have hit a terminal state so give it a zero reward
        if t < MAX_TIMESTEPS:
            blob.observe(S,A,0.,None)
        else:
            blob.observe(S,A,1.,None)
                
        avgreward.append(tot_R)
        avg_Q = 100* np.average(np.amax(blob.Q.model.predict(np.array(S_list)), axis=1))
        avgQ.append(avg_Q)
        avg_reward = np.mean(avgreward)
        viz_flag = True if avg_reward > .5 else False
        # update the xy data
        yQ.append(np.mean(avgQ))
        x.append(i_episode)
        y.append(avg_reward)
        if len(avgreward) > 10:
            maxsofar = max(maxsofar,np.mean(avgreward))
        if len(avgQ) > 85:
            maxQsofar = max(maxQsofar,np.mean(avgQ))
        
        #print(np.average(np.amax(blob.Q.model.predict(np.array(S_list)), axis=1)))
        if i_episode % 1000 == 0:
            print('Learning rate: {}, Memory size: {}, Batch size: {}, Q update: {}'.format(LR, MS, BS, WE))
            print("episode: {}, average reward: {}, Reward: {:.2f}, Memory: {}/{}, Epsilon: {:.2f}, Max: {:.2f}, Q: {:.2f}".format(i_episode,str(np.round(np.mean(avgreward),3)),tot_R, len(blob.experience_pr._experience), MS, blob.policy.epsilon,maxsofar,np.mean(avgQ)))
            blob.Q_est.model.save('model_{}_{}_{}_{}.h5'.format(LR, MS, BS, WE))
    string = 'Args: '+str(args[:-1])+'\n'
    string += 'Max R: '+str(maxsofar)+'\n'
    string += 'Max Q: '+str(maxQsofar)+'\n'
    target.write(string)
    
    if maxsofar > 0.25:
        send_email(string)
        model.save('my_model.h5')
    res = {
        'loss': -1*maxsofar,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'maxQ': maxQsofar
        }
    print (res)
    return res

objective()
