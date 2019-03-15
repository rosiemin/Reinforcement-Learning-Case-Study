import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from reinforce_funcs import *



''' Setup gym environment '''
ENV_NAME = 'MountainCar-v0'
seed = 2
env = gym.make(ENV_NAME)
np.random.seed(seed)
env.seed(seed)

pos_range, vel_range = [i for i in zip(env.observation_space.low,
                                       env.observation_space.high)]
# position range * 2 for positiive/negative velocity signs
n_states = int(((pos_range[1]-pos_range[0])*10 +1) * 2)
n_actions = env.action_space.n

''' Building Neural Network model '''
drop = 0.25
model = Sequential()
model.add(Dense(units=24, input_dim=n_states, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(n_actions, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


''' Training Model: Q learning '''
def train_model(env, model, n_episodes=1000, render=False, rend_mul=100):
    y = 0.95
    eps = 0.75
    decay_factor = 0.99
    records = []
    r_lst = []
    targ_avg_lst = []
    eye_s = np.eye(n_states)
    for episode in range(n_episodes):
        s_ = env.reset()
        s = convert_states(s_)
        eps*= decay_factor
        if episode % 100 == 0:
            print("Episode {} of {}".format(episode, n_episodes))
        done = False
        r_sum = 0
        targ_sum = 0
        for t in range(200):
            if render and episode % rend_mul == 0:
                env.render()
            if np.random.rand() < eps:
                a = np.random.randint(n_actions)
            else:
                a = rand_max(model.predict(eye_s[s:s+1]).reshape(-1))
            _s, r, done, _ = env.step(a)
            new_s = convert_states(_s)
            target = r + y * np.max(model.predict(eye_s[new_s:new_s+1])) \
                     + 100*(np.abs(_s[1]))
            targ_vec = model.predict(eye_s[s:s+1])[0]
            targ_vec[a] = target
            model.fit(eye_s[s:s+1],targ_vec.reshape(1,-1), epochs=1, verbose=0)
            records.append((*s_, a))
            s_ = _s
            s = new_s
            r_sum += r
            targ_sum += target
            if done:
                break
        r_lst.append(r_sum)
        targ_avg_lst.append(targ_sum / 200)
    return r_lst, targ_avg_lst, records


r_lst, targ_avg_lst, records = train_model(env, model)
r_lst = np.array(r_lst)
# plot rewards over episodes
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(r_lst+200, label='Total reward +200', color='g')
ax.plot(targ_avg_lst, label='Avg Target per action', color='b')
ax.legend()
ax.set_xlabel('Number of Episodes')
ax.set_ylabel('Average reward per episode')
ax.set_title('MLP Q Learning '+ ENV_NAME)

fig.show()

# # After training is done, we save the final weights.
model.save('../models/nn_{}_{}_weights.h5f'.format(ENV_NAME,
                                                input('Model nickname: ')),
                 overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)


# r_lst = [r*200+200 for r in r_avg_lst]
# fig2, ax = plt.subplots(figsize=(8,6))
# ax.plot(r_lst, label='Total reward +200', color='g')
# ax.plot(targ_avg_lst, label='Avg Target per action', color='b')
# ax.legend()
# ax.set_xlabel('Number of Episodes')
# ax.set_ylabel('Reward per episode')
# ax.set_title('MLP Q Learning '+ ENV_NAME)

### script for further training to 50,000 episodes with graphs
# r_lst_, targ_avg_lst_, records_ = train_model(env, model, 49000)
# model.save('../models/nn_{}nn_r_add_100velo_50kep_weights.h5f'.format(ENV_NAME), overwrite=True)
# r_lst = np.append(r_lst, r_lst_)
# targ_avg_lst += targ_avg_lst_
# records += records_
# pos, velo, act = [i for i in zip(*records)]
# plot_policy(pos,velo,act, save_path='../images/nn_policy_plot_50kep.png', show=False)
# fig, ax = plt.subplots(figsize=(8,6))
# ax.plot(r_lst+200, label='Total reward +200', color='g')
# ax.plot(targ_avg_lst, label='Avg Target per action', color='b')
# ax.legend()
# ax.set_xlabel('Number of Episodes')
# ax.set_ylabel('Average reward per episode')
# ax.set_title('MLP Q Learning '+ ENV_NAME)
# fig.savefig('../images/nn_r_add_100velo_50kep.png')
# print('=====DONE======')
