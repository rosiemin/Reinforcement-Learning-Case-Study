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

# pos_range, vel_range = [i for i in zip(env.observation_space.low,
#                                        env.observation_space.high)]
# position range * 2 for positiive/negative velocity signs
# n_states = ((pos_range[1]-pos_range[0])*10 +1) * 2
n_actions = env.action_space.n

''' Building Neural Network model '''
drop = 0.25
model = Sequential()
# model.add(Flatten(input_size=(1,n_states)))
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop))
model.add(Dense(n_actions, activation='linear'))
### Comment this compile out for keras-rl agent
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])


####### Similar to keras-rl cartPole example
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


''' Training Model: Q learning '''
# y = 0.95
# eps = 0.3


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_{}_weights.h5f'.format(ENV_NAME,
                                                input('Model nickname: ')),
                 overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=True)
