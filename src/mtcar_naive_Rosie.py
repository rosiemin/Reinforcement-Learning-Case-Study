import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


env = gym.make('MountainCar-v0')

# self.min_position = -1.2
# self.max_position = 0.6
# self.max_speed = 0.07
# self.goal_position = 0.5
# position, velocity = self.state

total_episodes = 1000
print(env.action_space)
print(env.observation_space)

#Naive Model with random selection:
rewards_lst = []
position_lst = []
velocity_lst = []
action_lst = []

for i_episode in range(total_episodes):
    env.reset()
    reward_n = 0
    for t in range(200):
        # env.render(mode = 'rgb_array')
        # print(f"Obs Before: {state}")
        action = env.action_space.sample()
        print(f"Action Taken: {action}")
        action_lst.append(action)
        state, reward, done, info = env.step(action)
        print(f"Obs After: {state}")
        position_lst.append(state[0])
        velocity_lst.append(state[1])
        print(f"Reward: {reward}")
        reward_n += reward
        print(f"Done?: {done}")
        print(f"info: {info}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            rewards_lst.append(reward_n)

            break
