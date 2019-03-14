import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        # pic = env.render()
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

plt.figure(2, figsize=[10,5])
p = pd.Series(position_lst)
plt.plot(p, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('Position')
plt.title('All Car Position')
plt.savefig('Final Position.png')
plt.show()

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
fig = plt.figure(3, figsize=[7,7])
ax = fig.gca()
plt.set_cmap('brg')
surf = ax.scatter(X,Y, c=Z)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Policy')
recs = []
for i in range(0,3):
     recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
plt.legend(recs,labels,loc=4,ncol=3)
# fig.savefig('Policy.png')
plt.show()
