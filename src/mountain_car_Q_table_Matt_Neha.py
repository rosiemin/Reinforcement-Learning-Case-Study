import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable

np.random.seed(2)

def plot_max_position(positions, rewards):
    plt.figure(1, figsize=[10,6])
    plt.ylim(-1, 1)
    plt.plot(positions[:,1], color='g') #positions[:,0]
    ma = pd.Series(positions[:,1]).rolling(100).mean()
    plt.plot(ma, color='b', label="100-Episode Rolling Average") #positions[:,0]
    plt.title("Mountain Car Max Positions")
    plt.xlabel('Episode')
    plt.ylabel('Max Position')
    plt.legend()
    plt.show()

def print_pretty_table(table):
    x = PrettyTable()
    x.field_names = ["STATE", "LEFT", "NEUTRAL", "RIGHT"]

    state_pos = np.round(np.linspace(0.6, -1.2, 19).reshape(-1,1),1)
    plus = np.where(np.zeros(len(state_pos)) == 0, "+", 0).reshape(-1,1)
    state_top = np.hstack((state_pos, plus))
    minus = np.where(np.zeros(len(state_pos)) == 0, "-", 0).reshape(-1,1)
    state_bottom = np.hstack((state_pos, minus))
    state_labels = np.vstack((state_top, state_bottom))

    for j,i in zip(state_labels, table):
        x.add_row([j, i[0],i[1],i[2]])

    print(x)

env = gym.make('MountainCar-v0')
env.seed(2)

# self.min_position = -1.2
# self.max_position = 0.6
# self.max_speed = 0.07
# self.goal_position = 0.5
# position, velocity = self.state

# action 0 = left
# action 1 = nothing
# action 2 = right

total_episodes = 50000
print(env.action_space)
print(env.observation_space)

max_position = -.4
positions = np.ndarray([0,2])
velocities = np.ndarray([0,2])
all_rewards = []
successful = []
time_successful = []

# initialize r_table
q_table = np.zeros((37, 3))

def convert_states(state):
    pos_rounded = round(state[0],1)
    vel_sign = 1 if state[1] >= 0 else 0
    return int((0.6 - pos_rounded) / 0.1 + 19 * (1-vel_sign))

def rand_max(a):
    idx = np.random.choice([0,1,2], size=3, replace=False)
    i_max = np.argmax(a[idx])
    return idx[i_max]

# initialize q_table
q_table = np.zeros((38, 3))

y = 0.95 # discount factor
eps = 0.5 # epsilon-greed explore threshold
lr = 0.05 # learning rate
decay_factor = 0.99999 # as time goes on, decrease willingness to explore

for episode in range(total_episodes):
    state = env.reset()
    running_reward = 0
    eps *= decay_factor
    max_position = -.4
    print(episode)
    for t in range(200):
        # env.render()

        state = convert_states(state)

        if np.random.random() < eps or np.sum(q_table[state, :]) == 0:
            action = np.random.randint(0, 3)
        else:
            #action = np.argmax(r_table[state, :]) # Exploit
            action = rand_max(q_table[state, :])

        new_state, reward, done, info = env.step(action)

        # Adjust reward based on car velocity
        reward += np.abs(new_state[1]) * 1000

        # also add reward if get to new max position
        if new_state[0] > max_position:
            max_position = new_state[0]
            reward += 10

        # create new_state conversion
        new_state_int = convert_states(new_state)

        q_table[state, action] = (1 - lr) * q_table[state, action] + lr * (reward + y * np.max(q_table[new_state_int, :]))
        running_reward += reward
        state = new_state

        if done:
            if new_state[0] >= 0.5:
                successful.append(episode)
            all_rewards.append(running_reward)
            break

    positions = np.append(positions, [[episode, max_position]], axis=0)

print(positions)
#print(f"Rewards per Episode {all_rewards}")
#print('Furthest Position: {}'.format(max_position))
print('Successful Episodes: {}'.format(np.count_nonzero(successful)))
print(q_table)
print(f"Successful Episodes {successful}")
plot_max_position(positions, all_rewards)
print_pretty_table(q_table)
