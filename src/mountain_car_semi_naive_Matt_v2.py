import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

def plot_max_position(positions, rewards):
    plt.figure(1, figsize=[8,6])
    plt.title("Mountain Car Max Positions and Rewards per Episode")
    plt.subplot(211)
    plt.plot(positions[:,1], color='g') #positions[:,0],
    plt.xlabel('Episode')
    plt.ylabel('Max Position')
    plt.subplot(212)
    plt.plot(rewards, color='g')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.show()


env = gym.make('MountainCar-v0')

# self.min_position = -1.2
# self.max_position = 0.6
# self.max_speed = 0.07
# self.goal_position = 0.5
# position, velocity = self.state

# action 0 = left
# action 1 = nothing
# action 2 = right


total_episodes = 10
print(env.action_space)
print(env.observation_space)

max_position = -.4
positions = np.ndarray([0,2])
velocities = np.ndarray([0,2])
all_rewards = []
successful = []
time_successful = []

# initialize r_table
r_table = np.zeros((37, 3))

def convert_states(state):
    pos_rounded = round(state[0],1)
    vel_sign = 1 if state[1] >= 0 else 0
    return int((0.6 - pos_rounded) / 0.1 + 19 * (1-vel_sign))

def rand_max(a):
    idx = np.random.choice([0,1,2], size=3, replace=False)
    i_max = np.argmax(a[idx])
    return idx[i_max]

for episode in range(total_episodes):
    state = env.reset()

    running_reward = 0
    for t in range(200):
        env.render()
        expected_reward = -1*t

        state = convert_states(state)

        if np.sum(r_table[state, :]) == 0:
            action = env.action_space.sample()

        else:
            #action = np.argmax(r_table[state, :]) # Exploit
            action = rand_max(r_table[state, :])

        new_state, reward, done, info = env.step(action)

        if new_state[0] > max_position:
            max_position = new_state[0]
            positions = np.append(positions, [[episode, max_position]], axis=0)

        r_table[state, action] += reward
        running_reward += reward
        state = new_state

        if done:
            if state[0] >= 0.5:
                successful.append(episode)
            all_rewards.append(running_reward)
            break

print(positions)
print(f"Rewards per Episode {all_rewards}")
print('Furthest Position: {}'.format(max_position))
print('Successful Episodes: {}'.format(np.count_nonzero(successful)))
print(r_table)

#plot_max_position(positions, all_rewards)
