import gym
import numpy as np
import matplotlib.pyplot as plt
import imageio


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

total_episodes = 10
print(env.action_space)
print(env.observation_space)

max_position = -.4
positions = np.ndarray([0,2])
velocities = np.ndarray([0,2])
print(f"Positions Initial {positions}")
print(f"Velocities Initial {velocities}")
all_rewards = []
successful = []
images = []


for episode in range(total_episodes):
    state = env.reset()
    print(f"initial position: {state}")
    total_rewards = 0
    for t in range(200):
        images.append(env.render("rgb_array"))
        # print(f"Obs Before: {observation}")
        action = env.action_space.sample()
        # print(f"Action Taken: {action}")
        state, reward, done, info = env.step(action)
        if state[0] > max_position:
            max_position = state[0]
            positions = np.append(positions, [[episode, max_position]], axis=0)
        # print(f"Obs After: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Done?: {done}")
        # print(f"info: {info}")
        total_rewards += reward
        if done:
            if state[0] >= 0.5:
                successful.append(episode)
            all_rewards.append(total_rewards)
            #print("Episode finished after {} timesteps".format(t+1))
            break

print(positions)
print(f"Rewards per Episode {all_rewards}")
print('Furthest Position: {}'.format(max_position))
print('Successful Episodes: {}'.format(np.count_nonzero(successful)))

plot_max_position(positions, all_rewards)

with imageio.get_writer("images/naive_model.gif",mode='I') as writer:
    for image in images:
        writer.append_data(image)
