# Deep learning case study
## Authors: Matt Devor, Nathan James, Rosie Martinez & Neha Rao

## Table of Contents
* [The Goal](#The-Goal)
* [The Plan](#The-Plan)
* [Implementing our code](#Implementing-our-code)
  * [The Environment](#The-Environment)
  * [The Actions](#The-Actions)
  * [The States](#The-States)
  * [The Rewards](#The-Rewards)
  * [Q-Learning](#Q-learning)
  * [The Resulting Policy](#The-Resulting-Policy)
* [Our Model](#Our-model-in-action)
* [The Results](#The-Results)
* [Final Thoughts & Next Steps](#Final-Thoughts-&-Next-Steps)

## The Goal:
The goal of this case study was to use reinforcement learning to train an agent to perform well in an environment using the OpenAI Gym [source]/(https://gym.openai.com/envs/). Based on our time constraints, we chose to tackle one of the classic environments:

<p align="center">
  <img src="images/mtcar.gif" width="550">
</p>

## Background:

<p align="center">
  <img src="images/environ_action.gif" width="550">
</p>


##### The Environment:
This is the place where the agent lives and interacts with.

**For our case study, the environment is the mountain.**

##### The Actions:
Action is usually based on the environment, different environments lead to different actions based on the agent. Set of valid actions for an agent are recorded in a space called an action space. These are usually finite in number.

**Our actions were Left, Neutral, and Right**

##### The States:
The state is a complete description of the world, they don’t hide any pieces of information that is present in the world. It can be a position, a constant or a dynamic. We mostly record these states in arrays, matrices or higher order tensors.

**Our states were position and velocity**

##### The Rewards:
The reward function R is the one which must be kept tracked all-time in reinforcement learning. It plays a vital role in tuning, optimizing the algorithm and stop training the algorithm. It depends on the current state of the world, the action just taken, and the next state of the world.

**We are rewarding for ...FILL IN**

##### Policies:
Policy is a rule used by an agent for choosing the next action, these are also called as agents brains.

**Our policies are ... FILL IN**

##### Reward Table or Q-Table (depending on model):
For our case study, we realized that our states were based on two continuous variables:

|          	| Minimum 	| Maximum 	|
|----------	|---------	|---------	|
| Position 	| -1.2    	| 0.6     	|
| Velocity 	| -0.07   	| 0.07    	|

## Naive Model:
The Naive Model randomly from the rewards table based on the maximum reward based on that state. (see reward table below). If two states had the same max value of reward, the actions were shuffled and one was chosen at random.

<p align="center">
  <img src="images/naive_lineplot.png" width="550">
  <img src="images/naive_policy.png" width="550">
</p>

```

                                +--------------+--------+---------+--------+
                                |    STATE     |  LEFT  | NEUTRAL | RIGHT  |
                                +--------------+--------+---------+--------+
                                | ['-1.2' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-1.1' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-1.0' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.9' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.8' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.7' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.6' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.5' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.4' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.3' '+'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.2' '+'] | -24.0  |  -23.0  | -23.0  |
                                | ['-0.1' '+'] | -283.0 |  -283.0 | -282.0 |
                                | ['0.0' '+']  |  -3.0  |   -3.0  |  -3.0  |
                                | ['0.1' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['0.2' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['0.3' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['0.4' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['0.5' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['0.6' '+']  |  0.0   |   0.0   |  0.0   |
                                | ['-1.2' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-1.1' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-1.0' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.9' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.8' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.7' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.6' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.5' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.4' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.3' '-'] |  0.0   |   0.0   |  0.0   |
                                | ['-0.2' '-'] | -23.0  |  -24.0  | -23.0  |
                                | ['-0.1' '-'] | -212.0 |  -212.0 | -212.0 |
                                | ['0.0' '-']  | -119.0 |  -119.0 | -119.0 |
                                | ['0.1' '-']  |  -3.0  |   -3.0  |  -4.0  |
                                | ['0.2' '-']  |  0.0   |   0.0   |  0.0   |
                                | ['0.3' '-']  |  0.0   |   0.0   |  0.0   |
                                | ['0.4' '-']  |  0.0   |   0.0   |  0.0   |
                                | ['0.5' '-']  |  0.0   |   0.0   |  0.0   |
                                +--------------+--------+---------+--------+

```


## Q-Table Model:

## Keras Model:
   3. What architecture you chose and why
   4. What final architecture you chose and why (how did you pick your hyperparameters?)


## The Results:

|                           	| Naive Model 	| Q-Table Model 	| Keras Model 	|
|---------------------------	|-------------	|---------------	|-------------	|
| Average Reward            	| -200        	|               	|             	|
| Total number of episodes  	| 1000        	|               	|             	|
| Episode of first success  	| 0           	|               	|             	|
| Total number of successes 	| 0           	|               	|             	|

## Final Thoughts & Next Steps:
