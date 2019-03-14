import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def plot_max_position(positions, rewards, save_path = None, show = True):
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
    if show:
        plt.show()
    else:
        plt.savefig(save_path)

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

def plot_policy(position, velocity, action, save_path = None, show = True):
    z = pd.Series(action)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = z.apply(lambda s:colors[s])
    labels = ['Left','Right','Nothing']

    fig = plt.figure(5, figsize=[7,7])
    ax = fig.gca()
    plt.set_cmap('brg')
    ax.scatter(position,velocity, c=z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    for i in range(0,3):
         recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)

    if not show:
        fig.savefig(save_path)
    else:
        plt.show()
