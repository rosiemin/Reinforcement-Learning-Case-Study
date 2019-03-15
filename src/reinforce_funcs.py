import numpy as np


def convert_states(state):
    pos_rounded = round(state[0],1)
    vel_sign = 1 if state[1] >= 0 else 0
    return int((0.6 - pos_rounded) / 0.1 + 19 * (1-vel_sign))

def rand_max(a):
    idx = np.random.choice(np.arange(a.shape[0]), size=a.shape[0], replace=False)
    i_max = np.argmax(a[idx])
    return idx[i_max]
