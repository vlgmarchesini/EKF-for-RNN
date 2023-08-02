from torch.utils.data import Dataset
import torch
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_data(datapath, vel_only=True, many_files=True, ncmds=2, n_states=7, propeller_dyn='False'):
    if many_files:
        files = glob(datapath)
        xtraining = ''
        for file in files:
            csv_data = pd.read_csv(file, header=None)
            if xtraining == '':
                xtraining = csv_data.to_numpy()[:, 0:n_states+1+ncmds]
                # print(f"xtraining.shape={xtraining.shape}", flush=True)
                # print(f"xtraining[0:3]={xtraining[0:3]}", flush=True)
            else:
                xtraining = np.concatenate((xtraining, csv_data.to_numpy()[:, 0:n_states+1+ncmds]), axis=0)

        tseries = xtraining[:, 0]
    else:
        csv_data = pd.read_csv(datapath, header=None)
        xtraining = csv_data.to_numpy()
        tseries = xtraining[:, 0]

    pdyn = 1 if propeller_dyn == 'True' else 0
    if vel_only:
        # x and y are sinusoids, hard to model
        state = xtraining[:, 4:n_states + pdyn] # if the propeller is not ignored xtraining[:, 4:n_states+1]
    else:
        state = xtraining[:, 1:n_states + pdyn] # if the propeller is not ignored xtraining[:, 4:n_states]

    u = xtraining[:, n_states+1:n_states+1+ncmds]
    dt = tseries[1] - tseries[0]
    _, num_features = state.shape
    return xtraining, tseries, state, u, dt, num_features


def normalization_parameters(state,u):
    _, n_states = state.shape
    _, n_cmd = u.shape
    std_states = np.zeros(n_states)
    mean_states = np.zeros(n_states)
    std_cmd = np.zeros(n_cmd)
    mean_cmd = np.zeros(n_cmd)

    for i in range(n_states):
      std_states[i] = np.std(state[:,i])
      mean_states[i] = np.mean(state[:,i])

    for i in range(n_cmd):
      std_cmd[i] = np.std(u[:,i])
      mean_cmd[i] = np.mean(u[:,i])

    return mean_states, std_states, mean_cmd, std_cmd


class TrajectoryGRU(Dataset):
    def __init__(self, state, u):

        len_x, n_of_states = state.shape
        len_u, n_of_controls = u.shape

        size_of_y = len_x - 3
        size_of_y = int(size_of_y)
        jump = 1
        self.jump = jump
        self.Y = torch.zeros((size_of_y, 1, n_of_states))
        self.cmds = torch.zeros((size_of_y, 1, n_of_controls))

        for i in tqdm(range(size_of_y)):
            self.Y[i, 0] = torch.tensor(state[i + 1])
            self.cmds[i, 0] = torch.tensor(u[i])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.cmds[idx], self.Y[idx]


