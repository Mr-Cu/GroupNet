import os, random, numpy as np, copy

from torch.utils.data import Dataset
import torch


def seq_collate(data):

    (past_traj, future_traj) = zip(*data)
    past_traj = torch.stack(past_traj,dim=0)
    future_traj = torch.stack(future_traj,dim=0)
    data = {
        'past_traj': past_traj,
        'future_traj': future_traj,
        'seq': 'nfl',
    }

    return data

class NFLDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        super(NFLDataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        if training:
            data_root = 'datasets/nfl/train.npy'
        else:
            data_root = 'datasets/nfl/test.npy'

        self.trajs = np.load(data_root) 

        # if training:
        #     self.trajs = self.trajs[:7500]
        # else:
        #     self.trajs = self.trajs[:2500]
        # print(len(self.trajs))
        # assert 1<0
        if training:
            self.trajs = self.trajs[:550]
        else:
            self.trajs = self.trajs[:240]

        self.batch_len = len(self.trajs)
        # print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        # print(self.traj_abs.shape) #torch.Size([550, 23, 15, 2])
        # print(self.traj_norm.shape) #torch.Size([550, 23, 15, 2])
        # assert 1<0

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        past_traj = self.traj_abs[index, :, :self.obs_len, :]
        future_traj = self.traj_abs[index, :, self.obs_len:, :]
        # print(past_traj.shape[0]) #很多个23
        # print(past_traj.shape) #torch.Size([23, 5, 2])，很多组
        # print(future_traj.shape) #torch.Size([23, 10, 2])
        # assert 1<0
        out = [past_traj, future_traj]
        return out
