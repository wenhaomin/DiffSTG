# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


def search_multihop_neighbor(adj, hops=5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list((np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]

class CleanDataset():
    def __init__(self, config):

        self.data_name = config.data.name
        self.feature_file = config.data.feature_file
        self.val_start_idx = config.data.val_start_idx
        self.adj = np.load(config.data.spatial)
        self.label, self.feature = self.read_data()

        #for stpgcn
        if config.model.get('alpha', None) is not None:
            self.alpha = config.model.alpha
            self.t_size = config.model.t_size
            self.spatial_distance = search_multihop_neighbor(self.adj, hops=self.alpha)
            self.range_mask = self.interaction_range_mask(hops=self.alpha, t_size=self.t_size)

    def read_data(self):
        if 'PEMS' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
        elif 'AIR' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
            data = np.nan_to_num(data, nan=0)
        elif 'Metro' in self.data_name:
            data = np.expand_dims(np.load(self.feature_file)[:, :, 0], -1)
            data = np.nan_to_num(data, nan=0)
        else:
            data = np.load(self.feature_file)
        # return data.astype('float32'), self.normalization(data).astype('float32')
        return self.normalization(data).astype('float32'), self.normalization(data).astype('float32')





    def normalization(self, feature):
        train = feature[:self.val_start_idx]
        # if 'Metro' in self.data_name:
        #     idx_lst = [i for i in range(train.shape[0]) if i % (24 * 6) >= 7 * 6 - 12]
        #     train = train[idx_lst]

        mean = np.mean(train)
        std = np.std(train)

        # since the feature is actual the flow, the mean and std of feature is also the label's mean and std
        self.mean = mean
        self.std = std
        return (feature - mean) / std

    def reverse_normalization(self, x):
        return self.mean + self.std * x

    # for stpgcn
    def interaction_range_mask(self, hops=2, t_size=3):
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()] * t_size, axis=-1)  # V,tV



class TrafficDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        self.data_name = clean_data.data_name


        self.label = np.array(clean_data.label) # (T_total, V, D), where T_all means the total time steps in the data
        self.feature = np.array(clean_data.feature)  # (T_total, V, D)

        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __getitem__(self, index):

        recent_idx = self.idx_lst[index]

        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)
        return label, node_feature, pos_w, pos_d

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            # only 6:00-24:00 for Metro data
            if 'Metro' in self.data_name:
                if label_start_idx % (24 * 6) < (7 * 6):
                    continue
                if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                    continue

            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data

            if recent:
                idx_lst.append(recent)
        return idx_lst

    #################################################


if __name__ == '__main__':
    pass
