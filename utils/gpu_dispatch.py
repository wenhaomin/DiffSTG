# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from utils.common_utils import ws, dir_check
import json, os

'''
This module is proposed for GPU dispatching. 
It can analysis the current GPU memory usage, and return the GPU/cuda_id that satisfy your memory requirement.

Let say your server has 4 GPU cards, and your requires at least 6000(MB) GPU memory for your code, then you can use:
GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3])
Then the above code will return an avaliable GPU id.

Doing so can save us the trouble of manually specifying the gpu id,
especially when running multiple instances in parallel.
'''

gpu_num = 8 # set the value to the GPU number in your device.
def cmd_lst(cmd_lst = []):
    # excute cmd list
    for l in cmd_lst:
        cmd(l)

def cmd(cmd_):
    # excute and print one cmd
    try:
        os.system(cmd_)
        print(cmd_)
    except:
        print(cmd_+ "failed")

import random
class GPU():
    def __init__(self):
        self.info_dict = {}
        self.log_file = ws + '/output/gpustat/'
        dir_check(self.log_file)
        self.info_path = self.log_file + '/info.npy'
        self.info_dict = {}

    def get_nvidia_smi(self,fin):
        f = open(fin)
        lines = [l for l in f]
        gpu_num = 8 # the GPU card number of your device
        lineno = [9 + 4 * i for i in range(gpu_num)]

        info_dict = {}
        for id, idx in enumerate(lineno):
            info = lines[idx]
            info = info.split('|')[2]
            info = info.rstrip(' ').lstrip(' ')
            used, total = info.split('/')
            used = int(used.rstrip('MiB '))
            total = int(total.rstrip('MiB').lstrip(' '))
            info_dict[int(id)] = [used, total]
        return info_dict

    def save(self):
        # save info_dict to file
        np.save(self.info_path, self.info_dict)

    def load(self):
        self.info_dict = np.load(self.info_path,allow_pickle=True).item()
        return self.info_dict

    def update_info_dict(self):
        stat_path = self.log_file + '/gpustat.txt'
        cmds = [f'nvidia-smi > {stat_path}']
        cmd_lst(cmds)

        self.info_dict = self.get_nvidia_smi(stat_path)

        self.save()
        return self.info_dict


    def load_info_dict(self):
        if not os.path.exists(self.info_path):
            self.info_dict = self.update_info_dict()

        else: # load it from the cathe file
            self.info_dict = self.load()
        return self.info_dict

    def get_usefuel_gpu(self, max_memory:int, condidate_gpu_id: list): #condidate_id:[0,1,2]
        # self.info_dict = self.load_info_dict()
        try:
            self.info_dict = self.update_info_dict()
        except Exception as err:
            print('wrong in load gpu info dict', err)
            self.info_dict = {}

        useful_id = []
        for id, (used, total) in self.info_dict.items():

            n = (total - used) / max_memory
            n = int(n)
            # if n > 1 and int(id) < 4: useful_id.append(int(id))
            if n > 1 and int(id) in condidate_gpu_id: useful_id.append(int(id))

        if len(useful_id) > 0:
            id = random.choice(useful_id)
            return id
        else:
            print('None gpu is avalible, try again later')
            return None



if __name__ == '__main__':

    GpuDispather = GPU()
    GpuDispather.update_info_dict()
    print(GpuDispather.info_dict)
    id = GpuDispather.get_usefuel_gpu(max_memory=2000, condidate_gpu_id=[0,1,2,3])
    print(GpuDispather.info_dict)
    print(id)
