import csv

def load_sonic_lvl_set(train=True):
    lvls = {}
    with open('data/sonic-train.csv' if train else 'data/sonic-val.csv', newline='') as lvl_file:
        reader = csv.reader(lvl_file)
        for row in reader:
            lvls[row[1]] = row[0] #lvls[lvl] = game
    return lvls

def get_game_from_sonic_lvl(lvl_id):
    lvl_set = all_sonic_lvls()
    if lvl_id in lvl_set: return lvl_set[lvl_id]
    else: raise KeyError("Level id not found in train or test set")

def is_sonic_lvl(lvl_id):
    return lvl_id in all_sonic_lvls()

def all_sonic_lvls():
    return {**load_sonic_lvl_set(True), **load_sonic_lvl_set(False)}

############################################################################################

import numpy as np

class FrameStack:
    
    def __init__(self, capacity, default_frame):
        self.capacity = capacity
        self.default_frame = default_frame
        self.reset()

    def append(self, frame):
        self._tensor = np.roll(self._tensor,1)
        self.__setitem__(0, frame)

    def reset(self):
        self._tensor = np.repeat(np.expand_dims(self.default_frame, axis=-1), self.capacity, axis=-1)

    def __getitem__(self, idx):
        return self._tensor[:,:,idx]

    def __setitem__(self, key, value):
        try:
            self._tensor[...,key] = value
        except:
            self._tensor[...,key] = np.squeeze(value)

    @property
    def shape(self):
        return self._tensor.shape        

    @property
    def ndim(self):
        return self._tensor.ndim

    @property
    def stack(self):
        return self._tensor

 














