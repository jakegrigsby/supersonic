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


import scipy

class Trajectory:
    """
    The storage and calculation needed for each training trajectory for a PPO
    agent.

    The trajectory is initialized. Each step adds information. Then end_trajectory
    is called and all the information needed to update the RND networks is 
    calculated and made available.
    """
    def __init__(self):
        #keeping track of attributes used in update_models
        self.states = []
        self.i_rews = []
        self.rews = []
        self.old_act_probs = []
        self.vals_e = []
        self.vals_i = []
        self.exp_targets = []

    def add(self, state, rew, i_rew, exp_target, act_probs, val_e, val_i):
        self.states.append(state)
        self.rews.append(rew)
        self.i_rews.append(i_rew)
        self.old_act_probs.append(act_probs)
        self.vals_e.append(val_e)
        self.vals_i.append(val_i)
        self.exp_targets.append(exp_target)

    def end_trajectory(self, gamma, lam, i_rew_coeff, e_rew_coeff):
        """calculate gaes, rewards-to-go, convert to numpy arrays."""
        #calculate advantages
        self.vals_next = self.vals[1:] + 0
        deltas = self.rews[:-1] + gamma * self.vals_next - self.vals
        e_adv = self.discount_cumsum(deltas, gamma * lam)
        deltas = self.i_rews[:-1] + gamma * self.vals_next - self.vals
        i_adv = self.discount_cumsum(deltas, gamma * lam)
        self.gaes = np.asarray(e_adv) + np.asarray(i_adv)
        self.states = np.assarray(self.states, dtype=np.uint8)
        i_rews = np.asarray(self.discount_cumsum(self.i_rews, gamma))
        e_rews = np.asarray(self.dicount_cumsum(self.rews, gamma))
        self.rews = (i_rew_coeff*i_rews) + (e_rew_coeff*e_rews)
        self.exp_targets = np.asarray(self.exp_targets)
        self.old_act_probs = np.asarray(self.old_act_probs)
        
    def discount_cumsum(self, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
