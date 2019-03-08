import csv
import os
import numpy as np
import scipy
import json
import cv2

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

############################################################################################

class Trajectory:
    """
    The storage and calculation needed for each training trajectory for a PPO
    agent.

    The trajectory is initialized. Each step adds information. Then end_trajectory
    is called and all the information needed to update the RND networks is
    calculated and made available.
    """
    def __init__(self, past_trajectory=None):
        # keeping track of attributes used in update_models
        self.states = []
        self.rews_i = []
        self.rews_e = []
        self.old_act_probs = []
        self.vals_e = []
        self.vals_i = []
        self.exp_targets = []

        # if given a past trajectory to resume from, the first elements in this trajectory will be the last from the old one
        if past_trajectory != None:
            i = len(past_trajectory.states) - 1
            self.states.append(past_trajectory.states[i])
            self.i_rews.append(past_trajectory.i_rews[i])
            self.rews.append(past_trajectory.rews[i])
            self.old_act_probs.append(past_trajectory.old_act_probs[i])
            self.vals_e.append(past_trajectory.vals_e[i])
            self.vals_i.append(past_trajectory.vals_i[i])
            self.exp_targets.append(past_trajectory.exp_targets[i])

    def add(self, state, rew_e, rew_i, exp_target, act_probs, val_e, val_i):
        self.states.append(np.squeeze(state, axis=0))
        self.rews_e.append(rew_e)
        self.rews_i.append(rew_i)
        self.old_act_probs.append(act_probs)
        self.vals_e.append(val_e)
        self.vals_i.append(val_i)
        self.exp_targets.append(exp_target)
    
    def _lists_to_ndarrays(self):
        self.states = np.asarray(self.states)
        self.rew_i = np.asarray(self.rews_i)
        self.rews_e = np.asarray(self.rews_e)
        self.old_act_probs = np.asarray(self.old_act_probs)
        self.vals_e = np.asarray(self.vals_e)
        self.vals_i = np.asarray(self.vals_i)
        self.exp_targets = np.asarray(self.exp_targets)
        self.vals_next_e = np.asarray(self.vals_next_e)
        self.vals_next_i = np.asarray(self.vals_next_i)

    def end_trajectory(self, gamma_i, gamma_e, lam, i_rew_coeff, e_rew_coeff):
        """calculate gaes, rewards-to-go, convert to numpy arrays."""
        #calculate advantages
        self.vals_next_e = self.vals_e[1:] + [0]
        self.vals_next_i = self.vals_i[1:] + [0]
        self._lists_to_ndarrays()
        deltas = self.rews_e[:-1] + gamma_e * self.vals_next_e[1:] - self.vals_e[:-1]
        e_adv = self.discount_cumsum(deltas, gamma_e * lam)
        deltas = self.rew_i[:-1] + gamma_i * self.vals_next_i[1:] - self.vals_i[:-1]
        i_adv = self.discount_cumsum(deltas, gamma_i * lam)
        self.gaes = np.asarray(e_adv) + np.asarray(i_adv)
        self.states = np.asarray(self.states, dtype=np.float32)
        i_rews = np.asarray(self.discount_cumsum(self.rews_i, gamma_i))
        e_rews = np.asarray(self.discount_cumsum(self.rews_e, gamma_e))
        self.rews = (i_rew_coeff*i_rews) + (e_rew_coeff*e_rews)
        self.exp_targets = np.asarray(self.exp_targets)
        self.old_act_probs = np.asarray(self.old_act_probs)

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

#########################################################################################

def save_hyp_dict_to_file(filename, hyp_dict):
    with open(filename, 'w') as f:
        json_hyp_dict = json.dump(hyp_dict, f)

def load_hyp_dict_from_file(filename):
    with open(filename, 'r') as f:
        json_hyp_dict = json.load(f)
    return json_hyp_dict

#########################################################################################

def get_lvl_map(lvl_id):
    maps_dir = os.path.join('data','lvl_maps')
    game = get_game_from_sonic_lvl(lvl_id)
    lvl_id = lvl_id.replace('.','') + '.png'
    lvl_path = os.path.join(game, lvl_id)
    lvl_path = os.path.join(maps_dir, lvl_path)
    img = cv2.imread(lvl_path)
    return np.transpose(img, (1, 0, 2)) #rotate image

def get_avg_lvl_map_dims():
    num_lvls = len(all_sonic_lvls().keys())
    x_sizes, y_sizes = np.zeros(num_lvls), np.zeros(num_lvls)
    for idx, lvl_id in enumerate(all_sonic_lvls().keys()):
        lvl_map = get_lvl_map(lvl_id)
        x_sizes[idx] = lvl_map.shape[0]
        y_sizes[idx] = lvl_map.shape[1]
    return (int(np.mean(x_sizes)), int(np.mean(y_sizes)))