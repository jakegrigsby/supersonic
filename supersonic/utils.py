import csv
import os
import json
import math

import numpy as np
import scipy.signal
import cv2


def load_sonic_lvl_set(train=True):
    lvls = {}
    with open(
        "data/sonic-train.csv" if train else "data/sonic-val.csv", newline=""
    ) as lvl_file:
        reader = csv.reader(lvl_file)
        for row in reader:
            lvls[row[1]] = row[0]  # lvls[lvl] = game
    return lvls


def get_game_from_sonic_lvl(lvl_id):
    lvl_set = all_sonic_lvls()
    if lvl_id in lvl_set:
        return lvl_set[lvl_id]
    else:
        raise KeyError("Level id not found in train or test set")


def is_sonic_lvl(lvl_id):
    return lvl_id in all_sonic_lvls()


def all_sonic_lvls():
    return {**load_sonic_lvl_set(True), **load_sonic_lvl_set(False)}


def load_atari_lvl_set():
    lvls = []
    with open("data/gym-atari.csv") as lvl_file:
        reader = csv.reader(lvl_file)
        for row in reader:
            lvls.append(row[0])
    return lvls


############################################################################################


class FrameStack:
    def __init__(self, capacity, default_frame, dtype=np.uint8):
        self.capacity = capacity
        self.default_frame = default_frame.astype(dtype)
        self.reset()

    def append(self, frame):
        self._tensor = np.roll(self._tensor, 1)
        self.__setitem__(0, frame)

    def reset(self):
        self._tensor = np.repeat(
            np.expand_dims(self.default_frame, axis=-1), self.capacity, axis=-1
        )

    def __getitem__(self, idx):
        return self._tensor[:, :, idx]

    def __setitem__(self, key, value):
        try:
            self._tensor[..., key] = value
        except:
            self._tensor[..., key] = np.squeeze(value)

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

    def __init__(self, rollout_length, i_rew_running_stats=None, past_trajectory=None):
        # keeping track of attributes used in update_models
        self.states = []
        self.rews_i = []
        self.rews_e = []
        self.old_act_probs = []
        self.vals_e = []
        self.vals_i = []
        self.exp_targets = []
        self.actions = []

        self.i_rew_running_stats = i_rew_running_stats

        self.rollout_length = rollout_length

        # if given a past trajectory to resume from, the first elements in this trajectory will be the last from the old one
        if past_trajectory != None:
            self.rews_i.append(past_trajectory.rews_i[-1])
            self.rews_e.append(past_trajectory.rews_e[-1])
            self.old_act_probs.append(past_trajectory.old_act_probs[-1])
            self.vals_e.append(past_trajectory.vals_e[-1])
            self.vals_i.append(past_trajectory.vals_i[-1])
            self.exp_targets.append(past_trajectory.exp_targets[-1])

    def add(self, state, rew_e, rew_i, exp_target, act_prob_tuple, val_e, val_i):
        self.states.append(np.squeeze(state))
        self.rews_e.append(rew_e)
        self.rews_i.append(rew_i)
        self.i_rew_running_stats.push(rew_i)
        self.old_act_probs.append(act_prob_tuple[0])
        self.actions.append(act_prob_tuple[1])
        self.vals_e.append(val_e)
        self.vals_i.append(val_i)
        self.exp_targets.append(np.squeeze(exp_target))

    def _lists_to_ndarrays(self):
        self.states = np.asarray(self.states)[: self.rollout_length]
        self.rews_i = np.asarray(self.rews_i)[: self.rollout_length]
        self.rews_e = np.asarray(self.rews_e)[: self.rollout_length]
        self.old_act_probs = np.asarray(self.old_act_probs)[: self.rollout_length]
        self.vals_e = np.asarray(self.vals_e)[: self.rollout_length + 1]
        self.vals_i = np.asarray(self.vals_i)[: self.rollout_length + 1]
        self.exp_targets = np.asarray(self.exp_targets)[: self.rollout_length]
        self.actions = np.asarray(self.actions)[: self.rollout_length]

    def normalize_rews_i(self, rews_i):
        return rews_i / (self.i_rew_running_stats.standard_deviation() + 1e-8)

    def end_trajectory(
        self, e_rew_coeff, i_rew_coeff, gamma_i, gamma_e, lam, last_val_i, last_val_e
    ):
        """calculate gaes, rewards-to-go, convert to numpy arrays."""
        self.vals_e.append(last_val_e)
        self.vals_i.append(last_val_i)
        self._lists_to_ndarrays()
        # normalize internal advantages, not external. Uses running mean and std
        self.rews_i = 0.1 * self.normalize_rews_i(self.rews_i)
        deltas = self.rews_e + gamma_e * self.vals_e[1:] - self.vals_e[:-1]
        e_adv = self.discount_cumsum(deltas, gamma_e * lam)
        deltas = self.rews_i + gamma_i * self.vals_i[1:] - self.vals_i[:-1]
        i_adv = self.discount_cumsum(deltas, gamma_i * lam)
        # calculate advantages and returns
        self.gaes = np.expand_dims(
            e_rew_coeff * np.asarray(e_adv) + i_rew_coeff * np.asarray(i_adv), axis=1
        ).astype(np.float32)
        self.rews_i = np.asarray(self.discount_cumsum(self.rews_i, gamma_i)).astype(
            np.float32
        )
        self.rews_e = np.asarray(self.discount_cumsum(self.rews_e, gamma_e)).astype(
            np.float32
        )

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


#########################################################################################


def save_hyp_dict_to_file(filename, hyp_dict):
    with open(filename, "w") as f:
        json_hyp_dict = json.dump(hyp_dict, f)


def load_hyp_dict_from_file(filename):
    with open(filename, "r") as f:
        json_hyp_dict = json.load(f)
    return json_hyp_dict


#########################################################################################


def random_actions(env, steps):
    obs, rew, done, info = None, 0, False, {}
    for step in range(steps):
        obs, rew, done, info = env.step(env.action_space.sample())
    return obs, rew, done, info


#########################################################################################


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())
