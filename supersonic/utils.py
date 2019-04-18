import csv
import os
import json
import operator
import random
import collections

import numpy as np
import scipy.signal
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

    def __init__(self, capacity, default_frame, dtype=np.uint8):
        self.capacity = capacity
        self.default_frame = default_frame.astype(dtype)
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
    def __init__(self, rollout_length, past_trajectory=None):
        # keeping track of attributes used in update_models
        self.states = []
        self.rews_i = []
        self.rews_e = []
        self.old_act_probs = []
        self.vals_e = []
        self.vals_i = []
        self.exp_targets = []
        self.actions = []

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
        self.old_act_probs.append(np.expand_dims(act_prob_tuple[0], axis=0))
        self.actions.append(act_prob_tuple[1])
        self.vals_e.append(val_e)
        self.vals_i.append(val_i)
        self.exp_targets.append(np.squeeze(exp_target))

    def _lists_to_ndarrays(self):
        self.states = np.asarray(self.states)[:self.rollout_length]
        self.rews_i = np.asarray(self.rews_i)[:self.rollout_length]
        self.rews_e = np.asarray(self.rews_e)[:self.rollout_length]
        self.old_act_probs = np.asarray(self.old_act_probs)[:self.rollout_length]
        self.vals_e = np.asarray(self.vals_e)[:self.rollout_length+1]
        self.vals_i = np.asarray(self.vals_i)[:self.rollout_length+1]
        self.exp_targets = np.asarray(self.exp_targets)[:self.rollout_length]
        self.actions = np.asarray(self.actions)[:self.rollout_length]

    def end_trajectory(self, gamma_i, gamma_e, lam, last_val_i, last_val_e):
        """calculate gaes, rewards-to-go, convert to numpy arrays."""
        #calculate advantages
        self.vals_e.append(last_val_e)
        self.vals_i.append(last_val_i)
        self._lists_to_ndarrays()
        deltas = self.rews_e + gamma_e * self.vals_e[1:] - self.vals_e[:-1]
        e_adv = self.discount_cumsum(deltas, gamma_e * lam)
        deltas = self.rews_i + gamma_i * self.vals_i[1:] - self.vals_i[:-1]
        i_adv = self.discount_cumsum(deltas, gamma_i * lam)
        self.gaes = np.expand_dims(np.asarray(e_adv) + np.asarray(i_adv), axis=1).astype(np.float32)
        self.i_rews = np.asarray(self.discount_cumsum(self.rews_i, gamma_i)).astype(np.float32)
        self.e_rews = np.asarray(self.discount_cumsum(self.rews_e, gamma_e)).astype(np.float32)

    def discount_cumsum(self, x, discount):
        """
        r = x[::-1]
        a = [1, -discount]
        b = [1]
        y = scipy.signal.lfilter(b, a, x=x)
        return np.squeeze(y[::-1])
        """
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

#########################################################################################

def random_actions(env, steps):
        obs, rew, done, info = None, 0, False, {}
        for step in range(steps):
            obs, rew, done, info = env.step(env.action_space.sample())
        return obs, rew, done, info

#########################################################################################
class SegmentTree:
    """
    Abstract SegmentTree data structure used to create PrioritizedMemory.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity, operation, neutral_element):

        #powers of two have no bits in common with the previous integer
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2"
        self._capacity = capacity

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * capacity)]

        self._operation = operation

        self.next_index = 0

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]
        

class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class PartitionedRingBuffer(object):
    """
    Buffer with a section that can be sampled from but never overwritten.
    Used for demonstration data (DQfD). Can be used without a partition,
    where it would function as a fixed-idxs variant of RingBuffer.
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.data = [None for _ in range(maxlen)]
        self.permanent_idx = 0
        self.next_idx = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            raise KeyError()
        return self.data[idx % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        self.data[(self.permanent_idx + self.next_idx)] = v
        self.next_idx = (self.next_idx + 1) % (self.maxlen - self.permanent_idx)

    def load(self, load_data):
        assert len(load_data) < self.maxlen, "Must leave space to write new data."
        for idx, data in enumerate(load_data):
            self.length += 1
            self.data[idx] = data
            self.permanent_idx += 1
            
class PrioritizedMemory:

    def __init__(self, limit, alpha=.4, start_beta=1., end_beta=1., steps_annealed=1, **kwargs):

        self.ignore_episode_boundaries = True

        #The capacity of the replay buffer
        self.limit = limit

        #Transitions are stored in individual RingBuffers, similar to the SequentialMemory.
        self.actions = PartitionedRingBuffer(limit)
        self.rewards = PartitionedRingBuffer(limit)
        self.terminals = PartitionedRingBuffer(limit)
        self.observations = PartitionedRingBuffer(limit)
        self.exp_targets = PartitionedRingBuffer(limit)

        assert alpha >= 0
        #how aggressively to sample based on TD error
        self.alpha = alpha
        #how aggressively to compensate for that sampling. This value is typically annealed
        #to stabilize training as the model converges (beta of 1.0 fully compensates for TD-prioritized sampling).
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps_annealed = steps_annealed

        #SegmentTrees need a leaf count that is a power of 2
        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2

        #Create SegmentTrees with this capacity
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.

        #wrapping index for interacting with the trees
        self.next_index = 0

    def append(self, observation, action, reward, terminal, exp_target):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.exp_targets.append(exp_target)
        #The priority of each new transition is set to the maximum
        self.sum_tree[self.next_index] = self.max_priority ** self.alpha
        self.min_tree[self.next_index] = self.max_priority ** self.alpha

        #shift tree pointer index to keep it in sync with RingBuffers
        self.next_index = (self.next_index + 1) % self.limit

    def _sample_proportional(self, batch_size):
        #outputs a list of idxs to sample, based on their priorities.
        idxs = list()

        for _ in range(batch_size):
            mass = random.random() * self.sum_tree.sum(0, self.limit - 1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            idxs.append(idx)

        return idxs

    def sample(self, batch_size, beta=1.):
        idxs = self._sample_proportional(batch_size)

        #importance sampling weights are a stability measure
        importance_weights = list()
        exp_targets = list()

        #The lowest-priority experience defines the maximum importance sampling weight
        prob_min = self.min_tree.min() / self.sum_tree.sum()
        max_importance_weight = (prob_min * self.nb_entries)  ** (-beta)
        obs_t, act_t, rews, obs_t1, dones = [], [], [], [], []

        experiences = list()
        for idx in idxs:
            terminal0 = self.terminals[idx]
            while terminal0:
                idx = np.random.choice(np.arange(0, len(self.actions)))
                terminal0 = self.terminals[idx]

            #probability of sampling transition is the priority of the transition over the sum of all priorities
            prob_sample = self.sum_tree[idx] / self.sum_tree.sum()
            importance_weight = (prob_sample * self.nb_entries) ** (-beta)
            #normalize weights according to the maximum value
            importance_weights.append(importance_weight/max_importance_weight)

            # Code for assembling stacks of observations and dealing with episode boundaries is borrowed from
            # SequentialMemory
            state0 = np.squeeze(self.observations[idx])
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal1 = self.terminals[idx+1]
            state1 = np.squeeze(self.observations[idx+1])

            exp_targets.append(self.exp_targets[idx])

            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size

        # Return a tuple whre the first batch_size items are the transititions
        # while -3 is the importance weights of those transitions and -2 is
        # the idxs of the buffer (so that we can update priorities later). -1
        # is the exploration targets, which we save so they do not have to be 
        # recalculated.
        return tuple(list(experiences)+ [importance_weights, idxs, exp_targets])

    def update_priorities(self, idxs, priorities):
        #adjust priorities based on new TD error
        for i, idx in enumerate(idxs):
            assert 0 <= idx < self.limit
            priority = priorities[i] ** self.alpha
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def calculate_beta(self, current_step):
        a = float(self.end_beta - self.start_beta) / float(self.steps_annealed)
        b = float(self.start_beta)
        current_beta = min(self.end_beta, a * float(current_step) + b)
        return current_beta

    def get_config(self):
        config = super(PrioritizedMemory, self).get_config()
        config['alpha'] = self.alpha
        config['start_beta'] = self.start_beta
        config['end_beta'] = self.end_beta
        config['beta_steps_annealed'] = self.steps_annealed

    @property
    def nb_entries(self):
        """Return number of observations
        # Returns
            Number of observations
        """
        return len(self.observations)
    
Experience = collections.namedtuple('Experience', 'state0, action, reward, state1, terminal1')