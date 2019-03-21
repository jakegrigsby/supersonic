import os

import numpy as np
from mpi4py import MPI

from supersonic import task_manager, utils, agent

class DiscreteSearchSpace:

    def __init__(self, iterable):
        self.space = list(iterable)
        self.probs = [(1/len(self.space)) for point in self.space]
    
    def sample(self, nb):
        point = np.random.choice(self.space, size=nb, replace=False, p=self.probs)
        return point if nb > 1 else point[0]

    def update(self, point, step):
        loc = self.space.index(point)
        adjustment = step / (len(self.space) - 1)
        self.probs = [prob - adjustment if idx != loc else prob + step for (idx, prob) in enumerate(self.probs)]


class PowerofNSearchSpace(DiscreteSearchSpace):
    """Discrete search space of powers of `n`. Range [n**low, n**high)"""
    def __init__(self, n, low, high):
        space = np.power(np.asarray([n for i in range(high-low)]), np.arange(low, high))
        super().__init__(space)


class ContinuousSearchSpace:

    def __init__(self, low, high, distrib='uniform', mean=None, std=None):
        self.range = [low, high]
        if distrib=='uniform':
            self.distrib = np.random.uniform
            self.uniform = True
        elif distrib=='normal':
            if mean==None or std==None:
                raise ValueError("args `mean` and `std` not optional for distrib type `normal`")
            self.uniform = False
            self.mean = mean
            self.std = std
        else:
            raise ValueError("param `distrib` must be `uniform` or `normal`")
    
    def sample(self, nb):
        if self.uniform:
            return np.random.uniform(self.range[0], self.range[1], nb)
        else:
            choices = np.clip(np.random.normal(self.mean, self.std, nb), self.range[0], self.range[1])
    
    def update(self, point, step):
        mid = (self.range[1] - self.range[0]) / 2
        if point < mid:
            if step > 0:
                self.range[1] = max(self.range[0], self.range[1] - step)
            else:
                self.range[0] = min(self.range[1], self.range[0] + step) 
        elif point > mid:
            if step > 0:
                self.range[0] = min(self.range[1], self.range[0] + step)
            else:
                self.range[1] = max(self.range[0], self.range[1] - step) 


def bucketize_space(space, buckets):
    assert isinstance(space, ContinuousSearchSpace)
    bucketized_range = np.linspace(space.range[0], space.range[1], buckets)
    return DiscreteSearchSpace(bucketized_range)


class AgentParamFinder:
    lvl = DiscreteSearchSpace(utils.all_sonic_lvls().keys())
    hyperparameters = [
    rollout_length = DiscreteSearchSpace(np.arange(64,2000)),
    ppo_batch_size = PowerofNSearchSpace(2, 1, 11),
    exp_batch_size = PowerofNSearchSpace(2, 1, 8),
    ppo_opt_steps = DiscreteSearchSpace(np.arange(3,30,2)),
    ppo_clip_value = DiscreteSearchSpace([.1,.2,.3]),
    e_rew_coeff = ContinuousSearchSpace(1.,3.),
    i_rew_coeff = ContinuousSearchSpace(1.,3.),
    exp_train_prop = ContinuousSearchSpace(.1,1.),
    exp_lr = ContinuousSearchSpace(1e-4, 1e-2),
    ppo_lr = ContinuousSearchSpace(1e-4, 1e-2),
    vis_model = DiscreteSearchSpace(['NatureVision']),
    policy_model = DiscreteSearchSpace(['NaturePolicy']),
    val_model = DiscreteSearchSpace(['VanillaValue']),
    exp_target_model = DiscreteSearchSpace(['NatureVision']),
    exp_train_model = DiscreteSearchSpace(['NatureVision']),
    ]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    round_num = 0

    def __init__(self, epochs):
        self.epochs = epochs
    
    def find_params(self):
        for epoch in range(self.epochs):
            self.round_num = epoch
            if self.rank == 0:
                hyp_dict = self.sample()
            hyp_dict = comm.bcast(hyp_dict, root=0) 
            log_dir = 'paramsearch/round{}/run{}'.format(self.round_num, self.rank)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            task_list = [task_manager.Task(self.lvl.sample(), hyp_dict, log_dir) for worker in range(self.size)]
            self.training_manager = task_manager.TrainingManager(task_list, agent.ppo_agent)
            self.deploy()
            utils.save_hyp_dict_to_file(log_dir + "_hyp_dict.json", hyp_dict)

    def deploy(self):
        """Deploy new hyperparameter settings on available nodes"""
        self.training_manager.train()

    def evaluate(self):
        """Evaluate results of most recent param deployment"""
        pass
    
    def adjust(self):
        """Adjust search spaces"""
        pass
    
    def sample(self):
        """Sample new params from the adjusted search spaces"""
        hyp_dict = {
            "exp_lr":self.exp_lr.sample(),
            "ppo_lr":self.ppo_lr.sample(),
            "vis_model":self.vis_model.sample(),
            "policy_model":self.policy_model.sample(),
            "val_model":self.val_model.sample(),
            "exp_target_model":self.exp_target_model.sample(),
            "exp_train_model":self.exp_train_model.sample(),
            "exp_net_opt_steps":self.exp_net_opt_steps.sample(),
            "gamma_i":self.gamma_i.sample(),
            "gamma_e":self.gamma_e.sample(),
            "rollout_length":self.rollout_length.sample(),
            "ppo_net_opt_steps":self.ppo_ne_opt_steps.sample(),
            "e_rew_coeff":self.e_rew_coeff.sample(),
            "i_rew_coeff":self.i_rew_coeff.sample(),
            "exp_train_prop":self.exp_train_prop.sample(),
            "lam":self.lam.sample(),
            "exp_batch_size":self.exp_batch_size.sample(),
            "ppo_batch_size":self.ppo_batch_size.sample(),
            "ppo_clip_value":self.ppo_clip_value.sample(),
            "update_mean_gae_until":self.update_mean_gae_until.sample(),
        }
        return hyp_dict