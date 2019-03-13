import numpy as np

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
    rollout_length = DiscreteSearchSpace(np.arange(64,2000))
    ppo_batch_size = PowerofNSearchSpace(2, 1, 11)
    exp_batch_size = PowerofNSearchSpace(2, 1, 8)
    ppo_opt_steps = DiscreteSearchSpace(np.arange(3,30,2))
    ppo_clip_value = DiscreteSearchSpace([.1,.2,.3])
    e_rew_coeff = ContinuousSearchSpace(1.,3.)
    i_rew_coeff = ContinuousSearchSpace(1.,3.)
    exp_train_prop = ContinuousSearchSpace(.1,1.)
    exp_lr = ContinuousSearchSpace(1e-4, 1e-2)
    ppo_lr = ContinuousSearchSpace(1e-4, 1e-2)
    #TODO: Add choices between different vision, policy, value and exploration models

    def __init__(self):
        raise NotImplementedError()
    
    def deploy(self):
        """Deploy new hyperparameter settings on available nodes"""
        raise NotImplementedError()
    
    def evaluate(self):
        """Evaluate results of most recent param deployment"""
        raise NotImplementedError()
    
    def adjust(self):
        """Adjust search spaces"""
        raise NotImplementedError()
    
    def sample(self):
        """Sample new params from the adjusted search spaces"""
        raise NotImplementedError()


