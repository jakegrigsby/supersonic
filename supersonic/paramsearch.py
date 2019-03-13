import numpy as np

class DiscreteSearchSpace:

    def __init__(self, iterable):
        self.space = iterable
        self.probs = [(1/len(self.space)) for point in self.space]
    
    def sample(self, nb):
        return np.random.choice(self.space, size=nb, replace=False, p=self.probs)

    def update(self, elem, step):
        loc = self.space.index(elem)
        adjustment = step / (len(self.space) - 1)
        self.probs = [prob - adjustment if idx != loc else prob + step for (idx, prob) in enumerate(self.probs)]


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