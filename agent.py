import tensorflow as tf

class BaseAgent:
    """
    Build a deep RL agent to learn on env generated from env_id.

    All the attributes can be specified by a python dictionary (hyp_dict). This should make hyperparameter search
    pretty straight forward.
    """
    def __init__(self, env_id, hyp_dict, log_dir):
        #self.log = EpisodeLog()
        pass

    def train(self, epochs, device):
        pass

    @property
    def weights(self):
        """return current model weights as numpy array."""
        pass

class MetaLearningAgent(BaseAgent):
    """
    The version of PPO used for meta learning could be different than sprinting.
    """

class SpeedLearningAgent(BaseAgent):
    """
    PPO version optimized for progress in < 1 million frames.
    """
