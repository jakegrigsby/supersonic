from mpi4py import MPI
import numpy as np

from supersonic import utils, task_manager

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class MetaLearner:
"""
Our idea for a metalearning algorithm which consists of a few parts:
    1) A Deep RL agent. Should be capable of starting with some initial set of weights, and training on
        some RL environment, improving until it arrives at some (better) set of weights.
    2) A task picker. Should look at a history of which levels in the task train set we have trained on so far
        in order to decide which one to train on next. How it makes this decision will be left as a black-boxed
        subroutine and is probably the key to this working. Choosing at random would transform this alg into something
        closely resembling Reptile. The current plan is to try some basic online RL algs, where the states are going to be
        some representation of each task (at least a task number, but probably a set of sample observations or other info).
        The actions are a deterministic choice to stop training an agent on a level and move it to a new one. The rewards are
        the difference between the current meta-weights (the weights the agent was initialized with) and the final weights after
        x epochs of training the agent on that level. The idea here is that the task picker will be incentivized to choose tasks
        it thinks the agent has 'forgotten'.
"""
    def __init__(self, heads):
        self.heads = heads
        self.train_lvl_set = utils.load_sonic_lvl_set()
        init_lvls = np.random.choice(self.train_lvl_set.keys(), heads)
        self.lvl_ids = {lvl_id : name for (lvl_id, name) in zip(range(heads), self.train_lvl_set.keys())}
        self.lvl_names = {value : key for (key, value) in self.lvl_ids.items()}
        self.hist = np.empty(shape=(heads, 1))
        self.hist[:,0] = self._lvl_names_to_ids(init_lvls)

         self.agent_hyp_dict = {
                'exp_lr' : ,
                'policy_lr': ,
                'val_lr': ,
                'vis_model' : ,
                'policy_model': ,
                'val_model': ,
                'exp_target_model': ,
                'exp_train_model': ,
                'exp_train_model': ,
                'exp_net_opt_steps': ,
                'gamma': ,
                'lam': ,
                }
         self.log_dir = 'log_dir' #todo

         self.agent_epochs_per_run = agent_epochs_per_run

         self.max_hist_sequence = max_hist_sequence

         self.model_dir = model_dir #where metaweights are saved
        
    
    def train(iterations):
        iteration = 0
        init_weights = agent.ppo_agent('GreenHillZone.Act1', self.agent_hyp_dict, self.log_dir).weights
        while iteration < iterations:
            new_envs = self.choose_new_envs()
            new_env_names = self._lvl_ids_to_names(new_envs)
            task_list = []
            for name in new_env_names:
               task = task_manager.Task(env_id = name,
                                hyp_dict = self.agent_hyp_dict,
                                log_dir = self.log_dir,
                                )
               task_list.append(task)

            trainer = task_manager.TrainingManager(task_list, agent.ppo_agent)
            trainer.send_weights(init_weights)
            trainer.train(self.agent_epochs_per_run)
            trainer.gather_weights()
            rew = trainer.new_weights - init_weights
            seq_len = min(self.hist.shape[1], self.max_hist_sequence)
            s = self.hist[:, -1-seq_len:-1]
            s_1 = self.hist[:, -seq_len:]
            a = self.hist[:,-1]
            self.train_heads(s, a, rew, s_1)
            init_weights +=  np.mean(trainer.new_weights - init_weights)
        np.savez(init_weights, self.model_dir)

            
    
    def _lvl_names_to_ids(names_iterable):
        return [self.lvl_ids[name] for name in names_iterable]

    def _lvl_ids_to_names(id_iterable):
        return [self.lvl_names[lvl_id] for lvl_id in id_iterable]


