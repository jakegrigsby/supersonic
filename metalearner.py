from mpi4py import MPI
import numpy as np

import utils
import task_manager

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class MetaLearner:

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


