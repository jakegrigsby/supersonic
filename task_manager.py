from collections import namedtuple

import tensorflow as tf
from mpi4py import MPI 

from agent import Agent


Task = namedtuple('Task','env_id hyp_dict log_dir')

class TrainingManager:

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank() 

    def __init__(self, task_list, agent_type):
        self.local_task = comm.scatter(task_list, root=0)
        self.agent = agent_type(self.local_task['env_id'], self.local_task['hyp_dict'], self.local_task['log_dir'])

        #there are 4 gpus per cluster. Assign one agent to each.
        self.device = 'gpu:{}'.format(rank % 4)
    
    def train(self, epochs):
        self.agent.train(epochs, self.device)
    
    def gather_logs(self):
        self.dataframes = comm.gather(self.agent.log, root=0)
        #can EpisodeLog object be made pickleable?
    
    def gather_weights(self):
        self.model_weights = comm.Gather(self.agent.weights, root=0)
    
    def send_weights(self, weights):
        self.agent.weigts = comm.Scatter(weights, root=0)




