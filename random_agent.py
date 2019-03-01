import random
import os

import environment
from logger import EpisodeLog, Logger
import camera



class SonicRandomAgent:

    def __init__(self, env_id, log_filepath=None):
        self.env_id = env_id
        self.env = environment.auto_env(env_id)
        self.logging = bool(log_filepath)
        self.log_filepath = log_filepath
        self.logger = None
        self.death_coords = []
        self.training_steps = 0

    def run(self, episodes, max_steps=100, render=False):
        if self.logging:
            log_filename = os.path.join(self.log_filepath, '{}_{}'.format(self.__class__.__name__, self.env_id))
            self.logger = Logger(log_filename)
        self.reset_ep_stats()
        for self.episode in range(episodes):
            if self.episode % 10 == 0: print('episode: {}/{}'.format(self.episode, episodes))
            self.env.reset()
            step = 0
            done = False
            while step < max_steps and not done:
                if step % 100 == 0: print('step: {}/{}'.format(step, max_steps))
                if render:
                    self.env.render()
                _, rew, done, info = self.env.step(self.choose_action())
                self.update_ep_stats(rew, done, info)
                step += 1

            if self.logging: 
                self.logger.log_episode(self.get_ep_log())
        
        if self.logging: 
            self.logger.close()

    def choose_action(self):
        if random.random() < 0.15:
            return self.env.action_space.sample()
        return 1

    def update_ep_stats(self, rew, done, info):
        self.ep_rew += rew
        self.score = info['score']
        self.max_x = max(self.max_x, info['x'])

    def get_ep_log(self):
        ep_dict = {
            'episode_num': self.episode,
            'death_coords': self.death_coords,
            'training_steps': self.training_steps,
            'max_x': self.max_x, 
            'reward': self.ep_rew, 
            'score': self.score
            }
        self.reset_ep_stats()
        return EpisodeLog(ep_dict)

    def reset_ep_stats(self):
        self.max_x = 0
        self.ep_rew = 0
        self.score = 0
        self.death_coords = []
        self.training_steps = 0


if __name__ == '__main__':
    x = SonicRandomAgent('GreenHillZone.Act1', log_filepath='logs/')
    cam = camera.Camera(x)
    cam.start_recording('testvideo.mov')
    x.run(40, render=True, max_steps=100)#4500)
    cam.stop_recording()