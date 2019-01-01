import random
import os

import environment
from episode_log import EpisodeLog
import camera



class SonicRandomAgent:

    def __init__(self, env_id, log_filepath=None):
        self.env_id = env_id
        self.env = environment.auto_env(env_id)
        self.fieldnames = ['max_x', 'reward', 'score']
        self.logging = bool(log_filepath)
        self.log_filepath = log_filepath

    def run(self, episodes, max_steps=100, render=False):
        if self.logging:
            log_filename = os.path.join(self.log_filepath, '{}_{}'.format(self.__class__.__name__, self.env_id))
            self.log = EpisodeLog(log_filename, self.fieldnames)
        self.reset_ep_stats()
        for ep in range(episodes):
            self.env.reset()
            step = 0
            done = False
            while step < max_steps and not done:
                if render:
                    self.env.render()
                _, rew, done, info = self.env.step(self.choose_action())
                self.update_ep_stats(rew, done, info)
                step += 1

            if self.logging: 
                self.log.write(self.get_ep_dict())
        self.log.close()

    def choose_action(self):
        if random.random() < 0.15:
            return self.env.action_space.sample()
        return 1

    def update_ep_stats(self, rew, done, info):
        self.ep_rew += rew
        self.score = info['score']
        self.max_x = max(self.max_x, info['x'])

    def get_ep_dict(self):
        ep_dict = {'max_x': self.max_x, 
         'reward': self.ep_rew, 
         'score': self.score}
        self.reset_ep_stats()
        return ep_dict

    def reset_ep_stats(self):
        self.max_x = 0
        self.ep_rew = 0
        self.score = 0


if __name__ == '__main__':
    x = SonicRandomAgent('GreenHillZone.Act1')
    cam = camera.Camera(x)
    cam.start_recording('testvideo.mov')
    x.run(1, render=False, max_steps=500)
    cam.stop_recording()