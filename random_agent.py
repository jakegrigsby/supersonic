import random
import os

import environment
from logger import EpisodeLog, Logger
import camera



class SonicRandomAgent:

    def __init__(self, env_id):
        self.env_id = env_id
        self.env = environment.auto_env(env_id)
        self.death_coords = []
        self.training_steps = 0

    def run(self, episodes, max_steps=100, render=False):
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
                step += 1

    def choose_action(self):
        if random.random() < 0.15:
            return self.env.action_space.sample()
        return 1

if __name__ == '__main__':
    x = SonicRandomAgent('GreenHillZone.Act1', log_filepath='logs/')
    cam = camera.Camera(x)
    cam.start_recording('testvideo.mov')
    x.run(40, render=True, max_steps=100)#4500)
    cam.stop_recording()