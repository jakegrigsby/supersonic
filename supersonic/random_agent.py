import random
import os

import retro

from supersonic import environment, utils, logger, camera


class SonicRandomAgent:
    def __init__(self, env_id, use_custom_env=True):
        self.env_id = env_id
        if use_custom_env:
            self.env = environment.auto_env(env_id)
        else:
            game = utils.get_game_from_sonic_lvl(env_id)
            self.env = environment.SonicDiscretizer(retro.make(game=game, state=env_id))
        self.death_coords = []
        self.training_steps = 0

    def run(self, episodes, max_steps=100, render=False):
        for self.episode in range(episodes):
            self.env.reset()
            step = 0
            done = False
            while step < max_steps and not done:
                if render:
                    self.env.render()
                _, rew, done, info = self.env.step(self.choose_action())
                print(rew)
                step += 1

    def choose_action(self):
        if random.random() < 0.15:
            return self.env.action_space.sample()
        return 1


if __name__ == "__main__":
    x = SonicRandomAgent("GreenHillZone.Act1", False)
    cam = camera.Camera(x)
    cam.start_recording("testvideo.mov")
    x.run(40, render=True, max_steps=1000)
    cam.stop_recording()
