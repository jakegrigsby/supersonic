import math

import numpy as np
import gym
import retro
import cv2
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros

from supersonic import utils

# don't use the gpu for frame scaling
USE_GPU_FOR_FRAME_SCALING = False
cv2.ocl.setUseOpenCL(USE_GPU_FOR_FRAME_SCALING)


ENV_BUILDER_REGISTRY = {}


def env_builder(keys):
    if type(keys) == str:
        keys = [keys]
    elif not type(keys) == list:
        keys = list(keys)

    def register(func):
        for key in keys:
            ENV_BUILDER_REGISTRY[key] = func
        return func

    return register


def auto_env(env_id, **kwargs):
    """
   automatically finds the right environment builder function for the input environment id.

   Connects id's to funcs using ENV_BUILDER_REGISTRY. Register a new function using the env_builder decorator:
    @env_builder([lvl1, lvl2, lvl3, ...])
    def this_builds_env():

    this will register lvl1, lvl2 etc and connect them to this_builds_env().
   """
    if env_id in ENV_BUILDER_REGISTRY:
        return ENV_BUILDER_REGISTRY[env_id](env_id, **kwargs)
    else:
        return base_env(env_id, **kwargs)


@env_builder(utils.all_sonic_lvls().keys())
def build_sonic(lvl):
    game = utils.get_game_from_sonic_lvl(lvl)
    env = base_env(game, lvl)
    env = WarpFrame(env)
    env = ClipScaleReward(env, scale=0.01, lower_bound=-1, upper_bound=1)
    env = ScaledFloatFrame(env)
    env = SonicDiscretizer(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = StickyActionEnv(env)
    env = FrameStackWrapper(env)
    env = SonicInfoWrapper(env)
    env = AllowBacktrackingAddMaxSteps(env)
    env.recognized = "Sonic"
    return env


@env_builder(utils.load_atari_lvl_set())
def build_atari(lvl):
    env = base_env(lvl)
    env = ClipScaleReward(env, scale=.01, lower_bound=-1, upper_bound=1.)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = StickyActionEnv(env)
    env = FrameStackWrapper(env)
    return env

@env_builder(utils.load_mario_lvl_set())
def build_mario(lvl):
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    env = base_env(lvl)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = WarpFrame(env)
    env = ClipScaleReward(env, scale=.01, lower_bound=-1, upper_bound=1.)
    env = ScaledFloatFrame(env)
    env = MaxAndSkipEnv(env)
    env = StickyActionEnv(env)
    env = FrameStackWrapper(env)
    env.recognized = "Mario"
    return env


def base_env(*args, **kwargs):
    """
    auto-switching between gym, gym-retro and gym-super-mario-bros
    
    EAFP
    """
    try:
        # regular gym
        env = gym.make(*args, **kwargs)
    except:
        try:
            # gym retro
            env = retro.make(*args, **kwargs)
        except:
            # gym-super-mario-bros
            env = gym_super_mario_bros.make(*args, **kwargs)
    env.recognized = None
    return env


class ClipScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale=0.1, lower_bound=-1, upper_bound=1):
        super().__init__(env)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scale = scale

    def reward(self, rew):
        return self.scale * self.clip(self.lower_bound, rew, self.upper_bound)

    def clip(self, min_val, value, max_val):
        return min(max(min_val, value), max_val)


class WarpFrame(gym.ObservationWrapper):
    """
    84 x 84, grayscale
    """

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return np.squeeze(frame)


class StickyActionEnv(gym.Wrapper):
    """
    Sticky-frames nondeterminism
    """

    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        action = self.last_action if np.random.uniform() < self.p else action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Look at every k'th frame. More memory efficient. More realistic (human) reaction time.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape[:-1], dtype=np.float32
        )
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        obs = obs.astype(np.float32)
        return obs / 255


class DynamicNormalize(gym.Wrapper):
    """
    Normalize observations and rewards using a running mean and std dev.

    adapt_until: adjust the mean and variance for this many steps
    """

    def __init__(self, env, obs_min=-5, obs_max=5, adapt_until=10000, normalize_rew=False):
        super().__init__(env)
        self.adapt_until = adapt_until
        self.reset_normalization()
        self.normalize_rew = normalize_rew
        self.obs_min = obs_min
        self.obs_max = obs_max

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs, rew = self.dynamic_normalize(obs, rew)
        return obs, rew, done, info

    def update_stats(self, obs, rew):
        self.count += 1
        delta = obs - self.o_mean
        self.o_mean += delta / self.count
        delta2 = obs - self.o_mean
        self.o_m2 += delta * delta2

        delta = rew - self.r_mean
        self.r_mean += delta / self.count
        delta2 = rew - self.r_mean
        self.r_m2 += delta * delta2

    def dynamic_normalize(self, obs, rew):
        obs = obs.astype(np.float32)
        if self.count < self.adapt_until:
            self.update_stats(obs, rew)
        obs_var = self.o_m2 / ((self.count - 1) + 1e-4)
        obs_std = np.sqrt(obs_var)
        obs = (obs - self.o_mean) / (obs_std + 1e-5)
        obs = np.clip(obs, self.obs_min, self.obs_max)

        if self.normalize_rew:
            rew_var = self.r_m2 / ((self.count - 1) + 1e-4)
            rew_std = math.sqrt(rew_var)
            rew = (rew - self.r_mean) / (rew_std + 1e-5)

        return obs, rew

    def reset_normalization(self):
        self.count = 0
        self.o_mean = np.squeeze(np.zeros(self.env.observation_space.shape).astype(np.float32))
        self.o_m2 = np.squeeze(np.zeros(self.env.observation_space.shape).astype(np.float32))
        self.o_variance = np.squeeze(np.zeros(self.env.observation_space.shape).astype(np.float32))
        self.r_mean, self.r_m2, self.r_var = 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStackWrapper(gym.Wrapper):
    """
    Most game-based RL envs are partially observable. It's common to compensate using a stack of the k most
    recent frames, giving the agent a sense of direction and speed. This wrapper keeps track of that frame stack
    and returns a numpy array that can be fed straight into the NN.
    """

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = utils.FrameStack(
            capacity=k, default_frame=self.env.reset(), dtype=np.float32
        )

    def reset(self):
        self.env.reset()
        self.frames.reset()
        return self.stack

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.stack, rew, done, info

    @property
    def stack(self):
        return np.expand_dims(self.frames.stack, axis=0)


class RewardScaler(gym.RewardWrapper):
    """
    Rescale rewards to best range for PPO
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktrackingAddMaxSteps(gym.Wrapper):
    """
    Let the agent go backwards without losing reward.
    Important for Sonic.
    """
    def __init__(self, env, initial_max_steps=300, final_max_steps=4500):
        super().__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.current_max_steps = initial_max_steps
        self.final_max_steps = final_max_steps
        self._step_count = 0
        self._last_info = {}

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        self._step_count = 0
        self._last_info = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._step_count >= self.current_max_steps:
            obs, rew, done, info = self.env.reset(), 0, True, self._last_info
            info["FORCED EXIT"] = True
            #increase max steps. helpful for high exploration environments
            self.current_max_steps = min(self.current_max_steps + 50, self.final_max_steps)
        else:
            obs, rew, done, info = self.env.step(action)
            self._cur_x += rew
            rew = max(0, self._cur_x - self._max_x)
            self._max_x = max(self._max_x, self._cur_x)
            self._step_count += 1
            self._last_info = info
        return obs, rew, done, info

class SonicInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["stage"] = self.env.statename
        return obs, rew, done, info

class Gauntlet(gym.Wrapper):
    """
    String multiple sonic levels together *as long as they're from the same game*
    """
    def __init__(self, *lvls):
        self.env = auto_env(lvls[0])
        super().__init__(self.env)
        self.recognized = self.env.recognized
        self._lvls = [*lvls]
        self.cur_lvl = 0
   
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            if info["screen_x"] >= info["screen_x_end"]:
            # the level is complete
                if self.cur_lvl < len(self._lvls) - 1:
                    # this isn't the last level in the progression. Don't stop
                    self.cur_lvl += 1
                    self.env.statename = self._lvls[self.cur_lvl]
                    self.env.load_state(self.env.statename, retro.data.Integrations.STABLE)
                    self.env.reset()
                    done = False
                else:
                    # time to reset the progression
                    self.reset()
                    done = True
        return obs, rew, done, info
    
    def reset(self):
        if self.cur_lvl > 0:
            self.cur_lvl = 0
            self.env.statename = self._lvls[self.cur_lvl]
            self.env.load_state(self.env.statename, retro.data.Integrations.STABLE)
        return self.env.reset()
    

@env_builder('GreenHillZoneComplete')
def greenhillzonecomplete(_):
    env = Gauntlet(*[f"GreenHillZone.Act{n}" for n in range(1,4)])
    return env
            
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = [
            "B",
            "A",
            "MODE",
            "START",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "C",
            "Y",
            "X",
            "Z",
        ]
        actions = [
            ["LEFT"],
            ["RIGHT"],
            ["LEFT", "DOWN"],
            ["RIGHT", "DOWN"],
            ["DOWN"],
            ["DOWN", "B"],
            ["B"],
        ]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()
