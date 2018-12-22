import copy
import numpy as np
import gym
import retro
import cv2
from numpy_ringbuffer import RingBuffer

#don't use the gpu for frame scaling
cv2.ocl.setUseOpenCL(False)


ENV_BUILDER_REGISTRY = {}


def env_builder(key):
    def register(func):
        ENV_BUILDER_REGISTRY[key] = func
        return func
    return register

def auto_env(**kwargs):
   """
   auto parsing of env id string to generate env for registered types using ENV_BUILDER_REGISTRY

   env = auto_env('GreenHillZoneAct1') returns correct SonicEnv
   env = auto_env('Montezumas') returns correct Montezuma, all with same func

   We don't really need this right now because it's all Sonic. But I set up the registry decorator so might as well.
   Will be nice later when we're switching between gym and retro a lot.
   """
   raise NotImplementedError()

class Camera(gym.Wrapper):
    """
    Recording gameplay footage.
    
    -Should be turned on and off automatically based on some set of rules
    we decide show if gameplay is worth recording.
    
    -Can decide to record something after it already happened. We'll need that for recoridng
    the first time it gets past tough obstacles in the level. (triggered by x and y pos). Will need
    a buffer that keeps the latest frames and can be saved to memory when somethign important happens.
    Otherwise it keeps getting overwritten. Use RingBuffer() probably. Search up numpy_ringbuffer

    -Also ability to start and stop recording like normal (things that haven't happened yet), and for as long
    as we need.

    -Implemented like the other wrappers below.

    -The info variable from env.step() is a dict with a bunch of game info like x pos, y pos, rings, lives, score and stuff like that.

    env = Camera(env)
    """
    def __init__(self, env):
        raise NotImplementedError()


@env_builder('Sonic')
def build_sonic(game, lvl):
    env = base_env(game, lvl)
    env = WarpFrame(env)
    env = AllowBacktracking(env)
    env = ClipReward(env, -5, 5)
    env = DynamicNormalize(env)
    env = SonicDiscretizer(env)
    env = StickyActionEnv(env)
    env = FrameStack(env)
    return env


def base_env(**kwargs):
    """
    auto-switching between gym and gym-retro
    """
    try:
        env = gym.make(**kwargs)
    except:
        env = retro.make(**kwargs)
    return env

class ClipReward(gym.RewardWrapper):

    def __init__(self, env, lower_bound = -1, upper_bound = 1):
        super().__init__(env)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.clip(self.lower_bound, rew, self.upper_bound), done, info

    def clip(min_val, value, max_val):
        return min(max(min_val, value), max_val)

class WarpFrame(gym.ObservationWrapper):
    """
    84 x 84, grayscale
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class StickyActionEnv(gym.Wrapper):
    """
    Sticky-frames nondeterminism
    """
    def __init__(self, env, p=.25):
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
        super().__init__(self, env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class DynamicNormalize(gym.ObservationWrapper):
    """
    Normalize observations and rewards using a running mean and std dev.

    adapt_until: adjust the mean and variance for this many steps
    """
    def __init__(self, env, adapt_until=10000):
        super().__init__(self, env)
        self.adapt_until = adapt_until
        self.reset_normalization()
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return dynamic_normalize(obs, rew), done, info

    def update_stats(self, obs, rew):
        self.count += 1

        delta = obs - self.o_mean
        self.o_mean += delta / self.count
        delta2 = obs - self.o_mean
        self.o_m2 += delta * delta2
        self.o_variance = self.o_m2 / self.count 

        delta = rew - self.r_mean
        self.r_mean += delta / self.count
        delta2 = rew - self.r_mean
        self.r_m2 += delta * delta2
        self.r_variance = self.r_m2 / self.count

    def dynamic_normalize(self, obs, rew):
        if self.count < self.adapt_until:
            self.update_stats(obs, rew)
        norm_obs = (obs - self.o_mean) / np.sqrt(self.o_var + 1e-5)
        norm_rew = (rew - self.r_mean) / (self.r_var + 1e-5)**(1/2) 
        return norm_obs, norm_rew

    def reset_normalization(self):
        self.count = 0
        self.o_mean, self.o_m2, self.o_variance = np.zeros(self.env.observation_space.shape).astype(np.uint8)
        self.r_mean, self.r_m2, self.r_var = .0

class FrameStack(gym.Wrapper):
    """
    Most game-based RL envs are partially observable. It's common to compensate using a stack of the k most
    recent frames, giving the agent a sense of direction and speed. This wrapper keeps track of that frame stack
    and returns a numpy array that can be fed straight into the NN.
    """ 
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = RingBuffer(capacity=k, dtype=np.uint8)

    def reset(self):
        reset_obs = self.env.reset()
        self.frames.extend([reset_obs] * k)
        return self.stack
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.stack, rew, done, info
    
    @property
    def stack(self):
        return np.array(self.frames)

        
class RewardScaler(gym.RewardWrapper):
    """
    Rescale rewards to best range for PPO
    """
    def reward(self, reward):
        return reward * .01


class AllowBacktracking(gym.Wrapper):
    """
    Let the agent go backwards without losing reward.
    Important for Sonic.
    """
    def __init__(self, env):
        super().__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


