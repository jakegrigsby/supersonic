import gym

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
        self.env = env


