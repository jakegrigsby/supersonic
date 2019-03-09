import os

import gym
import retro
import skvideo.io
import numpy as np

from supersonic import utils

class Camera(gym.Wrapper):
    """
    Recording gameplay footage.
    
    env: env to record. Can be a wrapped env.
    raw_frames: usually you want to record gameplay footage as seen by a human player (vs. the processed
                version shown to the agent.) If true this wrapper 'dives' down to become the lowest-level wrapper
                of your base env. That way, when it saves gameplay frames they are not altered. Defaults to true.
    highlight_buffer_capacity: If > 0, camera has ability to record save this many frames to video *after* they've already
                occured. Useful when the decision to record can only be made after a key event (like the first time a tough
                obstacle is cleared).
    """
    def __init__(self, agent, raw_frames=True, highlight_buffer_capacity=250):
        self.env = agent.env
        super().__init__(self.env)
        self.recording = False
        self.raw_frames = raw_frames
        self.record_that_enabled = bool(highlight_buffer_capacity)
        if raw_frames:
            self._dive_down()
        self._buffer = utils.FrameStack(capacity=highlight_buffer_capacity, default_frame=self.env.reset()) 

    def reset(self):
        self._buffer.reset()
        return self.env.reset()

    def start_recording(self, output_path):
        """
        Begin recording gameplay clip.

        output_path: file path for recorded video
        """
        filename, extension = os.path.splitext(output_path)
        filename += '.mp4'
        self.rec_output_path = filename
        self.recording = True
        self.clip, self.actions = [], []
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.record_that_enabled:
            self._buffer.append(obs)
        if self.recording:
            self.clip.append(obs)
            self.actions.append(action)
        return obs, rew, done, info

    def _dive_down(self):
        """
        If raw_frames is set to true, camera needs access to unwrapped env to record
        unaltered observations. The wrapper system works like a singly linked list.
        This function positions the Camera module just above the unwrapped env.
        """
        if not hasattr(self.env, "env"):
            #the env is the base env
            return
        inner_wrapper = self.env
        while not inner_wrapper.env.unwrapped == inner_wrapper.env:
            inner_wrapper = inner_wrapper.env
            inner_wrapper.record = self._enable_record
            inner_wrapper.stop_recording = self._enable_stop_recording
            inner_wrapper.record_that = self._enable_record_that
        self.env = inner_wrapper.env
        inner_wrapper.env = self

    def stop_recording(self):
        """
        stop recording and save video to output path provided when recording began.
        """
        self.clip = np.asarray(self.clip, dtype=np.uint8)
        skvideo.io.vwrite(self.rec_output_path, self.clip)
        self.clip, self.actions = [], []

    def record_that(self, output_path):
        """
        Save the last `highlight_buffer_capacity` frames to a video. Allows
        recording to take place after the event occured.
        """
        if self.record_that_enabled:
            skvideo.io.vwrite(output_path, self._buffer.stack)
            self._buffer.reset()
        else:
            raise AttributeError()
    
    def _enable_record(self, output_dir):
        self.env.record(output_dir)
    
    def _enable_stop_recording(self):
        self.env.stop_recording()

    def _enable_record_that(self, output_dir):
        self.env.record_that(output_dir)
    
    record = start_recording




