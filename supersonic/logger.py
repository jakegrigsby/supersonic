import csv, glob, json, os, time, webcolors
from datetime import datetime

import numpy as np
import pandas as pd

COUNTER_DIGITS = 2

# Folder format:
# - run00
#   - episode00
#       - episode_logs.csv
#       - trajectory_logs.csv
#   - episode01
#       - ...


def get_next_run_folder(folder_base):
    folders = glob.glob(folder_base + "*")
    folders.sort()
    # print(folder_base, 'folders:', folders)

    if not folders:
        last_run = -1
    else:
        last_run_filename = folders[-1]
        _i = last_run_filename.rfind("_")
        last_run = int(last_run_filename[_i + 1 :])

    return folder_base + run_num_str(last_run + 1) + "/"


def run_num_str(run_num):
    run = str(run_num)
    while len(run) < COUNTER_DIGITS:
        run = "0" + run
    return run


class Logger:
    """
    Allows users to save information to disk at the end of every Episode and Trajectory.
    Eventually, this will be fed into a Visdom dashboard that can read live from each file and display all the data.
    """

    # The time to wait before trying to connect to Visdom again, in seconds
    MAX_VISDOM_TIMEOUT = 5  # 60 * 5

    # The line plots to create for each episode
    EPISODE_LINE_PLOTS = [
        ("episode_num", "max_x", "blue"),
        ("episode_num", "external_reward", "green"),
        ("episode_num", "internal_reward", "purple"),
    ]

    def __init__(self, folder_base):
        # Make run directory
        self.run_folder = get_next_run_folder(folder_base + "_")
        os.makedirs(self.run_folder)
        # Keep cache of open files
        self.log_files = {}

    def _log(self, filepath, filename, dict_obj):
        # print('_log {} to {}'.format(dict_obj, filepath+filename))
        total_file_path = filepath + filename
        try:
            os.makedirs(filepath)
        except FileExistsError:
            pass
        if total_file_path not in self.log_files:
            self.log_files[total_file_path] = open(total_file_path, "a")
        self.log_files[total_file_path].write(json.dumps(dict_obj) + "\n")
        self.close()

    def log_trajectory(self, trajectory_log):
        episode_num = episode_log.episode_num
        filename = self.run_folder
        self._log(filename, "trajectory_logs.csv", vars(trajectory_log))

    def log_episode(self, episode_log):
        episode_num = episode_log.episode_num
        filename = self.run_folder
        episode_log_dict = vars(episode_log)
        self._log(filename, "episode_logs.csv", episode_log_dict)

    def close(self):
        for log_file in self.log_files:
            self.log_files[log_file].close()
        self.log_files.clear()


class EpisodeLog:
    """
    Contains information about one training run within a specific episode.
    """

    required_params = [
        "episode_num",
        "death_coords",
        "training_steps",
        "max_x",
        "score",
        "external_reward",
        "internal_reward",
        "action_count",
    ]

    def __init__(self, params):
        for required_param in self.required_params:
            if required_param not in params:
                raise ValueError(
                    "EpisodeLog constructor missing required param {}".format(
                        required_param
                    )
                )
        # Episode number
        self.episode_num = params["episode_num"]
        # The (x,y) coordinates of each of the agent's deaths.
        self.death_coords = params["death_coords"]
        # The total number of training steps so far
        self.training_steps = params["training_steps"]
        # The farthest x-coord reached by the agent with its y-coord @TODO add y-coord calc
        self.max_x = params["max_x"]
        # The max score so far
        self.score = params["score"]
        # The reward from this episode
        self.external_reward = params["external_reward"]
        self.internal_reward = params["internal_reward"]
        self.action_count = params["action_count"]
        # @TODO: implement action distribution, video playback buffer