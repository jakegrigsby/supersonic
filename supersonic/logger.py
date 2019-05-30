import csv, glob, json, os, time, webcolors
from datetime import datetime

import numpy as np

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
    def __init__(self, folder_base):
        # Make run directory
        self.run_folder = get_next_run_folder(folder_base + "_")
        os.makedirs(self.run_folder)
        # Keep cache of open files
        self.log_files = {}

    def _log(self, filepath, filename, dict_obj):
        total_file_path = filepath + filename
        try:
            os.makedirs(filepath)
        except FileExistsError:
            pass
        if total_file_path not in self.log_files:
            self.log_files[total_file_path] = open(total_file_path, "a")
        self.log_files[total_file_path].write(json.dumps(dict_obj) + "\n")
        self.close()

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
    def __init__(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])