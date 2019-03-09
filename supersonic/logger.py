import csv, glob, json, os, time, webcolors
from datetime import datetime

import numpy as np
import pandas as pd
from visdom import Visdom


COUNTER_DIGITS = 2

# Folder format:
# - run00
#   - episode00
#       - episode_logs.csv
#       - trajectory_logs.csv
#   - episode01
#       - ...

def get_next_run_folder(folder_base):
    folders = glob.glob(folder_base + '*')
    folders.sort()
    #print(folder_base, 'folders:', folders)

    if not folders:
        last_run = -1
    else:
        last_run_filename = folders[-1]
        _i = last_run_filename.rfind('_')
        last_run = int(last_run_filename[_i+1:])

    return folder_base + run_num_str(last_run + 1) + '/'


def run_num_str(run_num):
    run = str(run_num)
    while len(run) < COUNTER_DIGITS:
        run = '0' + run
    return run

def init_visdom():
    PORT = 8097
    HOST = 'http://localhost'
    viz = Visdom(port=PORT, server=HOST)
    return viz

class Logger:
    """
    Allows users to save information to disk at the end of every Episode and Trajectory.

    Eventually, this will be fed into a Visdom dashboard that can read live from each file and display all the data.
    """

    #The time to wait before trying to connect to Visdom again, in seconds
    MAX_VISDOM_TIMEOUT = 5 # 60 * 5

    #The line plots to create for each episode
    EPISODE_LINE_PLOTS = [
        ('episode_num', 'max_x', 'blue'),
        ('episode_num', 'reward', 'green'),
        ('episode_num', 'score', 'purple'),
    ]



    def __init__(self, folder_base):
        # Make run directory
        self.run_folder = get_next_run_folder(folder_base + '_')
        os.makedirs(self.run_folder)
        # Keep cache of open files
        self.log_files = {}
        # Write directly to Visdom if possible
        self.visdom = init_visdom()
        # Maintain cache of visdom plots
        self.viz_plots = {}

    def plot_episode_visdom(self, filename, dict_obj):
        # If we don't have visdom session, first try and reconnect
        if not self.visdom.check_connection():
            return
        # Then iterate through and create all plots
        for x_key, y_key, line_color in Logger.EPISODE_LINE_PLOTS:
            plot_title = '{} vs. {}'.format(y_key, x_key)
            plot_name = filename + plot_title
            if plot_name in self.viz_plots:
                plot = self.viz_plots[plot_name]
                opts = None
            else:
                plot = None
                # Convert 'navy' to (0,0,128)
                if isinstance(line_color, str):
                    line_color = webcolors.name_to_rgb(line_color)
                line_color_arg = np.array([line_color])
                opts = dict(
                    title=plot_title, 
                    xlabel=x_key, 
                    ylabel=y_key,
                    linecolor=line_color_arg,
                )

            x = dict_obj[x_key]
            y = dict_obj[y_key]
            self.plot_visdom_line(plot_name, x, y, plot=plot, opts=opts)

    def plot_visdom_line(self, plot_name, x, y, plot=None, opts=None):
        ''' Creates or appends an (x,y) coordinate to a line plot in Visdom. '''
        if opts:
            self.viz_plots[plot_name] = self.visdom.line(
                X=[x], 
                Y=[y], 
                opts=opts
            )
        elif plot:
            self.visdom.line(
                X=[x], 
                Y=[y], 
                win=plot, 
                update='append')
        else:
            raise ValueError('Need to provide a plot to append to or options for a new one')



    def _log(self, filepath, filename, dict_obj):
        # print('_log {} to {}'.format(dict_obj, filepath+filename))
        total_file_path = filepath + filename
        try:
            os.makedirs(filepath)
        except FileExistsError:
            pass
        if total_file_path not in self.log_files:
            self.log_files[total_file_path] = open(total_file_path, 'w')
        self.log_files[total_file_path].write(json.dumps(dict_obj) + '\n')


    def log_trajectory(self, trajectory_log):
        episode_num = episode_log.episode_num
        filename = self.run_folder
        self._log(filename, 'trajectory_logs.csv', vars(trajectory_log))

    def log_episode(self, episode_log):
        episode_num = episode_log.episode_num
        filename = self.run_folder
        episode_log_dict = vars(episode_log)
        self._log(filename, 'episode_logs.csv', episode_log_dict)
        # update Visdom plots
        self.plot_episode_visdom(filename, episode_log_dict)

    def close(self):
        for log_file in self.log_files:
            self.log_files[log_file].close()
        self.log_files.clear()


class EpisodeLog:
    """
    Contains information about one training run within a specific episode.
    """
    required_params = ['episode_num', 'death_coords', 'training_steps', 'max_x', 'score', 'external_reward', 'internal_reward']
    def __init__(self, params):
        for required_param in self.required_params: 
            if required_param not in params: 
                raise ValueError('EpisodeLog constructor missing required param {}'.format(required_param))
        # Episode number
        self.episode_num = params['episode_num']
        # The (x,y) coordinates of each of the agent's deaths.
        self.death_coords = params['death_coords']
        # The total number of training steps so far
        self.training_steps = params['training_steps']
        # The farthest x-coord reached by the agent with its y-coord @TODO add y-coord calc
        self.max_x = params['max_x']
        # The max score so far
        self.score = params['score']
        # The reward from this episode
        self.external_reward = params['external_reward']
        self.internal_reward = params['internal_reward']
        # @TODO: implement action distribution, video playback buffer

    # Function __dir()___ which list all  
    # the base attributes to be used. 
    # def __dir__(self): 
    #     return self.required_params

class TrajectoryLog:
    """
    Contains information about one trajectory.
    """
    def __init__(self, episode_num):
        self.episode_num = episode_num
        # @TODO: add more stuff here