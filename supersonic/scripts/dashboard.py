import os
os.chdir('..')
import argparse
import json

import pandas as pd
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np

from supersonic import utils

def load_if_dict(line):
        try:
            return json.loads(line)
        except:
            #not a dict line
            return line

def reformat_death_coords(death_coords, x_adjustment_func=lambda x : x, y_adjustment_func=lambda y : y):

    def dive_to_coords(nested_coords):
        coord = nested_coords
        while len(coord) < 2:
            try:
                coord = coord[0]
            except:
                return
        return coord
 
    death_coords = list(death_coords)
    x, y = [], []
    for coord in death_coords:
        coord = dive_to_coords(coord)
        if not coord: continue
        x.append(x_adjustment_func(coord[0]))
        y.append(y_adjustment_func(coord[1]))
    return x, y
   
def main():
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    args = parser.parse_args()

    #load log file and read it into a dataframe
    run_path = f"logs/{args.run}/episode_logs.csv"
    with open(run_path, "r") as log_file:
        *line_list, lvl = [load_if_dict(line) for line in log_file.readlines()]
    log_df = pd.DataFrame(line_list)

    #create matplotlib figure
    fig = plt.figure(figsize=(16,10))
    grid = matplotlib.gridspec.GridSpec(3,1)

    #plot level map and death coords scatter plot
    ax1 = plt.subplot(grid[0,:])
    if not type(lvl) == str: lvl = "GreenHillZone.Act1"
    game = utils.get_game_from_sonic_lvl(lvl)
    lvl_map = plt.imread(f"data/lvl_maps_sonic/{game}/{lvl.replace('.','')}.png")
    ax1.imshow(lvl_map, aspect='auto')
    x_func = lambda x : x + 150
    y_func = lambda y : y + 150 if y > 700 else y + 100
    death_coords_x, death_coords_y = reformat_death_coords(log_df["death_coords"], x_func, y_func)
    ax1.scatter(death_coords_x, death_coords_y,marker='x',color='red')
    fig.suptitle(lvl)
    
    #plot internal and external rewards
    ax2 = plt.subplot(grid[1,0])
    ax2.set_xticklabels([])
    smoothed_rews_e = utils.moving_average(log_df["external_reward"], min(len(line_list), 50))
    ax2.plot(log_df["episode_num"], smoothed_rews_e)
    ax2.set_title('External Reward')

    ax3 = plt.subplot(grid[2,0])
    ax3.plot(log_df["episode_num"], utils.moving_average(log_df["internal_reward"], min(len(line_list), 50)), color="orange")
    ax3.set_title('Internal Reward')
    ax3.set_xlabel("Episode Number")
    plt.show()

if __name__ == "__main__":
    main()
