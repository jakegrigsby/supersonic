import csv

def load_sonic_lvl_set(train=True):
    lvls = {}
    with open('sonic-train.csv' if train else 'sonic-val.csv', newline='') as lvl_file:
        reader = csv.reader(lvl_file)
        for row in reader:
            lvls[row[1]] = row[0] #lvls[lvl] = game
    return lvls

def get_game_from_sonic_lvl(lvl_id):
    lvl_set = all_sonic_lvls()
    try:
        return lvl_set[lvl_id]
    except KeyError:
        print("Level id not found in train or test set...")
        raise KeyError()

def is_sonic_lvl(lvl_id):
    return lvl_id in all_sonic_lvls()

def all_sonic_lvls():
    return {**load_sonic_lvl_set(True), **load_sonic_lvl_set(False)}
