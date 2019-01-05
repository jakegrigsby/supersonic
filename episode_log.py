import csv
import time
import uuid
import os

import pandas as pd

class EpisodeLog:

    def __init__(self, filename, fieldnames, new=True):
        self.id = uuid.uuid4()
        self.filename = filename +  '_{}'.format(self.id)
        self.log = open(self.filename, 'w+', newline='')
        self.fieldnames = fieldnames
        self.start_time = time.clock()

        self.writer = csv.DictWriter(self.log, fieldnames, dialect='unix')
        self.writer.writeheader()
        self.log.flush()

    def write(self, episode_dict):
        self.writer.writerow(episode_dict)
        self.log.flush()

    def close(self):
        time_elapsed = time.time() - self.start_time
        self.log.write('{}'.format(time_elapsed))
        self.log.close()

    def make_dataframe(self):
        self.df = pd.read_csv(self.filename)
