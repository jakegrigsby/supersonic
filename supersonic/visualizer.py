import csv
import time
from datetime import datetime

from visdom import Visdom
import pandas as pd

from supersonic.episode_log import EpisodeLog


class Visualizer:

    def __init__(self, episode_logs, update_interval, win):
        self.viz = Visdom()
        self.win = win
        self.time = time.time()

        self.filenames = episode_logs
        self.update_interval = update_interval

    def run(self):
        self.update_all()
        while True:
            current_time = time.time()
            if current_time - self.time >= self.update_interval:
                self.update_all()
                self.time = current_time

    def str_to_color(self, s):
        color_hash = hash(s)
        r = (color_hash & 0xff0000) >> 16
        g = (color_hash & 0x00ff00) >> 8
        b = (color_hash & 0x0000ff)
        return 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'

    def get_line(self, filename, data):
        color = self.str_to_color(filename)

        return dict(
            x=list(range(1, len(data) + 1)),
            y=data,
            mode='lines',
            type='custom',
            line=dict(color=color),
            name='test',
            showlegend=False
        )

    def get_layout(self, filename, metric):
        return dict(
            title=filename + ': ' + metric,
            xaxis={'title': 'episode'},
            yaxis={'title': metric}
        )

    def update_all(self):
        print('updating graphs:', datetime.now())
        for filename in self.filenames:
            self.update(filename)

    def update(self, filename):
        with open(filename, 'r+', newline='') as file:
            reader = csv.DictReader(file, dialect='unix')

            # store metrics
            metrics = reader.fieldnames
            if metrics is None:
                return

            # store data
            data = [[] for _ in metrics]
            for row in reader:
                for i in range(len(metrics)):
                    data[i].append(float(row[metrics[i]]))

            # update each metric's plot on Visdom
            for i in range(len(metrics)):
                line = self.get_line(filename, data[i])
                layout = self.get_layout(filename, metrics[i])
                self.viz._send({'data': [line], 'layout': layout, 'win': self.win + filename + metrics[i]})
