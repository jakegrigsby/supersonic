import csv
import time
from datetime import datetime
from visdom import Visdom

from episode_log import EpisodeLog


class Visualizer:

    def __init__(self, episode_logs, update_interval, win):
        self.viz = Visdom()
        self.win = win
        self.time = time.time()

        self.episode_logs = episode_logs
        self.update_interval = update_interval

    def run(self):
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
        for episode_log in self.episode_logs:
            self.update(episode_log)

    def update(self, episode_log):
        with open(episode_log.filename, 'r+', newline='') as file:
            reader = csv.DictReader(file, dialect='unix')

            # store data
            metrics = reader.fieldnames
            data = [[] for _ in metrics]
            for row in reader:
                for i in range(len(metrics)):
                    data[i].append(float(row[metrics[i]]))

            # update each metric's plot on Visdom
            for i in range(len(metrics)):
                line = self.get_line(episode_log.filename, data[i])
                layout = self.get_layout(episode_log.filename, metrics[i])
                self.viz._send({'data': [line], 'layout': layout, 'win': self.win + episode_log.filename + metrics[i]})


# def test():
#     filenames = ['data/test_data.csv', 'data/test_data2.csv']
#     fieldnames = ['score', 'ep_length', 'value_loss']
#     episode_logs = [EpisodeLog(filenames[0], fieldnames), EpisodeLog(filenames[1], fieldnames)]
#
#     for x in range(1, 201):
#         dict1 = {fieldnames[0]: x, fieldnames[1]: x ** (1/2), fieldnames[2]: 1 / x}
#         dict2 = {fieldnames[0]: x * (1-x), fieldnames[1]: x ** (3/2), fieldnames[2]: 10 / x}
#         episode_logs[0].write(dict1)
#         episode_logs[1].write(dict2)
#
#     update_interval = 2
#     win = 'test'
#     visualizer = Visualizer(episode_logs, update_interval, win)
#     visualizer.run()
#
# test()
