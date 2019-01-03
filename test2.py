from episode_log import EpisodeLog
import time

filenames = ['data/test_data.csv', 'data/test_data2.csv']
fieldnames = ['score', 'ep_length', 'value_loss']
episode_logs = [EpisodeLog(filenames[0], fieldnames), EpisodeLog(filenames[1], fieldnames)]

for x in range(1, 1001):
    dict1 = {fieldnames[0]: x, fieldnames[1]: x ** (1/2), fieldnames[2]: 1 / x}
    dict2 = {fieldnames[0]: x * (1-x), fieldnames[1]: x ** (3/2), fieldnames[2]: 10 / x}
    episode_logs[0].write(dict1)
    episode_logs[1].write(dict2)
    print('x =', x)
    time.sleep(1)
