import argparse, json
from visdom import Visdom

VISDOM_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
DEFAULT_RUN = '1'

EPISODE_FILENAME = 'episode_logs.txt'

LEVEL_NAME = 'SonicRandomAgent_GreenHillZone.Act1_'
LOG_FOLDER = 'logs/'

def get_dicts_from_json_file(filename):
	lines = open(filename).readlines()
	return [json.loads(line) for line in lines]

def make_episode_line_plot(viz, log_entries, x_key, y_key):
	x = [l[x_key] for l in log_entries]
	y = [l[y_key] for l in log_entries]
	title = '{} vs. {}'.format(y_key, x_key)
	viz.line(X=x, Y=y, opts=dict(title=title, showlegend=True))

def make_episode_plots(viz, episode_log):
	log_entries = get_dicts_from_json_file(episode_log)
	make_episode_line_plot(viz, log_entries, 'episode_num', 'reward')
	make_episode_line_plot(viz, log_entries, 'episode_num', 'score')
	make_episode_line_plot(viz, log_entries, 'episode_num', 'max_x')

def live_update_plots(viz):
	# @TODO: implement Sockets to talk to server


def main():
	parser = argparse.ArgumentParser(description='Visdom plotter arguments')
	parser.add_argument('-port', metavar='port', type=int, default=VISDOM_PORT,
	                    help='port the visdom server is running on.')
	parser.add_argument('-server', metavar='server', type=str,
	                    default=DEFAULT_HOSTNAME,
	                    help='Server address of the target to run the demo on')
	parser.add_argument('-run', metavar='run', type=str, default=DEFAULT_RUN,
	                    help='index of the run to visualize')
	parser.add_argument('-live', metavar='live', type=bool, default=False,
	                    help='whether data collection is live')
	FLAGS = parser.parse_args()

	viz = Visdom(port=FLAGS.port, server=FLAGS.server)

	episode_log = LOG_FOLDER + LEVEL_NAME + FLAGS.run + '/' + EPISODE_FILENAME

	# make_episode_plots(viz, episode_log)

	if FLAGS.live:
		live_update_plots(viz)


# viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

main()