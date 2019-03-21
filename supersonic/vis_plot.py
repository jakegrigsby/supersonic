import argparse, datetime, json, webcolors
import numpy as np
from visdom import Visdom

VISDOM_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
DEFAULT_RUN = '1'

EPISODE_FILENAME = 'episode_logs.csv'

LOG_FOLDER = 'logs/'

class VisdomLogger:

	#The line plots to create for each episode
	EPISODE_LINE_PLOTS = [
		('episode_num', 'max_x', 'orange'),
		('episode_num', 'external_reward', 'red'),
		('episode_num', 'internal_reward', 'purple'),
	]

	EPISODE_HISTOGRAMS = [
		'death_coords'
	]

	def __init__(self, server, port, run_name, logs):
		self.logs = logs
		self.run_name = run_name
		self.viz = Visdom(port=port, server=server, env=run_name)
		self.viz.delete_env(run_name) # Clear past runs with run_name.

	def convert_color(color):
		''' Convert colors to the format visdom likes them in. '''
		if not color or not isinstance(color, str): return color
		rgb = webcolors.name_to_rgb(color)
		return np.array([rgb])

	def make_episode_hist(self, data, title='Histogram'):
		opts = dict(
			title=title,
		)
		self.viz.histogram(X=data, opts=opts)

	def make_episode_line_plot(self, x_key, y_key, color=None):
		plot_title = '{} vs. {}'.format(y_key, x_key)
		opts = dict(
			title=self.run_name,
			xlabel=x_key,
			ylabel=y_key,
			linecolor=VisdomLogger.convert_color(color),
		)
		x = [l[x_key] for l in self.logs]
		y = [l[y_key] for l in self.logs]
		# title = '{} vs. {}'.format(y_key, x_key)
		self.viz.line(X=x, Y=y, opts=opts)

	def make_info_box(self):
		current_time = datetime.datetime.now()
		date = current_time.strftime("%Y-%m-%d %H:%M:%S")
		info_html = '<h4>Updated at {} with {} episodes.</h4>'.format(
			date, len(self.logs)
		)
		self.viz.text(info_html)

	def render(self):
		# Draw info.
		self.make_info_box()
		# Draw line charts.
		for x_key, y_key, color in VisdomLogger.EPISODE_LINE_PLOTS:
			self.make_episode_line_plot(x_key, y_key, color=color)
		# Draw scatter charts.
		for hist_key in VisdomLogger.EPISODE_HISTOGRAMS:
			data = [entry[hist_key][0][0] for entry in self.logs]
			self.make_episode_hist(data, title=hist_key)

def get_dicts_from_json_file(filename):
	lines = open(filename).readlines()
	return [json.loads(line) for line in lines]

def main():
	parser = argparse.ArgumentParser(description='Visdom plotter arguments')
	parser.add_argument('-port', metavar='port', type=int, default=VISDOM_PORT,
						help='port the visdom server is running on.')
	parser.add_argument('-server', metavar='server', type=str,
						default=DEFAULT_HOSTNAME,
						help='Server address of the target to run the demo on')
	parser.add_argument('-run', metavar='run', type=str, default=DEFAULT_RUN,
						help='index of the run to visualize')

	FLAGS = parser.parse_args()

	episode_log = LOG_FOLDER + FLAGS.run + '/' + EPISODE_FILENAME
	logs = get_dicts_from_json_file(episode_log)

	viz = VisdomLogger(FLAGS.server, FLAGS.port, FLAGS.run, logs)
	viz.render()

if __name__ == "__main__":
	main()
