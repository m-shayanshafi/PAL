import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
from absl import flags
from absl import app
import sys

# IMPORTANT: MAKE SURE COLUMN NAMES AND FLAGS ARE CONSISTENT

DATA_COLUMNS = ["number_nodes", "sample_ratio", "adversarial_client_control", "noise_multiplier", "delta", "committee_size", "epsilon", "rounds_adversary_observes", "rounds_adversary_majority", "epsilon_list"]

COLORS = ["red", "blue", "green", "purple", "orange", "brown"]

flags.DEFINE_integer('number_nodes', None, 'Total number of users in the system')

flags.DEFINE_float('sample_ratio', None, 'Number of users sampled each round')

flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')

flags.DEFINE_float('delta', None, 'Target delta')

flags.DEFINE_integer('committee_size', None, 'Committee Size')

flags.DEFINE_float('adversarial_client_control', None, 'Number of clients controlled by adversary')

flags.DEFINE_integer('rounds_adversary_observes', None, 'Number of rounds adversary observes')	

flags.DEFINE_integer('rounds_adversary_majority', None, 'Noise multiplier for DP-SGD')

flags.DEFINE_string('dep_var', 'epsilon', 'Dependant variable')

flags.DEFINE_string('ind_var', None, 'Independent variable')

flags.DEFINE_string('legend', None, 'Legend')

flags.DEFINE_string('data_file', None, 'Data file to read results from')

flags.mark_flag_as_required('ind_var')
flags.mark_flag_as_required('legend')
flags.mark_flag_as_required('data_file')

FLAGS = flags.FLAGS

PLOT_NAME = "results.jpg"

def main(argv):

	del argv

	dependant_variable = FLAGS.dep_var
	independent_variable = FLAGS.ind_var
	legends = FLAGS.legend 

	# results
	results = pd.read_csv(FLAGS.data_file)	

	# create filter maps
	filter_map = create_filter_map(FLAGS, results)
	flag_filtered_df = filter_data(filter_map, results)

	#plot
	create_plot(flag_filtered_df, dependant_variable, independent_variable, legends, PLOT_NAME, filter_map)


def create_filter_map(FLAGS, results):


	# get values not None
	filter_map = get_flags_passed(FLAGS)

	return filter_map


def get_flags_passed(FLAGS):

	filter_map = {}
	flag_dict = FLAGS.flag_values_dict()

	for DATA_COLUMN in DATA_COLUMNS:

		if DATA_COLUMN in flag_dict and flag_dict[DATA_COLUMN] is not None:
			filter_map[DATA_COLUMN] = flag_dict[DATA_COLUMN]

	return filter_map


def add_implied_filters(filter_map, results):

	filtered_data = None

	if "rounds_adversary_majority" in filter_map and not "committee_size" in filter_map: 

		print("Inside if condition")
		filtered_data = filter_by_least_committee_size(filter_map["rounds_adversary_majority"], results)
		
		print(filtered_data.shape)

	return filtered_data

def filter_by_least_committee_size(adversarial_majority, results):

	min_committee_size = min(results["committee_size"].unique())

	x = results.query("committee_size ==" + str(min_committee_size))

	return x


def filter_data(filter_map, results):

	query  = generate_query(filter_map)
	
	filtered_df = results.query(query)

	return filtered_df


def generate_query(filter_map):

	query = ''

	query_cols = len(filter_map.items())
	col_idx = 1 

	for column, value in filter_map.items():

		query = query  + column + "==" + str(value)		

		if query_cols == col_idx:
			break

		query = query + " & "	

		col_idx = col_idx + 1

	return query


def create_plot(filtered_df, dependant_variable, independant_variable, legends, plot_name, filter_map):

	fig, ax = plt.subplots(figsize=(10, 5))

	# find unique values in legend
	legend_unique_values = find_unique_values_legend(legends, filtered_df)

	lines = []

	min_x = 0
	max_x = 0

	min_y = 0
	max_y = 0

	color_idx = 0

	for legend_unique_value in legend_unique_values:

		query = legends + "==" + str(legend_unique_value)
		line_df = filtered_df.query(query)
		final_line_df = add_implied_filters(filter_map,line_df)

		x = final_line_df[independant_variable].unique()
		print(x)
		y = final_line_df[dependant_variable]
		print(y)

		min_x, max_x = find_min_max(x, min_x, max_x)
		min_y, max_y = find_min_max(y, min_y, max_y)

		line = mlines.Line2D(x, y, color=COLORS[color_idx], linewidth=3, linestyle='-', label=query)		

		lines.append(line)
		ax.add_line(line)

		color_idx = color_idx + 1

	plt.legend(handles= lines, loc='right', fontsize=18)
	plt.xlabel(independant_variable, fontsize=22)
	plt.ylabel(dependant_variable, fontsize=22)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	ax.set_ylim([0, max_y])
	ax.set_xlim([0, max_x])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	fig.tight_layout(pad=0.1)
	fig.savefig(plot_name)

def find_min_max(new_values, prev_min, prev_max):
	
	new_min = min(new_values)
	new_max = max(new_values)

	final_min = prev_min
	final_max = prev_max

	if new_min < prev_min:
		final_min= new_min

	if new_max > prev_max:
		final_max = new_max

	return final_min, final_max


def find_unique_values_legend(legends, results):

	return results[legends].unique()


if __name__ == '__main__':
  app.run(main)