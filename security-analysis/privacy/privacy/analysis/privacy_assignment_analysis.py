from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

from absl import app
from absl import flags
import pandas as pd
import random
import committee_analysis

sys.skip_tf_privacy_import = True

from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.analysis.rdp_accountant import compute_rdp
import compute_fed_biscotti_sgd_privacy as privacy_analysis
import utils

flags.DEFINE_integer('U', None, 'Total number of users in the system')
flags.DEFINE_float('sample_ratio', None, 'Number of users sampled each round')
flags.DEFINE_integer('steps', None, "Number of rounds to run training for")

flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
flags.DEFINE_float('delta', None, 'Target delta')

flags.DEFINE_integer('committee_size', None, 'Committee Size')
flags.DEFINE_float('adversarial_client_control', None, 'Number of clients controlled by adversary')

flags.DEFINE_string('output_file', 'results.csv', 'Output file to append results to')

flags.mark_flag_as_required('U')
flags.mark_flag_as_required('sample_ratio')
flags.mark_flag_as_required('steps')
flags.mark_flag_as_required('noise_multiplier')
flags.mark_flag_as_required('delta')
flags.mark_flag_as_required('committee_size')
flags.mark_flag_as_required('adversarial_client_control')

FLAGS = flags.FLAGS

def main(argv):

	del argv  # argv is not used.

	num_samples = FLAGS.U * FLAGS.sample_ratio
	num_adversaries = FLAGS.U * FLAGS.adversarial_client_control

	utils.write_header_if_file_empty(FLAGS.output_file)	

	#Get privacy guarantees
	final_epsilon, adversary_observes,epsilon_list = privacy_analysis.get_privacy_adversarial_guarantee(FLAGS.U, num_samples , FLAGS.steps, FLAGS.noise_multiplier, FLAGS.delta, FLAGS.committee_size, num_adversaries)
	
	adversary_majority=committee_analysis.get_num_rounds_adversary_majority(FLAGS.steps, num_adversaries, FLAGS.committee_size, FLAGS.U)

	print('DP-SGD with sampling rate = {:.3g}% , noise_multiplier = {}, nodes = {}, committee_size = {}, adversarial_client_control = {}  iterated'
	    ' over {} steps satisfies'.format(FLAGS.sample_ratio, FLAGS.noise_multiplier, FLAGS.U, FLAGS.committee_size, FLAGS.adversarial_client_control,  FLAGS.steps), end=' ')

	print('differential privacy with eps = {:.3g} and delta = {}.'.format(
	      final_epsilon, FLAGS.delta))

	print('The adversary observes rounds = {:.3g} and has a majority in = {}.'.format(
      adversary_observes, adversary_majority))
	
	# Write results
	fields = [FLAGS.U, FLAGS.sample_ratio, FLAGS.adversarial_client_control, FLAGS.committee_size, FLAGS.noise_multiplier,  FLAGS.delta, final_epsilon, adversary_observes, adversary_majority, epsilon_list]

	utils.write_output_to_file(FLAGS.output_file, fields)

if __name__ == '__main__':
  app.run(main)
