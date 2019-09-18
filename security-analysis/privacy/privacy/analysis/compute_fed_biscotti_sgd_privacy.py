# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Command-line script for computing privacy of a model trained with DP-SGD.

The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism. The mechanism's parameters are controlled by flags.

Example:
  compute_dp_sgd_privacy
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.12 \
    --epochs=60 \
    --delta=1e-5

The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""

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


#Output file
OUTPUT_FILE="privacy_analysis.csv"
COMMITTEE_SIZE=30
ADVERSARIAL_CLIENT_STAKE=30

# Opting out of loading all sibling packages and their dependencies.
sys.skip_tf_privacy_import = True

from privacy.analysis.rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from privacy.analysis.rdp_accountant import get_privacy_spent

# flags.mark_flag_as_required('epochs')

orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
          list(range(5, 64)) + [128, 256, 512])

def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  rdp = compute_rdp(q, sigma, steps, orders)

  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  # print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
  #       ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
  # print('differential privacy with eps = {:.3g} and delta = {}.'.format(
  #     eps, delta))
  # print('The optimal RDP order is {}.'.format(opt_order))

  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

  return eps

def apply_dp_sgd_analysis_biscotti(q, sigma, steps, orders, delta, prev_rdp):

  rdp = compute_rdp(q, sigma, 1, orders)

  new_rdp = prev_rdp + rdp

  eps, _, opt_order = get_privacy_spent(orders, new_rdp, target_delta=delta)

  # print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
  #     ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')

  # print('differential privacy with eps = {:.3g} and delta = {}.'.format(
  #     eps, delta))
  # print('The optimal RDP order is {}.'.format(opt_order))

  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
        'the set of orders.')

  return eps, new_rdp


def main(argv):

  FLAGS = flags.FLAGS

  flags.DEFINE_integer('N', None, 'Total number of examples')
  flags.DEFINE_integer('batch_size', None, 'Batch size')
  flags.DEFINE_integer('steps', 0, 'Steps')
  flags.DEFINE_float('noise_multiplier', None, 'Noise multiplier for DP-SGD')
  flags.DEFINE_float('epochs', None, 'Number of epochs (may be fractional)')
  flags.DEFINE_float('delta', 1e-6, 'Target delta')
  flags.DEFINE_integer('committee_size', 30, 'Target delta')
  flags.DEFINE_float('adversarial_client_control', 0.3, 'Adversarial Client Control')

  flags.mark_flag_as_required('N')
  flags.mark_flag_as_required('batch_size')
  flags.mark_flag_as_required('noise_multiplier')


  del argv  # argv is not used.
  q = FLAGS.batch_size / FLAGS.N  # q - the sampling ratio.

  COMMITTEE_SIZE=FLAGS.committee_size
  ADVERSARIAL_CLIENT_STAKE=FLAGS.adversarial_client_stake

  results_df = pd.DataFrame(columns=['system', 'round', 'epsilon'])

  if q > 1:
    raise app.UsageError('N must be larger than the batch size.')

  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])

  steps = FLAGS.steps

  if steps == 0:
    steps = int(math.ceil(FLAGS.epochs * FLAGS.N / FLAGS.batch_size))

  round_eps = []

  # For federated learning
  for rounds in xrange(1,steps+1):

      this_round_eps = apply_dp_sgd_analysis(q, FLAGS.noise_multiplier, rounds, orders, FLAGS.delta)

      results_df = results_df.append({'system': 'fed_learn', 'round': rounds, 'epsilon':this_round_eps},ignore_index=True)

      round_eps.append(this_round_eps)

  # For biscotti
  stake_map = generate_stake_map(FLAGS.N)
  adversarial_clients = get_adversarial_clients(ADVERSARIAL_CLIENT_STAKE,stake_map)

  secure_agg = True
  prev_rdp=0
  sigma = FLAGS.noise_multiplier
  sigma_agg = FLAGS.noise_multiplier*math.sqrt(FLAGS.batch_size)
  observation_list = get_observation_list(steps, FLAGS.adversarial_client_stake, FLAGS.committee_size, FLAGS.N)

  for rounds in xrange(1,steps+1):

    this_round_eps = 1

    if observation_list[rounds-1] == "no_adv" :      
      noise_multiplier = sigma_agg
    else:
      noise_multiplier = sigma

    this_round_eps, prev_rdp = apply_dp_sgd_analysis_biscotti(q, noise_multiplier, rounds, orders, FLAGS.delta, prev_rdp)

    results_df = results_df.append({'system': 'biscotti', 'round': rounds, 'epsilon':this_round_eps},ignore_index=True)  

  results_df.to_csv(OUTPUT_FILE)

def get_adversarial_clients(adversarial_client_stake, stake_map):

  sorted_map = sorted(stake_map.items(), key=lambda x: x[1])
  adversarial_clients = []

  for node, stake in sorted_map:
    adversarial_clients.append(node)    
    adversarial_client_stake = adversarial_client_stake - stake

    if adversarial_client_stake <= 0:
      break
      
  return adversarial_clients

def get_observation_list(rounds, adversary_control_nodes, committee_size, total_nodes):

  prob_adversary = adversary_control_nodes/total_nodes

  num_adversarial_rounds = committee_analysis.get_num_rounds_adversary_observes(rounds, prob_adversary, committee_size, total_nodes)

  x = ['adv'] * num_adversarial_rounds
  y = ['no_adv'] * (rounds - num_adversarial_rounds)

  observation_list = x + y

  random.shuffle(observation_list)

  return observation_list  


def check_if_adversary_observes_dep(stake_map, committee_size, num_clients, adversarial_clients):

  committee = select_committee(stake_map, committee_size)

  set_committee = set(committee)
  set_adversarial = set(adversarial_clients)

  if (set_committee & set_adversarial):
     return True
  else:
     return False 


def get_privacy_adversarial_guarantee(num_users, num_samples, steps, noise_multiplier, delta, committee_size, adversarial_client_control):

  q = num_samples / num_users 

  secure_agg = True
  prev_rdp=0
  
  sigma_agg = noise_multiplier*math.sqrt(num_samples)
  epsilon = 0
  adversary_observes = 0

  observation_list = get_observation_list(steps, adversarial_client_control, committee_size , num_users)

  epsilon_list = []

  for rounds in xrange(1,steps+1):

      this_round_eps = 1

      if observation_list[rounds-1] == "no_adv" :      
        sigma = sigma_agg
      else:
        sigma = noise_multiplier
        adversary_observes+=1

      this_round_eps, prev_rdp = apply_dp_sgd_analysis_biscotti(q, sigma, rounds, orders, delta, prev_rdp)

      epsilon = this_round_eps

      epsilon_list.append(this_round_eps)


  return epsilon, observation_list.count('adv'), epsilon_list 



def select_committee(stake_map, committee_size):

  sumStake = sum(stake_map.values())
  
  prob_stake = {node: (stake/sumStake) for node, stake in stake_map.iteritems()}

  cum_prob = 0
  cum_prob_stake = {}

  for node,node_prob in prob_stake.iteritems():

    cum_prob = cum_prob + node_prob    
    cum_prob_stake[node]= cum_prob

  # Select nodes
  committee = []
  for x in xrange(0,committee_size):
    rand_value = random.uniform(0,1)
    selected_node = get_node(rand_value,cum_prob_stake)
    committee.append(selected_node)

  return committee

def get_node(rand_value, cum_prob_stake):

  node_selected = 0

  prev_prob = 0.0
  
  for node,node_prob in cum_prob_stake.iteritems():

    if (rand_value > prev_prob) and (rand_value <= node_prob):
      node_selected = node
      break

    prev_prob = node_prob

  return node_selected


def generate_stake_map(numClients):

  stakeMap = {} 

  for node_id in xrange(0,numClients):   
    stakeMap[node_id] = 1

  return stakeMap


if __name__ == '__main__':
  app.run(main)