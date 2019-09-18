import math
from scipy.special import comb
import sys


def get_prob_observe_one_round(prob_adversary, committee_size, total_nodes):

	# print(prob_adversary)
	prob_no_adversary_chosen = 1

	prob_no_adversary = 1 - prob_adversary
	prob_one_client_chosen = 1.0/total_nodes

	for x in xrange(0,committee_size):

		prob_no_adversary_chosen = prob_no_adversary_chosen * prob_no_adversary

		prob_no_adversary = prob_no_adversary - prob_one_client_chosen

	prob_adversary_chosen = 1 - prob_no_adversary_chosen

	return prob_adversary_chosen


def get_num_rounds_adversary_observes(num_rounds, prob_adversary, committee_size, total_nodes):

	prob_adversary_chosen = get_prob_observe_one_round(prob_adversary, committee_size, total_nodes)


	return int(round(num_rounds*prob_adversary_chosen))

def get_num_rounds_adversary_majority(num_rounds, num_adversary, committee_size, total_nodes):

	prob_adversary = float(num_adversary)/total_nodes

	prob_adversary_majority = get_prob_observe_majority(prob_adversary, committee_size, total_nodes)

	# print(prob_adversary_majority)

	return int(round(num_rounds*prob_adversary_majority))

def get_prob_observe_majority(prob_adversary, committee_size, total_nodes):

	majority_nodes = int(math.ceil(committee_size/2.0))
	
	prob_select_client = 1/total_nodes

	prob_majority = 0

	for num_adversaries in xrange(majority_nodes,committee_size+1):
		
		prob_num_adversaries_committee= get_prob_num_adversary_committee(num_adversaries, committee_size, prob_adversary, prob_select_client)

		prob_majority = prob_num_adversaries_committee + prob_majority



	return prob_majority

def get_prob_num_adversary_committee(num_adversary, committee_size,prob_adversary, prob_select_client):

	# print(get_prob_k_of_n_adversary(num_adversary, committee_size, prob_adversary, prob_select_client))

	committee_probability = comb(committee_size, num_adversary, exact=True) * get_prob_k_of_n_adversary(num_adversary, committee_size, prob_adversary, prob_select_client)

	return committee_probability


def get_prob_k_of_n_adversary(num_adversary, committee_size, prob_adversary, prob_select_client):

	prob_k_adversary_chosen = 1
	prob_no_adversary = 1 - prob_adversary
	prob_adversary_1 = prob_adversary * 1

	for idx in xrange(1,committee_size+1):

		if idx <= num_adversary:

			prob_k_adversary_chosen = prob_k_adversary_chosen * prob_adversary
			prob_adversary_1 = prob_adversary_1 - prob_select_client
					
		else:

			prob_k_adversary_chosen = prob_k_adversary_chosen * prob_no_adversary
			prob_no_adversary = prob_no_adversary - prob_select_client


	return prob_k_adversary_chosen