import csv
import pandas as pd

COLUMNS = ["number_nodes", "sample_ratio", "adversarial_client_control", "noise_multiplier", "delta", "committee_size", "epsilon", "rounds_adversary_observes", "rounds_adversary_majority", "epsilon_list"]

def write_output_to_file(output_file, fields):

	with open(output_file,'a') as output:
		
		writer = csv.writer(output)	
		writer.writerow(fields)


def write_header_if_file_empty(output_file):

	try:		
		df = pd.read_csv(output_file)

	except pd.errors.EmptyDataError:

		with open(output_file,'a') as output:			
			writer = csv.writer(output)	
			writer.writerow(COLUMNS)