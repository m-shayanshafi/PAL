import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines

def main():

	fig, ax = plt.subplots(figsize=(10, 5))
	plt.rcParams["text.usetex"] =True

	results = pd.read_csv('privacy_analysis.csv')
	federated_learning_bool = results['system']=='fed_learn'
	federated_learning = results[federated_learning_bool]
	biscotti_bool = results['system']=='biscotti'
	biscotti = results[biscotti_bool]

	l1 = mlines.Line2D(federated_learning['round'], federated_learning['epsilon'], color='black', linewidth=3, linestyle='-', label="Federated Learning")

	print(biscotti['round'])

	l2 = mlines.Line2D(biscotti['round'], biscotti['epsilon'], color='red', linewidth=3, linestyle='-', label="Peer to Peer")

	ax.add_line(l1)
	ax.add_line(l2)

	plt.legend(handles=[l1,l2], loc='right', fontsize=18)

	plt.xlabel("Rounds", fontsize=22)

	plt.ylabel("Privacy Loss ($\epsilon$)", fontsize=22)

	ax.set_ylim([0, 100])
	ax.set_xlim([0, 100])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# plt.setp(ax.get_xticklabels(), fontsize=18)
	# plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)

	fig.savefig("privacy_loss.jpg")
	# plt.show()


if __name__ == '__main__':
  main()
