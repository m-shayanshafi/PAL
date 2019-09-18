#Hello World

numRounds="100"
numUsers="100"
numSample="40"
sigma="1.0"
delta="1e-5"

while getopts r:u:t:s: o
do	case "$o" in
	r)	numRounds="$OPTARG"
		;;
	u)	numUsers="$OPTARG"
		;;
	s)  numSample="$OPTARG"
		;;
	t)  sigma="$OPTARG"
		;;
	[?]) exit 1
		 ;;
	esac
done

python compute_fed_biscotti_sgd_privacy.py --N=100 --batch_size=40 --noise_multiplier=1.0 --steps=100 --delta=1e-6 --committee_size=10 --adversarial_client_stake=30

python plot.py 
	

