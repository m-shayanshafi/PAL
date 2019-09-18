delta="1e-6"
numRounds="100"

numUsers="100 200 300 400 500 600 700 800 900 1000"
adversaryRatios="0.05 0.1 0.15 0.2 0.25 0.3"
sigmas="1.0 2.0 3.0 4.0 5.0 6.0"
sampleRatios="0.1 0.2 0.3 0.4"
committeeSizes="3 5 7 9 11 12 15 17 19 21 23 25 27 30"

while getopts r:u:t:s:c:a:o: o
do	case "$o" in
	r)	numRounds="$OPTARG"
		;;
	u)	numUsers="$OPTARG"
		;;
	s)  sampleRatios="$OPTARG"
		;;
	t)  sigmas="$OPTARG"
		;;
	c) committeeSize="$OPTARG"
		;;
	a) adversarySize="$OPTARG"
		;;
	o) outputFile="$OPTARG"
		;;
	[?]) exit 1
		 ;;
	esac
done

> $outputFile

for numRound in $numRounds; do
	for numUser in $numUsers
	do

		for sampleRatio in $sampleRatios
		do

			for adversaryRatio in $adversaryRatios; do

				for sigma in $sigmas; do

					for committeeSize in $committeeSizes; do

						python privacy_assignment_analysis.py -U=$numUser --steps=$numRound --delta=$delta --committee_size=$committeeSize --adversarial_client_control=$adversaryRatio --noise_multiplier=$sigma --sample_ratio=$sampleRatio --output_file=$outputFile						

						# exit

					done

				done

			done
			
		done

	done
done


					