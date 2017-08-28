# echo "Generating data for $.2 (Training for $1 steps)"

# echo "------------------------------------------------"
# echo "Generating Q variant without shield"
# python3 watertank.py -c=watertank_no_shield --num-steps=$1 -o=0 -t=$2

# echo "------------------------------------------------"
# echo "Generating Q variant without shield and huge negative rewards"
# python2 watertank.py -c=watertank_no_shield_huge_neg --num-steps=$1 -o=0 -t=$2 -p

# echo "------------------------------------------------"
# echo "Generating Q variant with shield"
# python2 watertank.py -c=watertank_shield --num-steps=$1 -o=1 -t=$2

# echo "------------------------------------------------"
# echo "Generating Q variant with shield but negative rewards for unsafe ones"
# python2 watertank.py $1/$1.png -t=$2 -o=1 -c=$1/$1_1_neg_reward -n --num-steps=$3





# echo "------------------------------------------------"
# echo "Generating SARSA variant without shield"
# python2 watertank.py -c=watertank_sarsa_no_shield --num-steps=$1 -o=0 -t=$2 -r

# echo "------------------------------------------------"
# echo "Generating SARSA variant without shield and huge negative rewards"
# python2 watertank.py $1/$1.png -t=$2 -o=3 -c=$1/$1_0_huge_neg --num-steps=$3 -p

echo "------------------------------------------------"
echo "Generating SARSA variant with shield"
python3 watertank.py -c=watertank_sarsa_shield --num-steps=$1 -o=1 -t=$2 -r

# echo "------------------------------------------------"
# echo "Generating SARSA variant with shield but negative rewards for unsafe ones"
# python2 watertank.py $1/$1.png -t=$2 -o=3 -c=$1/$1_3_neg_reward_sarsa -n --num-steps=$3 -r
