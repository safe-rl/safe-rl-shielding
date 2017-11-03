echo "Generating data for $1 (Training for $2 steps)"
echo "------------------------------------------------"
echo "Generating variant without shield"
python3 simulator.py $1/$1.png -t=$2 -o=0 -c=$1/$1_0 --num-steps=$3 
echo "------------------------------------------------"
echo "Generating variant without shield and huge negative rewards"
python3 simulator.py $1/$1.png -t=$2 -o=3 -c=$1/$1_0_huge_neg --num-steps=$3 -p 
echo "------------------------------------------------"
echo "Generating variant with only 1 action"
python3 simulator.py $1/$1.png -t=$2 -o=1 -c=$1/$1_1 --num-steps=$3 
echo "------------------------------------------------"
echo "Generating variant with 3 actions in ranking"
python3 simulator.py $1/$1.png -t=$2 -o=3 -c=$1/$1_3 --num-steps=$3 
echo "------------------------------------------------"
echo "Generating variant with only 1 action but negative rewards for unsafe ones"
python3 simulator.py $1/$1.png -t=$2 -o=1 -c=$1/$1_1_neg_reward -n --num-steps=$3
echo "------------------------------------------------"
echo "Generating variant with 3 actions in ranking but negative rewards for unsafe ones"
python3 simulator.py $1/$1.png -t=$2 -o=3 -c=$1/$1_3_neg_reward -n --num-steps=$3 
# echo "------------------------------------------------"
# echo "Generating SARSA variant without shield"
# python2 simulator.py $1/$1.png -t=$2 -o=0 -c=$1/$1_0_sarsa --num-steps=$3 -r
# echo "------------------------------------------------"
# echo "Generating SARSA variant with only 1 action"
# python2 simulator.py $1/$1.png -t=$2 -o=1 -c=$1/$1_1_sarsa --num-steps=$3 -r
# echo "------------------------------------------------"
# echo "Generating SARSA variant with 3 actions in ranking"
# python2 simulator.py $1/$1.png -t=$2 -o=3 -c=$1/$1_3_sarsa --num-steps=$3 -r
# echo "------------------------------------------------"
# echo "Generating SARSA variant with only 1 action but negative rewards for unsafe ones"
# python2 simulator.py $1/$1.png -t=$2 -o=1 -c=$1/$1_1_neg_reward_sarsa -n --num-steps=$3 -r
# echo "------------------------------------------------"
# echo "Generating SARSA variant with 3 actions in ranking but negative rewards for unsafe ones"
# python2 simulator.py $1/$1.png -t=$2 -o=3 -c=$1/$1_3_neg_reward_sarsa -n --num-steps=$3 -r
