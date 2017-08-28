# Grid World

##Requirements:
 - Python 2.7
 - [Pygame](http://www.pygame.org/download.shtml)
 - [PyBrain](http://pybrain.org)
 - Some python packages (numpy, scipy, PIL, ..)
 
## Usage:
We provided 2 packages: **9x9\_illustrative** and **15x9_cycling**

E.g: python simulator.py 9x9\_illustrative/9x9\_illustrative.png -t=.6 -o=3 -c=9x9\_illustrative/9x9\_illustrative

 - The first parameter specifies which map should be loaded
 - The parameter **-t** sets the exploration factor (epsilon for the greedy one (standard), multiplication factor for UCB(needs small modifications in code))
 - The parameter **-o** sets the number of outputs from the learner to the shield, i.e. the size of the ranking. 0 disables the shield
 - The parameter **-c** allows to set the base name to collect date, like avg reward and episode length
 - The parameter **-n** enables punishments for unsafe actions corrected by the shield
 - The parameter **--num-steps** allows to set an upper bound for the simulation
 - The parameter **-g** generates new specification files and terminates (no simulation done)
 - Parameters **-s** and **-l** can be used to save/load the _Q_-tables
 
If there is the need to create a new package:
The first parameter specifies which map should be loaded. It has to be a indexed PNG file. In the same directory, there has to be a parameter file, with the same name, but the _.params_ extension. For each shield variant, a file with e.g. "\_1.py" for the 1-action variant has to be provided. Colors, bombs, etc. can be set in the parameter file. It's best to look at an existing one an copy most of it. 

## Generator script
gen\_data.sh _[base\_name]_ _[epsilon]_ _[number\_of\_iterations]_ generates data for no shield, shielded with 1 and 3 actions and shielded incl. punishments for 1 and 3 actions.

E.g: gen\_data.sh 9x9\_illustrative/9x9\_illustrative 0.6 100000