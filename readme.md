# Deep Double Oracle SSG
This is an implementation of the deep double oracle algorithm used for solving sequential zero sum SSGs with partial attacker observation.

## Getting Started

To get started, clone or download the repository to get a local copy of the repo on your machine.

### Prerequisites

This implementation uses both CPLEX as well as DOCPLEX. Set up CPLEX as described [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

### Installing

Run the following command from the project directory to install dependencies:  
```
pip install -r requirements.txt
```

## Running the tests

The test framework is still very much a WIP, but to run all unit tests, use the following command from the root of this project:  
```
python -m unittest discover .
```

## Code Structure and Parameters
The code is split into 6 main files (apart from the test files):  

### Oracle Files
attackerOracle.py and defenderOracle.py contain the code for the attacker and defender oracles respectively. These include a class definition as well as some functions.   
__init__ defines the layers of the network as well as the nonlinearities, including input and output sizes.   
__forward__ defines what a forward pass of the network looks like. This specifies the order the layers are applied. Of special note are the inputs to this function (two observation/action pairs for the two-sequence LSTM), and the output (selecting one of the two generated values to represent the action-value estimate).   
__getAction__ gets the highest value action estimate, given a game and observation.  
__getActionFromActions__ does the same, but uses a pre-specified list of available actions.  
__setState__ is used to set the neural network's weights and biases.  
__getState__ is used to get the neural network's weights and biases.

The __AttackerEquilibrium__ class is structured the same, but instead of being a neural network, it solves a small one-shot equilibrium over the action set by using the CoreLP.  

The __ReplayMemory__ class is used to define the replay memory structure used in actor-critic training. it stores a maximum number of transitions, and offers a sampling function and a push function.  

Finally, there are a couple utility functions offered in the file:  
__getInputTensor__ is a utility function that bundles observations and actions into a suitable tensor for the network to take as input.  
__train__ is used to train the networks using actor-critic methods. Firstly, there are several parameters that can be modified. N is the maximum number of transitions that the replaymemory can hold. batchSize is the number of transitions sampled in a minibatch. C is the number of steps before the targetNetwork is reset. Epochs is the number of games played. Optimizer is the pytorch optimizer used -- ADAM by default. Similarly, lossfunction is the lossFunction used -- MSELoss by default. showOutput is used to output debug information during training (may or may not do anything depending on the current state of development).  
Train is called in the experiment file, so any desired parameter changes can be made in the call there.

### CoreLP
coreLP.py contains the code for solving the equilibriums during the double oracle algorithm. This includes a few functions for defining linear programs that can be solved as optimization problems:  
__createDefenderModel__ defines a maximin problem for the defender given pure defender and attacker pure strategies and a payout matrix to determine a mixed strategy.  
__createAttackerModel__ defines a maximin problem for the attacker given pure defender and attacker pure strategies and a payout matrix to determine a mixed strategy.  
__createDefenderOneShotModel__ defines a maximin problem for the defender given defender and attacker actions and penalties/rewards.
__createAttackerOneShotModel__ defines a maximin problem for the attacker given defender and attacker actions and penalties/rewards.  

### SSG
ssg.py is the code for managing the state of the game as it is played. This includes manage viable actions, timestep, game history, and more. This includes a class defintion as well as several helper functions:  
__init__ initializes a game, given a number of targets, number of resources, defender rewards, and defender penalties.  
__restartGame__ resets the state of the game to unplayed, zeroing our most values and resetting the game history.  
__place_ones__ is a helper function that returns all possible defender resource placements given a number of targets (size) and a number of defender resources (count).  
 __getValidActions__ returns the valid actions for a player, based on the current state of the game.  
 __performActions__ performs the defender and attacker action specified and calculates the outcome. The actions and observations are stored in the history, and utility is calculated, as well as destroyed targets, available resources, etc. The new observations and game utility are returned.  
__getEmptyObservations__ returns a "null" observation for each player based on the game. These are used as the history for the first move of each game.  
__getActionScore__ returns the utility for the specified player given the attacker and defender moves, and the rewards and penalties.  
__getPayout__ returns the utility of a defender strategy played against an attacker strategy. Used for calculating the payout matrix.  
__getOracleScore__ gets the average utility of an oracle when played against an opponent mixed strategy.  
__getBestOracle__ uses the oracle score function to find the highest average scoring oracle against a mixed strategy.  

The utility functions are as follows:  
__generateRewardsAndPenalties__ generates random rewards and penalties for a game between bounds and for a specified number of targets.
__createRandomGame__ generates a game using random rewards and penalties, and a random number of resources and timesteps.  

### Experiment
experiment.py contains the flow of the experiment. It begins with some debug parameters and HyperParameters (i'd love to centralizes all the parameters to this point eventually). It starts by creating a game, seeding the initial strategies, and then proceeding through the following loop:  
* Generate mixed strategies  
* Find best pure defender
* clone that into a new oracle
* train oracle
* Find best pure attacker
* clone that into a new oracle
* train oracle
* Update pure pools and calculate payouts with each pure opponent
Currently testing is being done with fixed set of rewards and penalties for the sake of consistency.  

### Training Test
trainingTest.py contains code for testing the training process on an oracle. a simple game is constructed following the flow of the experiment code up to mixed strategy generation. when the oracle is trained, the average utility of the oracle against the mixed strategy is evaluated at every training step. This is used to generate a graph showing the effects of training on oracle at each timestep.
