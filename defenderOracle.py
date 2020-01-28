# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from random import Random

# Internal Imports
import ssg
import generalOracle as gO

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# ==============================================================================
# CLASSES
# ==============================================================================
class DefenderOracle(gO.Oracle):
    def __init__(self, targetNum):
        super(DefenderOracle, self).__init__(targetNum, ssg.DEFENDER_FEATURE_SIZE)
        self.targetNum = targetNum
        self.featureCount = ssg.DEFENDER_FEATURE_SIZE

    def inputFromGame(self, game):
        def buildInput(observation):
            old = torch.from_numpy(game.previousDefenderObservation).float().requires_grad_(True)
            new = torch.from_numpy(observation).float().requires_grad_(True)
            modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
            return modelInput
        return buildInput

class RandomDefenderOracle(gO.Oracle):
    def __init__(self, targetNum, game):
        super(RandomDefenderOracle, self).__init__(targetNum, ssg.DEFENDER_FEATURE_SIZE)
        self.targetNum = targetNum
        self.featureCount = ssg.DEFENDER_FEATURE_SIZE
        self.game = game

        self.random = Random()
        self.startState = self.random.getstate()

    # Define a forward pass of the network
    def forward(self, observation):
        # Get all the valid games
        validActions = self.game.getValidActions(ssg.DEFENDER)
        choice = self.random.choice(validActions)
        return choice

    def reset(self):
        self.random.setstate(self.startState)

    def inputFromGame(self, game):
        def buildInput(observation):
            old = torch.from_numpy(game.previousDefenderObservation).float().requires_grad_(True)
            new = torch.from_numpy(observation).float().requires_grad_(True)
            modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
            return modelInput
        return buildInput

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def makeDefenderPolicy(defenderRewards, defenderPenalties):
    """ Generates a random defender policy for testing """
    # [action, pastattacks, pastattackstatus, payoffs]
    attackerPolicy = {
    # Null round
    tuple(np.concatenate(([0,0,0, 0,0,0, 0,0,0], defenderRewards, defenderPenalties))): [1,0,0],

    # First round
    tuple(np.concatenate(([1,0,0, 1,0,0, 0,0,0], defenderRewards, defenderPenalties))): [0,1,0],   #2
    tuple(np.concatenate(([1,0,0, 1,0,0, 1,0,0], defenderRewards, defenderPenalties))): [0,1,0],   #1

    # Second round
    tuple(np.concatenate(([0,1,0, 1,2,0, 0,0,0], defenderRewards, defenderPenalties))): [0,0,1],   #1
    tuple(np.concatenate(([0,1,0, 1,2,0, 0,1,0], defenderRewards, defenderPenalties))): [0,0,1],

    tuple(np.concatenate(([0,1,0, 1,2,0, 0,1,0], defenderRewards, defenderPenalties))): [0,0,1],   #2
    tuple(np.concatenate(([0,1,0, 1,2,0, 1,0,0], defenderRewards, defenderPenalties))): [0,0,1],
    }
    return attackerPolicy

def testOracle(oracle, numTargets, numGames=15):
    correct = 0
    totalGuesses = 0
    for i in range (0, numGames):
        game, defenderRewards, defenderPenalties = ssg.createRandomGame(numTargets)
        createInput = gO.inputFromGame(game)
        mixedPolicy = makeDefenderPolicy(defenderRewards, defenderPenalties)

        aAction = [0]*numTargets
        dAction = [0]*numTargets
        dOb, aOb = game.getEmptyObservations()

        # Play a full game
        for timestep in range(game.timesteps):
            aAction = mixedPolicy[tuple(dOb)]  # Attacker action

            # Get the guess and label
            x = oracle(createInput(aOb)).view(2,3)[1]   # Defender action guess
            x = x.gt(0.5).int().detach().numpy()
            y, yScore = game.getBestActionAndScore(ssg.DEFENDER, dAction, defenderRewards, defenderPenalties) # Defender action true label
            if (np.array_equal(x,y)):
                correct += 1
            else:
                print(f"Inccorect guess: {x} instead of {y}")
            totalGuesses += 1

            game.performActions(x, aAction, dOb, aOb)

    print(f"Model tested. {correct}/{totalGuesses} guesses correct")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # ==============
    # Create Network
    # ==============
    numTargets = 3
    featureCount = 5
    oracle = gO.train(oracle=DefenderOracle(numTargets,featureCount), player=ssg.DEFENDER, targets=numTargets, makePolicy=makeDefenderPolicy, showOutput=True)
    print("Model initialized")
    testOracle(oracle, numTargets)


if __name__ == "__main__":
    main()
