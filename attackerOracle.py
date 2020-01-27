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

# Internal Imports
import ssg
import generalOracle as gO

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# ==============================================================================
# CLASSES
# ==============================================================================
class AttackerOracle(gO.Oracle):
    def __init__(self, targetNum, featureCount):
        super(AttackerOracle, self).__init__(targetNum, featureCount)

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def makeDefenderPolicy(defenderRewards, defenderPenalties):
    # [action, pastattacks, pastattackstatus, payoffs]
    defenderPolicy = {
    # Null round
    tuple(np.concatenate(([0,0,0, 0,0,0, 0,0,0], defenderRewards, defenderPenalties))): [1,0,0],

    # First round
    tuple(np.concatenate(([1,0,0, 1,0,0, 0,0,0], defenderRewards, defenderPenalties))): [0,0,0],   #1
    tuple(np.concatenate(([1,0,0, 0,1,0, 0,1,0], defenderRewards, defenderPenalties))): [1,0,0],   #2
    tuple(np.concatenate(([1,0,0, 0,0,1, 0,0,1], defenderRewards, defenderPenalties))): [1,0,0],   #3

    # Second round
    tuple(np.concatenate(([0,0,0, 1,2,0, 0,1,0], defenderRewards, defenderPenalties))): [0,0,0],   #1
    tuple(np.concatenate(([0,0,0, 1,0,2, 0,0,1], defenderRewards, defenderPenalties))): [0,0,0],

    tuple(np.concatenate(([1,0,0, 2,1,0, 0,1,0], defenderRewards, defenderPenalties))): [0,0,0],   #2
    tuple(np.concatenate(([1,0,0, 0,1,2, 0,1,1], defenderRewards, defenderPenalties))): [1,0,0],

    tuple(np.concatenate(([1,0,0, 2,0,1, 0,0,1], defenderRewards, defenderPenalties))): [0,0,0],   #3
    tuple(np.concatenate(([1,0,0, 0,2,1, 0,1,1], defenderRewards, defenderPenalties))): [1,0,0],
    }
    return defenderPolicy

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
            dAction = mixedPolicy[tuple(dOb)]  # Defender action

            # Get the guess and label
            x = oracle(createInput(aOb)).view(2,3)[1]   # Attacker action guess
            x = x.gt(0.5).int().detach().numpy()
            y, yScore = game.getBestActionAndScore(ssg.ATTACKER, dAction, defenderRewards, defenderPenalties) # Attacker action true label
            if (np.array_equal(x,y)):
                correct += 1
            else:
                print(f"Inccorect guess: {x} instead of {y}")
            totalGuesses += 1

            game.performActions(dAction, x, dOb, aOb)

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
    oracle = gO.train(oracle=AttackerOracle(numTargets,featureCount), player=ssg.ATTACKER, targets=numTargets, makePolicy=makeDefenderPolicy, showOutput=True)
    print("Model initialized")
    testOracle(oracle, numTargets)


if __name__ == "__main__":
    main()
