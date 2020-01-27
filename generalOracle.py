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

torch.manual_seed(1)
np.random.seed(1)

# ==============================================================================
# CLASSES
# ==============================================================================
class Oracle(nn.Module):
    def __init__(self, targetNum, featureCount):
        super(Oracle, self).__init__()
        self.observation_dim = targetNum * featureCount
        self.featureCount = featureCount

        self.linearLayer = nn.Linear(self.observation_dim, featureCount)
        self.ReLU = nn.ReLU()
        self.LSTM = nn.LSTM(2*featureCount, 2*targetNum*featureCount)
        self.outputLinearLayer = nn.Linear(2*targetNum*featureCount, targetNum)
        self.outputSoftmax = nn.Sigmoid()

    # Define a forward pass of the network
    def forward(self, observation):
        # Linear Layer
        linearOut = self.linearLayer(observation)
        # CReLU
        ReLUOld, ReLUNew = self.ReLU(linearOut)
        CReLUOld = torch.cat((ReLUOld, -ReLUOld),0)
        CReLUNew = torch.cat((ReLUNew, -ReLUNew),0)
        CReLU = torch.cat((CReLUOld,-CReLUNew),0).view(2,2*self.featureCount).unsqueeze(1)
        # LSTM
        LSTMOut, _ = self.LSTM(CReLU)
        sequenceSize, batchSize, numberOfOutputFeatures = LSTMOut.size(0), LSTMOut.size(1), LSTMOut.size(2)
        LSTMOut = LSTMOut.view(sequenceSize*batchSize, numberOfOutputFeatures)
        # Output
        linearOutput = self.outputLinearLayer(LSTMOut)
        output = self.outputSoftmax(linearOutput)
        return output

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def train(oracle, player, targets, makePolicy, epochs=10, lossThreshold=1e-7, optimizer=None, lossFunction=nn.SmoothL1Loss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.RMSprop(oracle.parameters())

    totalLoss = float("inf")
    while totalLoss > lossThreshold:
        if (showOutput):
            print(f"Avg loss for last {epochs} samples = {totalLoss}")
        totalLoss = 0
        game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets)
        inputFunction = inputFromGame(game)
        mixedPolicy = makePolicy(defenderRewards, defenderPenalties)

        for _ in range(0, epochs):
            dAction = [0]*game.numTargets
            aAction = [0]*game.numTargets
            dOb, aOb = game.getEmptyObservations()

            for timestep in range(game.timesteps):
                # Create model input
                if (player == ssg.DEFENDER):
                    aAction = mixedPolicy[tuple(dOb)]
                    guess = oracle(inputFunction(dOb)).view(2,3).squeeze(1).float().requires_grad_(True)
                    dAction, _ = game.getBestActionAndScore(ssg.DEFENDER, aAction, defenderRewards, defenderPenalties)
                    labelBit = np.concatenate((game.previousDefenderAction,dAction))
                    label = torch.from_numpy(labelBit).view(2,3).float().requires_grad_(True)

                elif(player == ssg.ATTACKER):
                    dAction = mixedPolicy[tuple(dOb)]
                    guess = oracle(inputFunction(aOb)).view(2,3).squeeze(1).float().requires_grad_(True)
                    aAction, _ = game.getBestActionAndScore(ssg.ATTACKER, dAction, defenderRewards, defenderPenalties)
                    labelBit = np.concatenate((game.previousAttackerAction,aAction))
                    label = torch.from_numpy(labelBit).view(2,3).float().requires_grad_(True)

                loss = lossFunction(guess, label)
                totalLoss += loss.item()

                optimizer.zero_grad()
                loss.backward() # compute the gradient of the loss with respect to the parameters of the model
                optimizer.step() # Perform a step of the optimizer based on the gradient just calculated

                dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

            game.restartGame()

        totalLoss = totalLoss/epochs
    return oracle

def inputFromGame(game):
    """ A curried function for creating neural net inputs from one observation """
    def mush(observation):
        old = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
        new = torch.from_numpy(observation).float().requires_grad_(True)
        modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
        return modelInput
    return mush

def getRandomOracle(player, targetNum, featureCount):
    # Create an Oracle
    oracle = Oracle(targetNum, featureCount)
    # Train it to guess valid moves

    return None
