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
class AttackerOracle(gO.Oracle):
    def __init__(self, targetNum):
        super(AttackerOracle, self).__init__(targetNum, ssg.ATTACKER_FEATURE_SIZE)
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE

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
        output = self.outputSoftmax(linearOutput).view(2,self.targetNum).squeeze(1).float().requires_grad_(True)[1]
        maxValIndex = torch.argmax(output)
        oneHot = torch.nn.functional.one_hot(maxValIndex, self.targetNum)
        return oneHot

    def inputFromGame(self, game):
        def buildInput(observation):
            old = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
            new = torch.from_numpy(observation).float().requires_grad_(True)
            modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
            return modelInput
        return buildInput

class RandomAttackerOracle(gO.Oracle):
    def __init__(self, targetNum, game):
        super(RandomAttackerOracle, self).__init__(targetNum, ssg.ATTACKER_FEATURE_SIZE)
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE
        self.game = game

        self.random = Random()
        self.startState = self.random.getstate()

    # Define a forward pass of the network
    def forward(self, observation):
        # Get all the valid games
        validActions = self.game.getValidActions(ssg.ATTACKER)
        choice = torch.from_numpy(np.asarray(self.random.choice(validActions))).float().requires_grad_(True)
        return choice

    def reset(self):
        self.random.setstate(self.startState)

    def inputFromGame(self, game):
        def buildInput(observation):
            old = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
            new = torch.from_numpy(observation).float().requires_grad_(True)
            modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
            return modelInput
        return buildInput

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def train(oracleToTrain, defenderPool, defenderMixedStrategy, game, epochs=10, iterations=25, optimizer=None, lossFunction=nn.SmoothL1Loss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.RMSprop(oracleToTrain.parameters())

    inputFunction = oracleToTrain.inputFromGame(game)

    totalLoss = float("inf")
    totalUtility = 0
    #while totalLoss > lossThreshold:
    for _ in range(iterations):
        if (showOutput):
            print(f"Avg loss for last {epochs} samples = {totalLoss}")
            print(f"Avg utility for last {epochs} samples = {totalUtility}")
        totalLoss = 0

        for _ in range(0, epochs):
            dAction = [0]*game.numTargets
            aAction = [0]*game.numTargets
            dOb, aOb = game.getEmptyObservations()

            for timestep in range(game.timesteps):
                # Create model input
                defenderAgent = np.random.choice(defenderPool, 1,
                              p=defenderMixedStrategy)[0]
                agentInputFunction = defenderAgent.inputFromGame(game)
                dAction = defenderAgent(agentInputFunction(dOb))

                guess = oracleToTrain(inputFunction(aOb))
                aAction, _ = game.getBestActionAndScore(ssg.ATTACKER, dAction, game.defenderRewards, game.defenderPenalties)
                label = torch.from_numpy(aAction).float().requires_grad_(True)

                loss = lossFunction(guess, label)
                totalLoss += loss.item()

                optimizer.zero_grad()
                loss.backward() # compute the gradient of the loss with respect to the parameters of the model
                optimizer.step() # Perform a step of the optimizer based on the gradient just calculated

                dOb, aOb = game.performActions(dAction, label, dOb, aOb)
            totalUtility += game.attackerUtility
            game.restartGame()
            for defender in defenderPool:
                defender.reset()

        totalLoss = totalLoss/epochs
        totalUtility = totalUtility/epochs
    return oracleToTrain

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass


if __name__ == "__main__":
    main()
