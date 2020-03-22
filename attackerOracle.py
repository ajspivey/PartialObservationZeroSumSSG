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

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# ==============================================================================
# CLASSES
# ==============================================================================
class AttackerOracle(nn.Module):
    def __init__(self, targetNum):
        super(AttackerOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE

    # Define a forward pass of the network
    def forward(self, observation):
        old = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
        new = torch.from_numpy(observation).float().requires_grad_(True)
        modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))

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
        return output

    def getAction(self, game, observation):
        return max([self.forward(observation, action) for action in game.getValidActions(game.ATTACKER)])

    def setState(self, state):
        if state is not None:
            self.load_state_dict(state, strict=True)

    def getState(self):
        state = self.state_dict()
        return state

class RandomAttackerOracle():
    def __init__(self, targetNum, game):
        super(RandomAttackerOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE
        self.game = game

        self.random = Random()
        self.startState = self.random.getstate()

    def getAction(self, game, observation):
        # Get all the valid games
        validActions = self.game.getValidActions(ssg.ATTACKER)
        choice = np.asarray(self.random.choice(validActions))
        return choice

    def reset(self):
        self.random.setstate(self.startState)

    def getState(self):
        return None

    def setState(self, state):
        pass

# class UniformAttackerOracle(gO.Oracle):
#     def __init__(self, targetNum, game):
#         super(RandomAttackerOracle, self).__init__(targetNum, ssg.ATTACKER_FEATURE_SIZE)
#         self.targetNum = targetNum
#         self.featureCount = ssg.ATTACKER_FEATURE_SIZE
#         self.game = game
#
#         self.random = Random()
#
#     # Define a forward pass of the network
#     def forward(self, observation):
#         # Get all the valid games
#         validActions = self.game.getValidActions(ssg.ATTACKER)
#         choice = torch.from_numpy(np.asarray(self.random.choice(validActions))).float().requires_grad_(True)
#         return choice
#
#     def inputFromGame(self, game):
#         def buildInput(observation):
#             old = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
#             new = torch.from_numpy(observation).float().requires_grad_(True)
#             modelInput = torch.cat((old.unsqueeze(0),new.unsqueeze(0)))
#             return modelInput
#         return buildInput

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def train(oracleToTrain, dIds, dMap, defenderMixedStrategy, game, alpha=0.15, epochs=10, iterations=100, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())

    inputFunction = oracleToTrain.inputFromGame(game)

    totalLoss = float("inf")
    totalUtility = 0

    # Choose whether to use the uniform distribution (explore) or the mixed distribution
    # distributionChoice = np.random.choice([0,1], 1, p=[1-alpha, alpha])[0]
    # distribution = distributions[distributionChoice]
    # uniformStrategy = [1/len(defenderMixedStrategy)] * len(defenderMixedStrategy)
    # distributions = [defenderMixedStrategy, uniformStrategy]
    for _ in range(iterations):
        if (showOutput):
            print(f"Avg loss for last {epochs} samples = {totalLoss}")
            print(f"Avg utility for last {epochs} samples = {totalUtility}")
        totalLoss = 0
        for _ in range(0, epochs):
            dAction = [0]*game.numTargets
            aAction = [0]*game.numTargets
            dOb, aOb = game.getEmptyObservations()

            distribution = defenderMixedStrategy
            defenderAgent = dMap[np.random.choice(dIds, 1,
            p=distribution)[0]]
            agentInputFunction = defenderAgent.inputFromGame(game)

            for timestep in range(game.timesteps):
                dAction = defenderAgent(agentInputFunction(dOb))
                dAction = game.makeLegalMove(ssg.DEFENDER, dAction)

                guess = oracleToTrain(inputFunction(aOb))
                guessScore = game.getActionScore(ssg.ATTACKER, guess, dAction, game.defenderRewards, game.defenderPenalties)
                aAction, labelScore = game.getBestActionAndScore(ssg.ATTACKER, dAction, game.defenderRewards, game.defenderPenalties)
                label = torch.from_numpy(aAction).float().requires_grad_(True)

                loss = lossFunction(torch.from_numpy(np.array(guessScore)).float().requires_grad_(True), torch.from_numpy(np.array(labelScore)).float().requires_grad_(True))
                totalLoss += loss.item()

                optimizer.zero_grad()
                loss.backward() # compute the gradient of the loss with respect to the parameters of the model
                optimizer.step() # Perform a step of the optimizer based on the gradient just calculated

                dOb, aOb = game.performActions(ssg.ATTACKER, label, dAction, aOb, dOb)
            totalUtility -= game.defenderUtility
            game.restartGame()
            for dId in dIds:
                dMap[dId].reset()

        totalLoss = totalLoss/epochs
        totalUtility = totalUtility/epochs

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass


if __name__ == "__main__":
    main()
