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

torch.autograd.set_detect_anomaly(True)

# ==============================================================================
# CLASSES
# ==============================================================================
class DefenderOracle(nn.Module):
    def __init__(self, targetNum):
        super(DefenderOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.DEFENDER_FEATURE_SIZE
        self.observation_dim = targetNum * self.featureCount
        self.input_size = self.observation_dim + targetNum

        self.linearLayer = nn.Linear(self.input_size, self.input_size*self.featureCount)
        self.ReLU = nn.ReLU()
        self.LSTM = nn.LSTM(2*self.input_size*self.featureCount, self.input_size*self.featureCount)
        self.outputLinearLayer = nn.Linear(2*self.input_size*self.featureCount, 1)


    # Define a forward pass of the network
    def forward(self, game, observation, action):
        inputTensor = self.getInputTensor(game, observation, action)
        # Linear Layer
        linearOut = self.linearLayer(inputTensor)
        # CReLU
        ReLUOld, ReLUNew = self.ReLU(linearOut)
        CReLUOld = torch.cat((ReLUOld, -ReLUOld),0)
        CReLUNew = torch.cat((ReLUNew, -ReLUNew),0)
        CReLU = torch.cat((CReLUOld,-CReLUNew),0).view(2,2*self.input_size*self.featureCount).unsqueeze(1)
        # LSTM
        LSTMOut, _ = self.LSTM(CReLU)
        # sequenceSize, batchSize, numberOfOutputFeatures = LSTMOut.size(0), LSTMOut.size(1), LSTMOut.size(2)
        LSTMOut = torch.flatten(LSTMOut)
        # Output
        output = self.outputLinearLayer(LSTMOut)
        return output[0]

    def getInputTensor(self, game, observation, action):
        oldActionTensor = torch.tensor(game.previousDefenderAction).float().requires_grad_(True)
        newActionTensor = torch.tensor(action).float().requires_grad_(True)
        oldObservationTensor = torch.from_numpy(game.previousDefenderObservation).float().requires_grad_(True)
        newObservationTensor = torch.from_numpy(observation).float().requires_grad_(True)

        old = torch.cat((oldObservationTensor, oldActionTensor),0)
        new = torch.cat((newObservationTensor, newActionTensor),0)

        return torch.cat((old.unsqueeze(0), new.unsqueeze(0)))

    def getAction(self, game, observation):
        validActions = game.getValidActions(ssg.DEFENDER)
        index = np.argmax([self.forward(game, observation, action) for action in validActions])
        action = validActions[index]
        return action

    def setState(self, state):
        if state is not None:
            self.load_state_dict(state, strict=True)

    def getState(self):
        state = self.state_dict()
        return state

    def reset(self):
        pass

class RandomDefenderOracle():
    def __init__(self, targetNum, game):
        super(RandomDefenderOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.DEFENDER_FEATURE_SIZE
        self.game = game

        self.random = Random()
        self.startState = self.random.getstate()

    def getAction(self, game, observation):
        # Get all the valid games
        validActions = self.game.getValidActions(ssg.DEFENDER)
        choice = np.asarray(self.random.choice(validActions))
        return choice

    def reset(self):
        self.random.setstate(self.startState)

    def getState(self):
        return None

    def setState(self, state):
        pass

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def train(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, alpha=0.15, epochs=10, iterations=10, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())

    # Choose whether to use the uniform distribution (explore) or the mixed distribution
    # distributionChoice = np.random.choice([0,1], 1, p=[1-alpha, alpha])[0]
    # distribution = distributions[distributionChoice]
    # uniformStrategy = [1/len(attackerMixedStrategy)] * len(attackerMixedStrategy)
    # distributions = [attackerMixedStrategy, uniformStrategy]

    for _ in range(iterations):
        avgLoss = 0
        for _ in range(0, epochs):
            dOb, aOb = game.getEmptyObservations()

            distribution = attackerMixedStrategy
            attackerAgent = aMap[np.random.choice(aIds, 1,
            p=distribution)[0]]

            for timestep in range(game.timesteps):
                aAction = attackerAgent.getAction(game, aOb)
                dAction = oracleToTrain.getAction(game, dOb)

                actions = game.getValidActions(ssg.DEFENDER)
                for action in actions:
                    qValueGuess = oracleToTrain(game, dOb, action)
                    qValueLabel = torch.tensor(game.getActionScore(ssg.DEFENDER, action, aAction, game.defenderRewards, game.defenderPenalties))
                    loss = lossFunction(qValueGuess,qValueLabel)
                    avgLoss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                avgLoss = avgLoss / len(actions)
                dOb, aOb = game.performActions(ssg.DEFENDER, dAction, aAction, dOb, aOb)

            game.restartGame()
            for aId in aIds:
                aMap[aId].reset()

            avgLoss = avgLoss / game.timesteps
        avgLoss = avgLoss / epochs
        print(f"avg loss over epoch: {avgLoss}")



# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass


if __name__ == "__main__":
    main()
