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
        self.observation_dim = targetNum * self.featureCount
        self.input_size = self.observation_dim + targetNum

        self.linearLayer = nn.Linear(self.input_size, self.input_size*self.featureCount)
        self.ReLU = nn.ReLU()
        self.LSTM = nn.LSTM(2*self.input_size*self.featureCount, self.input_size*self.featureCount)
        self.outputLinearLayer = nn.Linear(2*self.input_size*self.featureCount, 1)

    # Define a forward pass of the network
    def forward(self, oldObservation, observation, oldAction, action):
        inputTensor = getInputTensor(oldObservation, observation, oldAction, action)
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

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.ATTACKER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        index = np.argmax([self.forward(game.previousAttackerObservation, observation, game.previousDefenderAction, action) for action in actions])
        action = actions[index]
        return action

    def setState(self, state):
        if state is not None:
            self.load_state_dict(state, strict=True)

    def getState(self):
        state = self.state_dict()
        return state

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def getInputTensor(oldObservation, observation, oldAction, action):
    oldActionTensor = torch.tensor(oldAction).float().requires_grad_(True)
    newActionTensor = torch.tensor(action).float().requires_grad_(True)
    oldObservationTensor = torch.from_numpy(oldObservation).float().requires_grad_(True)
    newObservationTensor = torch.from_numpy(observation).float().requires_grad_(True)

    old = torch.cat((oldObservationTensor, oldActionTensor),0)
    new = torch.cat((newObservationTensor, newActionTensor),0)

    return torch.cat((old.unsqueeze(0), new.unsqueeze(0)))


def train(oracleToTrain, dIds, dMap, defenderMixedStrategy, game, alpha=0.15, epochs=10, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())

    for _ in range(0, epochs):
        avgLoss = 0
        defenderAgent = dMap[np.random.choice(dIds, 1, p=defenderMixedStrategy)[0]]

        dOb, aOb = game.getEmptyObservations()
        for timestep in range(game.timesteps):
            dAction = defenderAgent.getAction(game, dOb)
            actions = game.getValidActions(ssg.ATTACKER)
            aAction = oracleToTrain.getActionFromActions(game, actions, aOb)

            for action in actions:
                qValueGuess = oracleToTrain(game, dOb, action)
                qValueLabel = torch.tensor(game.getActionScore(ssg.ATTACKER, action, dAction, game.defenderRewards, game.defenderPenalties))
                loss = lossFunction(qValueGuess,qValueLabel)
                avgLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avgLoss /= len(actions)
            dOb, aOb, reward = game.performActions(dAction, aAction, dOb, aOb)

        game.restartGame()

        avgLoss /= game.timesteps
        if (showOutput):
            print(f"Avg loss for last epoch: {avgLoss}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass


if __name__ == "__main__":
    main()
