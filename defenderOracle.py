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
        actions = game.getValidActions(ssg.DEFENDER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        index = np.argmax([self.forward(game, observation, action) for action in actions])
        action = actions[index]
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
        actions = game.getValidActions(ssg.DEFENDER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        choice = np.asarray(self.random.choice(actions))
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

def train(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, alpha=0.15, epochs=10, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())

    for _ in range(0, epochs):
        avgLoss = 0
        attackerAgent = aMap[np.random.choice(aIds, 1, p=attackerMixedStrategy)[0]]

        dOb, aOb = game.getEmptyObservations()
        for timestep in range(game.timesteps):                                  # Play a full game
            aAction = attackerAgent.getAction(game, aOb)
            actions = game.getValidActions(ssg.DEFENDER)
            dAction = oracleToTrain.getActionFromActions(game, actions, dOb)

            for action in actions:                                              # Evaluate the qValue guess for each action
                qValueGuess = oracleToTrain(game, dOb, action)
                qValueLabel = torch.tensor(game.getActionScore(ssg.DEFENDER, action, aAction, game.defenderRewards, game.defenderPenalties))
                loss = lossFunction(qValueGuess,qValueLabel)
                avgLoss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avgLoss /= len(actions)
            dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

        game.restartGame()
        for aId in aIds:
            aMap[aId].reset()

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
