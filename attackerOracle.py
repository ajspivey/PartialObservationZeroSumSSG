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
        actions = game.getValidActions(ssg.ATTACKER)
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

class RandomAttackerOracle():
    def __init__(self, targetNum, game):
        super(RandomAttackerOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE
        self.game = game

        self.random = Random()
        self.startState = self.random.getstate()

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.ATTACKER)
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
            dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

        game.restartGame()
        for dId in dIds:
            dMap[dId].reset()

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
