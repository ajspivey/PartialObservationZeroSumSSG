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
from collections import namedtuple
import random
from random import Random

# Internal Imports
import ssg
from coreLP import createDefenderOneShotModel

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

torch.autograd.set_detect_anomaly(True)

Transition = namedtuple('Transition', ('ob0', 'action0', 'ob1', 'action1', 'reward', 'ob2', 'newStateActions'))
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
    def forward(self, previousObservation, observation, previousAction, action):
        inputTensor = getInputTensor(previousObservation, observation, previousAction, action)
        # Linear Layer
        linearOut = self.linearLayer(inputTensor)
        # CReLU
        ReLUOld, ReLUNew = self.ReLU(linearOut)
        CReLUOld = torch.cat((ReLUOld, -ReLUOld),0)
        CReLUNew = torch.cat((ReLUNew, -ReLUNew),0)
        CReLU = torch.cat((CReLUOld,-CReLUNew),0).view(2,2*self.input_size*self.featureCount).unsqueeze(1)
        # LSTM
        LSTMOut, _ = self.LSTM(CReLU)
        LSTMOut = torch.flatten(LSTMOut)
        # Output
        output = self.outputLinearLayer(LSTMOut)
        return output[0]

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.DEFENDER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        index = np.argmax([self.forward(game.previousDefenderObservation, observation, game.previousDefenderAction, action) for action in actions])
        action = actions[index]
        return action

    def setState(self, state):
        if state is not None:
            self.load_state_dict(state, strict=True)

    def getState(self):
        state = self.state_dict()
        return state

class DefenderEquilibrium():
    def __init__(self, targetNum):
        super(DefenderEquilibrium, self).__init__()
        self.targetNum = targetNum

    # Define a forward pass of the network
    def forward(self, previousObservation, observation, previousAction, action):
        return None

    def getAction(self, game, observation):
        defenderModel, actionDistribution, actions = createDefenderOneShotModel(game)
        defenderModel.solve()
        actionDistribution = [float(value) for value in actionDistribution.values()]
        index = np.random.choice(range(len(actions)), 1, p=actionDistribution)[0]
        return actions[index]

    def getActionFromActions(self, game, actions, observation):
        return self.getAction(game, observation)

    def setState(self, state):
        pass

    def getState(self):
        return None

# ------------------------------------------------------------------------------
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

# ------------------------------------------------------------------------------
def train(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, N=100, batchSize=15, C=20, epochs=10, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters(), lr=0.001)
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    # Initialize the replay memory with limited capacity N
    replayMemory = ReplayMemory(N)
    # Initialize target network with weights equal to the oracle to train
    targetNetwork = DefenderOracle(oracleToTrain.targetNum)
    targetNetwork.setState(oracleToTrain.getState())

    # An epoch is one iteration over all training data. In our case, that's the one
    # Game we're learning on.
    step = 0
    for epoch in range(0, epochs):
        print(f"epoch {epoch} of {epochs}")
        # initialize the starting values for the game
        dOb, aOb = game.getEmptyObservations()
        attackerAgent = aMap[np.random.choice(aIds, 1, p=attackerMixedStrategy)[0]]

        for timestep in range(game.timesteps):                                  # Play a full game
            # Choose an action based off of Q network (oracle to train)
            dAction = oracleToTrain.getAction(game, dOb)
            aAction = attackerAgent.getAction(game, aOb)

            # Execute that action and store the result in replay memory
            ob0 = game.previousDefenderObservation
            action0 = game.previousDefenderAction
            ob1 = dOb
            action1 = dAction
            dOb, aOb, reward = game.performActions(dAction, aAction, dOb, aOb)

            replayMemory.push(ob0, action0, ob1, action1, reward, dOb, game.getValidActions(ssg.DEFENDER))

            # Sample a random minibatch of transitions from replay memory
            if len(replayMemory) >= batchSize:
                minibatch = replayMemory.sample(batchSize)
                for sample in minibatch:
                    # For each thing in the minibatch, calculate the true label using Q^ (target network)
                    y = sample.reward
                    if timestep != game.timesteps -1:
                        y += max([targetNetwork.forward(sample.ob1, sample.ob2, sample.action1, newAction) for newAction in sample.newStateActions])
                    else:
                        y = torch.tensor(y)
                    guess = oracleToTrain.forward(sample.ob0, sample.ob1, sample.action0, sample.action1)
                    loss = lossFunction(guess, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # Every C steps, set Q^ = Q
            step += 1
            if step == C:
                targetNetwork.setState(oracleToTrain.getState())
                step = 0

        game.restartGame()



# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass


if __name__ == "__main__":
    main()
