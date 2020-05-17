"""
Contains elements necessary for Actor Critic learning.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
# External
from collections import namedtuple
import torch
import random
# Internal

# A named tuple to represent transitions that have been seen.
Transition = namedtuple('Transition', ('ob0', 'action0', 'ob1', 'action1', 'reward', 'ob2', 'newStateActions'))

# A class representing the replay memory of the agent being trained
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

# Packages observations and actions into an input tensor for the neural network
def getInputTensor(oldObservation, observation, oldAction, action):
    oldActionTensor = torch.tensor(oldAction).float().requires_grad_(True)
    newActionTensor = torch.tensor(action).float().requires_grad_(True)
    oldObservationTensor = torch.from_numpy(oldObservation).float().requires_grad_(True)
    newObservationTensor = torch.from_numpy(observation).float().requires_grad_(True)

    old = torch.cat((oldObservationTensor, oldActionTensor),0)
    new = torch.cat((newObservationTensor, newActionTensor),0)

    return torch.cat((old.unsqueeze(0), new.unsqueeze(0)))

# Learns from a batch in the replay memory and returns the average loss
def sampleMinibatch(replayMemory, game, targetNetwork, oracleToTrain, lossFunction, optimizer, timestep, batchSize=15):
    avgLoss = 0
    if len(replayMemory) >= batchSize:
        minibatch = replayMemory.sample(batchSize)
        optimizer.zero_grad()
        for sample in minibatch:
            # For each thing in the minibatch, calculate the true label using Q^ (target network)
            y = sample.reward
            if timestep != game.timesteps -1:
                y += max([targetNetwork.forward(sample.ob1, sample.ob2, sample.action1, newAction) for newAction in sample.newStateActions])
            else:
                y = torch.tensor(y)
            y = y.float()
            guess = oracleToTrain.forward(sample.ob0, sample.ob1, sample.action0, sample.action1)
            loss = lossFunction(guess, y)
            avgLoss += loss.item()
            loss.backward()
        optimizer.step()
    return avgLoss
