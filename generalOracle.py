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
        return None

    def reset(self):
        pass

    def inputFromGame(game):
        """ A curried function for creating neural net inputs from one observation """
        def mush(observation):
            return None
        return None
