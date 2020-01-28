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
        output = self.outputSoftmax(linearOutput).view(2,self.targetNum).squeeze(1).float().requires_grad_(True)[1]
        return output

    def reset(self):
        pass

    def inputFromGame(game):
        """ A curried function for creating neural net inputs from one observation """
        def mush(observation):
            return None
        return None
