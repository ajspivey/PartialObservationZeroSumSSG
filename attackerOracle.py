import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# Network definition
class AttackerOracle(nn.Module):
    def __init__(self, targetNum, featureCount):
        super(AttackerOracle, self).__init__()

        # Initizlize class variables
        # Example attacker observation
        #o_a = [0,0,1, 0,0,1, 0,0,1, 1.9,4.1,3.4]
        #      [action, past attacks, past attack status, payoffs]
        self.observation_dim = targetNum * featureCount
        self.featureCount = featureCount

        # Initialize pytorch components used (LSTM, Linear, softmax, and concatReLU)
        # LINEAR LAYER
        self.linearLayerLinear = nn.Linear(self.observation_dim, featureCount)
        self.linearLayerReLU = nn.ReLU()

        # PI LAYER
        self.piLayerLSTM = nn.LSTM(2*featureCount, 2*targetNum*featureCount)
        self.piLayerLinear = nn.Linear(2*targetNum*featureCount, targetNum)
        # self.piLayerReLU = nn.ReLU()
        # self.piLayerLinear2 = nn.Linear(in?,out?)   # output should be size targetNum?
        self.piLayerSoftmax = nn.Softmax(-1)

        # Q LAYER
        # self.qLayerLSTM = nn.LSTM(in?,out?)
        # self.qLayerLinear = nn.Linear(in?,out?)
        # self.qLayerReLU = nn.ReLU()
        # self.qLayerLinear2 = nn.Linear(in?,out?)    # output should be size targetNum.

        # How to implement a concatReLU:
        # >>> m = nn.ReLU()
        # >>> input = torch.randn(2).unsqueeze(0)
        # >>> output = torch.cat((m(input),m(-input)))

    # Define a forward pass of the network
    # TODO: find out when tensors need to be reshaped
    def forward(self, observation):
        # LINEAR LAYER OUTPUT (ll)
        llLinearOut = self.linearLayerLinear(observation)
        # llLinearOut = llLinearOut.view(1,4)
        # print(llLinearOut)

        # Simulates a CReLU
        llReLUOutPast, llReLUOutNew = self.linearLayerReLU(llLinearOut)
        llCReLUOutPast = torch.cat((llReLUOutPast, -llReLUOutPast),0)
        llCReLUOutNew = torch.cat((llReLUOutNew, -llReLUOutNew),0)
        llCReLUOut = torch.cat((llCReLUOutPast,-llCReLUOutNew),0).view(2,8).unsqueeze(1)

        # LSTM LAYER OUTPUT
        piLayerLSTMOut, _ = self.piLayerLSTM(llCReLUOut)
        sequenceSize, batchSize, numberOfOutputFeatures = piLayerLSTMOut.size(0), piLayerLSTMOut.size(1), piLayerLSTMOut.size(2)
        piLayerLSTMOut = piLayerLSTMOut.view(sequenceSize*batchSize, numberOfOutputFeatures)
        piLayerLinearOut = self.piLayerLinear(piLayerLSTMOut)
        piLayersoftMaxOut = self.piLayerSoftmax(piLayerLinearOut)
        print(piLayersoftMaxOut)
        return piLayersoftMaxOut

def generateRewards(numTargets, lowBound=1, highBound = 10):
    return np.random.randint(low=lowBound, high=highBound, size=numTargets)

def main():
    # Create Network
    model = AttackerOracle(3,4) # Create an attacker oracle to train on a game with 3 targets and 4 features
    lossFunction = nn.MSELoss() # Mean-squared error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.1) # Adam optimizer
    #      [action, past attacks, past attack status, payoffs]
    previousVector = torch.from_numpy(np.array([1,0,0, 0,0,1, 0,0,1, 1.9,4.1,3.4])).float().requires_grad_(True)
    featureVector = torch.from_numpy(np.array([0,1,0, 0,2,1, 0,0,1, 1.9,4.1,3.4])).float().requires_grad_(True)
    pastAndNew = torch.cat((previousVector.unsqueeze(0),featureVector.unsqueeze(0)))
    guessBeforeTraining = model(pastAndNew)
    # Train network
    #   Each epoch
    #       some number of times
    #           generate random mixed defender strategy
    #           Get prediction X
    #           compare to actual best response Y
    #               Error is cumulative difference in utility?
    # Use network to show prediction
    pass

if __name__ == "__main__":
    main()
