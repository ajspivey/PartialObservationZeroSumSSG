# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Internal Imports
import ssg
from actorCritic import getInputTensor, Transition, ReplayMemory, sampleMinibatch
from baseline import getBaselineScore

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

        # LSTM
        self.LSTM = nn.LSTM(self.input_size, self.input_size*self.featureCount)
        self.linearLayer = nn.Linear(2*self.input_size*self.featureCount, self.featureCount)
        self.outputLinearLayer = nn.Linear(2*self.featureCount, 1)
        self.ReLU = nn.ReLU()

        # LINEAR
        # self.inputLayer = nn.Linear(self.input_size, self.input_size*self.featureCount)
        # self.linearLayer1 = nn.Linear(self.input_size*self.featureCount, 10*self.input_size*self.featureCount)
        # self.linearLayer2 = nn.Linear(10*self.input_size*self.featureCount, self.input_size*self.featureCount)
        # self.outputLinearLayer = nn.Linear(self.input_size*self.featureCount, 1)
        # self.PReLU = nn.PReLU()

    # Define a forward pass of the network
    def forward(self, oldObservation, observation, oldAction, action):
        # LSTM
        inputTensor = getInputTensor(oldObservation, observation, oldAction, action)
        LSTMOutput, hiddenStates = self.LSTM(inputTensor)
        LSTMOutput = LSTMOutput[1]
        LSTMReLU = self.ReLU(LSTMOutput)
        CReLU = torch.cat((LSTMReLU, -LSTMReLU), 0).flatten().unsqueeze(0)
        linear = self.linearLayer(CReLU)
        linearReLU = self.ReLU(linear)
        CReLU2 = torch.cat((linearReLU, -linearReLU), 0).flatten().unsqueeze(0)
        output = self.outputLinearLayer(CReLU2)
        return output[0]
        
        # LINEAR
        # actionTensor = torch.tensor(action).float().requires_grad_(True)
        # observationTensor = torch.from_numpy(observation).float().requires_grad_(True)
        # inputTensor = torch.cat((observationTensor, actionTensor),0)
        # inputTensor = inputTensor.view(1, -1)
        # input = self.PReLU(self.inputLayer(inputTensor))
        # linear1 = self.PReLU(self.linearLayer1(input))
        # linear2 = self.PReLU(self.linearLayer2(linear1))
        # output = self.outputLinearLayer(linear2)
        # return output[0]

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

    def isSoftmax(self):
        return False

class DefenderParameterizedSoftmax():
    def __init__(self, targetNum):
        # Initalize random weights for parameterized softmx --
        # For the defender, we are parameterizing rewards and penalties at
        # remaining targets.
        self.targetNum = targetNum
        self.rewardsWeight = np.random.uniform(0,1)
        self.penaltiesWeight = np.random.uniform(0,1)

    def getActionDistribution(self, game, actions, observation):
        actionValues = [self.forward(game.previousDefenderObservation, observation, game.previousDefenderAction, action) for action in actions]
        estimateSum = sum(actionValues)
        distribution = [actionValue/estimateSum for actionValue in actionValues]
        return distribution

    def forward(self, previousObservation, observation, previousAction, action):
        rewards = observation[self.targetNum*3:self.targetNum*4]
        penalties = observation[self.targetNum*4:]
        # Combine the weights
        weightedRewards = sum(rewards * action * self.rewardsWeight)
        weightedPenalties = -sum(penalties * [1-x for x in action] * self.penaltiesWeight)
        return exp(weightedRewards + weightedPenalties)

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.DEFENDER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        distribution = self.getActionDistribution(game, actions, observation)
        return actions[np.random.choice(len(actions), 1, p=distribution)[0]]

    def setState(self, state):
        pass

    def getState(self):
        return None

    def isSoftmax(self):
        return True

# ==============================================================================
# FUNCTIONS
# ==============================================================================
# ------------------------------------------------------------------------------
def defenderTrain(oracleToTrain, aIds, aMap, aMix, game, dPool, N=300, batchSize=30, C=30, epochs=100, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False, trainingTest=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters(), lr=0.00001)
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    gameClone = ssg.cloneGame(game)

    if trainingTest:
        history = []
        lossHistory = []
        equilibriumHistory = []
        equilibriumScore = 0#getBaselineScore(ssg.DEFENDER, aIds, aMap, aMix, gameClone, dPool)

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
        attackerAgent = aMap[np.random.choice(aIds, 1, p=aMix)[0]]

        for timestep in range(game.timesteps):                                  # Play a full game
            # Choose an action based off of Q network (oracle to train)
            dAction = oracleToTrain.getAction(game, dOb)
            aAction = attackerAgent.getAction(game, aOb)

            # Execute that action and store the result in replay memory
            ob0 = game.previousDefenderObservation
            action0 = game.previousDefenderAction
            ob1 = dOb
            action1 = dAction
            dOb, aOb, dScore, aScore = game.performActions(dAction, aAction, dOb, aOb)
            replayMemory.push(ob0, action0, ob1, action1, dScore, dOb, game.getValidActions(ssg.DEFENDER))

            # Sample a random minibatch of transitions from replay memory
            avgLoss = sampleMinibatch(replayMemory, game, targetNetwork, oracleToTrain, lossFunction, optimizer, timestep, batchSize=batchSize)

            if trainingTest:
                oracleScore = ssg.expectedPureVMix(ssg.DEFENDER, oracleToTrain, aMap, aMix, gameClone)
                history.append(oracleScore)
                lossHistory.append(avgLoss/batchSize)
                equilibriumHistory.append(equilibriumScore)
            # Every C steps, set Q^ = Q
            step += 1
            if step == C:
                targetNetwork.setState(oracleToTrain.getState())
                step = 0

        game.restartGame()
    if trainingTest:
        return history, lossHistory, equilibriumHistory
    return ssg.expectedPureVMix(ssg.DEFENDER, oracleToTrain, aMap, aMix, gameClone)
