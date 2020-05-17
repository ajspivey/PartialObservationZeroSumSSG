# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Internal Imports
import ssg
from actorCritic import getInputTensor, Transition, ReplayMemory, sampleMinibatch
from baseline import getBaselineScore

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

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

# ==============================================================================
# FUNCTIONS
# ==============================================================================
# ------------------------------------------------------------------------------
def defenderTrain(oracleToTrain, aIds, aMap, aMix, game, dPool, N=100, batchSize=15, C=50, epochs=50, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False, trainingTest=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = []
    lossHistory = []
    equilibriumHistory = []

    gameClone = ssg.SequentialZeroSumSSG(game.numTargets, game.numResources, game.defenderRewards, game.defenderPenalties, game.timesteps)
    equilibriumScore = getBaselineScore(ssg.DEFENDER, aIds, aMap, aMix, gameClone, dPool)

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
                oracleScore = game.getOracleScore(ssg.DEFENDER, aIds, aMap, aMix, oracleToTrain)
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
    return game.getOracleScore(ssg.DEFENDER, aIds, aMap, aMix, oracleToTrain)
