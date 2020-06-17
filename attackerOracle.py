# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from math import exp
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
class AttackerOracle(nn.Module):
    def __init__(self, targetNum):
        super(AttackerOracle, self).__init__()
        self.targetNum = targetNum
        self.featureCount = ssg.ATTACKER_FEATURE_SIZE
        self.observation_dim = targetNum * self.featureCount
        self.input_size = self.observation_dim + targetNum

        # LSTM
        # self.LSTM = nn.LSTM(self.input_size, self.input_size*self.featureCount)
        # self.linearLayer = nn.Linear(2*self.input_size*self.featureCount, self.featureCount)
        # self.outputLinearLayer = nn.Linear(2*self.featureCount, 1)
        # self.ReLU = nn.ReLU()
        
        # LINEAR
        self.inputLayer = nn.Linear(self.input_size, self.input_size*self.featureCount)
        self.linearLayer1 = nn.Linear(self.input_size*self.featureCount, 10*self.input_size*self.featureCount)
        self.linearLayer2 = nn.Linear(10*self.input_size*self.featureCount, self.input_size*self.featureCount)
        self.outputLinearLayer = nn.Linear(self.input_size*self.featureCount, 1)
        self.PReLU = nn.PReLU()

    # Define a forward pass of the network
    def forward(self, oldObservation, observation, oldAction, action):
        # LSTM
        # inputTensor = getInputTensor(oldObservation, observation, oldAction, action)
        # LSTMOutput, hiddenStates = self.LSTM(inputTensor)
        # LSTMOutput = LSTMOutput[1]
        # LSTMReLU = self.ReLU(LSTMOutput)
        # CReLU = torch.cat((LSTMReLU, -LSTMReLU), 0).flatten().unsqueeze(0)
        # linear = self.linearLayer(CReLU)
        # linearReLU = self.ReLU(linear)
        # CReLU2 = torch.cat((linearReLU, -linearReLU), 0).flatten().unsqueeze(0)
        # output = self.outputLinearLayer(CReLU2)
        # return output[0]

        # LINEAR
        actionTensor = torch.tensor(action).float().requires_grad_(True)
        observationTensor = torch.from_numpy(observation).float().requires_grad_(True)
        inputTensor = torch.cat((observationTensor, actionTensor),0)
        inputTensor = inputTensor.view(1, -1)
        input = self.PReLU(self.inputLayer(inputTensor))
        linear1 = self.PReLU(self.linearLayer1(input))
        linear2 = self.PReLU(self.linearLayer2(linear1))
        output = self.outputLinearLayer(linear2)
        return output[0]


    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.ATTACKER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        estimates = [self.forward(game.previousAttackerObservation, observation, game.previousAttackerAction, action) for action in actions]
        index = np.argmax(estimates)
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

class AttackerParameterizedSoftmax():
    def __init__(self, targetNum):
        # Initalize random weights for parameterized softmx --
        # For the attacker, we are parameterizing rewards and penalties at
        # remaining targets.
        self.targetNum = targetNum
        self.rewardsWeight = np.random.uniform(0,1)
        self.penaltiesWeight = np.random.uniform(0,1)

    def getActionDistribution(self, game, actions, observation):
        actionValues = [self.forward(game.previousAttackerObservation, observation, game.previousAttackerAction, action) for action in actions]
        estimateSum = sum(actionValues)
        distribution = [actionValue/estimateSum for actionValue in actionValues]
        return distribution

    def forward(self, previousObservation, observation, previousAction, action):
        rewards = observation[self.targetNum*3:self.targetNum*4]
        penalties = observation[self.targetNum*4:]
        # Combine the weights
        weightedRewards = -sum(rewards * action * self.rewardsWeight)
        weightedPenalties = sum(penalties * action * self.penaltiesWeight)
        return exp(weightedRewards + weightedPenalties)

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.ATTACKER)
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
def attackerTrain(oracleToTrain, dIds, dMap, dMix, game, aPool, N=100, batchSize=15, C=100, epochs=100, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False, trainingTest=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    gameClone = ssg.cloneGame(game)

    if trainingTest:
        history = []
        lossHistory = []
        equilibriumHistory = []
        equilibriumScore = 0 #getBaselineScore(ssg.ATTACKER, dIds, dMap, dMix, gameClone, aPool)

    # Initialize the replay memory with limited capacity N
    replayMemory = ReplayMemory(N)
    # Initialize target network with weights equal to the oracle to train
    targetNetwork = AttackerOracle(oracleToTrain.targetNum)
    targetNetwork.setState(oracleToTrain.getState())

    # An epoch is one iteration over all training data. In our case, that's the one
    # Game we're learning on.
    step = 0
    for epoch in range(0, epochs):
        print(f"epoch {epoch} of {epochs}")
        # initialize the starting values for the game
        dOb, aOb = game.getEmptyObservations()
        defenderAgent = dMap[np.random.choice(dIds, 1, p=dMix)[0]]

        for timestep in range(game.timesteps):                                  # Play a full game
            # Choose an action based off of Q network (oracle to train)
            dAction = defenderAgent.getAction(game, dOb)
            aAction = oracleToTrain.getAction(game, aOb)

            # Execute that action and store the result in replay memory
            ob0 = game.previousAttackerObservation
            action0 = game.previousAttackerAction
            ob1 = aOb
            action1 = aAction
            dOb, aOb, dScore, aScore = game.performActions(dAction, aAction, dOb, aOb)
            replayMemory.push(ob0, action0, ob1, action1, aScore, dOb, game.getValidActions(ssg.ATTACKER))

            # Sample a random minibatch of transitions from replay memory
            avgLoss = sampleMinibatch(replayMemory, game, targetNetwork, oracleToTrain, lossFunction, optimizer, timestep, batchSize=batchSize)

            if trainingTest:
                oracleScore = ssg.expectedPureVMix(ssg.ATTACKER, oracleToTrain, dMap, dMix, gameClone)
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
    return ssg.expectedPureVMix(ssg.ATTACKER, oracleToTrain, dMap, dMix, gameClone)
