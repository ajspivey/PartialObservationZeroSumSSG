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
from actorCritic import getInputTensor, Transition, ReplayMemory

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

        self.linearLayer = nn.Linear(self.input_size, self.input_size*self.featureCount)
        self.ReLU = nn.ReLU()
        self.LSTM = nn.LSTM(2*self.input_size*self.featureCount, self.input_size*self.featureCount)
        self.outputLinearLayer = nn.Linear(2*self.input_size*self.featureCount, 1)

    # Define a forward pass of the network
    def forward(self, oldObservation, observation, oldAction, action):
        inputTensor = getInputTensor(oldObservation, observation, oldAction, action)
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

    def getAction(self, game, observation):
        actions = game.getValidActions(ssg.ATTACKER)
        return self.getActionFromActions(game, actions, observation)

    def getActionFromActions(self, game, actions, observation):
        estimates = [self.forward(game.previousAttackerObservation, observation, game.previousDefenderAction, action) for action in actions]
        index = np.argmax(estimates)
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
def attackerTrain(oracleToTrain, dIds, dMap, defenderMixedStrategy, game, aPool, N=100, batchSize=15, C=100, epochs=50, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False, trainingTest=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = []
    lossHistory = []
    equilibriumHistory = []

    gameClone = ssg.SequentialZeroSumSSG(game.numTargets, game.numResources, game.defenderRewards, game.defenderPenalties, game.timesteps)
    equilibriumScore = getBaselineScore(ssg.ATTACKER, dIds, dMap, defenderMixedStrategy, gameClone, aPool)

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
        defenderAgent = dMap[np.random.choice(dIds, 1, p=defenderMixedStrategy)[0]]

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
            if trainingTest:
                oracleScore = game.getOracleScore(ssg.ATTACKER, dIds, dMap, defenderMixedStrategy, oracleToTrain)
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
    return game.getOracleScore(ssg.ATTACKER, dIds, dMap, defenderMixedStrategy, oracleToTrain)
