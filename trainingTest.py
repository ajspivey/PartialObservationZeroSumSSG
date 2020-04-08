# =======
# IMPORTS
# =======
# External
import numpy as np
import csv
import matplotlib.pyplot as plt
# Internal
import ssg
import attackerOracle as aO
from attackerOracle import AttackerOracle, AttackerEquilibrium
import defenderOracle as dO
from defenderOracle import DefenderOracle, DefenderEquilibrium
import coreLP
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
from random import Random
# +++++++

Transition = namedtuple('Transition', ('ob0', 'action0', 'ob1', 'action1', 'reward', 'ob2', 'newStateActions'))

np.random.seed(1)

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

# ====
# MAIN
# ====
def main():
    # ===============
    # HyperParameters
    # ===============
    seedingIterations = 3
    targetNum = 4
    resources = 2
    timesteps = 2
    # +++++++++++++++


    # ==========================================================================
    # CREATE GAME
    # ==========================================================================
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    # Used to do consistent testing and comparisons
    game.defenderRewards = [21.43407823, 36.29590018,  1.00560437, 15.81429606]
    game.defenderPenalties = [10.12675036, 17.93247563, 0.44160624, 7.40201997]
    print(f"Defender Rewards: {defenderRewards}\n Defender penalties: {defenderPenalties}")
    payoutMatrix = {}
    attackerMixedStrategy = None
    defenderMixedStrategy = None
    newDefenderId = 0
    newAttackerId = 0
    attackerPureIds = []
    defenderPureIds = []
    attackerIdMap = {}
    defenderIdMap = {}
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==========================================================================
    # GENERATE INITIAL PURE STRATEGIES
    # ==========================================================================
    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
    for _ in range(seedingIterations):
        attackerOracle = aO.AttackerOracle(targetNum)
        defenderOracle = dO.DefenderOracle(targetNum)
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = attackerOracle
        defenderIdMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
    print("Strategies seeded.")

    # Compute the payout matrix for each pair of strategies
    for attackerId in attackerPureIds:
        pureAttacker = attackerIdMap[attackerId]
        for defenderId in defenderPureIds:
            pureDefender = defenderIdMap[defenderId]
            value = game.getPayout(pureDefender, pureAttacker).item()
            payoutMatrix[defenderId,attackerId] = value
            game.restartGame()
    print("Payout matrix computed.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==========================================================================
    # ALGORITHM ITERATIONS
    # ==========================================================================
    # ----------------------------------------------------------------------
    # CORELP
    # ----------------------------------------------------------------------
    # Compute the mixed defender strategy
    defenderModel, dStrategyDistribution, dUtility = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    defenderModel.solve()
    defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
    dUtility = float(dUtility)
    print("Defender mixed strategy computed.")
    # Compute the mixed attacker strategy
    attackerModel, aStrategyDistribution, aUtility = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    attackerModel.solve()
    attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
    aUtility = float(aUtility)
    print("Attacker mixed strategy computed.")
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ORACLES
    # ----------------------------------------------------------------------
    # --------
    # DEFENDER
    # --------
    # Find the best oracle we currently have (to base training off of)
    bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, attackerPureIds, attackerIdMap, attackerMixedStrategy, defenderIdMap.values())
    print(f"Best D Oracle: {bestDOracle}, bestUtility: {bestDOracleUtility}")
    parameters = bestDOracle.getState()
    newDOracle = dO.DefenderOracle(targetNum)
    newDOracle.setState(parameters)
    defenderTrain(newDOracle, attackerPureIds, attackerIdMap, attackerMixedStrategy, game)
    newDOracleScore = game.getOracleScore(ssg.DEFENDER, ids=attackerPureIds, map=attackerIdMap, mix=attackerMixedStrategy, oracle=newDOracle)
    print(f"New D Oracle Utility Computed: {newDOracleScore}")

    # --------

    # --------
    # ATTACKER
    # --------
    # Find the best oracle we currently have (to base training off of)
    # bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values())
    # print(f"Best A Oracle: {bestAOracle}, bestUtility: {bestAOracleUtility}")
    # # Train a new oracle
    # parameters = bestAOracle.getState()
    # newAOracle = aO.AttackerOracle(targetNum)
    # newAOracle.setState(parameters)
    # attackerTrain(newAOracle, defenderPureIds, defenderIdMap, defenderMixedStrategy, game)
    # newAOracleScore = game.getOracleScore(ssg.ATTACKER, ids=defenderPureIds, map=defenderIdMap, mix=defenderMixedStrategy, oracle=newAOracle)
    # print(f"New A Oracle Utility Computed: {newAOracleScore}")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def defenderTrain(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, N=100, batchSize=15, C=50, epochs=100, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = []
    lossHistory = []
    equilibriumHistory = []

    gameClone = ssg.SequentialZeroSumSSG(game.numTargets, game.numResources, game.defenderRewards, game.defenderPenalties, game.timesteps)

    equilibriumDefender = DefenderEquilibrium(oracleToTrain.targetNum)
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
                    guess = oracleToTrain.forward(sample.ob0, sample.ob1, sample.action0, sample.action1)
                    loss = lossFunction(guess, y)
                    # print(f"sample: {sample}")
                    # print(f"guess: {guess}")
                    # print(f"label: {y}")
                    # print(f"loss: {loss}")
                    # print()
                    avgLoss += loss.item()
                    loss.backward()
                # print("OPTIMIZER STEP\n\n")
                optimizer.step()
            oracleScore = gameClone.getOracleScore(ssg.DEFENDER, aIds, aMap, attackerMixedStrategy, oracleToTrain)
            equilibriumScore = gameClone.getOracleScore(ssg.DEFENDER, aIds, aMap, attackerMixedStrategy, equilibriumDefender)
            history.append(oracleScore)
            lossHistory.append(avgLoss/batchSize)
            equilibriumHistory.append(equilibriumScore)
            # Every C steps, set Q^ = Q
            step += 1
            if step == C:
                targetNetwork.setState(oracleToTrain.getState())
                step = 0

        game.restartGame()

    oracleScore = game.getOracleScore(ssg.DEFENDER, aIds, aMap, attackerMixedStrategy, oracleToTrain)
    print(f"ORACLE SCORE: {oracleScore}")
    fig1 = plt.figure(1)
    plt.plot(range(epochs * game.timesteps), history, 'g', label='Oracle Utility')
    plt.plot(range(epochs * game.timesteps), equilibriumHistory, 'r', label='Equilibrium Baseline Utility')
    plt.title('Utility')
    plt.xlabel('Minibatches Trained')
    plt.ylabel('Utility')
    plt.legend()

    fig2 = plt.figure(2)
    plt.plot(range(epochs * game.timesteps), lossHistory, 'r', label='Oracle Loss')
    plt.title('Oracle Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def attackerTrain(oracleToTrain, dIds, dMap, defenderMixedStrategy, game, N=100, batchSize=15, C=20, epochs=30, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters(), lr=0.001)
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = []

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
            dOb, aOb, reward = game.performActions(dAction, aAction, dOb, aOb)

            replayMemory.push(ob0, action0, ob1, action1, reward, dOb, game.getValidActions(ssg.ATTACKER))

            # Sample a random minibatch of transitions from replay memory
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
                    guess = oracleToTrain.forward(sample.ob0, sample.ob1, sample.action0, sample.action1)
                    loss = lossFunction(guess, y)
                    loss.backward()
                optimizer.step()
            oracleScore = game.getOracleScore(ssg.ATTACKER, dIds, dMap, defenderMixedStrategy, oracleToTrain)
            history.append(oracleScore)
            # Every C steps, set Q^ = Q
            step += 1
            if step == C:
                targetNetwork.setState(oracleToTrain.getState())
                step = 0

        game.restartGame()

    plt.plot(range(epochs * game.timesteps), history, 'g', label='Oracle Score')
    plt.title('Oracle Score')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
