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
from attackerOracle import AttackerOracle
import defenderOracle as dO
from defenderOracle import DefenderOracle
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
    seedingIterations = 5
    targetNum = 6
    resources = 2
    timesteps = 5
    # +++++++++++++++
    dEpochs = 50
    aEpochs = 50

    # ==========================================================================
    # CREATE GAME
    # ==========================================================================
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    # Used to do consistent testing and comparisons
    # defenderRewards = [21.43407823, 36.29590018,  1.00560437, 15.81429606]
    # defenderPenalties = [ 8.19103865,  5.52459114, 10.12675036, 17.93247563]

    print(defenderRewards)
    print(defenderPenalties)

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
    dHistory, dLossHistory, dMyopicHistory = defenderTrain(newDOracle, attackerPureIds, attackerIdMap, attackerMixedStrategy, game, defenderIdMap.values(), epochs=dEpochs)
    newDOracleScore = game.getOracleScore(ssg.DEFENDER, ids=attackerPureIds, map=attackerIdMap, mix=attackerMixedStrategy, oracle=newDOracle)
    print(f"New D Oracle Utility Computed: {newDOracleScore}")

    # --------

    # --------
    # ATTACKER
    # --------
    # Find the best oracle we currently have (to base training off of)
    bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values())
    print(f"Best A Oracle: {bestAOracle}, bestUtility: {bestAOracleUtility}")
    parameters = bestAOracle.getState()
    newAOracle = aO.AttackerOracle(targetNum)
    newAOracle.setState(parameters)
    aHistory, aLossHistory, aMyopicHistory = attackerTrain(newAOracle, defenderPureIds, defenderIdMap, defenderMixedStrategy, game, attackerIdMap.values(), epochs=aEpochs)
    newAOracleScore = game.getOracleScore(ssg.ATTACKER, ids=defenderPureIds, map=defenderIdMap, mix=defenderMixedStrategy, oracle=newAOracle)
    print(f"New A Oracle Utility Computed: {newAOracleScore}")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Plot the stuff
    print(f"DEFENDER ORACLE SCORE: {newDOracleScore}")
    print(f"DEFENDER BASELINE SCORE: {dMyopicHistory[0]}")
    fig1 = plt.figure(1)
    plt.plot(range(dEpochs * game.timesteps), dHistory, 'g', label='Defender Oracle Utility')
    plt.plot(range(dEpochs * game.timesteps), dMyopicHistory, 'r', label='Myopic Baseline Utility')
    plt.title('Defender Oracle Utility vs. Myopic Baseline')
    plt.xlabel('Minibatches Trained')
    plt.ylabel('Utility')
    plt.legend()

    fig2 = plt.figure(2)
    plt.plot(range(dEpochs * game.timesteps), dLossHistory, 'r', label='Defender Oracle Loss')
    plt.title('Defender Oracle Loss')
    plt.xlabel('Minibatches Trained')
    plt.ylabel('Loss')

    print(f"ATTACKER ORACLE SCORE: {newAOracleScore}")
    print(f"ATTACKER BASELINE SCORE: {aMyopicHistory[0]}")
    fig1 = plt.figure(3)
    plt.plot(range(aEpochs * game.timesteps), aHistory, 'g', label='Attacker Oracle Utility')
    plt.plot(range(aEpochs * game.timesteps), aMyopicHistory, 'r', label='Myopic Baseline Utility')
    plt.title('Attacker Oracle Utility vs. Myopic Baseline')
    plt.xlabel('Minibatches Trained')
    plt.ylabel('Utility')
    plt.legend()

    fig2 = plt.figure(4)
    plt.plot(range(aEpochs * game.timesteps), aLossHistory, 'r', label='Attacker Oracle Loss')
    plt.title('Attacker Oracle Loss')
    plt.xlabel('Minibatches Trained')
    plt.ylabel('Loss')
    plt.show()


def defenderTrain(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, dPool, N=100, batchSize=15, C=50, epochs=50, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
    if optimizer is None:
        optimizer = optim.Adam(oracleToTrain.parameters())
        optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = []
    lossHistory = []
    equilibriumHistory = []

    gameClone = ssg.SequentialZeroSumSSG(game.numTargets, game.numResources, game.defenderRewards, game.defenderPenalties, game.timesteps)
    equilibriumScore = getBaselineScore(ssg.DEFENDER, aIds, aMap, attackerMixedStrategy, gameClone, dPool)

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
            dOb, aOb, dScore, aScore = game.performActions(dAction, aAction, dOb, aOb)

            replayMemory.push(ob0, action0, ob1, action1, dScore, dOb, game.getValidActions(ssg.DEFENDER))

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
            oracleScore = gameClone.getOracleScore(ssg.DEFENDER, aIds, aMap, attackerMixedStrategy, oracleToTrain)
            history.append(oracleScore)
            lossHistory.append(avgLoss/batchSize)
            equilibriumHistory.append(equilibriumScore)
            # Every C steps, set Q^ = Q
            step += 1
            if step == C:
                targetNetwork.setState(oracleToTrain.getState())
                step = 0

        game.restartGame()

    return history, lossHistory, equilibriumHistory

def attackerTrain(oracleToTrain, dIds, dMap, defenderMixedStrategy, game, aPool, N=100, batchSize=15, C=100, epochs=50, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False):
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

    return history, lossHistory, equilibriumHistory

def getBaselineScore(player, ids, map, mix, game, pool):
    """
    Calculates the one-step-lookahead baseline utility. Each timestep, the opponent's
    mixed strategy is filtered by which pure strategies could have resulted in the
    observation seen. A best response is calculated against these pure strategies
    according to their normalized probability.
    """
    totalExpectedUtility = 0
    # Calculate expected utility for myopic play.
    for i in range(len(mix)):
        if mix[i] > 0:
            # For each enemy agent in the mixed strategy, calculate the expected myopic
            # play utility
            eAgent = map[i]
            equilibriumDistribution = mix.copy()
            distributionTotal = 1
            dOb, aOb = game.getEmptyObservations()
            # Play a game to get myopic utiltiy against this opponent
            for timestep in range(game.timesteps):
                if player == ssg.DEFENDER:
                    pOb = dOb
                    eOb = aOb
                else:
                    pOb = aOb
                    eOb = dOb
                eAction = eAgent.getAction(game, eOb)

                # we compute the best response against the
                # possible enemy agents, given the observation we saw -- the
                # distribution of these possible agents is normalized to calculate the response
                if (player == ssg.DEFENDER):
                    for eId in ids:
                        agent = map[eId]
                        action = agent.getAction(game, eOb)
                        if action != eAction:
                            equilibriumDistribution[eId] = 0
                else:
                    obAction = pOb[:game.numTargets]
                    target = eAction * obAction
                    for eId in ids:
                        agent = map[eId]
                        action = agent.getAction(game, eOb)
                        if not np.array_equal(action * obAction,target):
                            equilibriumDistribution[eId] = 0
                distributionTotal *= sum(equilibriumDistribution)
                equilibriumDistribution = [float(p)/sum(equilibriumDistribution) for p in equilibriumDistribution]

                # Given our new distribution, calculate the best response from the
                # expectations of the pure strategies available
                pActions = [playerAgent.getAction(game, dOb) for playerAgent in pool]
                if (player == ssg.DEFENDER):
                    actionScores = [game.getActionScore(pAction, eAction, game.defenderRewards, game.defenderPenalties)[0] * distributionTotal for pAction in pActions]
                else:
                    actionScores = [0] * len(pActions)
                    for i in range(len(equilibriumDistribution)):
                        if equilibriumDistribution[i] > 0:
                            actionScores = np.add(actionScores, [game.getActionScore(pAction, map[i].getAction(game, eOb), game.defenderRewards, game.defenderPenalties)[1] * equilibriumDistribution[i] for pAction in pActions])
                pAction = pActions[np.argmax(actionScores)]
                if player == ssg.DEFENDER:
                    dAction = pAction
                    aAction = eAction
                else:
                    dAction = eAction
                    aAction = pAction
                dOb, aOb, _, _ = game.performActions(dAction, aAction, dOb, aOb)
            totalExpectedUtility += game.defenderUtility * player * mix[i]
            game.restartGame()
    return totalExpectedUtility


if __name__ == "__main__":
    main()
