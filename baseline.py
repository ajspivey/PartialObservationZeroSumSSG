# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
import torch
import torch.nn as nn
import numpy as np
# Internal imports
import ssg

def getBaselineActionScores(observation, ids, map, mix, game, pool):
    """
    Returns the one step lookahead score of an action for the defender, given
    an attacker mixed strategy and an observation
    """
    pass

def createBaseline(oracleToTrain, aIds, aMap, attackerMixedStrategy, game, dPool, N=100, batchSize=15, C=50, epochs=50, optimizer=None, lossFunction=nn.MSELoss(), showOutput=False, trainingTest=True):
    """
    Trains a neural network to evaluate actions as the baseline would
    """
    pass

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
                    # For the defender, check that the same target is attacked.
                    for eId in ids:
                        agent = map[eId]
                        action = agent.getAction(game, eOb)
                        if action != eAction:
                            equilibriumDistribution[eId] = 0
                else:
                    # For the attacker, check that the attacked target is defended if the original
                    # action defends, or left undefended if the original action leaves it undefended
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
