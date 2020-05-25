# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
import torch
import torch.nn as nn
import numpy as np
# Internal imports
import ssg

def getBaselineScore(player, ids, map, mix, game, pool):
    """
    Calculates the one-step-lookahead baseline utility. Each timestep, the opponent's
    mixed strategy is filtered by which pure strategies could have resulted in the
    observation seen. A best response is calculated against these pure strategies
    according to their normalized probability.
    """
    if player == ssg.DEFENDER:
        return getDefenderBaseline(ids, map, mix, game, pool)
    else:
        return getAttackerBaseline(ids, map, mix, game, pool)

def getDefenderBaseline(aIds, aMap, aMix, game, dPool):
    expectedUtility = 0
    dOb, aOb = game.getEmptyObservations()
    undoClone = ssg.cloneGame(game)
    clone = ssg.cloneGame(game)
    editableAMix = aMix.copy()
    savedAction = None
    # Calculate the expected utility of attacker myopic play
    for agentIndex in range(len(aMap)):
        if aMix[agentIndex] > 0:
            aAgent = aMap[agentIndex]
            for timestep in range(game.timesteps):
                if timestep == 0:
                    # Calculate the best response against the entire mixed strategy and save it
                    savedAction = getBestResponseAction(ssg.DEFENDER, game, aMap, editableAMix, dPool, dOb, aOb)
                    # play the defender action and best response attacker action to obtain
                    # a set of observations
                    attackerAction = aAgent.getAction(game, aOb)
                    dOb, aOb, _, _ = game.performActions(savedAction, attackerAction, dOb, aOb)
                else:
                    # For each agent:
                    for i in range(len(editableAMix)):
                        if editableAMix[i] > 0:
                            #   Play the agent's action on clone with the saved action
                            attackerAction = aMap[i].getAction(clone, game.previousAttackerObservation)
                            dTestOb, _, _, _ = clone.performActions(savedAction, attackerAction, game.previousDefenderObservation, game.previousAttackerObservation)
                            #   Compare the resulting attacker observation. If they don't match set its odds to 0 in the editable mix
                            if not np.array_equal(dTestOb,dOb):
                                editableAMix[i] = 0
                            #   set clone to undoClone
                            ssg.cloneGameState(clone, undoClone)
                    # Using the filtered mix, compute the best response and save it
                    editableAMix = [float(p)/sum(editableAMix) for p in editableAMix]
                    savedAction = getBestResponseAction(ssg.DEFENDER, game, aMap, editableAMix, dPool, dOb, aOb)
                    # Set clone and undo clone to the current game
                    ssg.cloneGameState(clone, game)
                    ssg.cloneGameState(undoClone, game)
                    # perform the best response and agent action on the normal game
                    attackerAction = aAgent.getAction(game, aOb)
                    dOb, aOb, _, _ = game.performActions(savedAction, attackerAction, dOb, aOb)
            print(game.defenderUtility)
            expectedUtility += game.defenderUtility * aMix[agentIndex]
            editableAMix = aMix.copy()
            game.restartGame()
            ssg.cloneGameState(clone, game)
            ssg.cloneGameState(undoClone, game)
    return expectedUtility

def getAttackerBaseline(dIds, dMap, dMix, game, aPool):
    expectedUtility = 0
    dOb, aOb = game.getEmptyObservations()
    undoClone = ssg.cloneGame(game)
    clone = ssg.cloneGame(game)
    editableDMix = dMix.copy()
    savedAction = None
    # Calculate the expected utility of attacker myopic play
    for agentIndex in range(len(dMap)):
        if dMix[agentIndex] > 0:
            dAgent = dMap[agentIndex]
            for timestep in range(game.timesteps):
                if timestep == 0:
                    # Calculate the best response against the entire mixed strategy and save it
                    savedAction = getBestResponseAction(ssg.ATTACKER, game, dMap, editableDMix, aPool, aOb, dOb)
                    # play the defender action and best response attacker action to obtain
                    # a set of observations
                    dOb, aOb, _, _ = game.performActions(dAgent.getAction(game, dOb), savedAction, dOb, aOb)
                else:
                    # For each agent:
                    for i in range(len(editableDMix)):
                        if editableDMix[i] > 0:
                            #   Play the agent's action on clone with the attacker saved action
                            _, aTestOb, _, _ = clone.performActions(dMap[i].getAction(clone, game.previousDefenderObservation), savedAction, game.previousDefenderObservation, game.previousAttackerObservation)
                            #   Compare the resulting attacker observation. If they don't match set its odds to 0 in the editable mix
                            if not np.array_equal(aTestOb,aOb):
                                editableDMix[i] = 0
                            #   set clone to undoClone
                            ssg.cloneGameState(clone, undoClone)
                    # Using the filtered mix, compute the best response and save it
                    editableDMix = [float(p)/sum(editableDMix) for p in editableDMix]
                    savedAction = getBestResponseAction(ssg.ATTACKER, game, dMap, editableDMix, aPool, aOb, dOb)
                    # Set clone and undo clone to the current game
                    ssg.cloneGameState(clone, game)
                    ssg.cloneGameState(undoClone, game)
                    # perform the best response and agent action on the normal game
                    defenderAction = dAgent.getAction(game, dOb)
                    dOb, aOb, _, _ = game.performActions(defenderAction, savedAction, dOb, aOb)
            expectedUtility += game.defenderUtility * -1 * dMix[agentIndex]
            editableDMix = dMix.copy()
            game.restartGame()
            ssg.cloneGameState(clone, game)
            ssg.cloneGameState(undoClone, game)
    return expectedUtility

def getBestResponseAction(player, game, map, mix, pool, pOb, eOb):
    pActions = [playerAgent.getAction(game, pOb) for playerAgent in pool]
    eActions = [eAgent.getAction(game, eOb) for eAgent in map.values()]
    actionScores = []
    for pAction in pActions:
        actionScore = 0
        for i in range(len(eActions)):
            actionScore += game.getActionScore(pAction, eActions[i], game.defenderRewards, game.defenderPenalties)[player] * mix[i]
        actionScores.append(actionScore)
    return pActions[np.argmax(actionScores)]
