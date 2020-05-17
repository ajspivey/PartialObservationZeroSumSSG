# =======
# IMPORTS
# =======
# External
import numpy as np
import matplotlib.pyplot as plt
# Internal
from actorCritic import Transition, ReplayMemory, getInputTensor
from attackerOracle import AttackerOracle, attackerTrain
from defenderOracle import DefenderOracle, defenderTrain
# +++++++

def getTrainingGraph(player, game, ids, map, mix, pool, epochs=50):
    if player == ssg.DEFENDER:
        bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, ids, map, mix, pool)
        parameters = bestDOracle.getState()
        newDOracle = dO.DefenderOracle(game.numTargets)
        newDOracle.setState(parameters)
        return defenderTrain(newDOracle, ids, map, mix, game, pool, epochs=epochs)
        # return createBaseline(dO.DefenderOracle(game.numTargets), ids, map, mix, game, pool)
    else:
        bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, ids, map, mix, pool)
        parameters = bestAOracle.getState()
        newAOracle = aO.AttackerOracle(game.numTargets)
        newAOracle.setState(parameters)
        return attackerTrain(newAOracle, ids, map, mix, game, pool, epochs=epochs)

def showGraphs(graphs):
    i = 1
    for graph in graphs:
        player, graphSize, history, lossHistory, baselineHistory = graph
        playerName = "Defender"
        if player == ssg.ATTACKER:
            playerName = "Attacker"
        # Plot the stuff
        baselineGraph = plt.figure(i)
        plt.plot(range(graphSize), history, 'g', label=f'{playerName} Oracle Utility')
        plt.plot(range(graphSize), baselineHistory, 'r', label='Myopic Baseline Utility')
        plt.title(f'{playerName} Oracle Utility vs. Myopic Baseline')
        plt.xlabel('Minibatches Trained')
        plt.ylabel('Utility')
        plt.legend()

        lossGraph = plt.figure(i + 1)
        plt.plot(range(graphSize), lossHistory, 'r', label=f'{playerName} Oracle Loss')
        plt.title(f'{playerName} Oracle Loss')
        plt.xlabel('Minibatches Trained')
        plt.ylabel('Loss')

        i += 2
    plt.show()

# ====
# MAIN
# ====
def main():
    # ---------------
    # HyperParameters
    # ---------------
    seedingIterations = 5
    targetNum = 6
    resources = 2
    timesteps = 5
    timesteps2 = 2
    dEpochs = 50
    aEpochs = 50
    # ---------------
    # CREATE GAME
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    payoutMatrix = {}
    attackerMixedStrategy = None
    defenderMixedStrategy = None
    newDefenderId = 0
    newAttackerId = 0
    attackerPureIds = []
    defenderPureIds = []
    attackerIdMap = {}
    defenderIdMap = {}
    # Seeding
    for _ in range(seedingIterations):
        attackerOracle = aO.AttackerOracle(targetNum)
        defenderOracle = dO.DefenderOracle(targetNum)
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = attackerOracle
        defenderIdMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
    # payout matrix
    for attackerId in attackerPureIds:
        pureAttacker = attackerIdMap[attackerId]
        for defenderId in defenderPureIds:
            pureDefender = defenderIdMap[defenderId]
            value = game.getPayout(pureDefender, pureAttacker).item()
            payoutMatrix[defenderId,attackerId] = value
            game.restartGame()
    # coreLP
    defenderModel, dStrategyDistribution, dUtility = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    defenderModel.solve()
    defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
    dUtility = float(dUtility)
    attackerModel, aStrategyDistribution, aUtility = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    attackerModel.solve()
    attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
    aUtility = float(aUtility)
    # ----------------------------------------------------------------------
    # Get the defender and attacker graphs
    dHistory, dLossHistory, dBaselineHistory = getTrainingGraph(ssg.DEFENDER, game, attackerPureIds, attackerIdMap, attackerMixedStrategy, defenderIdMap.values(), epochs=dEpochs)
    aHistory, aLossHistory, aBaselineHistory = getTrainingGraph(ssg.ATTACKER, game, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values(), epochs=aEpochs)
    Get graphs for when the game only has 2 steps
    game.timesteps = timesteps2
    for attackerId in attackerPureIds:
        pureAttacker = attackerIdMap[attackerId]
        for defenderId in defenderPureIds:
            pureDefender = defenderIdMap[defenderId]
            value = game.getPayout(pureDefender, pureAttacker).item()
            payoutMatrix[defenderId,attackerId] = value
            game.restartGame()
    # coreLP
    defenderModel, dStrategyDistribution, dUtility = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    defenderModel.solve()
    defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
    dUtility = float(dUtility)
    attackerModel, aStrategyDistribution, aUtility = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
    attackerModel.solve()
    attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
    aUtility = float(aUtility)
    dHistory2, dLossHistory2, dBaselineHistory2 = getTrainingGraph(ssg.DEFENDER, game, attackerPureIds, attackerIdMap, attackerMixedStrategy, defenderIdMap.values(), epochs=dEpochs)
    aHistory2, aLossHistory2, aBaselineHistory2 = getTrainingGraph(ssg.ATTACKER, game, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values(), epochs=aEpochs)
    # Build the graphs
    graphs = [(ssg.DEFENDER, dEpochs*timesteps, dHistory, dLossHistory, dBaselineHistory), (ssg.ATTACKER, aEpochs*timesteps, aHistory, aLossHistory, aBaselineHistory), (ssg.DEFENDER, dEpochs*timesteps2, dHistory2, dLossHistory2, dBaselineHistory2), (ssg.ATTACKER, aEpochs*timesteps2, aHistory2, aLossHistory2, aBaselineHistory2)]
    # Show the graphs
    showGraphs(graphs)

if __name__ == "__main__":
    main()
