# =======
# IMPORTS
# =======
# External
import numpy as np
import matplotlib.pyplot as plt
# Internal
import ssg
from coreLP import getAttackerMixedStrategy, getDefenderMixedStrategy
from attackerOracle import AttackerOracle, attackerTrain
from defenderOracle import DefenderOracle, defenderTrain
from experiment import calculatePayoutMatrix, seedInitialPureStrategies

def getTrainingGraph(player, game, ids, map, mix, pool, batchSize=15, epochs=50):
    if player == ssg.DEFENDER:
        bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, ids, map, mix, pool)
        parameters = bestDOracle.getState()
        newDOracle = DefenderOracle(game.numTargets)
        newDOracle.setState(parameters)
        return defenderTrain(newDOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True)
    else:
        bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, ids, map, mix, pool)
        parameters = bestAOracle.getState()
        newAOracle = AttackerOracle(game.numTargets)
        newAOracle.setState(parameters)
        return attackerTrain(newAOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True)

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
    # =========
    # DEBUGGING
    # =========
    export = False
    # +++++++++
    # ---------------
    # HyperParameters
    # ---------------
    seedingIterations = 10
    targetNum = 4
    resources = 2
    timesteps = 2
    timesteps2 = 2
    dEpochs = 3
    aEpochs = 3
    # ---------------
    # CREATE GAME
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    game.defenderRewards = [10, 10, 10, 10]
    game.defenderPenalties =[10000, 10, 30, 10]
    newDefenderId, newAttackerId, dIds, aIds, dMap, aMap = seedInitialPureStrategies(seedingIterations, targetNum)
    payoutMatrix = calculatePayoutMatrix(dIds, aIds, dMap, aMap, game)
    # coreLP
    dMix, dMixUtility = getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
    aMix, aMixUtility = getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
    # ----------------------------------------------------------------------
    # Get the defender and attacker graphs
    dHistory, dLossHistory, dBaselineHistory = getTrainingGraph(ssg.DEFENDER, game, aIds, aMap, aMix, dMap.values(), batchSize=50, epochs=dEpochs)
    aHistory, aLossHistory, aBaselineHistory = getTrainingGraph(ssg.ATTACKER, game, dIds, dMap, dMix, aMap.values(), batchSize=50, epochs=aEpochs)
    # Get graphs for when the game only has 2 steps
    # game.timesteps = timesteps2
    # newDefenderId, newAttackerId, dIds, aIds, dMap, aMap = seedInitialPureStrategies(seedingIterations, targetNum)
    # payoutMatrix = calculatePayoutMatrix(dIds, aIds, dMap, aMap, game)
    # # coreLP
    # dMix, dMixUtility = getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
    # aMix, aMixUtility = getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
    # dHistory2, dLossHistory2, dBaselineHistory2 = getTrainingGraph(ssg.DEFENDER, game, attackerPureIds, attackerIdMap, attackerMixedStrategy, defenderIdMap.values(), epochs=dEpochs)
    # aHistory2, aLossHistory2, aBaselineHistory2 = getTrainingGraph(ssg.ATTACKER, game, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values(), epochs=aEpochs)
    # Build the graphs
    graphs = [(ssg.DEFENDER, dEpochs*timesteps, dHistory, dLossHistory, dBaselineHistory), (ssg.ATTACKER, aEpochs*timesteps, aHistory, aLossHistory, aBaselineHistory)]
    # , (ssg.DEFENDER, dEpochs*timesteps2, dHistory2, dLossHistory2, dBaselineHistory2), (ssg.ATTACKER, aEpochs*timesteps2, aHistory2, aLossHistory2, aBaselineHistory2)]
    # Show the graphs
    showGraphs(graphs)

if __name__ == "__main__":
    main()
