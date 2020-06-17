# =======
# IMPORTS
# =======
# External
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
# Internal
import ssg
from coreLP import getAttackerMixedStrategy, getDefenderMixedStrategy
from attackerOracle import AttackerOracle, attackerTrain
from defenderOracle import DefenderOracle, defenderTrain
from experiment import calculatePayoutMatrix, seedInitialPureStrategies

# =========
# FUNCTIONS
# =========
def getTrainingGraph(player, game, ids, map, mix, pool, batchSize=15, epochs=50, writer=None):
    if player == ssg.DEFENDER:
        bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, ids, map, mix, pool)
        parameters = bestDOracle.getState()
        newDOracle = DefenderOracle(game.numTargets)
        newDOracle.setState(parameters)
        linear, linearLoss = defenderTrain(newDOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True, writer=writer)
        writer.writerow([])
        writer.writerow(["LSTM"])
        newDOracle.setState(parameters)
        newDOracle.type = "LSTM"
        lstm, lstmLoss = defenderTrain(newDOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True, writer=writer)
        return linear, linearLoss, lstm, lstmLoss
    else:
        bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, ids, map, mix, pool)
        parameters = bestAOracle.getState()
        newAOracle = AttackerOracle(game.numTargets)
        newAOracle.setState(parameters)
        linear, linearLoss = attackerTrain(newAOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True, writer=writer)
        writer.writerow([])
        writer.writerow(["LSTM"])
        newAOracle.setState(parameters)
        newAOracle.type = "LSTM"
        lstm, lstmLoss = attackerTrain(newAOracle, ids, map, mix, game, pool, epochs=epochs, batchSize=batchSize, trainingTest=True, writer=writer)
        return linear, linearLoss, lstm, lstmLoss

def showGraphs(graphs):
    i = 1
    for graph in graphs:
        player, graphSize, history, lossHistory = graph
        playerName = "Defender"
        if player == ssg.ATTACKER:
            playerName = "Attacker"
        # Plot the stuff
        baselineGraph = plt.figure(i)
        plt.plot(range(graphSize), history, 'g', label=f'{playerName} Oracle Utility')
        plt.title(f'{playerName} Oracle Utility')
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

def displayTimes(times):
    START = 0
    PURE = 1
    PAYOUT = 2
    MIX = 3
    TRAINING = 4
    END = 5
    print(f"")
    print(f"Pure strategy generation time: {times[PURE] - times[START]}")
    print(f"Payout matrix generation time: {times[PAYOUT] - times[PURE]}")
    print(f"Mixed strategy generation time: {times[MIX] - times[PAYOUT]}")
    print(f"Training and baseline time: {times[TRAINING] - times[MIX]}")
    print(f"Total execution time: {times[END]-times[START]}")

# ====
# MAIN
# ====
def main():
    times = []
    times.append(time.time())
    # =========
    # DEBUGGING
    exportMixedStrategies = False
    # ---------------
    # HyperParameters
    dEpochs = 100
    aEpochs = 100
    seedingIterations = 20
    targetNum = 4
    resources = 2
    timesteps = 3
    # ---------------
    # CREATE GAME
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    # ---------------
    # CSV stuffs
    csvFileAttacker = open("outputFiles/attackerTraining.csv", "w", newline='')
    csvWriterAttacker = csv.writer(csvFileAttacker, delimiter=',')
    csvWriterAttacker.writerow(["defender rewards", "defender penalties"])
    csvWriterAttacker.writerow([f"{game.defenderRewards}", f"{game.defenderPenalties}"])
    csvWriterAttacker.writerow(["iteration", "action list", "network estimate list", "action chosen", "opponent action"])
    csvFileDefender = open("outputFiles/defenderTraining.csv", "w", newline='')
    csvWriterDefender = csv.writer(csvFileDefender, delimiter=',')
    csvWriterDefender.writerow(["defender rewards", "defender penalties"])
    csvWriterDefender.writerow([f"{game.defenderRewards}", f"{game.defenderPenalties}"])
    csvWriterDefender.writerow(["iteration", "action list", "network estimate list", "action chosen", "opponent action"])
    # ---------------
    print("Seeding Initial Strategies and Calculating Payout Matrix...")
    newDefenderId, newAttackerId, dIds, aIds, dMap, aMap = seedInitialPureStrategies(seedingIterations, targetNum)
    times.append(time.time())
    payoutMatrix = calculatePayoutMatrix(dIds, aIds, dMap, aMap, game)
    times.append(time.time())
    # coreLP
    print("Generating Mixed Strategies...")
    dMix, dMixUtility = getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, exportMixedStrategies)
    aMix, aMixUtility = getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, exportMixedStrategies)
    times.append(time.time())
    # ----------------------------------------------------------------------
    # Get the defender and attacker graphs
    print("Generating Training Graphs...")
    dHistoryLinear, dLossHistoryLinear, dHistoryLSTM, dLossHistoryLSTM  = getTrainingGraph(ssg.DEFENDER, game, aIds, aMap, aMix, dMap.values(), batchSize=100, epochs=dEpochs, writer=csvWriterDefender)
    aHistoryLinear, aLossHistoryLinear, aHistoryLSTM, aLossHistoryLSTM  = getTrainingGraph(ssg.ATTACKER, game, dIds, dMap, dMix, aMap.values(), batchSize=50, epochs=aEpochs, writer=csvWriterAttacker)
    times.append(time.time())
    # Build the graphs
    print("Building and displaying Graphs...")
    graphs = [(ssg.DEFENDER, dEpochs*timesteps, dHistoryLinear, dLossHistoryLinear), (ssg.DEFENDER, dEpochs*timesteps, dHistoryLSTM, dLossHistoryLSTM), (ssg.ATTACKER, aEpochs*timesteps, aHistoryLinear, aLossHistoryLinear), (ssg.ATTACKER, aEpochs*timesteps, aHistoryLSTM, aLossHistoryLSTM)]
    # Show the graphs
    times.append(time.time())
    csvFileDefender.close()
    csvFileAttacker.close()
    displayTimes(times)
    showGraphs(graphs)

if __name__ == "__main__":
    main()
