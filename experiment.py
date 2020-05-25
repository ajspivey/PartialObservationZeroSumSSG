# =======
# IMPORTS
# =======
# External
import numpy as np
import csv
# Internal
import ssg
from coreLP import getAttackerMixedStrategy, getDefenderMixedStrategy
from attackerOracle import AttackerOracle, attackerTrain
from defenderOracle import DefenderOracle, defenderTrain
# +++++++

np.random.seed(6)

# ====
# MAIN
# ====
def main():
    # =========
    # DEBUGGING
    # =========
    export = False
    # +++++++++
    # ===============
    # HyperParameters
    # ===============
    experimentIterations = 2
    seedingIterations = 3
    warmUpIterations = 50
    targetNum = 4
    resources = 2
    timesteps = 2
    # +++++++++++++++

    # ==========================================================================
    # CREATE GAME
    # ==========================================================================
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    # ==========================================================================
    # GENERATE INITIAL PURE STRATEGIES
    # ==========================================================================
    newDefenderId, newAttackerId, dIds, aIds, dMap, aMap = seedInitialPureStrategies(seedingIterations, targetNum)
    payoutMatrix = calculatePayoutMatrix(dIds, aIds, dMap, aMap, game)
    # ==========================================================================
    # WARM-UP ITERATIONS
    # ==========================================================================
    for _ in range(warmUpIterations):
        dMix, dMixUtility = getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
        aMix, aMixUtility = getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
        newDOracle = BaselineDefender(targetNum)
        newAOracle = BaselineAttacker(targetNum)
        newDefenderId, newAttackerId, payoutMatrix = updatePayoutMatrix(newDefenderId, newAttackerId, payoutMatrix, dIds, aIds, dMap, aMap, game, newDOracle, newAOracle)
    # ==========================================================================
    # ALGORITHM ITERATIONS
    # ==========================================================================
    for _ in range(experimentIterations):
        # ----------------------------------------------------------------------
        # CORELP
        # ----------------------------------------------------------------------
        dMix, dMixUtility = getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
        aMix, aMixUtility = getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export)
        # ----------------------------------------------------------------------
        # ORACLES
        # ----------------------------------------------------------------------
        # Defender
        bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, aIds, aMap, aMix, dMap.values())
        parameters = bestDOracle.getState()
        newDOracle = DefenderOracle(targetNum)
        newDOracle.setState(parameters)
        newDOracleScore = defenderTrain(oracleToTrain=newDOracle, aIds=aIds, aMap=aMap, aMix=aMix, game=game, dPool=dMap.values(), trainingTest=False)
        # Attacker
        bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, dIds, dMap, dMix, aMap.values())
        parameters = bestAOracle.getState()
        newAOracle = AttackerOracle(targetNum)
        newAOracle.setState(parameters)
        newAOracleScore = attackerTrain(oracleToTrain=newAOracle, game=game, dIds=dIds, dMap=dMap, dMix=dMix, aPool=aMap.values(), trainingTest=False)
        # ----------------------------------------------------------------------
        # UPDATE POOLS
        # ----------------------------------------------------------------------
        newDefenderId, newAttackerId, payoutMatrix = updatePayoutMatrix(newDefenderId, newAttackerId, payoutMatrix, dIds, aIds, dMap, aMap, game, newDOracle, newAOracle)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def seedInitialPureStrategies(seedingIterations, targetNum):
    newDefenderId = 0
    newAttackerId = 0
    aIds = []
    dIds = []
    aMap = {}
    dMap = {}
    for _ in range(seedingIterations):
        attackerOracle = AttackerOracle(targetNum)
        defenderOracle = DefenderOracle(targetNum)
        aIds.append(newAttackerId)
        dIds.append(newDefenderId)
        aMap[newAttackerId] = attackerOracle
        dMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
    return newDefenderId, newAttackerId, dIds, aIds, dMap, aMap

def calculatePayoutMatrix(dIds, aIds, dMap, aMap, game):
    payoutMatrix = {}
    for attackerId in aIds:
        pureAttacker = aMap[attackerId]
        for defenderId in dIds:
            pureDefender = dMap[defenderId]
            value = game.getPayout(pureDefender, pureAttacker).item()
            payoutMatrix[defenderId,attackerId] = value
            game.restartGame()
    return payoutMatrix

def updatePayoutMatrix(newDefenderId, newAttackerId, payoutMatrix, dIds, aIds, dMap, aMap, game, newDOracle, newAOracle):
    for aId in aIds:
        value = game.getPayout(newDOracle, aMap[aId])
        payoutMatrix[newDefenderId, aId] = value
    for dId in dIds:
        value = game.getPayout(dMap[dId], newAOracle)
        payoutMatrix[dId, newAttackerId] = value
    value = game.getPayout(newDOracle, newAOracle)
    payoutMatrix[newDefenderId, newAttackerId] = value
    aIds.append(newAttackerId)
    dIds.append(newDefenderId)
    aMap[newAttackerId] = newAOracle
    dMap[newDefenderId] = newDOracle
    newDefenderId += 1
    newAttackerId += 1
    return newDefenderId, newAttackerId, payoutMatrix


if __name__ == "__main__":
    main()



# csvFileThing = open("outputFiles/payoutMatrix.csv", "w", newline='')
# csvWriterThing = csv.writer(csvFileThing, delimiter=',')
# csvWriterThing.writerow(["Rewards",f"{defenderRewards}"])
# csvWriterThing.writerow(["Penalties",f"{defenderPenalties}"])
# csvWriterThing.writerow(["defenderID","attackerID","Value(Defender payoff)"])
# for key in payoutMatrix.keys():
#     defenderID, attackerID = key
#     csvWriterThing.writerow([defenderID, attackerID, payoutMatrix[key]])
# csvFileThing.close()
# if writeUtilityFile:
#     csvFile = open("outputFiles/utilities.csv", "w", newline='')
#     csvWriter = csv.writer(csvFile, delimiter=',')
#     csvWriter.writerow(["Defender Mixed Utility", "Expected Utility of Defender Oracle vs Attacker Mixed", "Expected Utility of Best Defender Pure Strategy", "Defender Mixed Strategy", "Attacker Mixed Utility", "Expected Utility of Attacker Oracle vs Defender Mixed", "Expected Utility of Best Attacker Pure Strategy", "Attacker Mixed Strategy"])
# if writeUtilityFile:
#     csvWriter.writerow([f"{dUtility}",f"{newDOracleScore}",f"{bestDOracleUtility}", f"{dMix}", f"{aUtility}", f"{newAOracleScore}",f"{bestAOracleUtility}", f"{aMix}"])
# if writeUtilityFile:
#     csvFile.close()
