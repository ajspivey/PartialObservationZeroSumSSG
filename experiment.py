# ======
# @TODO:
# ======
# Generalize attacker/defender specific functions
# * -ssg-
# * defender Oracle
# * attacker oracle
#
# Remove unused functions
# Standardize function parameter orders
# Standardize attacker/defender orders (defender before attacker?)
# Add docstrings to all functions
# Recomment
# Clean up whitespace/make flow of functions clear
# Create functions in experiment file?
# ++++++


# =======
# IMPORTS
# =======
# External
import numpy as np
import csv
# Internal
import ssg
import attackerOracle as aO
import defenderOracle as dO
import coreLP
import torch
from torch import autograd

# +++++++

np.random.seed(1)

# ====
# MAIN
# ====
def main():
    # =========
    # DEBUGGING
    # =========
    autograd.set_detect_anomaly(True)
    showFrameworkOutput = True
    showOracleTraining = True
    showUtilities = False
    showStrategies = False
    writeUtilityFile = True
    # +++++++++

    # ===============
    # HyperParameters
    # ===============
    experimentIterations = 10
    seedingIterations = 3
    targetNum = 4
    resources = 2
    timesteps = 2
    # +++++++++++++++


    # ==========================================================================
    # CREATE GAME
    # ==========================================================================
    # Create a game with 4 targets, 2 resources, and 2 timesteps
    if showFrameworkOutput:
        print("Creating game...")
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)
    # Used to do consistent testing and comparisons
    game.defenderRewards = [21.43407823, 36.29590018,  1.00560437, 15.81429606]
    game.defenderPenalties = [10.12675036, 17.93247563, 0.44160624, 7.40201997]

    if showFrameworkOutput:
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
    if showFrameworkOutput:
        print("Game created.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==========================================================================
    # GENERATE INITIAL PURE STRATEGIES
    # ==========================================================================
    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
    if showFrameworkOutput:
        print("Seeding initial attacker and defender pure strategies")
    for _ in range(seedingIterations):
        attackerOracle = aO.AttackerOracle(targetNum)
        defenderOracle = dO.DefenderOracle(targetNum)
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = attackerOracle
        defenderIdMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
    if showFrameworkOutput:
        print("Strategies seeded.")

    # Compute the payout matrix for each pair of strategies
    if showFrameworkOutput:
        print("Computing initial payout matrix for pure strategies...")
    for attackerId in attackerPureIds:
        pureAttacker = attackerIdMap[attackerId]
        for defenderId in defenderPureIds:
            print("Getting payout")
            pureDefender = defenderIdMap[defenderId]
            value = game.getPayout(pureDefender, pureAttacker).item()
            payoutMatrix[defenderId,attackerId] = value
            game.restartGame()
    if showFrameworkOutput:
        print("Payout matrix computed.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==========================================================================
    # ALGORITHM ITERATIONS
    # ==========================================================================
    if showFrameworkOutput:
        print("Beginning iteration:\n")
    if writeUtilityFile:
        csvFile = open("outputFiles/utilities.csv", "w", newline='')
        csvWriter = csv.writer(csvFile, delimiter=',')
        csvWriter.writerow(["Defender Mixed Utility","Attacker Mixed Utility", "Avg. Defender Oracle Utility vs Mixed", "Avg Score of Best Defender Pure Strategy", "Avg. Attacker Oracle Utility vs Mixed", "Avg Score of Best Attacker Pure Strategy"])
    for _ in range(experimentIterations):
        if showFrameworkOutput:
            print(f"iteration {_} of {experimentIterations}")
        # ----------------------------------------------------------------------
        # CORELP
        # ----------------------------------------------------------------------
        # Compute the mixed defender strategy
        if showFrameworkOutput:
            print("Computing defender mixed strategy...")
        defenderModel, dStrategyDistribution, dUtility = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        dUtility = float(dUtility)
        if showFrameworkOutput:
            print("Defender mixed strategy computed.")
        # Compute the mixed attacker strategy
        if showFrameworkOutput:
            print("Computing attacker mixed strategy...")
        attackerModel, aStrategyDistribution, aUtility = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]

        aUtility = float(aUtility)
        if showFrameworkOutput:
            print("Attacker mixed strategy computed.")
            print()
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # ORACLES
        # ----------------------------------------------------------------------
        # --------
        # DEFENDER
        # --------
        # Find the best oracle we currently have (to base training off of)
        if showFrameworkOutput:
            print("Finding highest utility defender oracle")
        bestDOracle, bestDOracleUtility = game.getBestOracle(ssg.DEFENDER, attackerPureIds, attackerIdMap, attackerMixedStrategy, defenderIdMap.values())
        print(f"Best Oracle: {bestDOracle}, bestUtility: {bestDOracleUtility}")

        # Train a new oracle
        if showFrameworkOutput:
            print("Computing defender oracle...")
        parameters = bestDOracle.getState()
        newDOracle = dO.DefenderOracle(targetNum)
        newDOracle.setState(parameters)
        dO.train(oracleToTrain=newDOracle, game=game, aIds=attackerPureIds, aMap=attackerIdMap, attackerMixedStrategy=attackerMixedStrategy, showOutput=showOracleTraining)
        # See the average payout of the new oracle
        if showFrameworkOutput:
            print("Defender oracle computed.")
            print("Testing score of new oracle")
        newDOracleScore = game.getOracleScore(ssg.DEFENDER, ids=attackerPureIds, map=attackerIdMap, mix=attackerMixedStrategy, oracle=newDOracle)
        if showFrameworkOutput:
            print(f"New Oracle Utility Computed: {newDOracleScore}")

        # --------

        # --------
        # ATTACKER
        # --------
        # Find the best oracle we currently have (to base training off of)
        if showFrameworkOutput:
            print("Finding highest utility attacker oracle")
        bestAOracle, bestAOracleUtility = game.getBestOracle(ssg.ATTACKER, defenderPureIds, defenderIdMap, defenderMixedStrategy, attackerIdMap.values())
        print(f"Best Oracle: {bestAOracle}, bestUtility: {bestAOracleUtility}")
        # Train a new oracle
        if showFrameworkOutput:
            print("Computing attacker oracle...")
        parameters = bestAOracle.getState()
        newAOracle = aO.AttackerOracle(targetNum)
        newAOracle.setState(parameters)
        aO.train(oracleToTrain=newAOracle, game=game, dIds=defenderPureIds, dMap=defenderIdMap, defenderMixedStrategy=defenderMixedStrategy, showOutput=showOracleTraining)
        if showFrameworkOutput:
            print("Attacker oracle computed")
            print("Testing score of new oracle")
        newAOracleScore = game.getOracleScore(ssg.ATTACKER, ids=defenderPureIds, map=defenderIdMap, mix=defenderMixedStrategy, oracle=newAOracle)
        if showFrameworkOutput:
            print(f"New Oracle Utility Computed: {newAOracleScore}")

        # ---------

        if showStrategies:
            print()
            print(f"Defender IDs: {defenderPureIds}")
            print(f"Defender mixed strategy: {defenderMixedStrategy}")
            print(f"Attacker IDs: {attackerPureIds}")
            print(f"Attacker mixed strategy: {attackerMixedStrategy}")
        if showUtilities:
            print()
            print(f"D Mixed Utility: {dUtility}")
            print(f"A Mixed Utility: {aUtility}")
            print(f"D Oracle utility (Should be > avg Mixed): {dOracleUtility}")
            print(f"A Oracle utility (Should be > avg Mixed): {aOracleUtility}")
        if writeUtilityFile:
            csvWriter.writerow([f"{dUtility}",f"{aUtility}",f"{newDOracleScore}",f"{bestDOracleUtility}",f"{newAOracleScore}",f"{bestAOracleUtility}"])
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # UPDATE POOLS
        # ----------------------------------------------------------------------
        if showFrameworkOutput:
            print("Updating pools and payout matrix...")
        for attackerId in attackerPureIds:
            # Run the game some amount of times
            value = game.getPayout(newDOracle, attackerIdMap[attackerId])
            payoutMatrix[newDefenderId, attackerId] = value
        for defenderId in defenderPureIds:
            # Run the game some amount of times
            value = game.getPayout(defenderIdMap[defenderId], newAOracle)
            payoutMatrix[defenderId, newAttackerId] = value
        value = game.getPayout(newDOracle, newAOracle)
        payoutMatrix[newDefenderId, newAttackerId] = value
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = newAOracle
        defenderIdMap[newDefenderId] = newDOracle
        newAttackerId += 1
        newDefenderId += 1
        if showFrameworkOutput:
            print("Pools and payout matrix updated.")
        if showFrameworkOutput or showOracleTraining or showUtilities or showStrategies:
            print("\n\n")
        # ----------------------------------------------------------------------

    if writeUtilityFile:
        csvFile.close()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Potential functions to make things clearer
def generateInitialPurePool():
    pass
def addToPayoutMatrix():
    pass

if __name__ == "__main__":
    main()
