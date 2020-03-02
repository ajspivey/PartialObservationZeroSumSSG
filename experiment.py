# ======
# @TODO:
# ======
# Refactor ssg to only have defender Utility
# Refactor player to 1/-1 instead of 0/1
# Generalize attacker/defender specific functions
# Remove unused functions
# Get rid of "generalOracle"
# Standardize function parameter orders
# Recomment
# Add docstrings to functions
# Clean up whitespace/make flow of functions clear
# Create functions in experiment file?


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

np.random.seed(1)

# ====
# MAIN
# ====
def main():
    # =========
    # DEBUGGING
    # =========
    showOracleTraining = False
    showFrameworkOutput = False
    showUtilities = False
    showStrategies = False
    writeUtilityFile = False

    # ===============
    # HyperParameters
    # ===============
    iterations = 10
    targetNum = 6
    resources = 2
    timesteps = 3

    # Create a game with 6 targets, 2 resources, and 2 timesteps
    if showFrameworkOutput:
        print("Creating game...")
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=resources, timesteps=timesteps)

    # Used to do consistent testing and comparisons
    defenderRewards = [21.43407823, 36.29590018,  1.00560437, 15.81429606,  8.19103865,  5.52459114]
    defenderPenalties = [10.12675036, 17.93247563, 20.44160624, 27.40201997, 21.54053121, 34.57575552]

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

    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
    if showFrameworkOutput:
        print("Seeding initial attacker and defender pure strategies")
    for _ in range(iterations):
        attackerOracle = aO.RandomAttackerOracle(targetNum, game)
        defenderOracle = dO.RandomDefenderOracle(targetNum, game)
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
            pureDefender = defenderIdMap[defenderId]
            # Run the game some amount of times
            value = ssg.getPayout(game, pureDefender, pureAttacker)
            payoutMatrix[attackerId,defenderId] = value
            game.restartGame()
            pureDefender.reset()
            pureAttacker.reset()
    if showFrameworkOutput:
        print("Payout matrix computed.")


    # Keep iterating as long as our score is improving by some threshold
    if showFrameworkOutput:
        print("Beginning iteration:\n")
    if writeUtilityFile:
        csvFile = open("utilities.csv", "w", newline='')
        csvWriter = csv.writer(csvFile, delimiter=',')
        csvWriter.writerow(["Defender Mixed Utility","Attacker Mixed Utility", "Defender Oracle Utility", "Attacker Oracle Utility", "Avg. Defender Oracle Utility vs Mixed", "Defender Oracle Correctness vs perfect play", "Avg Score of Best Defender Pure Strategy", "Avg. Attacker Oracle Utility vs Mixed", "Attacker Oracle Correctness vs perfect play", "Avg Score of Best Attacker Pure Strategy"])
    for _ in range(100):
        if showFrameworkOutput:
            print(f"iteration {_} of 100")
        # ------
        # CORELP
        # ------
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

        # -------
        # ORACLES
        # -------
        # Determine the best utility oracle we have against the attacker mixed strategy
        if showFrameworkOutput:
            print("Finding highest utility defender oracle")
        bestDOracle = None
        bestDUtility = 0
        for

        # Compute the defender oracle against the attacker mixed strategy
        if showFrameworkOutput:
            print("Computing defender oracle...")
        defenderOracle = dO.DefenderOracle(targetNum)
        dO.train(oracleToTrain=defenderOracle, game=game, aIds=attackerPureIds, aMap=attackerIdMap, attackerMixedStrategy=attackerMixedStrategy, showOutput=showOracleTraining)
        parameters = defenderOracle.getState()
        cloneOracle = dO.DefenderOracle(targetNum)
        cloneOracle.setState(parameters)
        ssg.compareOracles(game, defenderOracle, cloneOracle, attackerPureIds, attackerIdMap, attackerMixedStrategy)

        testedDUtility, testedDCor, testedDBestOpponent = ssg.testDefenderOracle(game, defenderOracle, attackerPureIds, attackerIdMap, attackerMixedStrategy, 100, defenderIdMap)
        if showFrameworkOutput:
            print("Defender oracle computed.")
        # Compute the attacker oracle against the defender mixed strategy
        if showFrameworkOutput:
            print("Computing attacker oracle...")
        attackerOracle = aO.AttackerOracle(targetNum)
        aOracleUtility, aOracleLoss = aO.train(oracleToTrain=attackerOracle, game=game, dIds=defenderPureIds, dMap=defenderIdMap, defenderMixedStrategy=defenderMixedStrategy, showOutput=showOracleTraining)
        testedAUtility, testedACor, testedABestOpponent = ssg.testAttackerOracle(game, attackerOracle, defenderPureIds, defenderIdMap, defenderMixedStrategy, 100, attackerIdMap)
        if showFrameworkOutput:
            print("Attacker oracle computed")
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
            csvWriter.writerow([f"{dUtility}",f"{aUtility}",f"{dOracleUtility}",f"{aOracleUtility}",f"{testedDUtility}",f"{testedDCor}",f"{testedDBestOpponent}",f"{testedAUtility}",f"{testedACor}",f"{testedABestOpponent}",])

        # ------------
        # UPDATE POOLS
        # ------------
        if showFrameworkOutput:
            print("Updating pools and payout matrix...")
        for attackerId in attackerPureIds:
            # Run the game some amount of times
            value = ssg.getPayout(game, defenderOracle, attackerIdMap[attackerId])
            payoutMatrix[newDefenderId, attackerId] = value
        for defenderId in defenderPureIds:
            # Run the game some amount of times
            value = ssg.getPayout(game, defenderIdMap[defenderId], attackerOracle)
            payoutMatrix[defenderId, newAttackerId] = value
        value = ssg.getPayout(game, defenderOracle, attackerOracle)
        payoutMatrix[newDefenderId, newAttackerId] = value
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = attackerOracle
        defenderIdMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
        if showFrameworkOutput:
            print("Pools and payout matrix updated.")
        if showFrameworkOutput or showOracleTraining or showUtilities or showStrategies:
            print("\n\n")

    if writeUtilityFile:
        csvFile.close()

if __name__ == "__main__":
    main()
