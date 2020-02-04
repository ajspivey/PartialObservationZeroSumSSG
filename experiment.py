# =======
# IMPORTS
# =======
# External
import numpy as np
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
    showUtilities = True
    showStrategies = True


    # Create a game with 6 targets, 2 resources, and 5 timesteps
    if showFrameworkOutput:
        print("Creating game...")
    targetNum = 6
    game, defenderRewards, defenderPenalties = ssg.createRandomGame(targets=targetNum, resources=2, timesteps=3)
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
    for _ in range(5):
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
    totalDefenderUtility = 0
    improvement = float('inf')
    if showFrameworkOutput:
        print("Beginning iteration:\n")
    for _ in range(5):
        # ------
        # CORELP
        # ------
        # Compute the mixed defender strategy
        if showFrameworkOutput:
            print("Computing defender mixed strategy...")
        defenderModel, dStrategyDistribution, dUtility = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        if showUtilities:
            print(f"D Mix from Core: {float(dUtility)}")
        if showFrameworkOutput:
            print("Defender mixed strategy computed.")
        if showStrategies:
            print(f"Defender IDs: {defenderPureIds}")
            print(f"Defender mixed strategy: {defenderMixedStrategy}")
        # Compute the mixed attacker strategy
        if showFrameworkOutput:
            print("Computing attacker mixed strategy...")
        attackerModel, aStrategyDistribution, aUtility = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
        if showFrameworkOutput:
            print("Attacker mixed strategy computed.")
        if showUtilities:
            print(f"A Mix from Core: {float(aUtility)}")
        if showStrategies:
            print(f"Attacker IDs: {attackerPureIds}")
            print(f"Attacker mixed strategy: {attackerMixedStrategy}")

        # ==========
        # EVALUATION
        # ==========
        # Compute the average defender and attacker utility with the mixed strategies
        # value = ssg.getAveragePayout(game, defenderMixedStrategy, defenderPureIds, defenderIdMap, attackerMixedStrategy, attackerPureIds, attackerIdMap)
        # if showUtilities:
            # print(f"Avg. D Mix against A Mix: {value}")

        # -------
        # ORACLES
        # -------
        # Compute the defender oracle against the attacker mixed strategy
        if showFrameworkOutput:
            print("Computing defender oracle...")
        defenderOracle = dO.DefenderOracle(targetNum)
        dOracleUtility, dOracleLoss = dO.train(oracleToTrain=defenderOracle, game=game, aIds=attackerPureIds, aMap=attackerIdMap, attackerMixedStrategy=attackerMixedStrategy, showOutput=showOracleTraining)
        if showFrameworkOutput:
            print("Defender oracle computed.")
        if showUtilities:
            print(f"D Oracle utility against A mix: {dOracleUtility}")
            print(f"D Oracle loss against A mix: {dOracleLoss}")
        # Compute the attacker oracle against the defender mixed strategy
        if showFrameworkOutput:
            print("Computing attacker oracle...")
        attackerOracle = aO.AttackerOracle(targetNum)
        aOracleUtility, aOracleLoss = aO.train(oracleToTrain=attackerOracle, game=game, dIds=defenderPureIds, dMap=defenderIdMap, defenderMixedStrategy=defenderMixedStrategy, showOutput=showOracleTraining)
        if showFrameworkOutput:
            print("Attacker oracle computed")
        if showUtilities:
            print(f"A Oracle utility against D mix: {aOracleUtility}")
            print(f"A Oracle loss against D mix: {aOracleLoss}")

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

    print(f"Average defender mixed strategy utility: {value}")
    print(f"Average defender mixed strategy utility: {value}")
    print(f"Using attacker mixed strategy {attackerMixedStrategy}")
    print(f"And defender mixed strategy {defenderMixedStrategy}")

if __name__ == "__main__":
    main()
