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
    # Create a game with 6 targets, 2 resources, and 5 timesteps
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
    print("Game created.")

    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
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
    print("Strategies seeded.")

    # Compute the payout matrix for each pair of strategies
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
    print("Payout matrix computed.")


    # Keep iterating as long as our score is improving by some threshold
    totalDefenderUtility = 0
    improvementThreshold = 1e-2
    improvement = float('inf')
    print("Beginning iteration:\n")
    while(improvement > improvementThreshold):
        # ------
        # CORELP
        # ------
        # Compute the mixed defender strategy
        print("Computing defender mixed strategy...")
        defenderModel, dStrategyDistribution = coreLP.createDefenderModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        print("Defender mixed strategy computed.")
        # Compute the mixed attacker strategy
        print("Computing attacker mixed strategy...")
        attackerModel, aStrategyDistribution = coreLP.createAttackerModel(attackerPureIds, attackerIdMap, defenderPureIds, defenderIdMap, payoutMatrix)
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
        print("Attacker mixed strategy computed.")
        # -------
        # ORACLES
        # -------
        # Compute the defender oracle against the attacker mixed strategy
        print("Computing defender oracle...")
        defenderOracle = dO.DefenderOracle(targetNum)
        dO.train(oracleToTrain=defenderOracle, game=game, aIds=attackerPureIds, aMap=attackerIdMap, attackerMixedStrategy=attackerMixedStrategy)
        print("Defender oracle computed.")
        # Compute the attacker oracle against the defender mixed strategy
        print("Computing attacker oracle...")
        attackerOracle = aO.AttackerOracle(targetNum)
        aO.train(oracleToTrain=attackerOracle, game=game, dIds=defenderPureIds, dMap=defenderIdMap, defenderMixedStrategy=defenderMixedStrategy)
        print("Attacker oracle computed")

        # ------------
        # UPDATE POOLS
        # ------------
        print("Updating pools and payout matrix...")
        for attackerId in attackerPureIds:
            # Run the game some amount of times
            value = ssg.getPayout(game, defenderOracle, attackerIdMap[attackerId])
            payoutMatrix[newDefenderId, attackerId] = value
            game.restartGame()
            defenderOracle.reset()
            attackerIdMap[attackerId].reset()
        for defenderId in defenderPureIds:
            # Run the game some amount of times
            value = ssg.getPayout(game, defenderIdMap[defenderId], attackerOracle)
            payoutMatrix[defenderId, newAttackerId] = value
            game.restartGame()
            defenderIdMap[defenderId].reset()
            attackerOracle.reset()
        value = ssg.getPayout(game, defenderOracle, attackerOracle)
        payoutMatrix[newDefenderId, newAttackerId] = value
        game.restartGame()
        defenderOracle.reset()
        attackerOracle.reset()
        attackerPureIds.append(newAttackerId)
        defenderPureIds.append(newDefenderId)
        attackerIdMap[newAttackerId] = attackerOracle
        defenderIdMap[newDefenderId] = defenderOracle
        newAttackerId += 1
        newDefenderId += 1
        print("Pools and payout matrix updated.")

        # TODO: Calculate improvement somehow. What is the end utility??

if __name__ == "__main__":
    main()
