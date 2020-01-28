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
    attackerPureStrategies = []
    defenderPureStrategies = []
    attackerMixedStrategy = None
    defenderMixedStrategy = None
    print("Game created.")

    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
    print("Seeding initial attacker and defender pure strategies")
    for _ in range(5):
        attackerOracle = aO.RandomAttackerOracle(targetNum, game)
        defenderOracle = dO.RandomDefenderOracle(targetNum, game)
        attackerPureStrategies.append(attackerOracle)
        defenderPureStrategies.append(defenderOracle)
    print("Strategies seeded.")

    # Compute the payout matrix for each pair of strategies
    print("Computing initial payout matrix for pure strategies...")
    for pureAttacker in attackerPureStrategies:
        for pureDefender in defenderPureStrategies:
            # Run the game some amount of times
            value = ssg.getPayout(game, pureDefender, pureAttacker)
            payoutMatrix[pureDefender,pureAttacker] = value
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
        defenderModel, dStrategyDistribution = coreLP.createDefenderModel(attackerPureStrategies, defenderPureStrategies, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        print("Defender mixed strategy computed.")
        # Compute the mixed attacker strategy
        print("Computing attacker mixed strategy...")
        attackerModel, aStrategyDistribution = coreLP.createAttackerModel(attackerPureStrategies, defenderPureStrategies, payoutMatrix)
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
        print("Attacker mixed strategy computed.")
        # -------
        # ORACLES
        # -------
        # Compute the defender oracle against the attacker mixed strategy
        print("Computing defender oracle...")
        defenderOracle = dO.DefenderOracle(targetNum)
        dO.train(oracleToTrain=defenderOracle, game=game, attackerPool=attackerPureStrategies, attackerMixedStrategy=attackerMixedStrategy)
        print("Defender oracle computed.")
        # Compute the attacker oracle against the defender mixed strategy
        print("Computing attacker oracle...")
        attackerOracle = aO.AttackerOracle(targetNum)
        aO.train(oracleToTrain=attackerOracle, game=game, defenderPool=defenderPureStrategies, defenderMixedStrategy=defenderMixedStrategy)
        print("Attacker oracle computed")

        # ------------
        # UPDATE POOLS
        # ------------
        print("Updating pools and payout matrix...")
        print("AttackerStrategies")
        for pureAttacker in attackerPureStrategies:
            # Run the game some amount of times
            value = ssg.getPayout(game, defenderOracle, pureAttacker)
            payoutMatrix[defenderOracle,pureAttacker] = value
            game.restartGame()
            defenderOracle.reset()
            pureAttacker.reset()
        print("Defender strateges")
        for pureDefender in defenderPureStrategies:
            # Run the game some amount of times
            value = ssg.getPayout(game, pureDefender, attackerOracle)
            payoutMatrix[pureDefender,attackerOracle] = value
            game.restartGame()
            pureDefender.reset()
            attackerOracle.reset()
        game.addPureStrategyToPool(ssg.ATTACKER, attackerOracle)
        game.addPureStrategyToPool(ssg.DEFENDER, defenderOracle)
        print("Pools and payout matrix updated.")

        # TODO: Calculate improvement somehow. What is the end utility??

if __name__ == "__main__":
    main()
