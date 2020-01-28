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
            value = ssg.getPayout(game, pureDefender, pureAttacker) # TODO: write this function
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
        defenderModel, dStrategyDistribution = coreLP.createDefenderModel(attackerPureStrategies, defenderPureStrategies, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        # Compute the mixed attacker strategy
        attackerModel, aStrategyDistribution = coreLP.createAttackerModel(attackerPureStrategies, defenderPureStrategies, payoutMatrix) #TODO: Write this function
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]

        print(f"Attacker mixed strategy: {attackerMixedStrategy}")
        print(f"Defender mixed strategy: {defenderMixedStrategy}")
        # -------
        # ORACLES
        # -------
        # Compute the defender oracle against the attacker mixed strategy
        defenderOracle = dO.DefenderOracle(targetNum, ssg.DEFENDER_FEATURE_SIZE)
        dO.train(oracleToTrain=defenderOracle, attackerPool=game.attackerPool, attackerMixedStrategy=attackerMixedStrategy) #TODO: Write this function
        # Compute the attacker oracle against the defender mixed strategy
        attackerOracle = dO.AttackerOracle(targetNum, ssg.ATTACKER_FEATURE_SIZE)
        dO.train(oracleToTrain=attackerOracle, defenderPool=game.defenderPool, defenderMixedStrategy=defenderMixedStrategy) #TODO: Write this function

        # ------------
        # UPDATE POOLS
        # ------------
        for pureAttacker in game.attackerPool:
            # Run the game some amount of times
            value = ssg.getAveragePayout(game, defenderOracle, pureAttacker) # TODO: write this function
            payoutMatrix[defenderOracle,pureAttacker] = value
        for pureDefender in game.defenderPool:
            # Run the game some amount of times
            value = ssg.getAveragePayout(game, pureDefender, attackerOracle) # TODO: write this function
            payoutMatrix[pureDefender,attackerOracle] = value
        game.addPureStrategyToPool(ssg.ATTACKER, attackerOracle)
        game.addPureStrategyToPool(ssg.DEFENDER, defenderOracle)

        # TODO: Calculate improvement somehow. What is the end utility??

if __name__ == "__main__":
    main()
