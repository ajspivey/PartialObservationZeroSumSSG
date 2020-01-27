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
if __name__ == "__main__":
    main()

def main():
    # Create a test game with 6 targets, 2 resources, and 5 timesteps
    targetNum = 6
    game = ssg.createRandomGame(targets=targetNum, resources=2, timesteps=5)
    payoutMatrix = {}

    # Start with a 5 random attacker pure strategies and 5 random defender pure strategies
    for _ in range(5):
        attackerOracle = aO.getRandomAttackerOracle(targetNum)
        defenderOracle = dO.getRandomDefenderOracle(targetNum)
        game.addPureStrategyToPool(ssg.ATTACKER, attackerOracle)
        game.addPureStrategyToPool(ssg.DEFENDER, defenderOracle)

    # Compute the payout matrix for each pair of strategies
    for pureAttacker in game.attackerPool:
        for pureDefender in game.defenderPool:
            # Run the game some amount of times
            value = ssg.getAveragePayout(game, pureDefender, pureAttacker) # TODO: write this function
            payoutMatrix[pureDefender,pureAttacker] = value


    # Keep iterating as long as our score is improving by some threshold
    totalDefenderUtility = 0
    improvementThreshold = 1e-2
    improvement = float('inf')
    while(improvement > improvementThreshold):
        # ------
        # CORELP
        # ------
        # Compute the mixed defender strategy
        defenderModel, dStrategyDistribution = coreLP.createDefenderModel(game.attackerPool, game.defenderPool, payoutMatrix)
        defenderModel.solve()
        defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
        # Compute the mixed attacker strategy
        attackerModel, aStrategyDistribution = coreLP.createAttackerModel(game.attackerPool, game.defenderPool, payoutMatrix) #TODO: Write this function
        attackerModel.solve()
        attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]

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
