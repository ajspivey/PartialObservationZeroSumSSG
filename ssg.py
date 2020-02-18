# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import combinations
import numpy as np
import random
import torch

np.random.seed(1)

# ==============================================================================
# CONSTANTS
# ==============================================================================
DEFENDER = 0
ATTACKER = 1
DEFENDER_FEATURE_SIZE = 5
ATTACKER_FEATURE_SIZE = 5

# ==============================================================================
# CLASSES
# ==============================================================================
class SequentialZeroSumSSG(object):
    def __init__(self, numTargets, numResources, defenderRewards, defenderPenalties, timesteps):
        global DEFENDER
        global ATTACKER
        self.numTargets = numTargets
        self.numResources = numResources
        self.defenderRewards = defenderRewards
        self.defenderPenalties = defenderPenalties
        self.timesteps = timesteps
        self.defenderUtility = 0
        self.attackerUtility = 0

        # Initialize internal values
        self.restartGame()

    # -------
    # Helpers
    # -------
    def restartGame(self):
        self.currentTimestep = 0
        self.targets = [1] * self.numTargets # Destroyed targets are a 0, otherwise a 1
        self.availableResources = self.numResources  # Resources are consumed if they stop an attack. This tracks remaining resources
        self.previousAttackerAction = np.array([0]*self.numTargets)
        self.previousDefenderAction = np.array([0]*self.numTargets)
        self.previousAttackerObservation = np.array([0]*self.numTargets*ATTACKER_FEATURE_SIZE)
        self.previousDefenderObservation = np.array([0]*self.numTargets*DEFENDER_FEATURE_SIZE)
        self.pastAttacks = [0]*self.numTargets
        self.pastAttackStatuses = [0]*self.numTargets
        self.defenderUtility = 0
        self.attackerUtility = 0

    def place_ones(self, size, count):
        """ Helper function for determining valid defender actions """
        for positions in combinations(range(size), count):
            p = [0] * size
            for i in positions:
                p[i] = 1
            yield p

    def getValidActions(self, player):
        """ Returns the valid actions for a player """
        # Actions are a vector of defender placements -- 0 if nothing is placed,
        # 1 if resources are placed.
        if (player == DEFENDER):
            if self.availableResources == 0:
                return [[0] * self.numTargets]
            allResourcePlacements = list(self.place_ones(self.numTargets, self.availableResources))
            viablePlacements = [placements for placements in allResourcePlacements if sum(np.multiply(self.targets,placements)) == self.availableResources]
            return viablePlacements
        elif (player == ATTACKER):
            actions = []
            for targetIndex in range(self.numTargets):
                if (self.targets[targetIndex]):
                    action = [0] * self.numTargets
                    action[targetIndex] = 1
                    if sum(np.multiply(action, self.pastAttacks)) == 0:
                        actions.append(action)
            return actions
        raise unknownPlayerError(f"Player is not Attacker or Defender. Player {player} unknown")


    # ---------------------
    # Performing game steps
    # ---------------------
    def performActions(self, defenderAction, attackerAction, oldDOb, oldAOb):
        defenderAction = defenderAction.detach().numpy()
        attackerAction = attackerAction.detach().numpy()

        self.currentTimestep += 1
        attackStatus = 1 - int(sum(np.multiply(attackerAction,defenderAction)))
        attackedTarget = np.where(np.array(attackerAction)==1)[0][0]
        self.availableResources = self.availableResources - (1 - attackStatus)
        self.pastAttacks[attackedTarget] = self.currentTimestep
        self.pastAttackStatuses = np.add(self.pastAttackStatuses, np.multiply(attackerAction, attackStatus))

        # Update actions and observations
        self.previousAttackerObservation = oldAOb
        self.previousDefenderObservation = oldDOb
        dObservation = np.concatenate((defenderAction, self.pastAttacks, self.pastAttackStatuses, self.defenderRewards, self.defenderPenalties))
        aObservation = np.concatenate((attackerAction, self.pastAttacks, self.pastAttackStatuses, self.defenderRewards, self.defenderPenalties))
        self.previousAttackerAction = attackerAction
        self.previousDefenderAction = defenderAction

        # Update utility scores
        defenderActionScore = self.getActionScore(DEFENDER, attackerAction, defenderAction, self.defenderRewards, self.defenderPenalties)
        self.defenderUtility += defenderActionScore
        self.attackerUtility -= defenderActionScore

        return (dObservation, aObservation)

    def getEmptyObservations(self):
        defenderObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.defenderRewards, self.defenderPenalties))
        attackerObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.defenderRewards, self.defenderPenalties))
        return (defenderObservation, attackerObservation)

    # -------------
    # Action Scores
    # -------------
    def getActionScore(self, player, aAction, dAction, defenderRewards, defenderPenalties):
        score = 0
        for targetIndex in range(len(dAction)):
            if aAction[targetIndex] and not dAction[targetIndex]:
                score -= defenderPenalties[targetIndex]
            elif aAction[targetIndex] and dAction[targetIndex]:
                score += defenderRewards[targetIndex]
        if player == ATTACKER:
            score = score * -1
        return score

    def getBestActionAndScore(self, player, eAction, defenderRewards, defenderPenalties):
        actions = self.getValidActions(player)
        bestAction = actions[0]
        bestActionScore = float("-inf")
        for action in actions:
            dAction = action
            aAction = eAction
            if (player == ATTACKER):
                dAction = eAction
                aAction = action
            actionScore = self.getActionScore(player, aAction, dAction, defenderRewards, defenderPenalties)
            if actionScore > bestActionScore:
                bestActionScore = actionScore
                bestAction = action
        return (np.array(bestAction), bestActionScore)

    def makeLegalAttackerMove(self, action):
        moveToPrune = action.detach().numpy()
        # Zero out any actions that are impossible and re-normalize
        for targetIndex in range(len(moveToPrune)):
            if self.pastAttackStatuses[targetIndex]:
                moveToPrune[targetIndex] = 0
        maxValIndex = torch.argmax(torch.from_numpy(moveToPrune))
        prunedMove = torch.nn.functional.one_hot(maxValIndex, self.numTargets)
        return prunedMove

    def makeLegalDefenderMove(self, action):
        moveToPrune = action.detach().numpy()
        # print(f"before: {moveToPrune}")
        # Zero out any actions that are impossible and re-normalize
        for targetIndex in range(len(moveToPrune)):
            if self.pastAttackStatuses[targetIndex]:
                moveToPrune[targetIndex] = 0
        # Pick the highest n remaining values, where n is the number of resources
        # left
        highest = np.argpartition(moveToPrune, -self.availableResources)[-self.availableResources:]
        if len(highest) == self.availableResources:
            for targetIndex in range(len(moveToPrune)):
                if targetIndex in highest:
                    moveToPrune[targetIndex] = 1
                else:
                    moveToPrune[targetIndex] = 0
        else:
            moveToPrune = np.array([0] * self.numTargets)
        return torch.from_numpy(moveToPrune)



# ------
# Helper
# ------
def generateRewardsAndPenalties(numTargets, lowBound=1, highBound = 50):
    rewards = np.random.uniform(low=lowBound, high=highBound, size=numTargets)
    penalties = np.random.uniform(low=lowBound, high=highBound, size=numTargets)
    return (rewards, penalties)

def createRandomGame(targets=5, resources=None, timesteps=None):
    defenderRewards, defenderPenalties = generateRewardsAndPenalties(targets)
    if resources is None:
        resources = np.random.randint(1, targets-1)
    if timesteps is None:
        timesteps = np.random.randint(1, targets-1)
    game = SequentialZeroSumSSG(targets, resources, defenderRewards, defenderPenalties, timesteps)
    return game, defenderRewards, defenderPenalties

def getPayout(game, defenderStrat, attackerStrat):
    """ Result is defender utility """
    aInput = attackerStrat.inputFromGame(game)
    dInput = defenderStrat.inputFromGame(game)
    aAction = [0]*game.numTargets
    dAction = [0]*game.numTargets
    dOb, aOb = game.getEmptyObservations()

    # Play a full game
    for timestep in range(game.timesteps):
        dAction = defenderStrat(dInput(dOb))
        dAction = game.makeLegalDefenderMove(dAction)
        aAction = attackerStrat(aInput(aOb))
        aAction = game.makeLegalAttackerMove(aAction)
        dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

    payout = game.defenderUtility

    defenderStrat.reset()
    attackerStrat.reset()
    game.restartGame()

    return payout

def getAveragePayout(game, defenderMixedStrategy, defenderPureIds, defenderIdMap, attackerMixedStrategy, attackerPureIds, attackerIdMap, iterations=100):
    """ Result is defender utility """
    totalDefenderUtility = 0
    # Play a certain number of games
    for iteration in range(iterations):
        aAction = [0]*game.numTargets
        dAction = [0]*game.numTargets
        dOb, aOb = game.getEmptyObservations()

        # Play a full game
        for timestep in range(game.timesteps):
            # Select the observations according to the mixed strategy
            defenderAgent = defenderIdMap[np.random.choice(defenderPureIds, 1,
                          p=defenderMixedStrategy)[0]]
            dAgentInputFunction = defenderAgent.inputFromGame(game)
            dAction = defenderAgent(dAgentInputFunction(dOb))
            dAction = game.makeLegalDefenderMove(dAction)

            attackerAgent = attackerIdMap[np.random.choice(attackerPureIds, 1,
                          p=attackerMixedStrategy)[0]]
            aAgentInputFunction = attackerAgent.inputFromGame(game)
            aAction = attackerAgent(aAgentInputFunction(aOb))
            aAction = game.makeLegalAttackerMove(aAction)

            dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

        totalDefenderUtility += game.defenderUtility
        game.restartGame()
        for defender in defenderIdMap.values():
            defender.reset()
        for attacker in attackerIdMap.values():
            attacker.reset()
    return totalDefenderUtility/iterations

def testAttackerOracle(game, oracle, ids, map, mix, iterations, aMap):
    print(f"TESTING ATTACKER ORACLE AGAINST MIXED DEFENDER")
    oracleUtility = 0
    correctness = 0
    # Get an average utility for this oracle
    for iteration in range(iterations):
        aAction = [0]*game.numTargets
        dAction = [0]*game.numTargets
        dOb, aOb = game.getEmptyObservations()
        # Play a full game
        for timestep in range(game.timesteps):
            # Select the observations according to the mixed strategy
            choice = np.random.choice(ids, 1, p=mix)[0]
            defenderAgent = map[choice]
            dAgentInputFunction = defenderAgent.inputFromGame(game)
            dAction = defenderAgent(dAgentInputFunction(dOb))
            dAction = game.makeLegalDefenderMove(dAction)
            aAgentInputFunction = oracle.inputFromGame(game)
            aAction = oracle(aAgentInputFunction(aOb))
            aAction = game.makeLegalAttackerMove(aAction)
            best, _ = game.getBestActionAndScore(ATTACKER, dAction, game.defenderRewards, game.defenderPenalties)
            compare = aAction.detach().numpy()
            if np.array_equal(compare,best):
                correctness += 1
            # print(f"Opponent action: {dAction}")
            # print(f"action: {aAction}")
            # print(f"best  : {best}")
            # print()
            dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
        oracleUtility += game.attackerUtility
        game.restartGame()
        for defender in map.values():
            defender.reset()
        oracle.reset()
    oracleUtility = oracleUtility / iterations
    correctness = correctness / (iterations*game.timesteps)
    print(f"Correctness: {correctness}")
    # Find the best average utility out of the other pure strategies
    avgs = []
    aIndex = 1
    aLen = len(aMap.values())
    for attacker in aMap.values():
        # print(f"Working with attacker {aIndex} out of {aLen}")
        avgUtility = 0
        for iteration in range(iterations):
            aAction = [0]*game.numTargets
            dAction = [0]*game.numTargets
            dOb, aOb = game.getEmptyObservations()
            # Play a full game
            for timestep in range(game.timesteps):
                # Select the observations according to the mixed strategy
                choice = np.random.choice(ids, 1,
                              p=mix)[0]
                defenderAgent = map[choice]
                dAgentInputFunction = defenderAgent.inputFromGame(game)
                dAction = defenderAgent(dAgentInputFunction(dOb))
                dAction = game.makeLegalDefenderMove(dAction)
                aAgentInputFunction = attacker.inputFromGame(game)
                aAction = attacker(aAgentInputFunction(aOb))
                aAction = game.makeLegalAttackerMove(aAction)
                dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
            avgUtility += game.attackerUtility
            game.restartGame()
            for defender in map.values():
                defender.reset()
            attacker.reset()
        avgUtility = avgUtility / iterations
        # print(f"attacker {aIndex} avg against defender: {avgUtility}")
        aIndex += 1
        avgs.append(avgUtility)
    print(f"Avg A oracle Utility against defender mix: {oracleUtility}")
    print(f"best average -- {max(avgs)}")

def testDefenderOracle(game, oracle, ids, map, mix, iterations, dMap):
    oracleUtility = 0
    correctness = 0
    print(f"TESTING DEFENDER ORACLE AGAINST MIXED ATTACKER")
    # Get an average utility for this oracle
    for iteration in range(iterations):
        aAction = [0]*game.numTargets
        dAction = [0]*game.numTargets
        dOb, aOb = game.getEmptyObservations()
        # Play a full game
        # print(f"game: {iteration}")
        for timestep in range(game.timesteps):
            # print(f"Timestep: {timestep}")
            # Select the observations according to the mixed strategy
            choice = np.random.choice(ids, 1, p=mix)[0]
            attackerAgent = map[choice]
            aAgentInputFunction = attackerAgent.inputFromGame(game)
            aAction = attackerAgent(aAgentInputFunction(aOb))
            aAction = game.makeLegalAttackerMove(aAction)
            dAgentInputFunction = oracle.inputFromGame(game)
            dAction = oracle(dAgentInputFunction(dOb))
            dAction = game.makeLegalDefenderMove(dAction)
            best, _ = game.getBestActionAndScore(DEFENDER, aAction, game.defenderRewards, game.defenderPenalties)
            compare = dAction.detach().numpy()
            if np.array_equal(compare,best):
                correctness += 1
            # print(f"Opponent action: {aAction}")
            # print(f"action: {dAction}")
            # print(f"best  : {best}")
            # print()
            dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
        oracleUtility += game.defenderUtility
        game.restartGame()
        for attacker in map.values():
            attacker.reset()
        oracle.reset()
    oracleUtility = oracleUtility / iterations
    correctness = correctness / (iterations*game.timesteps)
    print(f"Correctness: {correctness}")
    # Find the best average utility out of the other pure strategies
    avgs = []
    dIndex = 1
    dLen = len(dMap.values())
    for defender in dMap.values():
        # print(f"Working with defender {dIndex} out of {dLen}")
        avgUtility = 0
        for iteration in range(iterations):
            aAction = [0]*game.numTargets
            dAction = [0]*game.numTargets
            dOb, aOb = game.getEmptyObservations()
            # Play a full game
            for timestep in range(game.timesteps):
                choice = np.random.choice(ids, 1,
                              p=mix)[0]
                attackerAgent = map[choice]
                aAgentInputFunction = attackerAgent.inputFromGame(game)
                aAction = attackerAgent(aAgentInputFunction(aOb))
                aAction = game.makeLegalAttackerMove(aAction)
                dAgentInputFunction = defender.inputFromGame(game)
                dAction = defender(dAgentInputFunction(dOb))
                dAction = game.makeLegalDefenderMove(dAction)
                dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
            avgUtility += game.defenderUtility
            game.restartGame()
            for attacker in map.values():
                attacker.reset()
            defender.reset()
        avgUtility = avgUtility / iterations
        # print(f"defender {dIndex} avg against attacker: {avgUtility}")
        dIndex += 1
        avgs.append(avgUtility)
    print(f"Avg D oracle Utility against defender mix: {oracleUtility}")
    print(f"best average -- {max(avgs)}")
