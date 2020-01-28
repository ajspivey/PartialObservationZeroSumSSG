# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import combinations
import numpy as np
import random


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
            allResourcePlacements = list(self.place_ones(self.numTargets, self.availableResources))
            viablePlacements = [placements for placements in allResourcePlacements if sum(np.multiply(self.targets,placements)) == self.availableResources]
            return viablePlacements + [[0] * self.numTargets]
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
            else:
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
        aAction = attackerStrat(aInput(aOb))
        dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)

    return game.defenderUtility
