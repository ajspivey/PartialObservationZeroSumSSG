# ==============================================================================
# IMPORTS
# ==============================================================================
# External
from itertools import combinations
import numpy as np
import torch

np.random.seed(1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ==============================================================================
# CONSTANTS
# ==============================================================================
DEFENDER = 1
ATTACKER = -1
DEFENDER_FEATURE_SIZE = 5
ATTACKER_FEATURE_SIZE = 5
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ==============================================================================
# CLASSES
# ==============================================================================
class SequentialZeroSumSSG(object):
    # --------------------------------------------------------------------------
    def __init__(self, numTargets, numResources, defenderRewards, defenderPenalties, timesteps):
        """
        Creates a sequential zero-sum stackelberg security game with the given
        number of targets and number of defender resources, and the given
        defender rewards and defender penalties, as well as the specified timesteps
        """
        global DEFENDER
        global ATTACKER
        self.numTargets = numTargets
        self.numResources = numResources
        self.defenderRewards = defenderRewards
        self.defenderPenalties = defenderPenalties
        self.timesteps = timesteps
        self.defenderUtility = 0

        # Initialize internal values
        self.restartGame()

    # -----------------
    # Private functions
    # -----------------
    # --------------------------------------------------------------------------
    def restartGame(self):
        """
        Resets the internal state of the game to "unplayed"
        """
        self.currentTimestep = 0
        self.targets = [1] * self.numTargets
        self.availableResources = self.numResources
        self.pastAttacks = [0]*self.numTargets
        self.pastAttackStatuses = [0]*self.numTargets
        self.defenderUtility = 0

        self.previousAttackerAction = np.array([0]*self.numTargets)
        self.previousDefenderAction = np.array([0]*self.numTargets)
        self.previousAttackerObservation = np.array([0]*self.numTargets*ATTACKER_FEATURE_SIZE)
        self.previousDefenderObservation = np.array([0]*self.numTargets*DEFENDER_FEATURE_SIZE)
    # --------------------------------------------------------------------------
    def place_ones(self, size, count):
        """
        Helper function for determining valid defender actions
        """
        for positions in combinations(range(size), count):
            p = [0] * size
            for i in positions:
                p[i] = 1
            yield p

    # -------------------------------
    # Public Functions (External Use)
    # -------------------------------
    # --------------------------------------------------------------------------
    def getValidActions(self, player):
        """
        Returns a list of the valid actions for a player
        """
        actions = []
        if (player == DEFENDER):
            if self.availableResources == 0:
                return [[0] * self.numTargets]
            allResourcePlacements = list(self.place_ones(self.numTargets, self.availableResources))
            actions = [placements for placements in allResourcePlacements if sum(np.multiply(self.targets,placements)) == self.availableResources]
        elif (player == ATTACKER):
            for targetIndex in range(self.numTargets):
                if (self.targets[targetIndex]):
                    action = [0] * self.numTargets
                    action[targetIndex] = 1
                    if sum(np.multiply(action, self.pastAttacks)) == 0:
                        actions.append(action)
        return actions
    # --------------------------------------------------------------------------
    def performActions(self, dAction, aAction, dOb, aOb):
        """
        Performs the actions of the defender and attacker. Updates internal
        game state to reflect the new state of the game, and returns the new
        observations for the defender and attacker, as well as the defender's and
        attacker's scores for the move
        """
        # update game state
        self.currentTimestep += 1
        attackStatus = 1 - int(sum(np.multiply(aAction,dAction)))
        attackedTarget = np.where(np.array(aAction)==1)[0][0]
        self.availableResources = self.availableResources - (1 - attackStatus)
        self.pastAttacks[attackedTarget] = self.currentTimestep
        self.pastAttackStatuses = np.add(self.pastAttackStatuses, np.multiply(aAction, attackStatus))

        # Update actions and observations
        self.previousDefenderObservation = dOb
        self.previousDefenderAction = dAction
        dOb = np.concatenate((dAction, self.pastAttacks, self.pastAttackStatuses, self.defenderRewards, self.defenderPenalties))

        self.previousAttackerObservation = aOb
        self.previousAttackerAction = aAction
        aOb = np.concatenate((aAction, self.pastAttacks, self.pastAttackStatuses, self.defenderRewards, self.defenderPenalties))

        # Update utility scores
        defenderActionScore, attackerActionScore = self.getActionScore(dAction, aAction, self.defenderRewards, self.defenderPenalties)
        self.defenderUtility += defenderActionScore
        return (dOb, aOb, defenderActionScore, attackerActionScore)

    # --------------------------------------------------------------------------
    def getEmptyObservations(self):
        """
        Returns a set of empty observations, used by the LSTM layer for the first
        turn of a game
        """
        defenderObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.defenderRewards, self.defenderPenalties))
        attackerObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.defenderRewards, self.defenderPenalties))
        return (defenderObservation, attackerObservation)

    # --------------------------------------------------------------------------
    def getActionScore(self, dAction, aAction, defenderRewards, defenderPenalties):
        """
        Returns the score for defender and attacker if the two given actions are played
        """
        defenderReward = sum(np.multiply(np.multiply(dAction, aAction), defenderRewards))
        defenderPenalty = sum(np.multiply(aAction, defenderPenalties)) - sum(np.multiply(np.multiply(dAction, aAction), defenderPenalties))
        defenderScore = defenderReward - defenderPenalty
        attackerScore = defenderScore * -1
        return defenderScore, attackerScore

    # --------------------------------------------------------------------------
    def getPayout(self, defenderStrat, attackerStrat):
        """
        Returns the defender utility of the two given strategies played against
        each other.
        """
        dOb, aOb = self.getEmptyObservations()

        # Play a full game
        for timestep in range(self.timesteps):
            dAction = defenderStrat.getAction(self, dOb)
            aAction = attackerStrat.getAction(self, aOb)
            dOb, aOb, _, _ = self.performActions(dAction, aAction, dOb, aOb)

        defenderPayout = self.defenderUtility
        self.restartGame()

        return defenderPayout

    # --------------------------------------------------------------------------
    def getBestOracle(self, player, ids, map, mix, oracleList):
        """
        Returns the oracle in the list with the highest utility against the mixed
        strategy specified, and its utility
        """
        oracleList = list(oracleList)
        bestOracle = oracleList[0]
        bestUtility = expectedPureVMix(player, bestOracle, map, mix, cloneGame(self))

        # Calculate average utility for each oracle in the list
        for oracle in oracleList:
            if not oracle.isSoftmax():
                avgUtility = expectedPureVMix(player, oracle, map, mix, cloneGame(self))
                if (bestUtility is None or avgUtility > bestUtility):
                    bestUtility = avgUtility
                    bestOracle = oracle
        return (bestOracle, bestUtility)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =====================
# UTILITY SSG FUNCTIONS
# =====================
# ------------------------------------------------------------------------------
def generateRewardsAndPenalties(numTargets, lowBound=1, highBound = 50):
    """
    Generates a reward vector and a penalty vector within the given bounds
    """
    rewards = np.random.uniform(low=lowBound, high=highBound, size=numTargets)
    penalties = np.random.uniform(low=lowBound, high=highBound, size=numTargets)
    return (rewards, penalties)

# ------------------------------------------------------------------------------
def createRandomGame(targets=5, resources=None, timesteps=None):
    """
    Creates a game with the targets, resources, and timesteps specified, with
    random rewards and penalties
    """
    defenderRewards, defenderPenalties = generateRewardsAndPenalties(targets)
    if resources is None:
        resources = np.random.randint(1, targets-1)
    if timesteps is None:
        timesteps = np.random.randint(1, targets-1)
    game = SequentialZeroSumSSG(targets, resources, defenderRewards, defenderPenalties, timesteps)
    return game, defenderRewards, defenderPenalties
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------------------------------------------------
def cloneGame(game):
    """
    Clones the state of a game
    """
    gameClone = SequentialZeroSumSSG(game.numTargets, game.numResources, game.defenderRewards, game.defenderPenalties, game.timesteps)
    cloneGameState(gameClone, game)
    return gameClone
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cloneGameState(game1, game2):
    game1.currentTimestep = game2.currentTimestep
    game1.targets = game2.targets.copy()
    game1.availableResources = game2.availableResources
    game1.pastAttacks = game2.pastAttacks.copy()
    game1.pastAttackStatuses = game2.pastAttackStatuses.copy()
    game1.defenderUtility = game2.defenderUtility
    game1.previousAttackerAction = game2.previousAttackerAction.copy()
    game1.previousDefenderAction = game2.previousDefenderAction.copy()
    game1.previousAttackerObservation = game2.previousAttackerObservation.copy()
    game1.previousDefenderObservation = game2.previousDefenderObservation.copy()

#
def expectedPureVPure(dAgent, aAgent, gameClone, avgCount=10):
    expectedValue = playGame(gameClone, dAgent, aAgent)
    gameClone.restartGame()

    if (dAgent.isSoftmax() or aAgent.isSoftmax()):
        for _ in range(avgCount - 1):
            expectedValue += playGame(gameClone, dAgent, aAgent)
            gameClone.restartGame()
        expectedValue /= avgCount
    return expectedValue

def playGame(gameClone, dPure, aPure):
    dOb, aOb = gameClone.getEmptyObservations()
    # Play a full game
    for timestep in range(gameClone.timesteps):
        dAction = dPure.getAction(gameClone, dOb)
        aAction = aPure.getAction(gameClone, aOb)
        dOb, aOb, _, _ = gameClone.performActions(dAction, aAction, dOb, aOb)
    return gameClone.defenderUtility

def expectedPureVMix(player, pure, map, mix, gameClone):
    expectedValue = 0
    # For each agent in the mix, play a pureVpure game and multiply by the odds
    for agentIndex in range(len(map)):
        mixAgent = map[agentIndex]
        if player == DEFENDER:
            dAgent = pure
            aAgent = mixAgent
        else:
            dAgent = mixAgent
            aAgent = pure
        expectedValue += expectedPureVPure(dAgent, aAgent, gameClone) * mix[agentIndex]
    return expectedValue * player
