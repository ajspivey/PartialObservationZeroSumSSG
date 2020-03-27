# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import combinations
import numpy as np
import random
import torch
from torch.autograd import Variable

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
        self.previousAttackerAction = np.array([0]*self.numTargets)
        self.previousDefenderAction = np.array([0]*self.numTargets)
        self.previousAttackerObservation = np.array([0]*self.numTargets*ATTACKER_FEATURE_SIZE)
        self.previousDefenderObservation = np.array([0]*self.numTargets*DEFENDER_FEATURE_SIZE)
        self.pastAttacks = [0]*self.numTargets
        self.pastAttackStatuses = [0]*self.numTargets
        self.defenderUtility = 0
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
    def performActions(self, defenderAction, attackerAction, oldDOb, oldAOb):
        """
        Performs the actions of the player and their opponent. Updates internal
        game state to reflect the new state of the game, and returns the new
        observations for the player and opponent
        """
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
        defenderActionScore = self.getActionScore(DEFENDER, defenderAction, attackerAction, self.defenderRewards, self.defenderPenalties)
        self.defenderUtility += defenderActionScore

        return (dObservation, aObservation, self.defenderUtility)

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
    def getActionScore(self, player, pAction, eAction, defenderRewards, defenderPenalties):
        """
        Returns the score for a player if the two given actions are played
        """
        if (player == DEFENDER):
            dAction = pAction
            aAction = eAction
        else:
            aAction = pAction
            dAction = eAction
        defenderReward = sum(np.multiply(np.multiply(dAction, aAction), defenderRewards))
        defenderPenalty = sum(np.multiply(aAction, defenderPenalties)) - sum(np.multiply(np.multiply(dAction, aAction), defenderPenalties))
        score = defenderReward - defenderPenalty
        if player == ATTACKER:
            score = score * -1
        return score

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
            dOb, aOb, _ = self.performActions(dAction, aAction, dOb, aOb)

        payout = self.defenderUtility
        self.restartGame()

        return payout

    # --------------------------------------------------------------------------
    def getOracleScore(self, player, ids, map, mix, oracle, epochs=10):
        """
        Returns the utility of an oracle vs. a mixed strategy
        """
        bestUtility = None
        bestOracle = None

        # Calculate average utility for each oracle in the list
        avgUtility = 0
        if player == DEFENDER:
            dAgent = oracle
        else:
            aAgent = oracle
        for epoch in range(epochs):
            dOb, aOb = self.getEmptyObservations()

            # Choose the agent from the mixed strategy
            choice = np.random.choice(ids, 1,
            p=mix)[0]
            if player == DEFENDER:
                aAgent = map[choice]
            else:
                dAgent = map[choice]

            # Play a full game
            for timestep in range(self.timesteps):
                aAction = aAgent.getAction(self, aOb)
                dAction = dAgent.getAction(self, dOb)
                dOb, aOb, _ = self.performActions(dAction, aAction, dOb, aOb)
            avgUtility += self.defenderUtility * player
            self.restartGame()

        avgUtility = avgUtility / epochs
        return avgUtility

    # --------------------------------------------------------------------------
    def getBestOracle(self, player, ids, map, mix, oracleList):
        """
        Returns the oracle in the list with the highest utility against the mixed
        strategy specified, and its utility
        """
        bestUtility = None
        bestOracle = None

        # Calculate average utility for each oracle in the list
        for oracle in oracleList:
            avgUtility = self.getOracleScore(player, ids, map, mix, oracle)
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
