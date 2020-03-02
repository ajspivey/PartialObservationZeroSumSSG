# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import combinations
import numpy as np
import random
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

    # --------------------------------------------------------------------------
    def performActions(self, player, playerAction, enemyAction, oldPOb, oldEOb):
        """
        Performs the actions of the player and their opponent. Updates internal
        game state to reflect the new state of the game, and returns the new
        observations for the player and opponent
        """
        # Determine who is attacker and who is defender
        if (player == DEFENDER):
            defenderAction = playerAction.detach().numpy()
            attackerAction = enemyAction.detach().numpy()
            oldDOb = oldPOb
            oldAOb = oldEOb
        else:
            attackerAction = playerAction.detach().numpy()
            defenderAction = enemyAction.detach().numpy()
            oldAOb = oldPOb
            oldDOb = oldEOb

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

        # Determine who is attacker and who is defender
        if (player == DEFENDER):
            pObservation = dObservation
            eObservation = aObservation
        else:
            eObservation = dObservation
            pObservation = aObservation

        return (pObservation, eObservation)

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
        score = 0
        for targetIndex in range(len(dAction)):
            if aAction[targetIndex] and not dAction[targetIndex]:
                score -= defenderPenalties[targetIndex]
            elif aAction[targetIndex] and dAction[targetIndex]:
                score += defenderRewards[targetIndex]
        return score * player

    # --------------------------------------------------------------------------
    def getBestActionAndScore(self, player, eAction, defenderRewards, defenderPenalties):
        """
        Returns the best action and the score of the best action for a player given
        their opponent's action
        """
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

    # --------------------------------------------------------------------------
    def makeLegalMove(self, player, action):
        """
        Returns a move for the given player, modified to be legal
        """
        moveToPrune = action.detach().numpy()
        prunedMove = None
        # Zero out any actions that are impossible and re-normalize
        if player == DEFENDER:
            for targetIndex in range(len(moveToPrune)):
                if self.pastAttackStatuses[targetIndex]:
                    moveToPrune[targetIndex] = 0
            # Pick the highest n remaining values, where n is the number of resources left
            highest = np.argpartition(moveToPrune, -self.availableResources)[-self.availableResources:]
            if len(highest) == self.availableResources:
                for targetIndex in range(len(moveToPrune)):
                    if targetIndex in highest:
                        moveToPrune[targetIndex] = 1
                    else:
                        moveToPrune[targetIndex] = 0
            else:
                moveToPrune = np.array([0] * self.numTargets)
            prunedMove = torch.from_numpy(moveToPrune)
        else:
            for targetIndex in range(len(moveToPrune)):
                if self.pastAttackStatuses[targetIndex]:
                    moveToPrune[targetIndex] = 0
            maxValIndex = torch.argmax(torch.from_numpy(moveToPrune))
            prunedMove = torch.nn.functional.one_hot(maxValIndex, self.numTargets)
        return prunedMove

    # --------------------------------------------------------------------------
    def getPayout(self, defenderStrat, attackerStrat):
        """
        Returns the defender utility of the two given strategies played against
        each other.
        """
        aInput = attackerStrat.inputFromGame(self)
        dInput = defenderStrat.inputFromGame(self)
        aAction = [0]*self.numTargets
        dAction = [0]*self.numTargets
        dOb, aOb = self.getEmptyObservations()

        # Play a full game
        for timestep in range(self.timesteps):
            dAction = defenderStrat(dInput(dOb))
            dAction = self.makeLegalMove(DEFENDER, dAction)
            aAction = attackerStrat(aInput(aOb))
            aAction = self.makeLegalMove(ATTACKER, aAction)
            dOb, aOb = self.performActions(DEFENDER, dAction, aAction, dOb, aOb)

        payout = self.defenderUtility
        defenderStrat.reset()
        attackerStrat.reset()
        self.restartGame()

        return payout

    # --------------------------------------------------------------------------
    def getOracleScore(self, player, ids, map, mix, oracle, iterations=100):
        """
        Returns the oracle in the list with the highest utility against the mixed
        strategy specified, and its utility
        """
        bestUtility = None
        bestOracle = None

        # Calculate average utility for each oracle in the list
        avgUtility = 0
        pAgent = oracle
        pAgentIn = pAgent.inputFromGame(self)
        for iteration in range(iterations):
            playerAction = [0]*self.numTargets
            enemyAction = [0]*self.numTargets
            pOb, eOb = self.getEmptyObservations()

            # Choose the agent from the mixed strategy
            choice = np.random.choice(ids, 1,
            p=mix)[0]
            eAgent = map[choice]
            eAgentIn = eAgent.inputFromGame(self)

            # Play a full game
            for timestep in range(self.timesteps):
                eAction = eAgent(eAgentIn(eOb))
                eAction = self.makeLegalMove(-player, eAction)
                pAction = pAgent(pAgentIn(pOb))
                pAction = self.makeLegalMove(player, pAction)
                pOb, eOb = self.performActions(player, pAction, eAction, pOb, eOb)
            avgUtility += self.defenderUtility * player
            self.restartGame()

            for enemy in map.values():
                enemy.reset()
            oracle.reset()

        avgUtility = avgUtility / iterations
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













# THESE ARE LEFT OVER FROM TESTING. RIGID--DON'T WANT TO DELETE YET, BUT HARD TO
# MODIFY
# def testAttackerOracle(game, oracle, ids, map, mix, iterations, aMap):
#     print(f"TESTING ATTACKER ORACLE AGAINST MIXED DEFENDER")
#     oracleUtility = 0
#     correctness = 0
#     # print(f"Defender Dist: {mix}")
#     # Get an average utility for this oracle
#     for iteration in range(iterations):
#         aAction = [0]*game.numTargets
#         dAction = [0]*game.numTargets
#         dOb, aOb = game.getEmptyObservations()
#
#         # Select the observations according to the mixed strategy
#         choice = np.random.choice(ids, 1, p=mix)[0]
#         # print(f"Chosen defender agent: {choice}")
#         defenderAgent = map[choice]
#         dAgentInputFunction = defenderAgent.inputFromGame(game)
#
#         # Play a full game
#         for timestep in range(game.timesteps):
#             dAction = defenderAgent(dAgentInputFunction(dOb))
#             dAction = game.makeLegalDefenderMove(dAction)
#             aAgentInputFunction = oracle.inputFromGame(game)
#             aAction = oracle(aAgentInputFunction(aOb))
#             aAction = game.makeLegalAttackerMove(aAction)
#             best, _ = game.getBestActionAndScore(ATTACKER, dAction, game.defenderRewards, game.defenderPenalties)
#             compare = aAction.detach().numpy()
#             if np.array_equal(compare,best):
#                 correctness += 1
#             # print(f"Opponent action: {dAction}")
#             # print(f"action: {aAction}")
#             # print(f"best  : {best}")
#             # print()
#             dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
#         oracleUtility += game.attackerUtility
#         game.restartGame()
#         for defender in map.values():
#             defender.reset()
#         oracle.reset()
#     oracleUtility = oracleUtility / iterations
#     correctness = correctness / (iterations*game.timesteps)
#     # print(f"Correctness: {correctness}")
#     # Find the best average utility out of the other pure strategies
#     avgs = []
#     aIndex = 1
#     aLen = len(aMap.values())
#     for attacker in aMap.values():
#         # print(f"Working with attacker {aIndex} out of {aLen}")
#         avgUtility = 0
#         for iteration in range(iterations):
#             aAction = [0]*game.numTargets
#             dAction = [0]*game.numTargets
#             dOb, aOb = game.getEmptyObservations()
#
#             # Select the observations according to the mixed strategy
#             choice = np.random.choice(ids, 1,
#             p=mix)[0]
#             defenderAgent = map[choice]
#             dAgentInputFunction = defenderAgent.inputFromGame(game)
#
#             # Play a full game
#             for timestep in range(game.timesteps):
#                 dAction = defenderAgent(dAgentInputFunction(dOb))
#                 dAction = game.makeLegalDefenderMove(dAction)
#                 aAgentInputFunction = attacker.inputFromGame(game)
#                 aAction = attacker(aAgentInputFunction(aOb))
#                 aAction = game.makeLegalAttackerMove(aAction)
#                 dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
#             avgUtility += game.attackerUtility
#             game.restartGame()
#             for defender in map.values():
#                 defender.reset()
#             attacker.reset()
#         avgUtility = avgUtility / iterations
#         # print(f"attacker {aIndex} avg against defender: {avgUtility}")
#         aIndex += 1
#         avgs.append(avgUtility)
#     # print(f"Avg A oracle Utility against defender mix: {oracleUtility}")
#     # print(f"best average -- {max(avgs)}")
#     return oracleUtility, correctness, max(avgs)
#
# def testDefenderOracle(game, oracle, ids, map, mix, iterations, dMap):
#     oracleUtility = 0
#     correctness = 0
#     print(f"TESTING DEFENDER ORACLE AGAINST MIXED ATTACKER")
#     # Get an average utility for this oracle
#     # print(f"attacker Dist: {mix}")
#     for iteration in range(iterations):
#         aAction = [0]*game.numTargets
#         dAction = [0]*game.numTargets
#         dOb, aOb = game.getEmptyObservations()
#         # Play a full game
#         # print(f"game: {iteration}")
#         # Select the observations according to the mixed strategy
#         choice = np.random.choice(ids, 1, p=mix)[0]
#         # print(f"Chosen attacker agent: {choice}")
#         attackerAgent = map[choice]
#         aAgentInputFunction = attackerAgent.inputFromGame(game)
#
#         for timestep in range(game.timesteps):
#             # print(f"Timestep: {timestep}")
#             aAction = attackerAgent(aAgentInputFunction(aOb))
#             aAction = game.makeLegalAttackerMove(aAction)
#             dAgentInputFunction = oracle.inputFromGame(game)
#             dAction = oracle(dAgentInputFunction(dOb))
#             dAction = game.makeLegalDefenderMove(dAction)
#             best, _ = game.getBestActionAndScore(DEFENDER, aAction, game.defenderRewards, game.defenderPenalties)
#             compare = dAction.detach().numpy()
#             if np.array_equal(compare,best):
#                 correctness += 1
#             # print(f"Opponent action: {aAction}")
#             # print(f"action: {dAction}")
#             # print(f"best  : {best}")
#             # print()
#             dOb, aOb = game.performActions(dAction, aAction, dOb, aOb)
#         oracleUtility += game.defenderUtility
#         game.restartGame()
#         for attacker in map.values():
#             attacker.reset()
#         oracle.reset()
#     oracleUtility = oracleUtility / iterations
#     correctness = correctness / (iterations*game.timesteps)
#     # print(f"Correctness: {correctness}")
#     # print(f"Avg D oracle Utility against defender mix: {oracleUtility}")
#     return oracleUtility, correctness
