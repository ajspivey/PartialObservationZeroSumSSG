# ==============================================================================
# IMPORTS
# ==============================================================================
from itertools import combinations
import numpy as np

# ==============================================================================
# CLASSES
# ==============================================================================
class SequentialZeroSumSSG(object):
    def __init__(self, numTargets, numResources, targetRewards, timesteps):
        """Initialization method for oracle.
        Args:
            numTargets: The number of targets the defender must defend
            numResources: the number of defender resources available
            targetRewards: the reward (or penalty) for a succesful target attack
        """
        #
        # CONSTANTS
        #
        self.DEFENDER = 0
        self.ATTACKER = 1
        #
        # meta values
        #
        self.numTargets = numTargets
        self.numResources = numResources
        self.targetRewards = targetRewards
        self.timesteps = timesteps

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
        self.previousAttackerObservation = np.array([0]*self.numTargets*4)
        self.previousDefenderObservation = np.array([0]*self.numTargets*4)
        self.pastAttacks = [0]*self.numTargets
        self.pastAttackStatuses = [0]*self.numTargets

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
        if (player == self.DEFENDER):
            currentResources = self.resourcePlacements
            allResourcePlacements = list(self.place_ones(self.numTargets, self.availableResources))
            return [placements for placements in allResourcePlacements if sum(np.multiply(self.targets,placements)) == self.availableResources]
        elif (player == self.ATTACKER):
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
    def performActions(self, defenderAction, attackerAction):
        """ Performs a step of the game, using the given actions """
        self.currentTimestep += 1
        attackStatus = 1 - sum(np.multiply(attackerAction,defenderAction))
        attackedTarget = np.where(attackerAction==1)[0][0]
        self.availableResources = self.availableResources - attackStatus
        self.pastAttacks[attackedTarget] = self.currentTimestep
        self.pastAttackStatuses = np.add(self.pastAttackStatuses, np.multiply(attackerAction, attackStatus))
        # Update actions and observations
        dObservation, aObservation = self.getPlayerObservations(defenderAction, attackerAction)

        self.previousAttackerObservation = aObservation
        self.previousDefenderObservation = dObservation
        self.previousAttackerAction = attackerAction
        self.previousDefenderAction = defenderAction

    def getPlayerObservations(self, defenderAction, attackerAction):
        """
        Args:
            defenderAction: A binary vector representing the defender's action
            attackerAction: A one-hot vector representing the attacker's action
        Returns:
            A vector pair representing the attacker's observation and the defender's
            observation.
        """
        # if self.currentTimestep == 0:
        #     defenderObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.targetRewards))
        #     attackerObservation = np.concatenate(([0]*self.numTargets, [0]*self.numTargets, [0]*self.numTargets, self.targetRewards))
        #     return (defenderObservation, attackerObservation)
        # else:
        defenderObservation = np.concatenate((defenderAction, self.pastAttacks, self.pastAttackStatuses, self.targetRewards))
        attackerObservation = np.concatenate((attackerAction, self.pastAttacks, self.pastAttackStatuses, self.targetRewards))
        # Calculate the outcome of the turn
        for _ in range(self.numTargets):
            # Resource attacked & defended
            if defenderAction[_] and attackerAction[_]:
                pass
            # Resource attacked & destroyed
            if not defenderAction[_] and attackerAction[_]:
                pass
        return (defenderObservation, attackerObservation)


    # -------------
    # Action Scores
    # -------------
    def getActionScore(self, player, aAction, dAction, rewards):
        score = 0
        for targetIndex in range(len(dAction)):
            if aAction[targetIndex] and not dAction[targetIndex]:
                score += rewards[targetIndex]
        if player == self.DEFENDER:
            score = score * -1
        return score

    def getBestActionAndScore(self, player, eAction, rewards):
        actions = self.getValidActions(player)
        bestAction = actions[0]
        bestActionScore = float("-inf")
        for action in actions:
            dAction = action
            aAction = eAction
            if (player == self.ATTACKER):
                dAction = eAction
                aAction = action
            actionScore = self.getActionScore(player, aAction, dAction, rewards)
            if actionScore > bestActionScore:
                bestActionScore = actionScore
                bestAction = action
        return (np.array(bestAction), bestActionScore)
