from itertools import combinations
import numpy as np

class SequentialZeroSumSSG(object):
    def __init__(self, numTargets, numResources, targetRewards):
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
        # Destroyed targets are a 0, otherwise a 1
        self.targets = [1] * numTargets
        # Defender placements are 0 if nothing is placed, and 1 if resources are placed.
        self.resourcePlacements = [0] * numResources
        # Resources are consumed if they stop an attack. This tracks remaining resources
        self.availableResources = numResources

        #
        # Features
        #


    def place_ones(self, size, count):
        for positions in combinations(range(size), count):
            p = [0] * size
            for i in positions:
                p[i] = 1
            yield p

    def getValidActions(self, player):
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
                    actions.append(action)
            return actions
        raise unknownPlayerError(f"Player is not Attacker or Defender. Player {player} unknown")

    def performActions(self, defenderAction, attackerAction):
        self.resourcePlacements = defenderAction
        oldResourceCount = sum(self.availableResources)
        self.availableResources = np.multiply(attackerAction,defenderAction)
        # Target Defended
        if oldResourceCount > self.availableResources:
            pass
        # Target Destroyed
        else:
            pass
    def getPlayerObservations(self, defenderAction, attackerAction):
        """
        Args:
            defenderAction: A binary vector representing the defender's action
            attackerAction: A one-hot vector representing the attacker's action
        Returns:
            A vector pair representing the attacker's observation and the defender's
            observation.
        """
        defenderObservation = [] + defenderAction + self.resourcePlacements + self.pastAttacks + self.pastAttackStatuses + self.targetRewards
        attackerObservation = [] + attackerAction + self.pastAttacks + self.pastAttackStatuses + self.targetRewards
        # Calculate the outcome of the turn
        for _ in range(self.numTargets):
            # Resource attacked & defended
            if defenderAction[_] and attackerAction[_]:
                pass
            # Resource attacked & destroyed
            if not defenderAction[_] and attackerAction[_]:
                pass
        return (defenderObservation, attackerObservation)
