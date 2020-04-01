# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model
import numpy as np

# Internal Imports
import ssg

# Set the random seed
rnd = np.random
rnd.seed(0)

# ==============================================================================
# CLASSES
# ==============================================================================
def createDefenderModel(aIds, aMap, dIds, dMap, payoutMatrix):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mD = Model('defenderSolver')
    # Define the decision variables
    vD = mD.continuous_var(lb=float("-inf"),name='defenderUtility')                                  # The defender utility
    xD = mD.continuous_var_dict(keys=dIds) # The distribution over the defender pool
    # Constraints
    mD.add_constraint(sum(xD.values()) == 1)
    mD.add_constraints(xVal <= 1 for xVal in xD.values())
    mD.add_constraints(xVal >= 0 for xVal in xD.values())
    mD.add_constraints(vD <= sum([xD[dId] * payoutMatrix[dId, aId] for dId in dIds]) for aId in aIds)

    mD.maximize(vD)
    return mD, xD, vD

def createAttackerModel(aIds, aMap, dIds, dMap, payoutMatrix):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mA = Model('attackerSolver')
    # Define the decision variables
    vA = mA.continuous_var(lb=float("-inf"), name='attackerUtility')                                  # The attacker utility
    xA = mA.continuous_var_dict(keys=aIds) # The distribution over the attacker pool
    # Constraints
    mA.add_constraint(sum(xA.values()) == 1)
    mA.add_constraints(xVal <= 1 for xVal in xA.values())
    mA.add_constraints(xVal >= 0 for xVal in xA.values())
    mA.add_constraints(vA <= sum([xA[aId] * -payoutMatrix[dId, aId] for aId in aIds]) for dId in dIds)

    mA.maximize(vA)
    return mA, xA, vA

def createDefenderOneShotModel(game):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mD = Model('defenderSolver')

    attackerActions = game.getValidActions(ssg.ATTACKER)
    defenderActions = game.getValidActions(ssg.DEFENDER)
    defenderActions = [tuple(x) for x in defenderActions]

    rewards = game.defenderRewards
    penalties = game.defenderPenalties

    # Define the decision variables
    vD = mD.continuous_var(lb=float("-inf"),name='defenderUtility') # The defender utility
    xD = mD.continuous_var_dict(keys=defenderActions) # The distribution over the defender pool
    # Constraints
    mD.add_constraint(sum(xD.values()) == 1)
    mD.add_constraints(xVal <= 1 for xVal in xD.values())
    mD.add_constraints(xVal >= 0 for xVal in xD.values())
    mD.add_constraints(vD <= sum([xD[dAction] * game.getActionScore(ssg.DEFENDER, list(dAction), aAction, rewards, penalties) for dAction in defenderActions]) for aAction in attackerActions)

    mD.maximize(vD)
    return mD, xD, [list(action) for action in defenderActions]

def createAttackerOneShotModel(game):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mA = Model('attackerSolver')

    attackerActions = game.getValidActions(ssg.ATTACKER)
    attackerActions = [tuple(x) for x in attackerActions]
    defenderActions = game.getValidActions(ssg.DEFENDER)

    rewards = game.defenderRewards
    penalties = game.defenderPenalties

    # Define the decision variables
    vA = mA.continuous_var(lb=float("-inf"),name='attackerUtility') # The attacker utility
    xA = mA.continuous_var_dict(keys=attackerActions) # The distribution over the attacker pool
    # Constraints
    mA.add_constraint(sum(xA.values()) == 1)
    mA.add_constraints(xVal <= 1 for xVal in xA.values())
    mA.add_constraints(xVal >= 0 for xVal in xA.values())
    mA.add_constraints(vA <= sum([xA[aAction] * game.getActionScore(ssg.ATTACKER, dAction, list(aAction), rewards, penalties) for aAction in attackerActions]) for dAction in defenderActions)

    mA.maximize(vA)
    return mA, xA, [list(action) for action in attackerActions]

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    pass

if __name__ == "__main__":
    main()
