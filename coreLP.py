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
def createDefenderModel(attackerPool, defenderPool, payoutMatrix):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mD = Model('defenderSolver')
    # Define the decision variables
    vD = mD.continuous_var(name='defenderUtility')
    xD = mD.continuous_var_dict(keys=defenderPool) # The distribution over the defender pool
    # The objective function
    # Constraints
    mD.add_constraint(sum(xD.values()) == 1)
    mD.add_constraints(xVal <= 1 for xVal in xD.values())
    mD.add_constraints(xVal >= 0 for xVal in xD.values())
    mD.add_constraints(vD <= sum([xD[defenderPure] * payoutMatrix[defenderPure, attackerPure] for defenderPure in defenderPool]) for attackerPure in attackerPool)

    mD.maximize(vD)
    return mD, xD

def createAttackerModel(attackerPool, defenderPool, payoutMatrix):
    """Creates a model for solving the equilibrium of an ssg with a restricted
    domain of pure strategies (trained neural networks) for each player."""
    # Create a cplex model
    mA = Model('attackerSolver')
    # Define the decision variables
    vA = mA.continuous_var(name='attackerUtility')                                  # The attacker utility
    xA = mA.continuous_var_dict(keys=attackerPool) # The distribution over the attacker pool
    # Constraints
    mA.add_constraint(sum(xA.values()) == 1)
    mA.add_constraints(xVal <= 1 for xVal in xA.values())
    mA.add_constraints(xVal >= 0 for xVal in xA.values())
    mA.add_constraints(vA <= sum([xA[attackerPure] * payoutMatrix[defenderPure, attackerPure] for attackerPure in attackerPool]) for defenderPure in defenderPool)

    mA.maximize(vA)
    return mA, xA

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    dPool = ["a","b","c"]
    aPool = ["e","f","g"]
    payMatrix = {
        ("a","e"): 1,
        ("a","f"): -1,
        ("a","g"): 0,
        ("b","e"): 0,
        ("b","f"): 2,
        ("b","g"): 0,
        ("c","e"): 2,
        ("c","f"): -2,
        ("c","g"): -1,
    }
    core, xD = createDefenderModel(aPool, dPool, payMatrix)
    solution = core.solve(log_output=True)
    core.export_as_lp(path=".\\foo")
    print(solution)
    print([float(value) for value in xD.values()])

if __name__ == "__main__":
    main()
