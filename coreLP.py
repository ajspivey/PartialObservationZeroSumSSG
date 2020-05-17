# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
from docplex.mp.model import Model

# ==============================================================================
# CLASSES
# ==============================================================================
def createDefenderModel(dIds, dMap, aIds, aMap, payoutMatrix, export=False):
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
    if export:
        mD.export("defenderModel.lp")
    return mD, xD, vD

def getDefenderMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export=False):
    defenderModel, dStrategyDistribution, dUtility = createDefenderModel(dIds, dMap, aIds, aMap, payoutMatrix, export)
    defenderModel.solve()
    defenderMixedStrategy = [float(value) for value in dStrategyDistribution.values()]
    dUtility = float(dUtility)
    return defenderMixedStrategy, dUtility

def createAttackerModel(dIds, dMap, aIds, aMap, payoutMatrix, export=False):
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
    if export:
        mA.export("attackerModel.lp")
    return mA, xA, vA

def getAttackerMixedStrategy(dIds, dMap, aIds, aMap, payoutMatrix, export=False):
    attackerModel, aStrategyDistribution, aUtility = createAttackerModel(dIds, dMap, aIds, aMap, payoutMatrix, export)
    attackerModel.solve()
    attackerMixedStrategy = [float(value) for value in aStrategyDistribution.values()]
    aUtility = float(aUtility)
    return attackerMixedStrategy, aUtility
