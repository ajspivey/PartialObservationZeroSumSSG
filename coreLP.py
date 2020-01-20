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
class CoreLP():
    def __init__(self):
        pass

    def createModel(attackerPool, defenderPool, payoffMatrix):
        """Creates a model for solving the equilibrium of an ssg with a restricted
        domain of pure strategies (trained neural networks) for each player."""
        # Create a cplex model
        mD = Model('defenderSolver')
        # Define the decision variables
        vD = mD.continuous_var(name='defenderUtility')                                  # The defender utility
        xD = mD.continuous_var_dict(keys=defenderPool, name='defenderDistributionDict') # The distribution over the defender pool
        # The objective function
        # Constraints
        print(sum(xD.values()))
        mD.add_constraint(sum(xD.values()) == 1)
        mD.add_constraints(xVal <= 1 for xVal in xD.values())
        mD.add_constraints(xVal >= 0 for xVal in xD.values())
        mD.add_constraints(vD <= sum([xD[defenderPure] * payoffMatrix[defenderPure, attackerPure] for defenderPure in defenderPool]) for attackerPure in attackerPool)

        mD.maximize(vD)
        # Add constraints for the objective function
        #   TODO: Determine the utility for each strategy with function U
        #         (simulates games) between two neural networks
        # Add the rest of the constraints
        return mD, xD

    def getPhoneDeskProblemModel():
        # create one model instance, with a name
        m = Model(name='telephone_production')
        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        desk = m.continuous_var(name='desk')
        cell = m.continuous_var(name='cell')
        # write constraints
        # constraint #1: desk production is greater than 100
        m.add_constraint(desk >= 100)

        # constraint #2: cell production is greater than 100
        m.add_constraint(cell >= 100)

        # constraint #3: assembly time limit
        ct_assembly = m.add_constraint( 0.2 * desk + 0.4 * cell <= 400)

        # constraint #4: paiting time limit
        ct_painting = m.add_constraint( 0.5 * desk + 0.4 * cell <= 490)

        m.maximize(12 * desk + 20 * cell)
        m.print_information()
        return m

    def getCVRPProblemModel(n=10,Q=15):
        N = [i for i in range(1,n+1)]
        V = [0] + N
        q = {i:rnd.randint(1,10) for i in N}

        loc_x = rnd.rand(len(V))*200
        loc_y = rnd.rand(len(V))*100

        A = [(i,j) for i in V for j in V if i!=j]
        c = {(i,j):np.hypot(loc_x[i]-loc_x[j],loc_y[i]-loc_y[j]) for i,j in A}

        mdl = Model('CVRP')
        x = mdl.binary_var_dict(A,name='x')
        u = mdl.continuous_var_dict(N,ub=Q,name='u')
        mdl.minimize(mdl.sum(c[i,j]*x[i,j] for i,j in A))
        mdl.add_constraints(mdl.sum(x[i,j] for j in V if j!=i)==1 for i in N)
        mdl.add_constraints(mdl.sum(x[i,j] for i in V if i!=j)==1 for j in N)
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j],u[i]+q[j]==u[j]) for i,j in A if i!=0 and j!=0)
        mdl.add_constraints(u[i]>=q[i] for i in N)
        return mdl

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    dPool = ["a","b","c"]
    aPool = ["e","f","g"]
    payMatrix = {
        ("a","e"): 1,
        ("a","f"): 2,
        ("a","g"): 3,
        ("b","e"): 3,
        ("b","f"): 2,
        ("b","g"): 1,
        ("c","e"): 2,
        ("c","f"): 2,
        ("c","g"): 2,
    }
    core, xD = CoreLP.createModel(aPool, dPool, payMatrix)
    solution = core.solve(log_output=True)
    core.export_as_lp(path=".\\foo")
    print(solution)
    print([float(value) for value in xD.values()])

if __name__ == "__main__":
    main()
