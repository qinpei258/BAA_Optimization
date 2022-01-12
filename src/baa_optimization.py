import logging
from typing import List, Callable

import numpy as np
from scipy.optimize import Bounds, minimize

logging.basicConfig()
logger = logging.getLogger(__name__)


class Optimization:

    def compute_optimal_strategy_budgets(self, daily_budget, strategies, business_constraints):
        """
        Computes the optimal budget allocation at strategy level by solving an optimization problem with constraints.
        :return: Optional[np.ndarray]: The solution of the optimization problem, ie. an array with the updated budgets
        and slack variables for every strategy
        """

        try:
            # Below variables are the output of the optimization algorithm. Relaxations variables are used in case the
            # problem is not solvable (too many constraints).

            max_budget = np.round(0.4 * daily_budget, 2)     # 0.4 is just a rule, in order that the spending-capability
                                                             # is max_budget for a strategy which doesn't have a s-c
            spending_capabilities = np.array([s['spending_capability'] if s['spending_capability'] is not
                                                                          None else max_budget for s in strategies])
            budgets: np.ndarray = np.ones(spending_capabilities.shape[0])  # initialize to non null values
            relaxations: np.ndarray = np.ones(budgets.shape[0])  # initialize to non null values
            n_budgets, n_relaxations = budgets.shape[0], relaxations.shape[0]  #number of budgets and number of relaxations,
                                                                               # normally they are of the same value

            x_0 = np.hstack((budgets, relaxations))     # list of the terms which are in the constraints

            kpi_values = np.array([s['cpc'] for s in strategies])     # the elements of kpi_values are cpc(cost per click)
            kpi_values[kpi_values == 0] = max(kpi_values)             # if a value = 0, set it to max(kpi_values)

            # Build objective function
            # For the informations of this fonction, see https://armis.slite.com/app/channels/qlJCQckITi/notes/Sc_LEdsBWQ

            kpi_coeff, sc_coeff = (1, 100)                            # mu = kpi_coeff, lambda = sc_coeff

            default_optim_matrix = lambda: np.zeros((n_budgets + n_relaxations, n_budgets + n_relaxations))
            # a matrix of size n_budgets + n_relaxations, which will be used seperately  for the two terms in the objective fonction

            kpi_matrix = default_optim_matrix()
            kpi_matrix[0:n_budgets, 0:n_relaxations] = np.diag(kpi_values) # set the values of cpc to diagonal elements of kpi_matrix to
            kpi_objective = lambda x: np.linalg.norm(x=np.dot(kpi_matrix, x), ord=1) # x is a list containing the values of allocated budgets
                                                                                     #kpi_objective is the sum of the product of CPC and AD for each s

            sc_matrix = default_optim_matrix()
            sc_matrix[n_budgets:, n_budgets:] = np.diag(1 / np.sqrt(np.clip(a=spending_capabilities,
                                                                            a_min=0.1, a_max=None)))
            sc_objective = lambda x: np.linalg.norm(x=np.dot(sc_matrix, x), ord=2)  # x is a list containing the values of relaxation terms
                                                                                    # sc_objecttive is the sum of the value of A divided by sc for each s

            objective = lambda x: kpi_coeff * kpi_objective(x) + sc_coeff * sc_objective(x) # the fonction that we want to minimize

            constraints = self._build_optimization_constraints(budgets=budgets,
                                                               relaxations=relaxations,
                                                               spending_capabilities=spending_capabilities,
                                                               strategies=strategies,
                                                               business_constraints=business_constraints,
                                                               daily_budget=daily_budget)   # build constraints

            res = minimize(fun=objective, x0=x_0, constraints=constraints[0], bounds=constraints[1], method='SLSQP')  # solve the optimization problem
            if not res.success:
                raise Exception(f'Scipy unsuccessful result: {res}. Result: {res.x}')
        except:
            raise Exception('something wrong')

        return np.round(res.x, 2)

    def _build_optimization_constraints(self, budgets: np.ndarray,
                                        relaxations: np.ndarray,
                                        spending_capabilities: np.ndarray,
                                        strategies,
                                        business_constraints,
                                        daily_budget):
        """
        Builds constraints for optimization problem to be solved with Scipy.
        More info here: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
        :param budgets: np.array. First variable of the problem, representing allocated budgets per strategy.
        :param relaxations: Second variable of the problem, representing slacks to relax the optimization problem.
        :param spending_capabilities: np.ndarray: the computed spending capabilities for each strategy.
        :return: Tuple[LinearConstraint, Bounds], representing the scipy constraints for the optimization problem.
        """

        live_strategy_ids = [s['strategy_id'] for s in strategies]
        n_budgets, n_relaxations = budgets.shape[0], relaxations.shape[0]
        n_variables = n_budgets + n_relaxations

        business_constraints = [{'type': 'ineq', 'fun': self._build_scipy_constraint(strategy_ids=live_strategy_ids,
                                                                                     business_constraint=c['business_constraint'])}
                                for c in business_constraints]

        sc_constraints = [{'type': 'ineq', 'fun': lambda x: np.array([sc + x[i + n_budgets] - x[i] for i, sc in
                                                                      enumerate(spending_capabilities)])}]
        daily_budget_constraint = [
            {'type': 'eq', 'fun': lambda x: np.array([np.sum(x[0:n_budgets]) - daily_budget])}]
        positivity_constraints: Bounds = Bounds([0] * n_variables, [np.inf] * n_variables)

        return business_constraints + daily_budget_constraint + sc_constraints, positivity_constraints

    @staticmethod
    def _build_scipy_constraint(strategy_ids: List[int], business_constraint) -> Callable:
        constrained_strategy_indices = [strategy_ids.index(int(s)) for s in business_constraint['strategy_ids']]
        return lambda x: np.sum(x[constrained_strategy_indices]) - business_constraint['total_min_budget']
