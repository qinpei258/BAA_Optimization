from baa_optimization import Optimization
from input_generator import generate_daily_budget, generate_strategies, generate_business_constrains

if __name__ == '__main__':
    optimization = Optimization()

    daily_budget = generate_daily_budget()
    strategies = generate_strategies()
    business_constraints = generate_business_constrains()

    budgets = optimization.compute_optimal_strategy_budgets(daily_budget, strategies, business_constraints)

    print(budgets)
