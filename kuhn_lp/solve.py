import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyscipopt as scip

def solve_kuhn(n_players: int=2):
    """
    Function to solve the Kuhn poker game using linear programming.
    """
    A = pd.read_csv('kuhn_lp/kuhn.csv', header=None).values
    m, n = A.shape

    model = scip.Model("KuhnPoker")
    
    x = {}
    for i in range(m):
        x[i] = model.addVar(f"x_{i}", vtype="C", lb=0.0, ub=1.0)
    
    v = model.addVar("v", vtype="C", lb=-model.infinity(), ub=model.infinity())
    
    model.setObjective(v, "maximize")
    
    for j in range(n):
        model.addCons(scip.quicksum(A[i, j] * x[i] for i in range(m)) >= v)

    model.addCons(scip.quicksum(x[i] for i in range(m)) == 1.0)
    
    # Solve the model
    model.optimize()
    
    # Extract solution
    if model.getStatus() == "optimal":
        game_value = model.getVal(v)
        strategy = np.array([model.getVal(x[i]) for i in range(m)])
        
        print(f"Game value: {game_value:.6f}")
        print(f"Player 1 optimal strategy: {strategy}")
        
        dual_model = scip.Model("KuhnPokerDual")
        
        y = {}
        for j in range(n):
            y[j] = dual_model.addVar(f"y_{j}", vtype="C", lb=0.0, ub=1.0)
        
        u = dual_model.addVar("u", vtype="C", lb=-dual_model.infinity(), ub=dual_model.infinity())
        
        dual_model.setObjective(u, "minimize")
        
        for i in range(m):
            dual_model.addCons(scip.quicksum(A[i, j] * y[j] for j in range(n)) <= u)

        dual_model.addCons(scip.quicksum(y[j] for j in range(n)) == 1.0)
        
        dual_model.optimize()
        
        if dual_model.getStatus() == "optimal":
            dual_game_value = dual_model.getVal(u)
            dual_strategy = np.array([dual_model.getVal(y[j]) for j in range(n)])
            
            print(f"Dual game value: {dual_game_value:.6f}")
            print(f"Player 2 optimal strategy: {dual_strategy}")
            print(f"Duality gap: {abs(game_value - dual_game_value):.8f}")
            
            # Save
            np.save('kuhn_lp/strategies/p1_strategy.npy', strategy)
            np.save('kuhn_lp/strategies/p2_strategy.npy', dual_strategy)

            return {
                'game_value': game_value,
                'player1_strategy': strategy,
                'player2_strategy': dual_strategy,
                'payoff_matrix': A
            }
        else:
            print("Dual problem could not be solved optimally")
            return None
    else:
        print("Primal problem could not be solved optimally")
        return None

def interpret_kuhn_strategies(result):
    """
    Interpret the optimal strategies in terms of Kuhn poker actions.
    """
    if result is None:
        return
    
    print("\n" + "="*60)
    print("KUHN POKER STRATEGY INTERPRETATION")
    print("="*60)
    
    p1_strategy = result['player1_strategy']
    p2_strategy = result['player2_strategy']
    
    print(f"Game Value: {result['game_value']:.6f}")
    print(f"(Positive favors Player 1, Negative favors Player 2)")
    
    print(f"\nPlayer 1 Strategy (length: {len(p1_strategy)}):")
    for i, prob in enumerate(p1_strategy):
        if prob > 0.001:  # Only show non-zero strategies
            print(f"  Information set {i}: {prob:.4f}")
    
    print(f"\nPlayer 2 Strategy (length: {len(p2_strategy)}):")
    for j, prob in enumerate(p2_strategy):
        if prob > 0.001:  # Only show non-zero strategies
            print(f"  Information set {j}: {prob:.4f}")

if __name__ == "__main__":
    # Solve Kuhn poker
    result = solve_kuhn()
    
    # Interpret the results
    if result:
        interpret_kuhn_strategies(result)
        
        # Expected game value for Kuhn poker is -1/18 ≈ -0.0556
        theoretical_value = -1/18
        computed_value = result['game_value']
        
        print(f"\nComparison with theoretical result:")
        print(f"Theoretical game value: {theoretical_value:.6f}")
        print(f"Computed game value: {computed_value:.6f}")
        print(f"Difference: {abs(theoretical_value - computed_value):.8f}")


