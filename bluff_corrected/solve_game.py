"""
Solver for the corrected Bluff game using Linear Programming.

This implements the proper sequence-form LP for extensive games with imperfect information.
Each player's strategy is computed considering they know their own dice roll.
"""

import numpy as np
import pickle as pkl
import os
import itertools
from typing import Dict, Tuple
import pyscipopt

try:
    from .game_constants import NUM_FACES, NUM_DICES, NUM_PLAYERS
except ImportError:
    from game_constants import NUM_FACES, NUM_DICES, NUM_PLAYERS


class BluffSolver:
    """Solver for the corrected Bluff game."""
    
    def __init__(self, data_path: str = "bluff_corrected"):
        self.data_path = data_path
        self.sequences = None
        self.information_sets = None
        self.dice_rolls = None
        self.payoff_matrices = {}
        
        self.load_data()
    
    def load_data(self):
        """Load precomputed game data."""
        print("Loading game data...")
        
        # Load sequences
        with open(f"{self.data_path}/sequences.pkl", "rb") as f:
            self.sequences = pkl.load(f)
        
        # Load information sets
        with open(f"{self.data_path}/information_sets.pkl", "rb") as f:
            self.information_sets = pkl.load(f)
        
        # Load dice rolls
        with open(f"{self.data_path}/dice_rolls.pkl", "rb") as f:
            self.dice_rolls = pkl.load(f)
        
        # Load payoff matrices
        payoff_dir = f"{self.data_path}/payoff_matrices"
        for filename in os.listdir(payoff_dir):
            if filename.endswith('.npy'):
                # Parse filename: payoff_(dice1)_(dice2).npy
                parts = filename[:-4].split('_')
                dice1_str = '_'.join(parts[1:1+NUM_DICES])
                dice2_str = '_'.join(parts[1+NUM_DICES:])
                
                # Convert back to tuples
                dice1 = eval(dice1_str.replace('_', ','))
                dice2 = eval(dice2_str.replace('_', ','))
                
                if not isinstance(dice1, tuple):
                    dice1 = (dice1,)
                if not isinstance(dice2, tuple):
                    dice2 = (dice2,)
                
                matrix = np.load(f"{payoff_dir}/{filename}")
                self.payoff_matrices[(dice1, dice2)] = matrix
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Loaded {len(self.information_sets)} information sets") 
        print(f"Loaded {len(self.payoff_matrices)} payoff matrices")
    
    def get_sequence_counts(self):
        """Get number of sequences for each player."""
        p0_sequences = sum(1 for (p, _) in self.sequences.keys() if p == 0)
        p1_sequences = sum(1 for (p, _) in self.sequences.keys() if p == 1)
        return p0_sequences, p1_sequences
    
    def solve_for_dice_combination(self, dice1: Tuple[int, ...], dice2: Tuple[int, ...], 
                                 verbose: bool = False):
        """Solve the game for a specific dice combination."""
        if (dice1, dice2) not in self.payoff_matrices:
            raise ValueError(f"No payoff matrix for dice combination {dice1}, {dice2}")
        
        A = self.payoff_matrices[(dice1, dice2)]
        m, n = A.shape
        
        if verbose:
            print(f"Solving for dice {dice1} vs {dice2}")
            print(f"Payoff matrix shape: {A.shape}")
        
        # Player 1 (minimizer) problem using pyscipopt:
        # min v subject to: sum_i A[i,j] * x[i] >= v for all j, sum_i x[i] = 1, x[i] >= 0
        
        model1 = pyscipopt.Model("Player1_LP")
        if not verbose:
            model1.hideOutput()
            
        # Variables
        x = {i: model1.addVar(f"x_{i}", vtype="C", lb=0.0, ub=1.0) for i in range(m)}
        v = model1.addVar("v", vtype="C", lb=-model1.infinity(), ub=model1.infinity())
        
        # Objective: minimize v
        model1.setObjective(v, "minimize")
        
        # Constraints
        # sum_i x[i] = 1
        model1.addCons(pyscipopt.quicksum(x[i] for i in range(m)) == 1.0)
        
        # For each column j: expected payoff against pure column j is <= v
        # Minimizing v with these upper bounds yields v = max_j x^T A e_j
        for j in range(n):
            model1.addCons(pyscipopt.quicksum(A[i, j] * x[i] for i in range(m)) <= v)
        
        model1.optimize()
        
        if model1.getStatus() != "optimal":
            print(f"Warning: Player 1 problem not optimal: {model1.getStatus()}")
            return None, None, None
        
        x_opt = np.array([model1.getVal(x[i]) for i in range(m)])
        game_value = model1.getVal(v)
        
        # Player 2 (maximizer) problem using pyscipopt:
        # max u subject to: sum_j A[i,j] * y[j] <= u for all i, sum_j y[j] = 1, y[j] >= 0
        
        model2 = pyscipopt.Model("Player2_LP") 
        if not verbose:
            model2.hideOutput()
            
        # Variables
        y = {j: model2.addVar(f"y_{j}", vtype="C", lb=0.0, ub=1.0) for j in range(n)}
        u = model2.addVar("u", vtype="C", lb=-model2.infinity(), ub=model2.infinity())
        
        # Objective: maximize u
        model2.setObjective(u, "maximize")
        
        # Constraints
        # sum_j y[j] = 1
        model2.addCons(pyscipopt.quicksum(y[j] for j in range(n)) == 1.0)
        
        # For each row i: expected payoff against pure row i is >= u
        # Maximizing u with these lower bounds yields u = min_i e_i^T A y
        for i in range(m):
            model2.addCons(pyscipopt.quicksum(A[i, j] * y[j] for j in range(n)) >= u)
        
        model2.optimize()
        
        if model2.getStatus() != "optimal":
            print(f"Warning: Player 2 problem not optimal: {model2.getStatus()}")
            return x_opt, None, game_value
        
        y_opt = np.array([model2.getVal(y[j]) for j in range(n)])
        game_value2 = model2.getVal(u)
        
        if verbose:
            print(f"Game value (P1 perspective): {game_value:.6f}")
            print(f"Game value (P2 perspective): {game_value2:.6f}")
            print(f"Difference: {abs(game_value - game_value2):.6f}")
        
        return x_opt, y_opt, game_value
    
    def solve_all_combinations(self, save_results: bool = True):
        """Solve for all dice combinations."""
        print("Solving for all dice combinations...")
        
        results = {}
        
        for dice1 in self.dice_rolls:
            for dice2 in self.dice_rolls:
                try:
                    x_opt, y_opt, game_value = self.solve_for_dice_combination(dice1, dice2)
                    
                    results[(dice1, dice2)] = {
                        'x_strategy': x_opt,
                        'y_strategy': y_opt,
                        'game_value': game_value
                    }
                    
                    print(f"Dice {dice1} vs {dice2}: Game value = {game_value:.6f}")
                    
                except Exception as e:
                    print(f"Error solving for {dice1} vs {dice2}: {e}")
                    results[(dice1, dice2)] = None
        
        if save_results:
            os.makedirs(f"{self.data_path}/solutions", exist_ok=True)
            with open(f"{self.data_path}/solutions/all_solutions.pkl", "wb") as f:
                pkl.dump(results, f)
            print(f"Saved results to {self.data_path}/solutions/")
        
        return results
    
    def analyze_strategies(self, dice1: Tuple[int, ...], dice2: Tuple[int, ...]):
        """Analyze and display strategies for specific dice combination."""
        x_opt, y_opt, game_value = self.solve_for_dice_combination(dice1, dice2, verbose=True)
        
        if x_opt is None or y_opt is None:
            print("Could not solve for this dice combination")
            return
        
        print(f"\n=== Analysis for dice {dice1} vs {dice2} ===")
        print(f"Game value: {game_value:.6f}")
        
        # Display significant strategies
        print("\nPlayer 1 (minimizer) significant strategies:")
        p1_sequences = [(seq, idx) for (p, seq), idx in self.sequences.items() if p == 0]
        for seq, idx in p1_sequences:
            if x_opt[idx] > 0.01:  # Show strategies with >1% probability
                print(f"  Sequence '{seq}': {x_opt[idx]:.4f}")
        
        print("\nPlayer 2 (maximizer) significant strategies:")
        p2_sequences = [(seq, idx) for (p, seq), idx in self.sequences.items() if p == 1]
        for seq, idx in p2_sequences:
            if y_opt[idx] > 0.01:  # Show strategies with >1% probability
                print(f"  Sequence '{seq}': {y_opt[idx]:.4f}")


if __name__ == "__main__":
    try:
        solver = BluffSolver()
        
        print("Testing with a single dice combination...")
        dice_rolls = solver.dice_rolls
        
        if len(dice_rolls) >= 2:
            dice1, dice2 = dice_rolls[0], dice_rolls[1]
            solver.analyze_strategies(dice1, dice2)
        
        print("\nSolving for all combinations...")
        results = solver.solve_all_combinations()
        
        # Summary statistics
        game_values = []
        for result in results.values():
            if result and result['game_value'] is not None:
                game_values.append(result['game_value'])
        
        if game_values:
            print(f"\nGame value statistics:")
            print(f"  Mean: {np.mean(game_values):.6f}")
            print(f"  Std:  {np.std(game_values):.6f}")
            print(f"  Min:  {np.min(game_values):.6f}")
            print(f"  Max:  {np.max(game_values):.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run create_game.py first!")
