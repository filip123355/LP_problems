import numpy as np
import itertools
from bluff_cfr.cfr import CFRTrainer, Node
from bluff_cfr.constants import (N_DICES, N_DIE_SIDES, N_PLAYERS)

def test_nash_equilibrium_property(trainer: CFRTrainer, avg_strat: dict, tolerance=1e-3):
    """
    Test 1: Nash Equilibrium Property
    In Nash equilibrium, no player can improve by unilaterally changing strategy.
    For each infoset, check if any single action can beat the current mixed strategy.
    """
    print("=== Testing Nash Equilibrium Property ===")
    violations = 0
    
    for infoset_key, node in trainer.nodes.items():
        parts = infoset_key.split("|")
        player = int(parts[0])
        dice = eval(parts[1])  
        history = [int(x) for x in parts[2].split(",")] if parts[2] else []
        
        mixed_value = compute_infoset_value(trainer, avg_strat, player, dice, history)
        
        actions = get_valid_actions(trainer, history)
        current_strategy = node.get_average_strategy()
        
        for i, action in enumerate(actions):
            pure_strategy = np.zeros(len(actions))
            pure_strategy[i] = 1.0
            
            pure_value = compute_action_value(trainer, avg_strat, player, dice, history, action)
            
            if pure_value > mixed_value + tolerance:
                print(f"Nash violation at {infoset_key}: action {action} gives {pure_value:.6f} vs mixed {mixed_value:.6f}")
                violations += 1
    
    print(f"Nash equilibrium violations: {violations}")
    return violations == 0

def test_dominated_strategies(trainer: CFRTrainer, avg_strat: dict, tolerance=1e-3):
    """
    Test 3: No Dominated Strategies
    Optimal strategies should not assign positive probability to strictly dominated actions.
    """
    print("=== Testing for Dominated Strategies ===")
    dominated_violations = 0
    
    for infoset_key, node in trainer.nodes.items():
        parts = infoset_key.split("|")
        player = int(parts[0])
        dice = eval(parts[1])
        history = [int(x) for x in parts[2].split(",")] if parts[2] else []
        
        actions = get_valid_actions(trainer, history)
        strategy = node.get_average_strategy()
        
        for i, action_i in enumerate(actions):
            if strategy[i] > tolerance: 
                value_i = compute_action_value(trainer, avg_strat, player, dice, history, action_i)
                
                for j, action_j in enumerate(actions):
                    if i != j:
                        value_j = compute_action_value(trainer, avg_strat, player, dice, history, action_j)
                        
                        if value_j > value_i + tolerance:
                            print(f"Dominated strategy at {infoset_key}: action {action_i} (prob {strategy[i]:.4f}) dominated by {action_j}")
                            dominated_violations += 1
                            break
    
    print(f"Dominated strategy violations: {dominated_violations}")
    return dominated_violations == 0

def test_probability_consistency(trainer: CFRTrainer, avg_strat: dict, tolerance=1e-6):
    """
    Test 4: Probability Consistency
    All strategies should sum to 1 and be non-negative.
    """
    print("=== Testing Probability Consistency ===")
    consistency_violations = 0
    
    for infoset_key, strategy in avg_strat.items():
        if np.any(strategy < -tolerance):
            print(f"Negative probability at {infoset_key}: {strategy}")
            consistency_violations += 1
        
        if abs(np.sum(strategy) - 1.0) > tolerance:
            print(f"Probabilities don't sum to 1 at {infoset_key}: sum = {np.sum(strategy)}")
            consistency_violations += 1
    
    print(f"Probability consistency violations: {consistency_violations}")
    return consistency_violations == 0

def test_convergence_stability(trainer1: CFRTrainer, trainer2: CFRTrainer, tolerance=1e-2):
    """
    Test 5: Convergence Stability
    Compare strategies from two different CFR runs with more iterations.
    They should converge to the same solution.
    """
    print("=== Testing Convergence Stability ===")
    
    common_keys = set(trainer1.nodes.keys()) & set(trainer2.nodes.keys())
    max_diff = 0.0
    significant_diffs = 0
    
    for key in common_keys:
        strat1 = trainer1.nodes[key].get_average_strategy()
        strat2 = trainer2.nodes[key].get_average_strategy()
        
        diff = np.max(np.abs(strat1 - strat2))
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"Convergence instability at {key}: max diff = {diff:.6f}")
            significant_diffs += 1
    
    print(f"Maximum strategy difference: {max_diff:.6f}")
    print(f"Infosets with significant differences: {significant_diffs}")
    return significant_diffs == 0

def get_valid_actions(trainer: CFRTrainer, history: list) -> list:
    """Get valid actions given game history."""
    last_claim = history[-1] if len(history) > 0 else -1
    
    if last_claim == -1:
        return list(range(0, trainer.claims - 1))
    else:
        return list(range(last_claim + 1, trainer.claims))

def compute_infoset_value(trainer: CFRTrainer, avg_strat: dict, player: int, dice: tuple, history: list) -> float:
    """Compute expected value for an infoset under current strategies."""
    actions = get_valid_actions(trainer, history)
    infoset_key = f"{player}|{dice}|{','.join(map(str, history))}"
    
    strategy = avg_strat[infoset_key]
    
    expected_value = 0.0
    for i, action in enumerate(actions):
        action_value = compute_action_value(trainer, avg_strat, player, dice, history, action)
        expected_value += strategy[i] * action_value
    
    return expected_value

def compute_action_value(trainer: CFRTrainer, avg_strat: dict, player: int, dice: tuple, history: list, action: int) -> float:
    """Compute expected value for a specific action."""
    new_history = history + [action]
    
    if action == trainer.claims - 1:
        total_value = 0.0
        count = 0
        
        for opponent_dice in itertools.product(range(1, trainer.numSides + 1), repeat=trainer.numDices):
            if player == 0:
                full_dice = np.array([dice, opponent_dice])
            else:
                full_dice = np.array([opponent_dice, dice])
                
            claimant = (len(new_history) - 2) % N_PLAYERS
            utility = trainer.get_utility(full_dice, new_history, claimant, player)
            total_value += utility
            count += 1
        
        return total_value / count if count > 0 else 0.0
    
    next_player = len(new_history) % N_PLAYERS
    
    total_value = 0.0
    count = 0
    
    for opponent_dice in itertools.product(range(1, trainer.numSides + 1), repeat=trainer.numDices):
        opponent_infoset = f"{next_player}|{opponent_dice}|{','.join(map(str, new_history))}"
        

        opponent_strategy = avg_strat[opponent_infoset]
        opponent_actions = get_valid_actions(trainer, new_history)
        opponent_value = 0.0
        
        for j, opponent_action in enumerate(opponent_actions):
            if j < len(opponent_strategy):
                if next_player == player:
                    recursive_value = compute_action_value(trainer, avg_strat, player, dice, new_history, opponent_action)
                else:
                    opponent_new_history = new_history + [opponent_action]
                    if opponent_action == trainer.claims - 1:
                        if player == 0:
                            full_dice = np.array([dice, opponent_dice])
                        else:
                            full_dice = np.array([opponent_dice, dice])
                        claimant = (len(opponent_new_history) - 2) % N_PLAYERS
                        recursive_value = trainer.get_utility(full_dice, opponent_new_history, claimant, player)
                    else:
                        recursive_value = compute_infoset_value(trainer, avg_strat, player, dice, opponent_new_history)
                
                opponent_value += opponent_strategy[j] * recursive_value
        
        total_value += opponent_value
        count += 1
    
    return total_value / count if count > 0 else 0.0

def run_optimality_tests(strategy_file: str):
    """Run all optimality tests on a strategy file."""
    print("Running CFR Optimality Tests")
    print("=" * 50)
    
    trainer = CFRTrainer()
    trainer.load_strategies(strategy_file)
    avg_strat = {k: node.get_average_strategy() for k, node in trainer.nodes.items()}
    
    tests_passed = 0
    total_tests = 4
    
    if test_probability_consistency(trainer, avg_strat):
        tests_passed += 1
    
    if test_nash_equilibrium_property(trainer, avg_strat):
        tests_passed += 1
    
    
    if test_dominated_strategies(trainer, avg_strat):
        tests_passed += 1
    
    
    # For convergence test, you'd need two different strategy files
    # if test_convergence_stability(trainer1, trainer2):
    #     tests_passed += 1
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests-1}")  # -1 because we skip convergence test
    
    if tests_passed == total_tests - 1:
        print("✅ All tests passed! Strategies appear optimal.")
    else:
        print("❌ Some tests failed. Strategies may not be optimal.")
    
    return tests_passed == total_tests - 1

if __name__ == "__main__":
    run_optimality_tests(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
