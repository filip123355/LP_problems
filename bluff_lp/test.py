import numpy as np
from bluff_lp.constants import NUM_FACES, NUM_DICES
import os

def test_solution(face: int, tol=1e-5) -> float:
    print(f"\n--- Testing solution for face {face} ---")
    
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{face}_{NUM_FACES}.npy")
    E = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/x.npy")
    F = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/y.npy")
    
    # solution_path = f"bluff_lp/solutions/{NUM_FACES}/solution_{face}_{NUM_FACES}.lp"
    # if not os.path.exists(solution_path):
    #     raise FileNotFoundError(f"No solution file found at {solution_path}")

    x = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}_dual.npy")

    print("Checking x is valid probability vector...")
    prob_sum = np.abs(np.sum(x) - 1)
    assert prob_sum < tol, f"Sum of x = {np.sum(x)} ≠ 1"

    assert np.all(x >= -tol), "Some entries of x are negative"

    e = np.zeros((E.shape[0],))
    e[0] = 1.0
    print(x)
    ex = E @ x
    assert np.allclose(ex, e, atol=tol), f"E @ x != e:\nExpected: {e}\nGot: {ex}"

    game_value = x @ A @ y
    print(f"Estimated value of the game: {game_value:.4f}")

    print("All tests passed.")
    return game_value

if __name__ == "__main__":
    for face in range(1, NUM_FACES + 1):
        try:
            val = test_solution(face)
            print(f"Game value for face {face}: {val:.4f}")
        except AssertionError as e:
            print(f"Test failed for face {face}: {e}")
