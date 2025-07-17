import numpy as np
from bluff_lp.constants import (NUM_FACES, NUM_DICES, ROLL)

def test_mixed_vs_pure(face: int, tol=1e-5):
    print(f"\n--- Testing mixed strategy vs pure strategies for face {face} ---")
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{face}_{NUM_FACES}.npy")
    x = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}_dual.npy")

    v = x @ A @ y

    # Row player: check that no pure row strategy does better against y
    for i in range(1, A.shape[0]):
        pure_row = np.zeros_like(x)
        pure_row[i] = 1.0
        pure_val = pure_row @ A @ y
        assert pure_val <= v + tol, f"Pure row strategy {i} does better: {pure_val} > {v}"

    # Column player: check that no pure column strategy does better against x
    for j in range(1, A.shape[1]):
        pure_col = np.zeros_like(y)
        pure_col[j] = 1.0
        pure_val = x @ A @ pure_col
        assert pure_val >= v - tol, f"Pure column strategy {j} does better: {pure_val} < {v}"

    print("Mixed strategy is at least as good as any pure strategy.")

def game_value(face: int):
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{face}_{NUM_FACES}.npy")
    x = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}_dual.npy")
    return x @ A @ y

if __name__ == "__main__":
    fail = False
    for face in range(1, NUM_FACES + 1):
        try:
            val = game_value(face)
            print(f"Game value for face {face}: {val:.4f}")
            test_mixed_vs_pure(face)
        except AssertionError as e:
            fail = True
            print(f"Test failed for face {face}: {e}")
    print("\n--------------------")
    if not fail:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above.")
    print("\n--- End of tests ---")