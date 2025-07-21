import numpy as np
import itertools
from bluff_lp.constants import (NUM_FACES, NUM_DICES)

def test_mixed_vs_pure(roll, tol=1e-5):
    print(f"\n--- Testing mixed strategy vs pure strategies for roll {roll} ---")
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{roll}_{NUM_FACES}.npy")
    x = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_{roll}_{NUM_FACES}.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_{roll}_{NUM_FACES}_dual.npy")

    A_, x_, y_ = A[1:, 1:], x[1:], y[1:]
    v = x_ @ A_ @ y_

    row_failures = []
    col_failures = []

    # Row player: check that no pure row strategy does better against y
    for i in range(A_.shape[0]):
        pure_row = np.zeros_like(x_)
        pure_row[i] = 1.0
        pure_val = pure_row @ A_ @ y_
        if pure_val > v + tol:
            row_failures.append((i, pure_val, v, pure_row.copy(), y_.copy()))

    # Column player: check that no pure column strategy does better against x
    for j in range(A_.shape[1]):
        pure_col = np.zeros_like(y_)
        pure_col[j] = 1.0
        pure_val = x_ @ A_ @ pure_col
        if pure_val < v - tol:
            col_failures.append((j, pure_val, v, pure_col.copy(), x_.copy()))

    if row_failures or col_failures:
        raise AssertionError(f"{len(row_failures)} row failures, {len(col_failures)} column failures.")

    print("Mixed strategy is at least as good as any pure strategy.")


def game_value(roll):
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{roll}_{NUM_FACES}.npy")
    x = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_{roll}_{NUM_FACES}.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_{roll}_{NUM_FACES}_dual.npy")
    return x[1:] @ A[1:, 1:] @ y[1:]


if __name__ == "__main__":
    fail = False
    faces = range(1, NUM_FACES + 1)
    for roll in itertools.product(faces, repeat=NUM_DICES):
        try:
            val = game_value(roll)
            print(f"Game value for roll {roll}: {val:.4f}")
            test_mixed_vs_pure(roll)
        except AssertionError as e:
            fail = True
            print(f"Test failed for roll {roll}: {e}")
    print("\n--------------------")
    if not fail:
        print("\n\033[92m✔ All tests passed successfully!")
    else:
        print("\n\033[91m✖ Some tests failed. Please check the output above.")
    print("\n--- End of tests ---")