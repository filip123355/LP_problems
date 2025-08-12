import numpy as np
import pyscipopt
from bluff_lp.constants import (NUM_FACES, NUM_DICES)


def _load_all():
    A_raw = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/big_gm.npy")
    chance_scale = (NUM_FACES ** (2 * NUM_DICES))
    A = (1.0 / chance_scale) * A_raw
    E = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/x.npy")
    F = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/y.npy")
    x = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy.npy")
    y = np.load(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_dual.npy")
    return A, E, F, x, y


def _best_response_row(A, E, e, y):
    # Maximize x^T A y subject to E^T x = e, x >= 0
    m = E.shape[1]
    n = y.shape[0]
    A_ = A[1:, 1:]
    y_ = y[1:]

    # objective coefficients over full x (include x0 with zero coeff)
    c = np.zeros(m)
    c[1:] = A_.dot(y_)

    lp = pyscipopt.Model()
    vars_x = [lp.addVar(name=f"x[{i}]", vtype="C", lb=0.0, ub=1.0) for i in range(m)]
    # E^T x = e
    for i in range(E.shape[0]):
        lp.addCons(pyscipopt.quicksum(E.T[j, i] * vars_x[j] for j in range(m)) == e[i, 0])
    lp.setObjective(pyscipopt.quicksum(c[j] * vars_x[j] for j in range(m)))
    lp.setMaximize()
    lp.optimize()
    val = lp.getObjVal()
    return val


def _best_response_col(A, F, f, x):
    # Minimize x^T A y subject to F y = f, y >= 0
    m = x.shape[0]
    n = F.shape[1]
    A_ = A[1:, 1:]
    x_ = x[1:]

    d = np.zeros(n)
    d[1:] = x_.dot(A_)

    lp = pyscipopt.Model()
    vars_y = [lp.addVar(name=f"y[{i}]", vtype="C", lb=0.0, ub=1.0) for i in range(n)]
    # F y = f
    for i in range(F.shape[0]):
        lp.addCons(pyscipopt.quicksum(F[i, j] * vars_y[j] for j in range(n)) == f[i, 0])
    lp.setObjective(pyscipopt.quicksum(d[j] * vars_y[j] for j in range(n)))
    lp.setMinimize()
    lp.optimize()
    val = lp.getObjVal()
    return val


def test_best_responses(tol=1e-6):
    print("\n--- Testing equilibrium via best responses (sequence-form) ---")
    A, E, F, x, y = _load_all()
    e = np.zeros((E.shape[0], 1)); e[0, 0] = 1.0
    f = np.zeros((F.shape[0], 1)); f[0, 0] = 1.0

    v = float(x[1:] @ A[1:, 1:] @ y[1:])
    row_br = _best_response_row(A, E, e, y)
    col_br = _best_response_col(A, F, f, x)

    if row_br > v + tol or col_br < v - tol:
        raise AssertionError(f"Row BR {row_br:.6g} vs v {v:.6g}, Col BR {col_br:.6g} vs v {v:.6g}")
    print("Best responses consistent with equilibrium.")


def game_value():
    A, E, F, x, y = _load_all()
    return float(x[1:] @ A[1:, 1:] @ y[1:])


if __name__ == "__main__":
    fail = False
    try:
        val = game_value()
        print(f"Game value: {val:.4f}")
        test_best_responses()
    except AssertionError as e:
            fail = True
            print(f"Test failed: {e}")
    print("\n--------------------")
    if not fail:
        print("\n\033[92m✔ All tests passed successfully!")
    else:
        print("\n\033[91m✖ Some tests failed. Please check the output above.")
    print("\n--- End of tests ---")