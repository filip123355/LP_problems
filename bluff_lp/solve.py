import numpy as np
import pyscipopt 
import os
import itertools

from bluff_lp.constants import (NUM_FACES, NUM_DICES)

def solve_game_matrix(save=True, verbose=False):
    # Primary
    # Load the full big matrix with null row/column and apply proper chance averaging.
    A_raw = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/big_gm.npy")
    # Each player has NUM_FACES**NUM_DICES possible private outcomes, so pairs are squared.
    chance_scale = (NUM_FACES ** (2 * NUM_DICES))
    A = (1.0 / chance_scale) * A_raw
    E = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/x.npy")
    F = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/y.npy")

    k, m = E.shape
    l, n = F.shape
    e = np.zeros((k, 1), dtype=np.float32)
    f = np.zeros((l, 1), dtype=np.float32)
    e[0, 0], f[0, 0] = 1.0, 1.0
    
    lp = pyscipopt.Model()
    x = {i: lp.addVar(f"x[{i}]", vtype="C", lb=0.0, ub=1.0) for i in range(m)}
    q = {k: lp.addVar(f"q[{k}]", vtype="C", lb=-lp.infinity(), ub=lp.infinity()) 
        for k in range(f.shape[0])}
    
    lp.setObjective(-pyscipopt.quicksum(e[k, 0] * q[k] for k in range(f.shape[0])))
    lp.setMaximize()
        
    for i in range(n):
        lp.addCons(pyscipopt.quicksum(-A[j, i] * x[j] for j in range(m)) -
                pyscipopt.quicksum(F[k, i] * q[k] for k in range(f.shape[0])) <= 0)

    for i in range(E.shape[0]):
        lp.addCons(pyscipopt.quicksum(E.T[j, i] * x[j] for j in range(m)) == e[i, 0])
        
    lp.optimize()
    
    if verbose:
        print(f"\nPrimary strategy for the game matrix:")
        for i in range(m):
            print(f"{i}: {lp.getVal(x[i])}")
        
    strategy = np.array([lp.getVal(x[i]) for i in range(m)])
        
    if save:
        os.makedirs(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}", exist_ok=True)
        np.save(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy.npy", strategy)
        
    # Dual
    lp = pyscipopt.Model()
    y = {i: lp.addVar(f"y[{i}]", vtype="C", lb=0.0, ub=1.0) for i in range(n)}
    p = {k: lp.addVar(f"p[{k}]", vtype="C", lb=-lp.infinity(), ub=lp.infinity()) for k in range(e.shape[0])}
    
    lp.setObjective(pyscipopt.quicksum(e[k, 0] * p[k] for k in range(e.shape[0])))
    lp.setMinimize()
    
    for i in range(m):
        lp.addCons(pyscipopt.quicksum(-A[i, j] * y[j] for j in range(n)) +
                pyscipopt.quicksum(E[k, i] * p[k] for k in range(e.shape[0])) >= 0)

    for i in range(F.shape[0]):
        lp.addCons(pyscipopt.quicksum(F[i, j] * y[j] for j in range(n)) == f[i, 0])
        
    lp.optimize()
    
    if verbose:
        print(f"\nDual strategy for the game matrix:")
        for i in range(n):
            print(f"{i}: {lp.getVal(y[i])}")
            
    dual_strategy = np.array([lp.getVal(y[i]) for i in range(n)])
        
    if save:
        np.save(f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}/strategy_dual.npy", dual_strategy)

if __name__ == "__main__":
    solve_game_matrix(save=True, verbose=False)
