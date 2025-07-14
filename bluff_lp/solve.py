import numpy as np
import pyscipopt 
import matplotlib.pyplot as plt
import os

from bluff_lp.constants import (NUM_FACES, NUM_DICES)
from bluff_lp.game_matrix import GameMatrix
from tqdm import tqdm

def solve_game_matix(face: int,
                    save: bool=True) -> None:
    """Solves the game matrix for Bluff using LP flow approach.

    Args:
        path (str, optional): Path of the game matrix for roll=FACE.
    """
    # Primary
    A = np.load(f"bluff_lp/game_matrices/{NUM_DICES}_{NUM_FACES}f/{face}_{NUM_FACES}.npy")
    E = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/x.npy")
    F = np.load(f"bluff_lp/game_constraints/{NUM_DICES}_{NUM_FACES}f/y.npy")
    
    k, m = E.shape
    l, n = F.shape
    e = np.zeros((k, 1), dtype=np.float32)
    f = np.zeros((l, 1), dtype=np.float32)
    e[0, 0], f[0, 0] = 1.0, 1.0
    
    print(e)
    
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
    
    print(f"\nPrimary strategy for face {face}:")
    for i in range(m):
        print(f"{i}: {lp.getVal(x[i])}")
        
    strategy = np.array([lp.getVal(x[i]) for i in range(m)])
        
    if save:
        os.makedirs(f"bluff_lp/solutions/{NUM_FACES}", exist_ok=True)
        # lp.writeProblem(f"bluff_lp/solutions/{NUM_FACES}/solution_{face}_{NUM_FACES}.lp")
        np.save(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}.npy", strategy)
        # print(f"\nSolution saved under bluff_lp/solution_{face}_{NUM_FACES}.lp")
        
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
    
    print(f"\nDual strategy for face {face}:")
    for i in range(n):
        print(f"{i}: {lp.getVal(y[i])}")
        
    dual_strategy = np.array([lp.getVal(y[i]) for i in range(n)])
        
    if save:
        # lp.writeProblem(f"bluff_lp/solutions/{NUM_FACES}/solution_{face}_{NUM_FACES}.lp")
        np.save(f"bluff_lp/solutions/{NUM_FACES}/strategy_{face}_{NUM_FACES}_dual.npy", dual_strategy)
        # print(f"\nSolution saved under bluff_lp/solution_{face}_{NUM_FACES}.lp")
        
    
def solve_player():
    for face in tqdm(range(1, NUM_FACES + 1)):
        print(f"\nSolving game matrix for face {face}...")
        solve_game_matix(face=face, 
                        save=True)
        
if __name__ == "__main__":
    solve_player()
    