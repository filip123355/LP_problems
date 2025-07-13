import numpy as np
import pyscipopt 
import matplotlib.pyplot as plt
import os

from bluff_lp.constants import (PATH, FACE, NUM_FACES, NUM_DICES)
from bluff_lp.game_matrix import GameMatrix
from tqdm import tqdm

def solve_game_matix(path: str=PATH,
                    save: bool=True) -> None:
    """Solves the game matrix for Bluff using LP flow approach.

    Args:
        path (str, optional): Path of the game matrix for roll=FACE.
    """
    A = np.load(path)
    E = np.load(f"./game_constraints/{NUM_FACES}f/x.npy")
    F = np.load(f"./game_constraints/{NUM_FACES}f/y.npy")
    
    _, m = E.shape
    _, n = F.shape
    e = np.zeros((1, m), dtype=np.float32)
    f = np.zeros((1, n), dtype=np.float32)
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
    
    for i in range(m):
        print(f"{i}: {lp.getVal(x[i])}")
        
    if save:
        os.makedirs(f"bluff_lp/solutions/{NUM_FACES}", exist_ok=True)
        lp.writeProblem(f"bluff_lp/solutions/{NUM_FACES}/solution_{FACE}_{NUM_FACES}.lp")
        print(f"\nSolution saved under bluff_lp/solution_{FACE}_{NUM_FACES}.lp")
        
def solve_player():
    for face in tqdm(range(1, NUM_FACES + 1)):
        print(f"\nSolving game matrix for face {face}...")
        solve_game_matix(path=f"bluff_lp/game_matrices/{NUM_FACES}f/{face}_{NUM_FACES}.npy",
                        save=True)
        
if __name__ =="__main__":
    solve_player()
    