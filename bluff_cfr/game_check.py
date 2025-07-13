import itertools 
import numpy as np 
from bluff_cfr.cfr import (CFRTrainer, Node, N_DICES, N_DIE_SIDES, N_PLAYERS) 

def load_trainer(path: str) -> CFRTrainer:
    trainer = CFRTrainer()
    trainer.load_strategies(path)
    return trainer

def get_average_strategies(trainer: CFRTrainer) -> dict:
    return {k: node.get_average_strategy() for k, node in trainer.nodes.items()}

def self_play_value(trainer: CFRTrainer, avg_strat: dict) -> float:
    total, count = 0.0, 0
    faces = range(1, trainer.numSides + 1)
    for dice0 in itertools.product(faces, repeat=trainer.numDices):
        for dice1 in itertools.product(faces, repeat=trainer.numDices):
            dices = (dice0, dice1)
            total += rollout_mixed(trainer, avg_strat, dices, history=[])
            count += 1
    return total / count if count > 0 else 0.0

def rollout_mixed(trainer: CFRTrainer, avg_strat: dict, dices: np.ndarray, 
                history: list|None=None) -> float:
    if history and history[-1] == trainer.claims - 1:
        claimant = (len(history)) % N_PLAYERS
        return trainer.get_utility(np.array(dices), history, claimant, 0)
    owner = len(history) % N_PLAYERS
    infoset = f"{owner}|{tuple(dices[owner])}|{','.join(map(str, history))}"
    strat = avg_strat[infoset]
    last = history[-1] if history else -1
    actions = list(range(last + 1, trainer.claims)) if last > -1 else list(range(0, trainer.claims - 1))
    ev = 0.0
    for i, a in enumerate(actions):
        ev += strat[i] * rollout_mixed(trainer, avg_strat, dices, history + [a])
    return ev

def best_response_value(trainer: CFRTrainer, avg_strat: dict, player: int) -> float:
    faces = range(1, trainer.numSides + 1)
    total, count = 0.0, 0
    for dice0 in itertools.product(faces, repeat=trainer.numDices):
        for dice1 in itertools.product(faces, repeat=trainer.numDices):
            dices = (dice0, dice1)
            total += rollout_br(trainer, avg_strat, dices, history=[], br_player=player)
            count += 1
    return total / count if count > 0 else 0.0 

def rollout_br(trainer: CFRTrainer, avg_strat: dict, dices: np.ndarray, 
            history: list|None, br_player: int) -> float:
    if history and history[-1] == trainer.claims - 1:
        claimant = (len(history)) % N_PLAYERS
        return trainer.get_utility(np.array(dices), history, claimant, br_player)
    owner = len(history) % N_PLAYERS
    infoset = f"{owner}|{tuple(dices[owner])}|{','.join(map(str, history))}"
    last = history[-1] if history else -1
    actions = list(range(last + 1, trainer.claims)) if last > -1 else list(range(0, trainer.claims - 1))

    if owner == br_player:
        best = -np.inf
        for a in actions:
            val = rollout_br(trainer, avg_strat, dices, history + [a], br_player)
            if val > best:
                best = val
        return best
    else:
        strat = avg_strat[infoset]
        ev = 0.0
        for i,a in enumerate(actions):
            ev += strat[i] * rollout_br(trainer, avg_strat, dices, history + [a], br_player)
        return ev

if __name__ == "__main__":
    trainer = load_trainer(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    avg_strat = get_average_strategies(trainer)
    
    V = self_play_value(trainer, avg_strat)
    Vbr0 = best_response_value(trainer, avg_strat, player=0)
    Vbr1 = best_response_value(trainer, avg_strat, player=1)

    print(f"Self‐play value: {V:.6f}")
    print(f"Player 0 best‐response: {Vbr0:.6f}, exploitability: {Vbr0 - V:.6f}")
    print(f"Player 1 best‐response: {Vbr1:.6f}, exploitability: {Vbr1 + V:.6f}")

    tol = 1e-3
    assert abs(Vbr0 - V) < tol and abs(Vbr1 + V) < tol, \
        "High exploitability: CFR likely incorrect"
