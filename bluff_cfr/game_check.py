import pickle 
import itertools 
import numpy as np 

from functools import lru_cache 
from bluff_cfr.cfr import (CFRTrainer, Node, N_DICES, N_DIE_SIDES, N_PLAYERS) 

def load_trainer(path: str) -> CFRTrainer:
    trainer = CFRTrainer()
    trainer.load_strategies(path)
    return trainer

def get_average_strategies(trainer: CFRTrainer) -> dict:
    return {k: node.get_averate_strategy() for k, node in trainer.nodes.items()}

def rollout_mixed(trainer, avg_strat, dices, history):
    # terminal?
    if history and history[-1] == trainer.claims-1:
        claimant = (len(history)-1) % N_PLAYERS
        return trainer.get_utility(np.array(dices), history, claimant, 0)
    owner = len(history) % N_PLAYERS
    infoset = f"{owner}|{tuple(dices[owner])}|{','.join(map(str,history))}"
    strat = avg_strat[infoset]
    last = history[-1] if history else -1
    actions = list(range(last+1, trainer.claims))
    ev = 0.0
    for i,a in enumerate(actions):
        ev += strat[i] * rollout_mixed(trainer, avg_strat, dices, history+[a])
    return ev

def best_response_value(trainer, avg_strat, player):
    """Best‐pure‐response value for `player`, opponent fixed to avg_strat."""
    faces = range(1, trainer.numSides+1)
    total, count = 0.0, 0
    for dice0 in itertools.product(faces, repeat=trainer.numDices):
        for dice1 in itertools.product(faces, repeat=trainer.numDices):
            dices = (dice0, dice1)
            total += rollout_br(trainer, avg_strat, dices, history=[], br_player=player)
            count += 1
    return total / count

def self_play_value(trainer: CFRTrainer, avg_strat: dict) -> float:
    total, count = 0.0, 0.0
    faces = range(1, trainer.numSides + 1)
    for dice0 in itertools.product(faces, repeat=trainer.numDices):
        for dice1 in itertools.product(faces, repeat=trainer.numDices):
            dices = (dice0, dice1)
            total += rollout_mix(trainer, avg_strat, dices, history=[])
            count += 1.0
    return total / count if count > 0 else 0.0
@lru_cache(maxsize=None)
def rollout_br(trainer, avg_strat, dices, history, br_player):
    # terminal?
    if history and history[-1] == trainer.claims-1:
        claimant = (len(history)-1) % N_PLAYERS
        return trainer.get_utility(np.array(dices), history, claimant, br_player)
    owner = len(history) % N_PLAYERS
    infoset = f"{owner}|{tuple(dices[owner])}|{','.join(map(str,history))}"
    last = history[-1] if history else -1
    actions = list(range(last+1, trainer.claims))

    if owner == br_player:
        # choose the action with max you-rollout value
        best = -np.inf
        for a in actions:
            val = rollout_br(trainer, avg_strat, dices, history+[a], br_player)
            if val > best:
                best = val
        return best
    else:
        # opponent follows avg_strat
        strat = avg_strat[infoset]
        ev = 0.0
        for i,a in enumerate(actions):
            ev += strat[i] * rollout_br(trainer, avg_strat, dices, history+[a], br_player)
        return ev

if __name__ == "__main__":
    trainer = load_trainer("bluff_cfr/strategies/strategy_2_2.pkl")
    avg_strat = get_average_strategies(trainer)

    V = self_play_value(trainer, avg_strat)
    Vbr0 = best_response_value(trainer, avg_strat, player=0)
    Vbr1 = best_response_value(trainer, avg_strat, player=1)

    print(f"Self‐play value: {V:.6f}")
    print(f"Player 0 best‐response: {Vbr0:.6f}  → exploitability₀ = {Vbr0 - V:.6f}")
    print(f"Player 1 best‐response: {Vbr1:.6f}  → exploitability₁ = {Vbr1 + V:.6f}")

    tol = 1e-3
    assert abs(Vbr0 - V) < tol and abs(Vbr1 + V) < tol, \
        "High exploitability: CFR likely incorrect"
