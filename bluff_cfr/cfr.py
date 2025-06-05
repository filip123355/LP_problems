# The code implements a Liar's Die game using the Counterfactual Regret Minimization (CFR) algorithm
# for two players.
import numpy as np
from numba import jit, njit, prange
from tqdm import tqdm
from time import time

"""Liar's Die game implementation using Counterfactual Regret Minimization (CFR) algorithm."""

N_DIE_SIDES = 2
N_PLAYERS = 2
N_DICES = 1

class Node:
    """Represents a node in the CFR tree."""
    def __init__(self, 
                 numActions: int,):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.ones(numActions, dtype=np.float32) / numActions
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        self.p_player = 0.0
        self.p_opponent = 0.0
        
    def get_strategy(self) -> np.ndarray:
        """Returns the current strategy for this node. Uses the regret matching inside."""
        positive_indices = self.regretSum > 0
        self.strategy[~positive_indices] = 0.0 # Taking only the positive regrets
        normalizingSum = np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum # Scaling the strategy
        else:
            self.strategy = np.ones_like(self.strategy) / len(self.strategy) # Uniform distribution if no positive regrets
        self.strategySum += self.strategy
        return self.strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Returns the average strategy for this node."""
        normalizingSum = np.sum(self.strategySum)
        if normalizingSum > 0:
            return self.strategySum / normalizingSum
        else:
            return np.ones_like(self.strategySum) / len(self.strategySum)
        
    def sample_action(self) -> int:
        """Samples the action basing on the current local strategy."""
        return np.random.choice(
            len(self.strategy),
            p=self.strategy / np.sum(self.strategy),
        )        
        
class CFRTrainer:
    """Trainer for the CFR algorithm."""
    def __init__(self, numSides: int=N_DIE_SIDES,
                numDices: int=N_DICES,):
        self.numSides = numSides
        self.numDices = numDices
        self.claims = numDices * 2 * numSides  # + 1 for bluff
        self.nodes = {}
    
    def get_utility(self, dices: np.ndarray, string_history: str, claimant: int,
                    cfr_player: int) -> float:
        """Computes utility for the CFR player, considering claimant and player roles."""
        last_claim = int(string_history.split(',')[-1])
        quantity = last_claim // self.numSides + 1
        face = last_claim % self.numSides + 1
        count = np.count_nonzero(dices == face)
        if count >= quantity:
            outcome = 1.0  # claimant wins
        else:
            outcome = -1.0  # claimant loses
        return outcome if claimant == cfr_player else -outcome
    
    def cfr(self, 
            dices: np.ndarray,
            player: int,
            owner: int,
            p_player: float=1.0,
            p_opponent: float=1.0,
            string_history: str="") -> float:
        """The CFR algorithm."""
        player_dice = tuple(dices[player, :]) # Getting our player's dices
        last_claim = int(string_history.split(',')[-1]) if string_history  != "" else -1
        infoset_key = f"{player}:{player_dice}:{string_history}"
        
        # Checks if the state is terminal
        if last_claim == self.claims - 1:
            climant = 1 - owner
            return self.get_utility(dices, string_history, climant, player)
        
        # Adding the node if info set not present
        actions = list(range(last_claim + 1, self.claims))
        numActions = len(actions)
        if infoset_key not in self.nodes.keys():
            self.nodes[infoset_key] = Node(numActions)
        node = self.nodes[infoset_key]
    
        # Initiate counterfactual values and counterfactual values with fixed actions
        cf_value = 0.0
        cf_fixed_a_value = np.zeros(numActions, dtype=np.float32)
        
        # Iterate over all actions an recursively call cfr for each action
        for i, action in enumerate(actions):
            next_string_history = string_history + "," + str(action) if string_history != "" else str(action)
            if owner == 0: # TODO: Better distinction between owner and player
                cf_fixed_a_value[i] = self.cfr(
                    dices, player, 1 - owner,
                    p_player * node.strategy[i], p_opponent,
                    next_string_history,
                )
            elif owner == 1:
                cf_fixed_a_value[i] = self.cfr(
                    dices, player, 1 - owner,
                    p_player, p_opponent * node.strategy[i],
                    next_string_history,
                )
            cf_value += node.strategy[i] * cf_fixed_a_value[i] 
            
        # Update regret. Numpy arrays allow to do it for every action at once
        if owner == player:
            regrets = cf_fixed_a_value - cf_value
            weight = p_opponent if player == 0 else p_player
            counter_weight = p_player if player == 0 else p_opponent
            node.regretSum += regrets * weight
            node.strategySum += node.strategy * counter_weight
            
        return cf_value         
        
    def solve(self, n_steps: int=10000):
        """Solves the Liar's Die game using CFR."""
        util = 0
        start = time()
        for step in tqdm(range(n_steps)):
            # each steps simulates a game
            np.random.seed(step)
            for player in range(2):
                dices = np.random.randint(1, self.numSides + 1, size=(2, self.numDices))
                util += self.cfr(dices, player, player, 1.0, 1.0, string_history="")
        end = time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Average game value: {util / 2 * n_steps}")
        # TODO: Add more stats
        
if __name__ == "__main__":
    # Solving Bluff by creation and solving CFR trainers
    trainer = CFRTrainer(numSides=N_DIE_SIDES, 
                         numDices=N_DICES)
    trainer.solve(n_steps=10000)
    