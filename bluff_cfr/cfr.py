import numpy as np
from tqdm import tqdm
from time import time

N_DIE_SIDES = 2
N_PLAYERS = 2
N_DICES = 1

class Node:
    def __init__(self, numActions: int):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.ones(numActions, dtype=np.float32) / numActions
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        
    def get_strategy(self) -> np.ndarray:
        positive_indices = self.regretSum > 0
        self.strategy[~positive_indices] = 0.0
        normalizingSum = np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.ones_like(self.strategy) / len(self.strategy)
        return self.strategy
    
    def get_average_strategy(self) -> np.ndarray:
        normalizingSum = np.sum(self.strategySum)
        if normalizingSum > 0:
            return self.strategySum / normalizingSum
        else:
            return np.ones_like(self.strategySum) / len(self.strategySum)

class CFRTrainer:
    def __init__(self, numSides: int=N_DIE_SIDES, numDices: int=N_DICES):
        self.numSides = numSides
        self.numDices = numDices
        self.claims = numDices * 2 * numSides  # Total possible claims
        self.nodes = {}
    
    def get_utility(self, dices: np.ndarray, history: list, claimant: int, cfr_player: int) -> float:
        last_claim = history[-1]
        quantity = last_claim // self.numSides + 1
        face = last_claim % self.numSides + 1
        # Count across ALL dice (both players)
        count = np.count_nonzero(dices == face)
        if count >= quantity:
            outcome = 1.0  # claimant wins
        else:
            outcome = -1.0  # claimant loses
        return outcome if claimant == cfr_player else -outcome
    
    def cfr(self, dices: np.ndarray, player: int, owner: int,
            p_player: float=1.0, p_opponent: float=1.0, history: list=None) -> float:
        if history is None:
            history = []
            
        player_dice = tuple(dices[player, :])
        infoset_key = f"{player}:{player_dice}:{','.join(map(str, history))}"
        
        # Terminal condition - game ends when highest claim is made
        if len(history) > 0 and history[-1] == self.claims - 1:
            claimant = (len(history) - 1) % 2  # Alternating claimants
            return self.get_utility(dices, history, claimant, player)
        
        # Get valid actions (can only bid higher than last claim)
        last_claim = history[-1] if len(history) > 0 else -1
        actions = list(range(last_claim + 1, self.claims))
        numActions = len(actions)
        
        # Get or create node
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(numActions)
        node = self.nodes[infoset_key]
        node.get_strategy()  # Update strategy
        
        # Counterfactual values
        cf_value = 0.0
        action_utils = np.zeros(numActions, dtype=np.float32)
        
        for i, action in enumerate(actions):
            new_history = history + [action]
            if owner == 0:
                util = self.cfr(dices, player, 1, 
                              p_player * node.strategy[i], p_opponent,
                              new_history)
            else:
                util = self.cfr(dices, player, 0,
                              p_player, p_opponent * node.strategy[i],
                              new_history)
            action_utils[i] = util
            cf_value += node.strategy[i] * util
        
        # Update regret and strategy
        if owner == player:
            regrets = action_utils - cf_value
            weight = p_opponent if player == 0 else p_player
            node.regretSum += regrets * weight
            node.strategySum += node.strategy * weight
            
        return cf_value
         
    def solve(self, n_steps: int=10000):
        util = 0
        start = time()
        for step in tqdm(range(n_steps)):
            np.random.seed(step)
            for player in range(2):
                dices = np.random.randint(1, self.numSides + 1, size=(2, self.numDices))
                util += self.cfr(dices, player, player)
        end = time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Average game value: {util / (2 * n_steps)}") 

if __name__ == "__main__":
    trainer = CFRTrainer(numSides=N_DIE_SIDES, numDices=N_DICES)
    trainer.solve(n_steps=10000)