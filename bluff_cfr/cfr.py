# The code implements a Liar's Die game using the Counterfactual Regret Minimization (CFR) algorithm
# for two players.
import numpy as np
from tqdm import tqdm

"""Liar's Die game implementation using Counterfactual Regret Minimization (CFR) algorithm."""

BLUFF = 0
N_DIE_SIDES = 6
N_PLAYERS = 2

class Node:
    """Represents a node in the CFR tree."""
    def __init__(self, 
                 numActions: int, 
                 owner: int,
                 string_history: str = ""):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.zeros(numActions, dtype=np.float32)
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        self.owner = owner
        self.u = 0.0
        self.p_player = 0.0
        self.p_opponent = 0.0
        self.string_history = string_history
        
    def get_strategy(self, realizationWeights: np.ndarray) -> np.array:
        """Returns the current strategy for this node."""
        normalizingSum = 0.0
        positive_indices = self.regretSum > 0
        self.strategy[~positive_indices] = 0.0
        normalizingSum += np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.ones_like(self.strategy) / len(self.strategy)
        self.strategySum += realizationWeights * self.strategy
        return self.strategy
        
    
    def get_average_strategy(self) -> np.array:
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
                numPlayers: int=N_PLAYERS,
                numDices: int=1,):
        self.numSides = numSides
        self.numPlayers = numPlayers
        self.numDices = numDices
        self.claims = numDices * numPlayers * numSides + 1  # +1 for bluff
        self.nodes = {}
    
    def get_utility(self, dices: np.ndarray, string_history: str) -> float:
        """Computing the utility for the first player"""
        last_claim = int(string_history[-1])
        count = np.count_nonzero(dices == (last_claim % self.numSides + 1))
        return 1.0 if count >= (last_claim // self.numSides + 1) else -1.0
    
    def cfr(self, 
            dices: np.ndarray,
            player: int,
            p_player: float=1.0,
            p_opponent: float=1.0,
            node: Node=None,) -> float:
        """The CFR algorithm."""
        player_dice = tuple(sorted(dices[player*self.numDices:(player+1)*self.numDices]))
        infoset_key = f"{player_dice}:{node.string_history}"
        
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(numActions, node.owner, node.string_history)
        node = self.nodes[infoset_key]
        # Checks if the state is terminal
        if node.string_history.split(",")[-1] == str(self.claims - 1):
            return self.get_utility(dices, node.string_history)
        
        # Check if the node is a chance node
        elif node.string_history[-1] == "":
            action = node.sample_action()
            return self.cfr(
                dices, 
                player,
                p_player,
                p_opponent, 
                Node(
                    numActions=node.numActions, 
                    owner=node.owner,
                    string_history=node.string_history + "," + str(action)
                )
            )
            
        cf_value = 0.0
        cf_fixed_a_value = np.zeros(node.numActions, dtype=np.float32)
        # Iterate over all actions
        last_claim = int(node.string_history.split(',')[-1]) if node.string_history else -1
        numActions = self.claims - last_claim - 1  # Actions: last_claim+1 to claims-1
        infoset_key = f"{player_dice}:{node.string_history}"
        for action in range(node.numActions):
            if node.owner == 0:
                cf_fixed_a_value[action] = self.cfr(
                    dices,
                    player,
                    p_player * node.get_strategy(p_player)[action],
                    p_opponent,
                    self.nodes[infoset_key],
                )
            elif node.owner == 2:
                cf_fixed_a_value[action] = self.cfr(
                    dices,
                    player,
                    p_player,
                    p_opponent * node.get_strategy(p_opponent)[action],
                    self.nodes[infoset_key],
                )
            cf_value += node.strategySum[action] * cf_fixed_a_value[action]
            
        # Update regret
        if node.owner == player:
            for action in range(node.numActions):
                regret = cf_fixed_a_value[action] - cf_value
                node.regretSum[action] += regret * p_opponent if node.owner == 0 else p_player
                node.strategySum[action] += node.strategy[action] * p_player if node.owner == 0 else p_opponent 
            node.strategy = node.get_strategy(p_player)
            
        return cf_value         
        
    def solve(self, n_steps: int=10000):
        """Solves the Liar's Die game using CFR."""
        util = 0
        for step in tqdm(range(n_steps)):
            dices = np.random.randint(1, self.numSides + 1, size=self.numPlayers * self.numDices)
            root = Node(numActions=self.claims, owner=0, string_history="")
            util += self.cfr(dices, "", 1.0, 1.0, root)
        print(f"\nAverage game value: {util / n_steps}")
        # TODO: Add more stats
        
if __name__ == "__main__":
    trainer = CFRTrainer(numSides=N_DIE_SIDES, numPlayers=N_PLAYERS)
    trainer.solve(n_steps=10000)