import numpy as np
import concurrent.futures
import copy 
import matplotlib.pyplot as plt
import pickle as pkl

from tqdm import tqdm
from time import time

N_DIE_SIDES = 6
N_PLAYERS = 2
N_DICES = 1

class Node:
    def __init__(self, numActions: int):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.ones(numActions, dtype=np.float32) / numActions
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        self.visits = 0
        
    def get_strategy(self) -> np.ndarray:
        self.strategy = np.maximum(self.regretSum, 0)
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
        self.claims = numDices * 2 * numSides + 1# Total possible claims. + 1 for bluff
        self.nodes = {}
    
    def get_utility(self, dices: np.ndarray, history: list, claimant: int, cfr_player: int) -> float:
        last_claim = history[-2] # Last number represents bluff
        quantity = last_claim // self.numSides + 1
        face = last_claim % self.numSides + 1
        # Count across ALL dice (both players)
        count = np.count_nonzero(dices == face)
        if count >= quantity:
            outcome = 1.0  # claimant wins
        else:
            outcome = -1.0  # claimant loses
        return outcome if claimant == cfr_player else -outcome
    
    def cfr(self, dices: np.ndarray, player: int, 
            p0: float=1.0, p1: float=1.0, history: list=None) -> float:
        if history is None:
            history = []
        owner = (len(history)) % 2
        owner_dice = tuple(dices[owner, :])
        infoset_key = f"{owner}|{owner_dice}|{','.join(map(str, history))}"
        
        # Terminal condition - game ends when highest claim is made
        if len(history) > 0 and history[-1] == self.claims - 1:
            claimant = (len(history)) % 2  # Alternating claimants
            return self.get_utility(dices, history, claimant, player)
        
        # Get valid actions (can only bid higher than last claim)
        last_claim = history[-1] if len(history) > 0 else -1
        if last_claim == -1:
            actions = list(range(0, self.claims - 1)) # No possibility to bluff at start
        else:
            actions = list(range(last_claim + 1, self.claims))
        numActions = len(actions)
        
        # Get or create node
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(numActions)
        node = self.nodes[infoset_key]
        
        # Counterfactual values
        cf_value = 0.0
        action_utils = np.zeros(numActions, dtype=np.float32)
        
        # Calculate counterfactual values for each action
        for i, action in enumerate(actions):
            new_history = history + [action]
            if owner == 0:
                action_utils[i] = self.cfr(dices, player,
                            p0 * node.strategy[i], p1,
                            new_history)
            else:
                action_utils[i] = self.cfr(dices, player,
                            p0, p1 * node.strategy[i],
                            new_history)
            cf_value += node.strategy[i] * action_utils[i]
        
        # Update regret and strategy
        if owner == player:
            regrets = action_utils - cf_value
            node.regretSum += regrets * (p1 if player == 0 else p0)
            node.strategySum += node.strategy * (p0 if player ==0 else p1)
            node.get_strategy() # Regret matching here
            node.visits += 1
                
        return cf_value
    
    def solve(self, n_steps: int=10000):
        util = 0
        utils = []
        start = time()
        for step in tqdm(range(n_steps)):
            dices = np.random.randint(1, self.numSides + 1, size=(2, self.numDices))
            for player in range(N_PLAYERS):
                round_util = self.cfr(dices, player)
                if player == 0:
                    util += round_util
                else:
                    util -= round_util
                utils.append(util / ((2 * step + 1)))
        end = time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Average game value: {util / (2 * n_steps)}") 
        plt.plot(range(len(utils)), utils)
        plt.title(f"Average game return: {util / (2 * n_steps)}")
        plt.xlabel("Iteration step")
        plt.hlines(y=0.0, xmin=0, xmax=len(utils), colors='r', linestyles='dotted', alpha=0.5)
        plt.show()
        
    def save_strategies(self, filename: str):
        with open(filename, "wb") as file:
            pkl.dump(trainer.nodes, file)
        print(f"Strategies have been saved to: {filename}")
        
    def load_strategies(self, filename: str):
        with open(filename, "rb") as file:
            self.nodes = pkl.load(file)
        print(f"Strategies have been loaded from: {filename}")
        
    def visualize_strategies(self, num_to_vis: int, nrows: int=4):
        total_infosets = len(self.nodes.keys())
        actual_num_to_vis = min(num_to_vis, total_infosets)
        ncols = max(actual_num_to_vis // nrows + (1 if actual_num_to_vis % nrows else 0), 1)
        
        print(f"Total infosets available: {total_infosets}")
        print(f"Attempting to visualize: {num_to_vis}")
        print(f"Actually visualizing: {actual_num_to_vis}")
        print(f"Grid size: {nrows}x{ncols}")
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
        axs = np.atleast_2d(axs)  # Force 2D array even for single row/col
        
        plot_count = 0
        for key, node in self.nodes.items():
            if plot_count >= actual_num_to_vis:
                break
                
            row = plot_count // ncols
            col = plot_count % ncols
            
            # Parse infoset key and get actions
            bids = key[7:]
            if bids == "":
                args = list(range(0, self.claims - 1))
            else:
                
                args = list(range([int(bid) for bid in bids.split(",")][-1] + 1, self.claims))
            
            avg_strategy = node.get_average_strategy()
            
            if len(args) != len(avg_strategy):
                print(f"Warning: length mismatch for {key}: actions={len(args)}, strategy={len(avg_strategy)}")
                continue
            
            ax = axs[row, col]
            ax.bar(args, avg_strategy)
            ax.set_title(f"Infoset: {key}\nVisits: {node.visits}", fontsize=8)
            ax.set_ylabel("Average Strategy")
            
            plot_count += 1
        
        # Hide unused subplots
        # for i in range(plot_count, nrows * ncols):
        #     row = i // ncols
        #     col = i % ncols
        #     axs[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
                
def merge_regrets(nodes: dict, worker_nodes_list: list) -> None:
    for key in nodes.keys():
        
        for worker_nodes in worker_nodes_list:
            if key in worker_nodes:
                nodes[key].regretSum += worker_nodes[key].regretSum
                nodes[key].strategySum += worker_nodes[key].strategySum
                nodes[key].get_strategy()
        
def run_cfr_batch(trainer: CFRTrainer, n_steps: int=10000, seed: int=0):
    np.random.seed(seed)
    util = 0
    utils = []
    for step in range(n_steps):
        dices = np.random.randint(1, trainer.numSides + 1, size=(2, trainer.numDices))
        for player in range(N_PLAYERS):
            round_util = trainer.cfr(dices, player)
            if player == 0:
                util += round_util
            else:
                util -= round_util
            utils.append(util / ((2 * step + 1)))
    return util / (2 * n_steps), utils, copy.deepcopy(trainer.nodes)
    
def solve_concurrent(trainer: CFRTrainer, n_steps: int=10000, n_workers: int=24, 
                    sync_points: int=5):
        steps_per_worker_per_batch = int(n_steps / n_workers / 2)
        seeds = np.random.randint(0, 1e9, size=n_workers)
        start = time()
        full_utils_list = [[] * sync_points]
        total_util = 0.0
        for s_point in tqdm(range(sync_points)):
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(run_cfr_batch, trainer, steps_per_worker_per_batch, 
                                        int(seeds[i])) for i in range(n_workers)]
                utils_sum = [future.result()[0] for future in futures]
                utils_lists =  [future.result()[1] for future in futures]
                worker_nodes_list = [future.result()[2] for future in futures]
            total_util += sum(utils_sum)
            merge_regrets(trainer.nodes, worker_nodes_list)
            full_utils_list = [full_util + utils for full_util, utils in zip(full_utils_list, utils_lists)]
        end = time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Average game value in concurrent setup: {total_util / (2 * n_steps)}") 
        for i in range(0, len(full_utils_list), 2):
            plt.plot(range(len(full_utils_list[i])), full_utils_list[i], label=f"Worker {i+1}")
        plt.legend()
        plt.title(f"Average game return: {total_util / (2 * n_steps)}")
        plt.xlabel("Iteration step")
        plt.show()
        
if __name__ == "__main__":    
    
    # Definition of a trainer
    trainer = CFRTrainer(numSides=N_DIE_SIDES, numDices=N_DICES)
    
    # Normal run
    trainer.solve(n_steps=10000)
    trainer.save_strategies(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    
    
    # Concurrent run
    # Something does not feel right. Every batch s
    # Seems like it is trained from scratch
    # solve_concurrent(trainer, n_steps=240, n_workers=24, sync_points=10) # Requires large n_steps and quite a sizable batch. 
    # Denser sync points ensure stability
    
    # Loading strategies and analysis
    trainer.load_strategies(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    trainer.visualize_strategies(num_to_vis=32, nrows=4)
    for key, node in trainer.nodes.items():
        # if "|(1,)|" in key:
        print(f"Infoset: {key}, Average Strategy: {node.get_average_strategy()}")
            
    # TODO: Second player is not optimised. Always playing from the perspective of the
    # first player. Need to implement alternating players. The player Y is defined as the one
    # playing the second in the round.