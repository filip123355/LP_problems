import numpy as np
import concurrent.futures
import copy 
import matplotlib.pyplot as plt
import pickle as pkl
import networkx as nx

from tqdm import tqdm
from time import time
from bluff_cfr.constants import (N_DICES, N_DIE_SIDES, N_PLAYERS)   

class Node:
    def __init__(self, numActions: int):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.ones(numActions, dtype=np.float32) / numActions
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        self.visits = 0
        
    def regret_matching(self):
        self.strategy = np.maximum(self.regretSum, 0)
        normalizingSum = np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.ones_like(self.strategy) / len(self.strategy)
    
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
        self.claims = numDices * 2 * numSides + 1 # Total possible claims. + 1 for bluff
        self.nodes = {}
    
    def get_utility(self, dices: np.ndarray, history: list, 
                    claimant: int, cfr_player: int) -> float:
        # The last action in history should be "bluff" (self.claims - 1)
        # The claim being challenged is the second-to-last action
        if len(history) < 2:
            return 0.0  # Should not happen in valid terminal states
            
        challenged_claim = history[-2]  # The claim being challenged
        quantity = challenged_claim // self.numSides + 1
        face = challenged_claim % self.numSides + 1
        
        # Count across ALL dice (both players)
        total_dice = dices.flatten()
        count = np.count_nonzero(total_dice == face)
        
        if count >= quantity:
            outcome = 1.0  # claimant (who made the claim) wins
        else:
            outcome = -1.0  # challenger wins
            
        # Return utility from Player 0's perspective always
        # If claimant is Player 0: return outcome directly
        # If claimant is Player 1: return -outcome (Player 0 gets opposite)
        return outcome if claimant == cfr_player else -outcome

    def cfr(self, dices: np.ndarray, player: int,
            p0: float=1.0, p1: float=1.0, history: list|None=None) -> float:
        if history is None:
            history = []
        owner = (len(history)) % 2
        owner_dice = tuple(dices[owner, :])
        infoset_key = f"{owner}|{owner_dice}|{','.join(map(str, history))}"
        
        # Terminal condition - game ends when someone calls bluff
        if len(history) > 0 and history[-1] == self.claims - 1:
            # The player who made the claim being challenged
            claimant = (len(history) - 2) % N_PLAYERS # -2 because we need the player who made the challenged claim
            return self.get_utility(dices, history, claimant, player)
        
        # Get valid actions
        last_claim = history[-1] if len(history) > 0 else -1
        
        if last_claim == -1:
            # First move: can make any claim from 0 to claims-2, but not bluff
            actions = list(range(0, self.claims - 1))
        else:
            # Can make higher claims or call bluff
            actions = list(range(last_claim + 1, self.claims))  # Includes bluff as last action
        numActions = len(actions)
        
        # Get or create node
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(numActions)
        node = self.nodes[infoset_key]
        
        # Counterfactual values
        cf_value = 0.0
        action_utils = np.zeros(numActions, dtype=np.float32)
        
        # Get the current strategy before recursion
        strategy = node.strategy
        
        # Calculate counterfactual values for each action
        for i, action in enumerate(actions):
            new_history = history + [action]
            if owner == 0:
                action_utils[i] = self.cfr(dices, player,
                            p0 * strategy[i], p1,
                            new_history)
            else:
                action_utils[i] = self.cfr(dices, player,
                            p0, p1 * strategy[i],
                            new_history)
            cf_value += strategy[i] * action_utils[i]
        
        # Update regret and strategy
        if owner == player:
            regrets = action_utils - cf_value
            node.regretSum += regrets * (p1 if player == 0 else p0)
            node.strategySum += strategy * (p0 if player == 0 else p1)
            node.regret_matching() # Regret matching here
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
        # plt.savefig(f"/images/convergance/cfr_training_plot_{self.numDices}_{self.numSides}.png")
        
    def save_strategies(self, filename: str):
        with open(filename, "wb") as file:
            pkl.dump(self.nodes, file)
        print(f"Strategies have been saved to: {filename}")
        
    def load_strategies(self, filename: str):
        with open(filename, "rb") as file:
            self.nodes = pkl.load(file)
        print(f"Strategies have been loaded from: {filename}")
        
    def visualize_strategies(self, num_to_vis:int|None=None, nrows: int=4):
        total_infosets = len(self.nodes.keys())
        actual_num_to_vis = min(num_to_vis, total_infosets)
        ncols = max(actual_num_to_vis // nrows + (1 if actual_num_to_vis % nrows else 0), 1)
        
        if num_to_vis is None:
            num_to_vis = total_infosets
            actual_num_to_vis = total_infosets
        
        print(f"Total infosets available: {total_infosets}")
        print(f"Attempting to visualize: {num_to_vis}")
        print(f"Actually visualizing: {actual_num_to_vis}")
        print(f"Grid size: {nrows}x{ncols}")
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
        axs = np.atleast_2d(axs)  # Force 2D array even for single row/col
        
        plot_count = 0
        for key, node in self.nodes.items():
            if plot_count >= actual_num_to_vis:
                break
                
            row = plot_count // ncols
            col = plot_count % ncols
            
            # Parse infoset key and get actions
            bids = key.split("|")[-1]
            if bids == "":
                args = list(range(0, self.claims - 1))
            else:
                args = list(range([int(bid) for bid in bids.split(",")][-1] + 1, self.claims))
            
            avg_strategy = node.get_average_strategy()
            
            if len(args) != len(avg_strategy):
                print(f"Warning: length mismatch for {key}: actions={len(args)}, strategy={len(avg_strategy)}")
                continue
            
            key = "x" + key[1:] if key[0] == "0" else "y" + key[1:]
            
            key = self.decode_strategy(key)
            
            for i, arg in enumerate(args):
                if i != len(args):
                    dice_call, face_call = self.decode(arg)
                    args[i] = f"({dice_call},{face_call})"
                if key.split("|")[-1] != "" and i == len(args) - 1:
                    args[i] = "Bluff"
                    
            ax = axs[row, col]
            ax.bar(args, avg_strategy)
            ax.set_title(f"Infoset: {key}\nVisits: {node.visits}", fontsize=8)
            ax.set_ylabel("Average Strategy")
            
            plot_count += 1
        
        plt.tight_layout()
        plt.show()        
    
    @staticmethod    
    def decode(incex: int) -> tuple[int, int]:
        total_faces = N_DIE_SIDES
        dice_call = incex // total_faces + 1
        face_call = incex % total_faces + 1
        return dice_call, face_call
    
    def decode_strategy(self, key: str) -> str:
        
        strategy = key.split("|")[-1].split(",")
        bids = ""
        
        if strategy != [""]:
            for i, bid in enumerate(strategy):
                bid = self.decode(int(bid))
                bids += f"({bid[0]},{bid[1]})"
        
        return "|".join(key.split("|")[:-1]) + "|" + bids if bids != "" else "|".join(key.split("|")[:-1]) + "|"
    
    def format_node_label(self, key: str) -> str:

        parts = key.split("|")
        if len(parts) != 3:
            return key  

        player_index, rolled_dice, bids = parts
        return f"{player_index} | {rolled_dice} | {bids}"

    def visualize_strategy_tree(self, max_depth: int = 5, max_nodes: int = 200):

        def format_node_label(key: str) -> str:
            parts = key.split("|")
            if len(parts) == 3:
                player, dice, bids = parts
                return f"P{player} | {dice} | {bids or '∅'}"
            return key

        G = nx.DiGraph()
        visited = set()

        # Find true root nodes (empty history for both players)
        root_keys = [k for k in self.nodes.keys()
                    if k.split("|")[2] == ""]  

        queue = [(rk, 0) for rk in root_keys]

        while queue and len(visited) < max_nodes:
            current_key, depth = queue.pop(0)
            if current_key in visited or depth > max_depth:
                continue
            visited.add(current_key)

            # Parse current node information
            parts = current_key.split("|")
            current_player = int(parts[0])
            current_dice = parts[1]
            history_str = parts[2]
            
            # Parse history
            if history_str == "":
                history = []
            else:
                history = [int(x) for x in history_str.split(",")]
            
            # Calculate valid actions using same logic as CFR
            last_claim = history[-1] if len(history) > 0 else -1
            
            if last_claim == -1:
                # First move: can make any claim from 0 to claims-2, but not bluff
                actions = list(range(0, self.claims - 1))
            else:
                # Can make higher claims or call bluff
                actions = list(range(last_claim + 1, self.claims))
            
            if current_key in self.nodes:
                node = self.nodes[current_key]
                avg_strat = node.get_average_strategy()
                
                for i, action in enumerate(actions):
                    new_history = history + [action]
                    
                    # Determine next player
                    next_player = (len(new_history)) % 2
                    
                    # Create next key - need to find the appropriate dice configuration
                    # For visualization, we'll use the same dice for simplicity
                    next_key = f"{next_player}|{current_dice}|{','.join(map(str, new_history))}"
                    
                    prob = avg_strat[i] if i < len(avg_strat) else 0.0
                    
                    # Add edge with action and probability labels
                    action_label = f"A{action}" if action < self.claims - 1 else "Bluff"
                    G.add_edge(current_key, next_key, 
                            label=f"{action_label}\n{prob:.2f}", 
                            weight=prob,
                            action=action)

                    # Add to queue if this is a valid continuation and not terminal
                    if action != self.claims - 1:  # Not a bluff call (terminal)
                        queue.append((next_key, depth + 1))

                    # Add to queue if this is a valid continuation and not terminal
                    if action != self.claims - 1:  # Not a bluff call (terminal)
                        queue.append((next_key, depth + 1))

        # Create layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            # Fallback to spring layout if graphviz is not available
            pos = nx.spring_layout(G, k=3, iterations=50)
            
        plt.figure(figsize=(16, 12))

        # Draw nodes with different colors for different players
        node_colors = []
        for node in G.nodes():
            player = int(node.split("|")[0])
            if player == 0:
                node_colors.append('lightblue')
            else:
                node_colors.append('lightcoral')

        widths = [max(0.5, G[u][v]['weight'] * 3) for u, v in G.edges()]

        nx.draw(G, pos,
                with_labels=False,
                arrows=True,
                node_size=800,
                node_color=node_colors,
                width=widths,
                edge_color='gray')

        # Draw node labels
        labels = {n: format_node_label(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Draw edge labels with action and probability
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels=edge_labels,
                                    font_color='red',
                                    font_size=6)

        plt.title(f"Strategy Tree (depth ≤ {max_depth}, nodes ≤ {max_nodes})\nBlue=Player0, Red=Player1")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print(f"Tree contains {len(G.nodes())} nodes and {len(G.edges())} edges")
        print(f"Root nodes: {len(root_keys)}")
        if len(G.nodes()) > 0:
            depths = {}
            for node in G.nodes():
                history_str = node.split("|")[2]
                depth = len(history_str.split(",")) if history_str else 0
                depths[depth] = depths.get(depth, 0) + 1
            print(f"Nodes by depth: {depths}")

                
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
    old_trainer = CFRTrainer(numSides=N_DIE_SIDES, numDices=N_DICES)
    old_trainer.load_strategies(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    
    # Normal uni-thread run
    trainer.solve(n_steps=1000000)
    trainer.save_strategies(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    
    # for key in old_trainer.nodes.keys():
    #     old_node = old_trainer.nodes[key]
    #     new_node = trainer.nodes[key]
    #     old_strat = old_node.get_average_strategy()
    #     new_strat = new_node.get_average_strategy()
    #     if not np.isclose(old_strat, new_strat, atol=1e-2).all():
    #         diff = np.sum(np.abs(old_strat - new_strat)) / 2
    #         print(f"Significant difference in infoset {key}: total diff = {diff}")
    #         actions = list(range(len(old_strat)))
    #         plt.figure(figsize=(8,4))
    #         plt.bar([str(a) for a in actions], old_strat, alpha=0.5, label='old')
    #         plt.bar([str(a) for a in actions], new_strat, alpha=0.5, label='new')
    #         plt.title(f"Infoset: {key}\nTotal diff: {diff:.4f}")
    #         plt.xlabel("Action index")
    #         plt.ylabel("Average strategy")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()
            
    # Concurrent run
    # Something does not feel right. Every batch 
    # Seems like it is trained from scratch
    # solve_concurrent(trainer, n_steps=240, n_workers=24, sync_points=10) # Requires large n_steps and quite a sizable batch. 
    # Denser sync points ensure stability
    
    # Loading strategies and display analysis
    # trainer.load_strategies(f"bluff_cfr/strategies/strategy_{N_DICES}_{N_DIE_SIDES}.pkl")
    # trainer.visualize_strategies(num_to_vis=32, nrows=4)
    # trainer.visualize_strategy_tree(max_depth=4, max_nodes=100)
    
    # Print average strategies for all infosets
    # for key, node in trainer.nodes.items():
    #     print(f"Infoset: {key}, Average Strategy: {node.get_average_strategy()}")