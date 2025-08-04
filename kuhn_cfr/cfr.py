import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import networkx as nx

from tqdm import tqdm
from time import time

plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = False 
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['mathtext.fontset'] = 'cm' 

class Node:
    def __init__(self, numActions: int=2):
        self.numActions = numActions
        self.regretSum = np.zeros(numActions, dtype=np.float32)
        self.strategy = np.ones(numActions, dtype=np.float32) / numActions
        self.strategySum = np.zeros(numActions, dtype=np.float32)
        self.visits = 0
        self.isterminal = False
        
    def regret_matching(self):
        self.strategy = np.maximum(self.regretSum, 0)
        normalizingSum = np.sum(self.strategy)
        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.ones_like(self.strategy) / len(self.strategy)
    
    def get_average_strategy(self) -> np.ndarray|str:
        normalizingSum = np.sum(self.strategySum)
        if normalizingSum > 0 and not self.isterminal:
            return self.strategySum / normalizingSum
        elif normalizingSum <= 0  and not self.isterminal:
            return np.ones_like(self.strategySum) / len(self.strategySum)
        else:
            return "Terminal Node"

def deal(numPlayers:int=2, numCards: int=3) -> list[int]:
    deck = np.arange(numCards)
    np.random.shuffle(deck)
    return deck[:numPlayers].tolist()

class KuhnCFRTrainer:
    def __init__(self, numPlayers: int=2, numCards: int=3):
        self.numPlayers = numPlayers
        self.numCards = numCards
        self.pot = 0
        self.nodes = {}

    def cfr(self, cards: list[int], player: int, 
            p0: float=1.0, p1: float=1.0, history: list|None=None) -> float:
        if history is None:
            history = []
        owner = (len(history)) % 2
        owner_cards = cards[owner]
        infoset_key = f"{owner}|{owner_cards}|{','.join(map(str, history))}"
        
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(numActions=2)
            
        node = self.nodes[infoset_key]    
        
        # Terminal conditions for Kuhn Poker
        if self.is_terminal(history):
            node.isterminal = True  
            return self.get_utility(cards, history, player)
        
        # Counterfactual values
        cf_value = 0.0
        action_utils = np.zeros(2, dtype=np.float32)

        # Get the current strategy before recursion
        strategy = node.strategy
        
        # CFR reccursion
        for i, action in enumerate(range(node.numActions)):
            new_history = history + [action]
            if owner == 0:
                action_utils[i] = self.cfr(cards, player,
                        p0 * strategy[i], p1,
                        new_history)
            else:
                action_utils[i] = self.cfr(cards, player,
                            p0, p1 * strategy[i],
                            new_history)
            cf_value += strategy[i] * action_utils[i]
        
         # Update regret and strategy
        if owner == player:
            regrets = action_utils - cf_value
            node.regretSum += regrets * (p1 if player == 0 else p0)
            node.strategySum += strategy * (p0 if player == 0 else p1)
            node.regret_matching() 
            node.visits += 1
                
        return cf_value
        
    def is_terminal(self, history: list[int]) -> bool:
        """
        Check if the game has reached a terminal state.
        0 = Pass/Check, 1 = Bet/Call
        """
        if len(history) == 0:
            return False
            
        # Game ends after both players pass (PP)
        if len(history) == 2 and history == [0, 0]:
            return True
            
        # Game ends if someone folds after a bet
        if len(history) == 2 and history == [1, 0]:  # P1 bets, P2 folds
            return True
            
        # Game ends if both players bet (call scenario)
        if len(history) == 2 and history == [1, 1]:  # P1 bets, P2 calls
            return True
            
        # Game ends after P1 passes, P2 bets, then P1 folds
        if len(history) == 3 and history == [0, 1, 0]:  # P1 pass, P2 bet, P1 fold
            return True
            
        # Game ends after P1 passes, P2 bets, then P1 calls
        if len(history) == 3 and history == [0, 1, 1]:  # P1 pass, P2 bet, P1 call
            return True
            
        return False
        
    def get_utility(self, cards: list[int], history: list[int], player: int) -> float:
        """
        Calculate utility for the given player.
        0 = Pass/Check, 1 = Bet/Call
        Utility is from the perspective of the given player.
        """
        p1_card, p2_card = cards[0], cards[1]
        
        # Determine pot size and winner
        if history == [0, 0]:  # Both pass - showdown with 1 chip pot
            pot_size = 1
            winner = 0 if p1_card > p2_card else 1
            
        elif history == [1, 0]:  # P1 bets, P2 folds - P1 wins 1 chip
            pot_size = 1
            winner = 0
            
        elif history == [1, 1]:  # P1 bets, P2 calls - showdown with 2 chip pot
            pot_size = 2
            winner = 0 if p1_card > p2_card else 1
            
        elif history == [0, 1, 0]:  # P1 pass, P2 bet, P1 fold - P2 wins 1 chip
            pot_size = 1
            winner = 1
            
        elif history == [0, 1, 1]:  # P1 pass, P2 bet, P1 call - showdown with 2 chip pot
            pot_size = 2
            winner = 0 if p1_card > p2_card else 1
            
        else:
            raise ValueError(f"Invalid terminal history: {history}")
        
        if player == winner:
            return pot_size
        else:
            return -pot_size

    def solve(self, numIterations: int=10000, 
              path:str|None="kuhn_cfr", verbose: bool=True) -> dict:
        start = time()
        game_value = 0.0
        game_values = []
        for i in tqdm(range(numIterations)):
            for player in range(self.numPlayers):
                cards = deal(self.numPlayers, self.numCards)
                if player == 0:
                    game_value += self.cfr(cards, player)
                else:
                    game_value -= self.cfr(cards, player)
                game_values.append(game_value / (2 * i + 1))
        stop = time()
        print(f"Training completed in {stop - start:.2f} seconds.")
        print(f"Average game value: {game_value / (2 * numIterations):.4f}")
        if verbose:
            plt.plot(game_values, label="Game Values", color="firebrick")
            plt.xlabel("Iterations")
            plt.legend()
            plt.hlines(y=-1/18, xmin=0, xmax=2 * numIterations, colors='red', linestyles='dashed')
            plt.hlines(y=0, xmin=0, xmax=2 * numIterations, colors='blue', linestyles='dashed')
            plt.show()
        average_strategies = {key: node.get_average_strategy() for key, node in self.nodes.items()}
        if path is not None:
            with open(f"{path}/strategies/kuhn_cfr_model.pkl", 'wb') as file:
                pkl.dump(self.nodes, file)
            with open(f"{path}/game_values/game_values.pkl", "wb") as file:
                pkl.dump(game_values, file)
            print(f"Model saved to '{path}'")
        return average_strategies, game_values
    
    def plot_strategies(self, strategies: dict):
        """
        Plot the Kuhn Poker game tree with learned strategies.
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        G = nx.DiGraph()
        
        # Add chance root node
        G.add_node("Chance", type="chance", label="Chance", level=0)
        
        # Create proper game tree structure
        pos = {}
        card_names = {0: 'J', 1: 'Q', 2: 'K'}
        
        # Level positions
        chance_y = 4
        p1_initial_y = 3
        p2_y = 2
        p1_second_y = 1
        terminal_y = 0
        
        pos["Chance"] = (0, chance_y)
        
        # Track x positions for each level
        x_counters = {3: -6, 2: -8, 1: -4, 0: -12}
        
        # Process each information set
        information_sets = {}
        for infoset_key, strategy in strategies.items():
            player, card, history = infoset_key.split('|')
            player = int(player)
            card = int(card)
            
            # Create information set node
            card_name = card_names[card]
            if history == '':
                # Initial decision nodes
                node_id = f"P{player+1}_{card_name}_initial"
                label = f"P{player+1}\n{card_name}"
                y_pos = p1_initial_y if player == 0 else p2_y
                level = 1 if player == 0 else 2
            else:
                # Second decision nodes (after P1 pass, P2 bet)
                node_id = f"P{player+1}_{card_name}_second"
                label = f"P{player+1}\n{card_name}\nafter PB"
                y_pos = p1_second_y
                level = 3
            
            if node_id not in G.nodes():
                x_pos = x_counters[level]
                x_counters[level] += 2
                pos[node_id] = (x_pos, y_pos)
                
                node_type = f"player{player+1}"
                G.add_node(node_id, type=node_type, label=label, level=level)
                information_sets[node_id] = strategy
        
        # Add edges to show game flow (simplified)
        # Connect chance to P1 initial nodes
        for node_id in G.nodes():
            if "P1_" in node_id and "_initial" in node_id:
                G.add_edge("Chance", node_id)
        
        # Define colors
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'chance')
            if node_type == 'chance':
                node_colors.append('#FFD700')  # Gold
                node_sizes.append(1200)
            elif node_type == 'player1':
                node_colors.append('#87CEEB')  # Sky blue
                node_sizes.append(1000)
            elif node_type == 'player2':
                node_colors.append('#FFA07A')  # Light salmon
                node_sizes.append(1000)
            else:
                node_colors.append('gray')
                node_sizes.append(800)
        
        # Draw the graph
        nx.draw(G, pos, ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=False,
                arrows=True,
                edge_color='black',
                width=1.5,
                arrowsize=15)
        
        # Add node labels with strategy information
        for node_id, (x, y) in pos.items():
            base_label = G.nodes[node_id].get('label', node_id)
            
            if node_id in information_sets:
                strategy = information_sets[node_id]
                strategy_text = f"Strategy: {strategy}"
                full_label = base_label + strategy_text
            else:
                full_label = base_label
            
            ax.text(x, y, full_label, fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', 
                      markersize=12, label='Chance'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', 
                      markersize=12, label='Player 1'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA07A', 
                      markersize=12, label='Player 2')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.set_title("Kuhn Poker CFR Learned Strategies", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print strategy summary
        # print("\nStrategy Summary:")
        # print("="*50)
        # for infoset_key, strategy in sorted(strategies.items()):
        #     player, card, history = infoset_key.split('|')
        #     card_name = card_names[int(card)]
        #     history_desc = "initial" if history == '' else f"after {history}"
        #     print(f"Player {int(player)+1}, Card {card_name}, {history_desc}:")
        #     print(f"  Pass: {strategy[0]:.3f}, Bet: {strategy[1]:.3f}")

if __name__ == "__main__":
    trainer = KuhnCFRTrainer(numPlayers=2, numCards=3)
    strategies, game_values = trainer.solve(numIterations=10000, path="kuhn_cfr/strategies")

    with open("kuhn_cfr/strategies/kuhn_cfr_model.pkl", 'rb') as file:
        strategies = pkl.load(file)

    
    for infoset, strategy in strategies.items():
        print(f"Infoset: {infoset}, Average Strategy: {strategy.get_average_strategy()}")

    # trainer.plot_strategies(strategies=strategies)
