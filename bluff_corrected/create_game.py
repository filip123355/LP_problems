"""
Corrected implementation of Bluff (Liar's Dice) game matrix generation.

Key fixes:
1. Each player knows their own dice roll
2. Game matrix should be computed for ALL combinations of both players' dice
3. Proper information sets based on (own_dice, history)
4. Correct payoff computation considering both players' dice
"""

import numpy as np
import itertools
import os
import pickle as pkl
from typing import Dict, Tuple, List
from time import time

try:
    from .game_constants import NUM_FACES, NUM_DICES, NUM_PLAYERS, TOTAL_CLAIMS, EPSILON
except ImportError:
    from game_constants import NUM_FACES, NUM_DICES, NUM_PLAYERS, TOTAL_CLAIMS, EPSILON


class InformationSet:
    """Represents an information set in the extensive form game."""
    def __init__(self, player: int, own_dice: Tuple[int, ...], history: Tuple[int, ...]):
        self.player = player
        self.own_dice = own_dice
        self.history = history
        self.key = f"{player}|{own_dice}|{history}"
        
    def get_valid_actions(self) -> List[int]:
        """Get valid actions from this information set."""
        if not self.history:
            # First move: can claim any quantity-face combination
            return list(range(TOTAL_CLAIMS))
        else:
            last_claim = self.history[-1]
            if last_claim >= TOTAL_CLAIMS:
                return []  # Game is over
            # Can make higher claims or call bluff
            return list(range(last_claim + 1, TOTAL_CLAIMS + 1))  # +1 for bluff call
    
    def __str__(self):
        return self.key
    
    def __hash__(self):
        return hash(self.key)
    
    def __eq__(self, other):
        return self.key == other.key


class BluffGameMatrix:
    """
    Corrected implementation of Bluff game using proper extensive form.
    Each player knows their own dice roll but not opponent's.
    """
    
    def __init__(self, num_dices: int = NUM_DICES, num_faces: int = NUM_FACES):
        self.num_dices = num_dices
        self.num_faces = num_faces
        self.num_players = NUM_PLAYERS
        
        # All possible dice rolls for one player
        self.dice_rolls = list(itertools.product(range(1, num_faces + 1), repeat=num_dices))
        
        # Information sets for each player
        self.information_sets: Dict[str, InformationSet] = {}
        
        # Sequence form: maps (player, sequence) to index
        self.sequences: Dict[Tuple[int, str], int] = {}
        self.sequence_count = [0, 0]  # Count for each player
        
        # Payoff matrices for each combination of dice rolls
        self.payoff_matrices: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}
        
    def claim_to_quantity_face(self, claim: int) -> Tuple[int, int]:
        """Convert claim index to (quantity, face) pair."""
        if claim >= TOTAL_CLAIMS:
            return (-1, -1)  # Bluff call
        quantity = claim // self.num_faces + 1
        face = claim % self.num_faces + 1
        return quantity, face
    
    def is_claim_true(self, dice1: Tuple[int, ...], dice2: Tuple[int, ...], 
                      quantity: int, face: int) -> bool:
        """Check if claim is true given both players' dice."""
        all_dice = list(dice1) + list(dice2)
        actual_count = all_dice.count(face)
        return actual_count >= quantity
    
    def build_information_sets(self):
        """Build all information sets for both players."""
        print("Building information sets...")
        
        def explore_game_tree(history: List[int], current_player: int):
            """Recursively explore the game tree to find all information sets."""
            if len(history) > 0 and history[-1] >= TOTAL_CLAIMS:
                # Game ended with bluff call
                return
                
            if len(history) > 10:  # Limit game length
                return
                
            # Create information sets for current player with each possible dice roll
            for dice_roll in self.dice_rolls:
                info_set = InformationSet(current_player, dice_roll, tuple(history))
                if info_set.key not in self.information_sets:
                    self.information_sets[info_set.key] = info_set
                
                # Get valid actions and continue exploration
                valid_actions = info_set.get_valid_actions()
                for action in valid_actions:
                    new_history = history + [action]
                    next_player = 1 - current_player
                    explore_game_tree(new_history, next_player)
        
        # Start exploration from empty history, player 0
        explore_game_tree([], 0)
        print(f"Created {len(self.information_sets)} information sets")
    
    def build_sequences(self):
        """Build sequence form representation."""
        print("Building sequence form...")
        
        # Add empty sequence for each player
        for player in range(self.num_players):
            self.sequences[(player, "")] = self.sequence_count[player]
            self.sequence_count[player] += 1
        
        # Build sequences by extending from information sets
        for info_set in self.information_sets.values():
            player = info_set.player
            
            # Convert history to sequence string
            if len(info_set.history) == 0:
                parent_seq = ""
            else:
                # Get the sequence of actions by this player
                player_actions = []
                for i, action in enumerate(info_set.history):
                    if i % 2 == player:  # This player's turn
                        player_actions.append(str(action))
                parent_seq = ",".join(player_actions)
            
            # Add sequences for each valid action
            valid_actions = info_set.get_valid_actions()
            for action in valid_actions:
                if parent_seq == "":
                    new_seq = str(action)
                else:
                    new_seq = parent_seq + "," + str(action)
                
                seq_key = (player, new_seq)
                if seq_key not in self.sequences:
                    self.sequences[seq_key] = self.sequence_count[player]
                    self.sequence_count[player] += 1
        
        print(f"Player 0 sequences: {self.sequence_count[0]}")
        print(f"Player 1 sequences: {self.sequence_count[1]}")
    
    def compute_payoff_matrix(self, dice1: Tuple[int, ...], dice2: Tuple[int, ...]) -> np.ndarray:
        """Compute payoff matrix for specific dice combination."""
        n_seq_0 = self.sequence_count[0]
        n_seq_1 = self.sequence_count[1]
        payoff_matrix = np.zeros((n_seq_0, n_seq_1))
        
        # For each combination of sequences, compute the expected payoff
        for (p0_seq, p0_idx) in [(seq, idx) for (p, seq), idx in self.sequences.items() if p == 0]:
            for (p1_seq, p1_idx) in [(seq, idx) for (p, seq), idx in self.sequences.items() if p == 1]:
                # Reconstruct the game history from both sequences
                payoff = self.compute_sequence_payoff(dice1, dice2, p0_seq, p1_seq)
                payoff_matrix[p0_idx, p1_idx] = payoff
        
        return payoff_matrix
    
    def compute_sequence_payoff(self, dice1: Tuple[int, ...], dice2: Tuple[int, ...], 
                               seq0: str, seq1: str) -> float:
        """Compute payoff for specific sequence combination."""
        if seq0 == "" or seq1 == "":
            return 0.0  # No complete game
        
        # Parse sequences into actions
        actions0 = [] if seq0 == "" else [int(x) for x in seq0.split(",")]
        actions1 = [] if seq1 == "" else [int(x) for x in seq1.split(",")]
        
        # Reconstruct full game history
        history = []
        i0 = i1 = 0
        player = 0
        
        while i0 < len(actions0) or i1 < len(actions1):
            if player == 0 and i0 < len(actions0):
                history.append(actions0[i0])
                i0 += 1
            elif player == 1 and i1 < len(actions1):
                history.append(actions1[i1])
                i1 += 1
            else:
                break
            player = 1 - player
        
        if not history:
            return 0.0
        
        # Check if game ended with bluff call
        last_action = history[-1]
        if last_action >= TOTAL_CLAIMS:  # Bluff call
            if len(history) < 2:
                return 0.0  # Invalid game
            
            challenged_claim = history[-2]
            quantity, face = self.claim_to_quantity_face(challenged_claim)
            
            if quantity == -1:  # Invalid claim
                return 0.0
            
            # Determine who made the claim and who called bluff
            claimant = (len(history) - 2) % 2
            challenger = (len(history) - 1) % 2
            
            # Check if claim is true
            claim_is_true = self.is_claim_true(dice1, dice2, quantity, face)
            
            if claim_is_true:
                # Claimant wins
                return 1.0 if claimant == 0 else -1.0
            else:
                # Challenger wins
                return 1.0 if challenger == 0 else -1.0
        
        return 0.0  # Game not finished
    
    def build_all_payoff_matrices(self):
        """Build payoff matrices for all dice combinations."""
        print("Building payoff matrices...")
        
        total_combinations = len(self.dice_rolls) ** 2
        count = 0
        
        for dice1 in self.dice_rolls:
            for dice2 in self.dice_rolls:
                self.payoff_matrices[(dice1, dice2)] = self.compute_payoff_matrix(dice1, dice2)
                count += 1
                if count % 10 == 0:
                    print(f"Computed {count}/{total_combinations} payoff matrices")
    
    def save_matrices(self, path: str = "bluff_corrected"):
        """Save all generated matrices and data."""
        os.makedirs(path, exist_ok=True)
        
        # Save sequences
        with open(f"{path}/sequences.pkl", "wb") as f:
            pkl.dump(self.sequences, f)
        
        # Save information sets
        info_set_data = {key: {
            'player': info_set.player,
            'own_dice': info_set.own_dice,
            'history': info_set.history
        } for key, info_set in self.information_sets.items()}
        
        with open(f"{path}/information_sets.pkl", "wb") as f:
            pkl.dump(info_set_data, f)
        
        # Save payoff matrices
        os.makedirs(f"{path}/payoff_matrices", exist_ok=True)
        for (dice1, dice2), matrix in self.payoff_matrices.items():
            filename = f"payoff_{dice1}_{dice2}.npy"
            np.save(f"{path}/payoff_matrices/{filename}", matrix)
        
        # Save dice rolls
        with open(f"{path}/dice_rolls.pkl", "wb") as f:
            pkl.dump(self.dice_rolls, f)
        
        print(f"Saved all data to {path}/")
    
    def build_all(self):
        """Build complete game representation."""
        start_time = time()
        
        self.build_information_sets()
        self.build_sequences()
        self.build_all_payoff_matrices()
        
        end_time = time()
        print(f"Total build time: {end_time - start_time:.2f} seconds")
        
        return self


if __name__ == "__main__":
    print("Building corrected Bluff game representation...")
    
    game = BluffGameMatrix(num_dices=NUM_DICES, num_faces=NUM_FACES)
    game.build_all()
    game.save_matrices()
    
    print("Done!")
    print(f"Information sets: {len(game.information_sets)}")
    print(f"Player 0 sequences: {game.sequence_count[0]}")
    print(f"Player 1 sequences: {game.sequence_count[1]}")
    print(f"Payoff matrices: {len(game.payoff_matrices)}")
