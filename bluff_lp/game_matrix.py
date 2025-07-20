import numpy as np 
import matplotlib.pyplot as plt 
import os
import pickle as pkl
import itertools

from time import time
from bluff_lp.constants import (NUM_FACES, NUM_DICES)
from tqdm import tqdm
class Buffer: 
    """
    The buffer for keeping track what index corresponds to what strategy.
    """
    def __init__(self):
        self.data = dict()
        self.last_ind: int = 0
    
    def add(self, code: str):
        if code not in self.data.keys():
            self.data[code] = self.last_ind
            self.last_ind += 1
            
    def __getitem__(self, code: int):
        return self.data[code]
    
    def __len__(self):
        return len(self.data)

class GameMatrix:
    """
    Class with the game matrix representing a normal form a Bluff through network flow approach.
    """
    game_matrix: np.ndarray
    hashmap: dict
    x_vec: np.ndarray
    y_vec: np.ndarray
    x_constraints: np.ndarray
    y_constraints: np.ndarray
    
    def __init__(self, 
                num_dices: int, 
                num_faces: int,
                x_roll: tuple|None=None):
        self.num_dices = num_dices
        self.num_faces = num_faces
        self.b = num_faces * num_dices * 2 # Multiplied by two take into account two players
        # Initialize fresh buffers per instance
        self.x_buffer = Buffer()
        self.y_buffer = Buffer()
        if x_roll is None:
            self.x_roll = tuple(np.random.randint(1, self.num_faces + 1, size=self.num_dices))
        else:
            self.x_roll = x_roll if isinstance(x_roll, (tuple, list, np.ndarray)) else (x_roll,)
        
    def roll(self):
        self.x_roll = tuple(np.random.randint(1, self.num_faces + 1, size=self.num_dices))

    def bid_decode(self, index: int) -> tuple[int, int]:
        total_faces = self.num_faces
        dice_call = index // total_faces + 1
        face_call = index % total_faces + 1
        return dice_call, face_call

    @staticmethod
    def zero_even_ones(binary_str: str):
        result = []
        one_count = 0
        for bit in binary_str[2:]:
            if bit == '1':
                one_count += 1
                result.append('0' if one_count % 2 == 0 else '1')
            else:
                result.append('0')
        return ''.join(result)
    
    def decode_strategy(self, 
                        strategy_rep:str,
                        ) -> tuple[str, str]:
        strategy_ind = int(strategy_rep, 2)
        odd = self.zero_even_ones(strategy_rep)
        strategy_odd_ind = int(odd, 2)
        strategy_even_ind = strategy_ind - strategy_odd_ind
        even = bin(strategy_even_ind)[2:]
        return odd, even

    def game_value(self, 
                dice_call: int, 
                face_call: int,
                player: bool,
                ) -> float:
        g_v = 0
        for y_roll in itertools.product(range(1, self.num_faces + 1), repeat=self.num_dices):
            rolls = list(y_roll) + list(self.x_roll)
            actual_count = rolls.count(face_call)
            if actual_count >= dice_call:
                g_v += 1
            else:
                g_v -= 1
        g_v /= self.num_faces ** self.num_dices
        return g_v if player else -g_v
    
    def build_constraints(self, verbose: bool=False):
        
        def _build(buffer: Buffer) -> np.ndarray:
            constraints = np.zeros((2, len(buffer) + 1))
            constraints[0, 0] = 1
            for root in buffer.data.keys():
                list_root = list(root.zfill(self.b + 1))
                unos_root = [i for i, x in enumerate(list_root) if x == '1']
                if len(unos_root) == 1:
                    constraints[1, 0] = -1
                    constraints[1, buffer.data[root] + 1] = 1
                to_add = np.zeros_like(constraints[0, :])
                ind = False
                for element in buffer.data.keys():
                    list_element = list(element.zfill(self.b + 1))
                    unos_element = [i for i, x in enumerate(list_element) if x == '1']
                    el_ind = unos_element[-1]
                    list_root_changed = list_root.copy()
                    list_root_changed[el_ind] = '1'
                    if list_element == list_root_changed and list_element != list_root:
                        to_add[buffer.data[root] + 1] = -1
                        to_add[buffer.data[element] + 1] = 1
                        ind = True
                if ind:
                    constraints = np.concatenate((constraints, to_add[np.newaxis, :]), axis=0)
            return constraints
        
        start = time()     
        self.x_constraints = _build(self.x_buffer)
        self.y_constraints = _build(self.y_buffer)
        self.x_vec = np.repeat(0, self.x_constraints.shape[0])
        self.y_vec = np.repeat(0, self.y_constraints.shape[0])
        self.x_vec[0] = 1
        self.y_vec[0] = 1
        stop = time()
        if verbose:
            print(f"\nFilling constraint matrices in time: {stop - start} s")
        
    def build_buffers(self):
        for index in range(1, 2 ** self.b):
            strategy_rep = bin(index) + '1'
            x_strategy, y_strategy = self.decode_strategy(strategy_rep)
            if x_strategy not in self.x_buffer.data.keys():
                self.x_buffer.add(x_strategy)
            if y_strategy not in self.y_buffer.data.keys():
                self.y_buffer.add(y_strategy)
        
    def build(self, verbose: bool=False):
        
        start_fill_game_matrix = time()
        self.game_matrix = np.zeros((len(self.x_buffer), len(self.y_buffer)))

        for index in range(1, 2 ** self.b):
            strategy_rep = bin(index)
            strategy_rep_list = list(strategy_rep[2:].zfill(self.b)) 
            unos = [i for i, x in enumerate(strategy_rep_list) if x == '1']
            last_one = unos[-1]
            player = True if len(unos) % 2 == 1 else False # True if the first player bids last
            dice_call, face_call = self.bid_decode(last_one)
            g_v = self.game_value(dice_call, face_call, player)
            strategy_rep += '1'
            x_strategy, y_strategy = self.decode_strategy(strategy_rep)
            x_ind, y_ind = self.x_buffer.data[x_strategy], self.y_buffer.data[y_strategy]
            self.game_matrix[x_ind, y_ind] = g_v
        
        # Adding blank row and column for null node   
        self.game_matrix = np.concatenate((np.repeat(0, self.game_matrix.shape[0]).reshape(-1, 1), self.game_matrix), axis=1)
        self.game_matrix = np.concatenate((np.repeat(0, self.game_matrix.shape[1]).reshape(1, -1), self.game_matrix), axis=0)
        stop_fill_game_matrix = time()
        
        if verbose:
            print(f"\nFilling game matrix in time: {stop_fill_game_matrix - start_fill_game_matrix} s")
            
    def plot(self):
        plt.matshow(self.game_matrix)
        plt.colorbar()
        plt.show()
    
    def save(self, 
            name: str,
            path :str="game_matrices",
            verbose: bool=False):
        np.save(f"{path}/{name}", self.game_matrix)
        if verbose:
            print(f"\nMatrix saved under: {path}/{name}")
    
    def save_constraints(self,
                        path : str="game_constraints", 
                        verbose: bool=False):
        np.save(f"{path}/x.npy", self.x_constraints)
        np.save(f"{path}/y.npy", self.y_constraints)
        if verbose: 
            print("\nConstraints saved")

if __name__ == "__main__":
    num_faces = NUM_FACES
    num_dices = NUM_DICES
    verbose = False
    for x_roll in itertools.product(range(1, num_faces + 1), repeat=num_dices):
        gm = GameMatrix(num_dices=num_dices, num_faces=num_faces, x_roll=x_roll)
        gm.build_buffers()
        gm.build(verbose=verbose)
        os.makedirs(f"bluff_lp/game_matrices/{num_dices}_{num_faces}f", exist_ok=True)
        gm.save(name=f"{x_roll}_{num_faces}.npy",
                path=f"bluff_lp/game_matrices/{num_dices}_{num_faces}f", verbose=verbose)
        
    gm.build_constraints()
    os.makedirs(f"bluff_lp/game_constraints/{num_dices}_{num_faces}f", exist_ok=True)
    gm.save_constraints(path=f"bluff_lp/game_constraints/{num_dices}_{num_faces}f")
    # print(gm.x_buffer.data)
    # print(gm.y_buffer.data)
    # gm.plot()
    
    # Saving the buffers
    os.makedirs(f"bluff_lp/buffers/{num_dices}_{num_faces}f", exist_ok=True)
    with open(f"bluff_lp/buffers/{num_dices}_{num_faces}f/x_buffer.pkl", "wb") as f:
        pkl.dump(gm.x_buffer.data, f)
    with open(f"bluff_lp/buffers/{num_dices}_{num_faces}f/y_buffer.pkl", "wb") as f:
        pkl.dump(gm.y_buffer.data, f)