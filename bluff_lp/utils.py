import pickle as pkl
import numpy as np
import os
from bluff_lp.constants import NUM_FACES, NUM_DICES

def decode(code: str, num_faces: int, num_dices: int) -> list:
    """
    Decode a binary string into a tuple of (dice_call, face_call).
    """
    code_filled = code.zfill(num_faces * num_dices * 2 + 1)
    ones = [i for i, c in enumerate(code_filled) if c == '1']
    return [(i // num_faces + 1, i % num_faces + 1) if i < num_faces * num_dices * 2
            else "Bluff" for i in ones]

def convert_buffer_to_strategies(num_faces: int, 
                                num_dices: int,
                                ) -> tuple[dict, dict]:
    """Convert the buffer ont o a proper strategy."""
    x_buffer = pkl.load(open(
        f"bluff_lp/buffers/{num_dices}_{num_faces}f/x_buffer.pkl", "rb"))
    y_buffer = pkl.load(open(
        f"bluff_lp/buffers/{num_dices}_{num_faces}f/y_buffer.pkl", "rb"))
    x_strategies, y_strategies = dict(), dict()
    for code, ind in x_buffer.items():
        x_strategies[ind + 1] = decode(code, num_faces, num_dices)
    for code, ind in y_buffer.items():
        y_strategies[ind + 1] = decode(code, num_faces, num_dices)
    
    os.makedirs(f"bluff_lp/strategies/{num_dices}_{num_faces}f", exist_ok=True)
    with open(f"bluff_lp/strategies/{num_dices}_{num_faces}f/x_strategies.pkl", "wb") as f:
        pkl.dump(x_strategies, f)
    with open(f"bluff_lp/strategies/{num_dices}_{num_faces}f/y_strategies.pkl", "wb") as f:
        pkl.dump(y_strategies, f)
    
    return x_strategies, y_strategies

if __name__ == "__main__":
    """Running come tests."""
    x_strategies, y_strategies = convert_buffer_to_strategies(
        num_faces=NUM_FACES, num_dices=NUM_DICES)
    # print(f"x_strategies: {x_strategies}")
    # print(f"y_strategies: {y_strategies}")
