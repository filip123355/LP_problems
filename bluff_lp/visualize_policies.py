import os
import math
import pickle as pkl
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from bluff_lp.constants import NUM_DICES, NUM_FACES


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_strategy(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return np.load(path)


def _load_buffers(num_dices: int, num_faces: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    buf_dir = f"bluff_lp/buffers/{num_dices}_{num_faces}f"
    with open(os.path.join(buf_dir, "x_buffer.pkl"), "rb") as fx:
        x_buf = pkl.load(fx)
    with open(os.path.join(buf_dir, "y_buffer.pkl"), "rb") as fy:
        y_buf = pkl.load(fy)
    return x_buf, y_buf


def _decode_last_bid_from_code(code: str, num_faces: int, b: int) -> Optional[Tuple[int, int]]:
    """
    Given a buffer key `code` (bitstring of varying length), zero-left-pad to b+1, then:
    - Ignore the trailing sentinel bit at index b if present.
    - Take the last remaining '1' position as the last chosen action.
    - Map index -> (dice_call, face_call).
    Returns None if no valid action is encoded.
    """
    s = code.zfill(b + 1)
    ones = [i for i, ch in enumerate(s) if ch == "1"]
    # Drop potential sentinel at the end
    ones = [i for i in ones if i < b]
    if not ones:
        return None
    last_idx = ones[-1]
    dice_call = last_idx // num_faces + 1
    face_call = last_idx % num_faces + 1
    return dice_call, face_call


def _aggregate_action_mass(
    strat: np.ndarray,
    base_buffer: Dict[str, int],
    num_dices: int,
    num_faces: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (per_block_action_mass, overall_action_mass):
    - per_block_action_mass shape: [B, num_dices, num_faces]
    - overall_action_mass shape: [num_dices, num_faces]
    where B = num_dices * num_faces (as used in constraint expansion).
    """
    B = num_dices * num_faces
    b = num_dices * num_faces * 2  # as in GameMatrix.b (two players)

    # infer base_len and validate shape
    base_len = len(base_buffer)
    if strat.shape[0] < 1 + base_len:
        raise ValueError(
            f"Strategy length {strat.shape[0]} too small for base buffer length {base_len}."
        )

    # Try to infer the block count from data if needed
    if (strat.shape[0] - 1) % base_len != 0:
        # Fallback to a best-effort: use integer division
        inferred_B = (strat.shape[0] - 1) // base_len
    else:
        inferred_B = (strat.shape[0] - 1) // base_len

    if inferred_B != B:
        # Work with inferred_B but warn via print; common when generalizing beyond 1 die
        print(f"[warn] Expected blocks B={B}, but inferred {inferred_B} from strategy length. Using inferred value.")
        B = inferred_B

    # Map index->code
    idx2code = {idx: code for code, idx in base_buffer.items()}

    # Drop the root variable, reshape into blocks
    body = strat[1:]
    blocks = body.reshape(B, base_len)

    per_block = np.zeros((B, num_dices, num_faces), dtype=float)

    for bidx in range(B):
        for j in range(base_len):
            mass = blocks[bidx, j]
            if mass <= 0:
                continue
            code = idx2code[j]
            dec = _decode_last_bid_from_code(code, num_faces, b)
            if dec is None:
                continue
            dice_call, face_call = dec
            if 1 <= dice_call <= num_dices and 1 <= face_call <= num_faces:
                per_block[bidx, dice_call - 1, face_call - 1] += mass

    overall = per_block.sum(axis=0)
    return per_block, overall


def _normalize_safe(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    if axis is None:
        s = arr.sum()
        return arr / s if s > 0 else arr
    s = arr.sum(axis=axis, keepdims=True)
    out = np.divide(arr, s, out=np.zeros_like(arr), where=s > 0)
    return out


def _plot_heatmap(mat: np.ndarray, title: str, save_path: Optional[str] = None):
    plt.figure(figsize=(1.6 * mat.shape[1] + 2, 1.6 * mat.shape[0] + 2))
    im = plt.imshow(mat, cmap="viridis")
    plt.title(title)
    plt.xlabel("face_call (1..F)")
    plt.ylabel("dice_call (1..D)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(mat.shape[1]), [str(i + 1) for i in range(mat.shape[1])])
    plt.yticks(range(mat.shape[0]), [str(i + 1) for i in range(mat.shape[0])])
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def _plot_blocks_grid(per_block: np.ndarray, title_prefix: str, out_dir: str):
    B, D, F = per_block.shape
    cols = math.ceil(math.sqrt(B))
    rows = math.ceil(B / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols + 1, 2.2 * rows + 1))
    axes = np.array(axes).reshape(rows, cols)
    vmax = per_block.max() if per_block.size else 1.0
    for bidx in range(B):
        r, c = divmod(bidx, cols)
        ax = axes[r, c]
        im = ax.imshow(per_block[bidx], cmap="viridis", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
        ax.set_title(f"block {bidx}")
        ax.set_xlabel("face")
        ax.set_ylabel("dice")
        ax.set_xticks(range(F))
        ax.set_xticklabels([str(i + 1) for i in range(F)])
        ax.set_yticks(range(D))
        ax.set_yticklabels([str(i + 1) for i in range(D)])
    # Hide unused axes
    for bidx in range(B, rows * cols):
        r, c = divmod(bidx, cols)
        axes[r, c].axis("off")
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{title_prefix.replace(' ', '_').lower()}_blocks.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def visualize_policies(save_dir: str | None = None, normalize: bool = True):
    save_dir = save_dir or f"bluff_lp/images/strategies/{NUM_DICES}_{NUM_FACES}f"
    _ensure_dir(save_dir)

    # Load strategies
    sol_dir = f"bluff_lp/solutions/{NUM_DICES}_{NUM_FACES}"
    x_path = os.path.join(sol_dir, "strategy.npy")
    y_path = os.path.join(sol_dir, "strategy_dual.npy")

    x_strat = _load_strategy(x_path)
    y_strat = _load_strategy(y_path)

    if x_strat is None and y_strat is None:
        raise FileNotFoundError("No strategies found. Run bluff_lp.solve first.")

    # Load buffers
    x_buf, y_buf = _load_buffers(NUM_DICES, NUM_FACES)

    if x_strat is not None:
        xb_per_block, xb_overall = _aggregate_action_mass(x_strat, x_buf, NUM_DICES, NUM_FACES)
        if normalize:
            xb_per_block = _normalize_safe(xb_per_block, axis=(1, 2))
            xb_overall = _normalize_safe(xb_overall)
        _plot_blocks_grid(xb_per_block, "X policy per block", save_dir)
        _plot_heatmap(xb_overall, f"X overall policy D={NUM_DICES} F={NUM_FACES}",
                      os.path.join(save_dir, "x_overall_policy.png"))

    if y_strat is not None:
        yb_per_block, yb_overall = _aggregate_action_mass(y_strat, y_buf, NUM_DICES, NUM_FACES)
        if normalize:
            yb_per_block = _normalize_safe(yb_per_block, axis=(1, 2))
            yb_overall = _normalize_safe(yb_overall)
        _plot_blocks_grid(yb_per_block, "Y policy per block", save_dir)
        _plot_heatmap(yb_overall, f"Y overall policy D={NUM_DICES} F={NUM_FACES}",
                      os.path.join(save_dir, "y_overall_policy.png"))

    print(f"Saved policy visualizations to: {save_dir}")


if __name__ == "__main__":
    visualize_policies()
