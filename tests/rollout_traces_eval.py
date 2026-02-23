"""
Autoregressive Rollout Evaluation: Predicted vs Actual Calcium Traces

Loads a trained GAT-LSTM model and autoregressively rolls it out far beyond
the training horizon to visualise how predictions diverge from ground truth.

Generates:
  1. Neuron trace plots: predicted (orange) vs actual (blue) for selected neurons
  2. Per-step error curve: MSE and R² at every rollout step
  3. Neuron heatmap: side-by-side actual vs predicted for all neurons

Usage:
    python tests/rollout_traces_eval.py                       # defaults (60 steps)
    python tests/rollout_traces_eval.py --rollout-steps 90    # 30 seconds
    python tests/rollout_traces_eval.py --checkpoint models/checkpoints/gat_lstm_best.pt
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.connectomes.connectome_loader import load_connectome
from data.functional.functional_loader import load_functional_data
from data.preprocessing import align_worm_timesteps
from models.gat import GATLSTM


DT = 1 / 3  # seconds per timestep (3 Hz sampling)
OUTPUT_DIR = Path(__file__).parent / "plots"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """Load GATLSTM from a training checkpoint (contains config + state_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", None)
    model = GATLSTM(config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    info = {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "val_r2": ckpt.get("val_r2"),
        "target_horizon": ckpt.get("target_horizon"),
    }
    return model, info


def get_test_worms(df, seed=42):
    """Reproduce the train/val/test worm split used during training."""
    worms = sorted(df["worm"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(worms)
    n = len(worms)
    n_val = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    n_train = n - n_val - n_test
    return list(worms[n_train + n_val:])


@torch.no_grad()
def autoregressive_rollout(model, window, edge_index, edge_attr, rollout_steps, device):
    """
    Roll the model out for *rollout_steps*, feeding each prediction back.

    Args:
        window:       np.ndarray [N, W]  initial input window (calcium values)
        edge_index:   Tensor on device
        edge_attr:    Tensor on device
        rollout_steps: int
        device:       torch device

    Returns:
        preds: np.ndarray [N, rollout_steps]
    """
    x = torch.tensor(window, dtype=torch.float32, device=device)  # [N, W]
    preds = []

    for _ in range(rollout_steps):
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        pred = model(data)  # [N, 1]
        preds.append(pred.squeeze(-1).cpu().numpy())

        if x.shape[1] > 1:
            x = torch.cat([x[:, 1:], pred], dim=1)
        else:
            x = pred

    return np.stack(preds, axis=1)  # [N, rollout_steps]


def select_neurons(matrix, mask, start, rollout_steps, num_neurons=8):
    """Pick neurons with highest variance that are valid throughout the segment."""
    T, N = matrix.shape
    end = min(start + 1 + rollout_steps, T)

    segment = matrix[start:end]
    seg_mask = mask[start:end]
    fully_valid = seg_mask.all(axis=0)

    valid_idx = np.where(fully_valid)[0]
    if len(valid_idx) == 0:
        valid_idx = np.arange(N)

    var = segment[:, valid_idx].var(axis=0)
    top = var.argsort()[::-1][:num_neurons]
    return valid_idx[top]


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_neuron_traces(actual, predicted, neuron_ids, neuron_names,
                       train_horizon, output_path):
    """Plot predicted vs actual calcium traces for selected neurons."""
    n = len(neuron_ids)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    H = predicted.shape[1]
    time_axis = np.arange(1, H + 1) * DT

    for i, nid in enumerate(neuron_ids):
        ax = axes[i // cols, i % cols]
        ax.plot(time_axis, actual[:H, nid], color="#3B82F6", linewidth=1.2,
                label="Actual", alpha=0.85)
        ax.plot(time_axis, predicted[nid], color="#F97316", linewidth=1.2,
                label="Predicted", alpha=0.85)

        if train_horizon is not None:
            bx = train_horizon * DT
            ylo, yhi = ax.get_ylim()
            ax.axvline(bx, color="#94A3B8", ls="--", lw=0.9, alpha=0.7)
            ax.text(bx + 0.15, yhi, f"t+{train_horizon}",
                    fontsize=8, color="#64748B", va="top")

        name = neuron_names[nid] if neuron_names is not None else f"Neuron {nid}"
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Ca\u00B2\u207A", fontsize=9)
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Time (s)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_visible(False)

    fig.suptitle("Autoregressive Rollout: Predicted vs Actual Calcium Traces",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}")


def plot_error_by_step(actual, predicted, mask_segment, train_horizon, output_path):
    """Plot MSE and R² at each rollout step."""
    H = predicted.shape[1]
    mses, r2s = [], []

    for s in range(H):
        pred_s = predicted[:, s]
        true_s = actual[s]
        m = mask_segment[s].astype(bool) if mask_segment is not None else np.ones(len(true_s), dtype=bool)
        p, t = pred_s[m], true_s[m]
        if len(p) == 0:
            mses.append(np.nan)
            r2s.append(np.nan)
            continue
        mse = float(np.mean((p - t) ** 2))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mses.append(mse)
        r2s.append(r2)

    time_s = np.arange(1, H + 1) * DT

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(time_s, mses, color="#EF4444", linewidth=1.5)
    ax1.set_xlabel("Prediction horizon (s)", fontsize=11)
    ax1.set_ylabel("MSE", fontsize=11)
    ax1.set_title("Error Accumulation Over Rollout", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.25)

    ax2.plot(time_s, r2s, color="#10B981", linewidth=1.5)
    ax2.set_xlabel("Prediction horizon (s)", fontsize=11)
    ax2.set_ylabel("R\u00B2", fontsize=11)
    ax2.set_title("R\u00B2 Over Rollout", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        if train_horizon is not None:
            bx = train_horizon * DT
            ax.axvline(bx, color="#94A3B8", ls="--", lw=0.9, alpha=0.7)
            yl = ax.get_ylim()
            ax.text(bx + 0.15, yl[1], f"training horizon (t+{train_horizon})",
                    fontsize=8, color="#64748B", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}")


def plot_heatmap(actual, predicted, neuron_names, train_horizon, output_path,
                 max_neurons=60):
    """Side-by-side heatmaps: actual vs predicted activity for all neurons."""
    H = predicted.shape[1]
    N = predicted.shape[0]
    idx = np.arange(min(N, max_neurons))

    act_seg = actual[:H, idx].T   # [n, H]
    pred_seg = predicted[idx, :]   # [n, H]

    vmin = min(float(act_seg.min()), float(pred_seg.min()))
    vmax = max(float(act_seg.max()), float(pred_seg.max()))

    time_s = np.arange(1, H + 1) * DT
    names = [neuron_names[i] if neuron_names is not None else str(i) for i in idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(idx) * 0.18)),
                                    sharey=True)

    im1 = ax1.imshow(act_seg, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                      extent=[time_s[0], time_s[-1], len(idx) - 0.5, -0.5])
    ax1.set_title("Actual", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron")
    ax1.set_yticks(range(0, len(idx), max(1, len(idx) // 20)))

    im2 = ax2.imshow(pred_seg, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                      extent=[time_s[0], time_s[-1], len(idx) - 0.5, -0.5])
    ax2.set_title("Predicted (Autoregressive)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")

    for ax in (ax1, ax2):
        if train_horizon is not None:
            ax.axvline(train_horizon * DT, color="black", ls="--", lw=0.8, alpha=0.6)

    fig.colorbar(im2, ax=[ax1, ax2], label="Ca\u00B2\u207A activity", shrink=0.6)
    fig.suptitle("Neuron Activity Heatmap: Actual vs Autoregressive Prediction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate autoregressive rollout with trace visualisation")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/gat_lstm_best.pt",
                        help="Path to training checkpoint")
    parser.add_argument("--rollout-steps", type=int, default=60,
                        help="Total steps to unroll (default 60 ~ 20 s)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Input window size (must match training)")
    parser.add_argument("--max-worms", type=int, default=None,
                        help="Max worms to load (None = all)")
    parser.add_argument("--worm-idx", type=int, default=0,
                        help="Which test-set worm to visualise (0 = first)")
    parser.add_argument("--start-t", type=int, default=None,
                        help="Starting timestep inside the worm (default: 25%% into series)")
    parser.add_argument("--num-neurons", type=int, default=8,
                        help="Number of neuron traces to plot")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = setup_device()
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUTOREGRESSIVE ROLLOUT EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Rollout: {args.rollout_steps} steps ({args.rollout_steps * DT:.1f} s)")
    print(f"Window:  {args.window_size}")

    # 1. Load model ---------------------------------------------------------
    print("\n1. Loading model...")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent / ckpt_path
    if not ckpt_path.exists():
        print(f"   ERROR: checkpoint not found at {ckpt_path}")
        return

    model, info = load_model(ckpt_path, device)
    train_horizon = info.get("target_horizon")
    print(f"   Params:  {model.num_params:,}")
    print(f"   Epoch:   {info.get('epoch')}")
    print(f"   Val R2:  {info.get('val_r2', '?')}")
    print(f"   Trained: t+{train_horizon}")

    # 2. Load data ----------------------------------------------------------
    print("\n2. Loading data...")
    connectome = load_connectome(verbose=False)
    df = load_functional_data(connectome=connectome, verbose=False)
    neuron_list = list(connectome.node_label)
    edge_index = connectome.edge_index.to(device)
    edge_attr = connectome.edge_attr.to(device)
    N = len(neuron_list)
    print(f"   Neurons: {N}, Edges: {connectome.num_edges}")
    print(f"   Worms available: {df['worm'].nunique()}")

    # 3. Pick a test worm ---------------------------------------------------
    print("\n3. Selecting test worm...")
    test_worms = get_test_worms(df)
    if len(test_worms) == 0:
        print("   ERROR: no test worms found")
        return
    worm_idx = min(args.worm_idx, len(test_worms) - 1)
    worm_id = test_worms[worm_idx]
    print(f"   Test worms: {test_worms}")
    print(f"   Selected:   {worm_id} (index {worm_idx})")

    worm_df = df[df["worm"] == worm_id]
    matrix, mask, timestamps = align_worm_timesteps(worm_df, neuron_list)
    if matrix is None:
        print("   ERROR: could not align worm timesteps")
        return
    T = matrix.shape[0]
    print(f"   Timesteps: {T} ({T * DT:.1f} s)")

    # 4. Determine start point ---------------------------------------------
    W = args.window_size
    H = args.rollout_steps
    latest_safe_start = T - 1 - H
    earliest_start = W - 1

    if latest_safe_start < earliest_start:
        H = T - 1 - earliest_start
        latest_safe_start = earliest_start
        print(f"   (clamped rollout to {H} steps to fit timeseries)")

    if args.start_t is not None:
        t0 = max(earliest_start, min(args.start_t, latest_safe_start))
    else:
        t0 = earliest_start + (latest_safe_start - earliest_start) // 4

    print(f"   Start: t={t0} ({t0 * DT:.1f} s)")
    print(f"   Rollout: {H} steps -> t={t0 + H} ({(t0 + H) * DT:.1f} s)")

    # 5. Run rollout --------------------------------------------------------
    print("\n4. Running autoregressive rollout...")
    window_init = matrix[t0 - W + 1: t0 + 1].T  # [N, W]
    predicted = autoregressive_rollout(
        model, window_init, edge_index, edge_attr, H, device
    )  # [N, H]

    actual_segment = matrix[t0 + 1: t0 + 1 + H]  # [H, N]
    mask_segment = mask[t0 + 1: t0 + 1 + H]       # [H, N]
    print(f"   Done. Predicted shape: {predicted.shape}")

    # 6. Select interesting neurons -----------------------------------------
    neuron_ids = select_neurons(matrix, mask, t0, H, num_neurons=args.num_neurons)
    print(f"   Selected neurons: {[neuron_list[i] for i in neuron_ids]}")

    # 7. Plot ---------------------------------------------------------------
    print("\n5. Generating plots...")
    tag = f"ar_K{train_horizon}" if train_horizon else "ar"

    plot_neuron_traces(
        actual_segment, predicted, neuron_ids, neuron_list,
        train_horizon=train_horizon,
        output_path=out_dir / f"{tag}_neuron_traces.png",
    )

    plot_error_by_step(
        actual_segment, predicted, mask_segment, train_horizon,
        output_path=out_dir / f"{tag}_error_by_step.png",
    )

    plot_heatmap(
        actual_segment, predicted, neuron_list, train_horizon,
        output_path=out_dir / f"{tag}_heatmap.png",
    )

    # 8. Summary metrics ----------------------------------------------------
    print("\n6. Summary metrics")
    print(f"   {'Horizon':<10} {'MSE':>10} {'R2':>10}")
    print(f"   {'-' * 30}")
    for label, step in [("t+1", 0), ("t+5", 4), ("t+10", 9),
                        ("t+20", 19), ("t+30", 29), ("t+60", 59)]:
        if step >= H:
            break
        p = predicted[:, step]
        t_gt = actual_segment[step]
        m = mask_segment[step].astype(bool)
        p, t_gt = p[m], t_gt[m]
        if len(p) == 0:
            continue
        mse = float(np.mean((p - t_gt) ** 2))
        ss_res = float(np.sum((t_gt - p) ** 2))
        ss_tot = float(np.sum((t_gt - t_gt.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"   {label:<10} {mse:>10.4f} {r2:>10.4f}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
