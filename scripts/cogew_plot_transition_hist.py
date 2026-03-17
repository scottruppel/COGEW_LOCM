"""Plot histogram of per-episode transition-window max Q for LOCM vs conventional."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogram of max Q in transition window (per episode).")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing episode_distribution.npz (default experiments/cogew_demo).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (PDF saved alongside).",
    )
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    exp_dir = root / "experiments" / "cogew_demo"
    data_dir = Path(args.data_dir) if args.data_dir else exp_dir
    out_path = Path(args.out) if args.out else (data_dir / "transition_maxQ_hist.png")

    d = np.load(data_dir / "episode_distribution.npz")
    locm = d["locm_max_Q_window"].astype(float)
    conv = d["conv_max_Q_window"].astype(float)
    adapt_ms = float(d["adapt_time_ms"])
    half_w = float(d["window_half_width_ms"])
    n = int(d["n_episodes"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.hist(locm, bins=args.bins, alpha=0.65, color="#9467bd", label=f"LOCM-EW (n={n})")
    ax.hist(conv, bins=args.bins, alpha=0.55, color="#1f77b4", label=f"Conventional (n={n})")

    ax.set_title(f"Per-episode max Q in transition window (±{half_w:.3f} ms around {adapt_ms:.1f} ms)")
    ax.set_xlabel("max Q in window")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    print("Saved", out_path, "and", out_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

