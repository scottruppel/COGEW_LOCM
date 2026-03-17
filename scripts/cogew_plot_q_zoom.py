"""Plot a zoomed-in Q(t) panel around the transition time.

Creates a single-panel plot over +/- window_ms around adapt_time_ms (default 150 ms).
Intended to make the per-episode separation visible near the conventional gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot zoomed-in track quality Q(t) around transition.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing eval_locm_ew.npz and eval_conventional.npz (or worst-episode files).",
    )
    parser.add_argument(
        "--use-worst",
        action="store_true",
        help="If set, plot eval_locm_match_worst.npz vs eval_conventional_worst.npz from data-dir.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for PNG (PDF written alongside).",
    )
    parser.add_argument(
        "--adapt-ms",
        type=float,
        default=None,
        help="Override adaptation time (ms). If omitted, uses adapt_time_ms from eval file.",
    )
    parser.add_argument(
        "--pre-ms",
        type=float,
        default=None,
        help="If set with --post-ms, plot [adapt_ms-pre_ms, adapt_ms+post_ms] instead of +/- window-ms.",
    )
    parser.add_argument(
        "--post-ms",
        type=float,
        default=None,
        help="If set with --pre-ms, plot [adapt_ms-pre_ms, adapt_ms+post_ms] instead of +/- window-ms.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=5.0,
        help="Half-width of zoom window in ms (default 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Suppression threshold line (default 0.3).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    exp_dir = root / "experiments" / "cogew_demo"
    data_dir = Path(args.data_dir) if args.data_dir else exp_dir
    out_path = Path(args.out) if args.out else (exp_dir / "q_zoom.png")

    if args.use_worst:
        locm = np.load(data_dir / "eval_locm_match_worst.npz", allow_pickle=True)
        conv = np.load(data_dir / "eval_conventional_worst.npz", allow_pickle=True)
        if args.out is None:
            out_path = data_dir / "q_zoom_worst.png"
    else:
        locm = np.load(data_dir / "eval_locm_ew.npz", allow_pickle=True)
        conv = np.load(data_dir / "eval_conventional.npz", allow_pickle=True)

    t_ms = locm["t_ms"].astype(float)
    q_locm = locm["Q"].astype(float)
    q_conv = conv["Q"].astype(float)

    adapt_ms = (
        float(args.adapt_ms)
        if args.adapt_ms is not None
        else (float(locm["adapt_time_ms"]) if "adapt_time_ms" in locm.files else 150.0)
    )
    if args.pre_ms is not None or args.post_ms is not None:
        if args.pre_ms is None or args.post_ms is None:
            raise SystemExit("If using --pre-ms/--post-ms you must provide both.")
        pre = float(args.pre_ms)
        post = float(args.post_ms)
        t_lo = adapt_ms - pre
        t_hi = adapt_ms + post
        window_desc = f"-{pre:.3f}ms/+{post:.3f}ms"
    else:
        w = float(args.window_ms)
        t_lo = adapt_ms - w
        t_hi = adapt_ms + w
        window_desc = f"±{w:.3f}ms"

    mask = (t_ms >= t_lo) & (t_ms <= t_hi)
    if not np.any(mask):
        raise RuntimeError(f"No samples in zoom window: [{t_lo}, {t_hi}] ms around adapt_ms={adapt_ms}")

    t = t_ms[mask]
    ql = q_locm[mask]
    qc = q_conv[mask]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    ax.plot(t, ql, label="LOCM-EW", color="#9467bd", lw=2)
    ax.plot(t, qc, label="Conventional", color="#1f77b4", ls="--", lw=1.8)
    ax.axvline(adapt_ms, color="gray", ls="--", alpha=0.9, label="TWS→STT")
    ax.axhline(float(args.threshold), color="red", alpha=0.7, label="Suppression threshold")

    title = f"Threat radar track quality Q(t) — zoom {window_desc} around {adapt_ms:.3f} ms"
    if args.use_worst and "worst_episode_index" in conv.files:
        title += f" (worst_conventional_ep={int(conv['worst_episode_index'])})"
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Q")
    ax.set_xlim(t_lo, t_hi)

    y_min = float(np.min([ql.min(), qc.min(), 0.0]))
    y_max = float(np.max([ql.max(), qc.max(), 1.0]))
    ax.set_ylim(max(-0.05, y_min - 0.05), min(1.05, y_max + 0.05))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncols=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    print("Saved", out_path, "and", out_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

