"""Generate the money-shot three-panel figure (PRD §5.1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Dir with eval_locm_ew.npz, eval_conventional.npz")
    parser.add_argument("--out", type=str, default=None, help="Output path for figure")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments" / "cogew_demo"
    data_dir = Path(args.data_dir) if args.data_dir else exp_dir
    out_path = Path(args.out) if args.out else exp_dir / "moneyshot.png"

    locm = np.load(data_dir / "eval_locm_ew.npz", allow_pickle=True)
    conv = np.load(data_dir / "eval_conventional.npz", allow_pickle=True)
    t = locm["t_ms"]
    Q_threshold = 0.3
    adapt_ms = float(locm["adapt_time_ms"]) if "adapt_time_ms" in locm.files else 150.0

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Panel 1: LOCM-EW transmitted power (continuous)
    ax1 = axes[0]
    power_locm = locm["power_db"]
    ax1.fill_between(t, power_locm, alpha=0.7, color="#9467bd")
    ax1.plot(t, power_locm, color="black", lw=0.8, alpha=0.8)
    ax1.axvline(adapt_ms, color="gray", ls="--", alpha=0.8)
    ax1.set_ylabel("TX power (dB rel)")
    ax1.set_title("LOCM-EW transmitted power (continuous)")
    finite1 = power_locm[np.isfinite(power_locm)]
    y1_bottom = (min(finite1.min(), -5) - 2) if len(finite1) else -25
    ax1.set_ylim(bottom=y1_bottom)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Conventional transmitted power (gap)
    ax2 = axes[1]
    power_conv = conv["power_db"]
    ax2.fill_between(t, power_conv, alpha=0.7, color="#1f77b4")
    ax2.plot(t, power_conv, color="black", lw=0.8, alpha=0.8)
    ax2.axvline(adapt_ms, color="gray", ls="--", alpha=0.8)
    ax2.set_ylabel("TX power (dB rel)")
    ax2.set_title("Conventional jammer transmitted power (gap during switch)")
    finite2 = power_conv[np.isfinite(power_conv)]
    y2_bottom = (min(finite2.min(), -5) - 2) if len(finite2) else -25
    ax2.set_ylim(bottom=y2_bottom)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Threat radar track quality Q(t)
    ax3 = axes[2]
    ax3.plot(t, locm["Q"], label="LOCM-EW", color="#9467bd", lw=2)
    ax3.plot(t, conv["Q"], label="Conventional", color="#1f77b4", ls="--", lw=1.5)
    ax3.axhline(Q_threshold, color="red", ls="-", alpha=0.7, label="Suppression threshold")
    ax3.axvline(adapt_ms, color="gray", ls="--", alpha=0.8)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Track quality Q")
    ax3.set_title("Threat radar track quality")
    ax3.legend(loc="upper right")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=150)
    plt.close(fig)
    print("Saved", out_path, "and", out_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
