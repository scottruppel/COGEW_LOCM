"""Generate supporting figures S1–S4 (PRD §5.2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    exp_dir = ROOT / "experiments" / "cogew_demo"
    data_dir = Path(args.data_dir) if args.data_dir else exp_dir
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # S1/S2 prefer dedicated spectral log file when present
    spectral_path = data_dir / "eval_locm_spectral.npz"
    if spectral_path.exists():
        locm = np.load(spectral_path)
        t_ms = locm["t_ms"]
        if "y_tilde" in locm.files and locm["y_tilde"].size:
            y = locm["y_tilde"]  # (T, r)
            mode_energy = np.square(y).T
            title = "S1: Spectral mode energy over time (y_tilde)"
            y_label = "Spectral mode"
        else:
            mode_energy = np.zeros((1, len(t_ms)))
            title = "S1: Spectral mode energy over time (missing)"
            y_label = "Spectral mode"
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            mode_energy,
            aspect="auto",
            extent=[t_ms[0], t_ms[-1], mode_energy.shape[0] - 0.5, -0.5],
            cmap="viridis",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Energy")
        fig.savefig(out_dir / "spectral_modes.png", dpi=150)
        plt.close(fig)
        print("Saved", out_dir / "spectral_modes.png")
    elif (data_dir / "eval_locm_ew.npz").exists():
        locm = np.load(data_dir / "eval_locm_ew.npz")
        actions = locm["actions"]  # (T, 32)
        t_ms = locm["t_ms"]
        mode_energy = np.square(actions).T
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mode_energy, aspect="auto", extent=[t_ms[0], t_ms[-1], 31.5, -0.5], cmap="viridis")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Channel (spectral mode proxy)")
        ax.set_title("S1: Channel energy over time (proxy)")
        plt.colorbar(im, ax=ax, label="Energy")
        fig.savefig(out_dir / "spectral_modes.png", dpi=150)
        plt.close(fig)
        print("Saved", out_dir / "spectral_modes.png")

    # S2: Damping profile (gamma_tilde) if logged
    if spectral_path.exists():
        locm = np.load(spectral_path)
        if "gamma_tilde_last" in locm.files and locm["gamma_tilde_last"].size:
            g = locm["gamma_tilde_last"]
            t_ms = locm["t_ms"]
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(g.T, aspect="auto", extent=[t_ms[0], t_ms[-1], g.shape[1] - 0.5, -0.5], cmap="plasma")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Spectral mode")
            ax.set_title("S2: Damping profile (gamma_tilde, last block)")
            plt.colorbar(im, ax=ax, label="gamma")
            fig.savefig(out_dir / "damping_profile.png", dpi=150)
            plt.close(fig)
            print("Saved", out_dir / "damping_profile.png")
    # S3: Spectrogram
    if (data_dir / "eval_locm_ew.npz").exists():
        locm = np.load(data_dir / "eval_locm_ew.npz")
        conv = np.load(data_dir / "eval_conventional.npz")
        t_ms = locm["t_ms"]
        psd_locm = np.square(locm["actions"])  # (T, 32)
        psd_conv = np.square(conv["actions"]) if "actions" in conv.files else np.tile(np.expand_dims(np.maximum(conv["power_db"], -10) / 10.0, 1), (1, 32))
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].imshow(psd_locm.T, aspect="auto", extent=[t_ms[0], t_ms[-1], 31.5, -0.5], cmap="magma")
        axes[0].set_ylabel("Channel")
        axes[0].set_title("LOCM-EW spectrogram (channel energy)")
        axes[1].imshow(psd_conv.T, aspect="auto", extent=[t_ms[0], t_ms[-1], 31.5, -0.5], cmap="magma")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Channel")
        axes[1].set_title("Conventional spectrogram")
        fig.savefig(out_dir / "spectrogram.png", dpi=150)
        plt.close(fig)
        print("Saved", out_dir / "spectrogram.png")

    # S4: Training curve
    if (data_dir / "bc_losses.json").exists():
        import json
        with open(data_dir / "bc_losses.json") as f:
            bc_losses = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bc_losses, color="blue", label="BC loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("S4: BC training curve")
        ax.legend()
        if (data_dir / "rl_returns.json").exists():
            with open(data_dir / "rl_returns.json") as f:
                rl_rewards = json.load(f)
            ax2 = ax.twinx()
            ax2.plot(rl_rewards, color="green", alpha=0.6, label="RL episode return")
            ax2.set_ylabel("Episode return")
            ax2.legend(loc="upper right")
        fig.savefig(out_dir / "training_curve.png", dpi=150)
        plt.close(fig)
        print("Saved", out_dir / "training_curve.png")
    else:
        print("No bc_losses.json found; skipping S4")


if __name__ == "__main__":
    main()
