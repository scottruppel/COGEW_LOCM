"""Build and save 32-channel RF spectral graph and spectrum (PRD Step 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cogew_locm.spectral_graph import build_rf_graph, build_and_save_spectrum
from locm.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CogEW spectral coupling graph")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "cogew.yaml"))
    parser.add_argument("--out-dir", type=str, default=None, help="Override data output dir")
    parser.add_argument("--plot", action="store_true", help="Plot topology and eigenmodes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir or str(ROOT / cfg.get("data_dir", "data/cogew"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    W_rf, channel_freqs, channel_bandwidths = build_rf_graph(
        n_channels=cfg["n_channels"],
        freq_range=(cfg["freq_range_ghz"][0] * 1e9, cfg["freq_range_ghz"][1] * 1e9),
        spectral_adjacency_weight=cfg.get("spectral_adjacency_weight", 1.0),
        next_neighbor_weight=cfg.get("next_neighbor_weight", 0.5),
        hardware_contention_weight=cfg.get("hardware_contention_weight", -0.5),
        threat_band_weight=cfg.get("threat_band_weight", 2.0),
        threat_center_ghz=cfg.get("radar_freq_ghz"),
        threat_bandwidth_ghz=cfg.get("radar_bandwidth_mhz", 50) / 1000.0,
    )

    U_r, Lambda_r = build_and_save_spectrum(
        W_rf,
        r=cfg.get("r", 32),
        out_dir=out_dir,
        metadata={"n_channels": cfg["n_channels"], "freq_range_ghz": cfg["freq_range_ghz"]},
    )

    # Save channel metadata for plots
    import numpy as np
    np.save(Path(out_dir) / "channel_freqs.npy", channel_freqs)
    np.save(Path(out_dir) / "channel_bandwidths.npy", channel_bandwidths)

    logger.info("Spectrum saved to %s (U_r %s, Lambda_r %s)", out_dir, U_r.shape, Lambda_r.shape)
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.bar(range(len(Lambda_r)), np.sort(Lambda_r))
            ax.set_xlabel("Eigenvalue index")
            ax.set_ylabel("Eigenvalue")
            ax.set_title("RF graph Laplacian spectrum")
            fig.savefig(Path(out_dir) / "eigenvalues.png", dpi=150)
            plt.close(fig)
            logger.info("Saved eigenvalues.png")
        except Exception as e:
            logger.warning("Plot failed: %s", e)


if __name__ == "__main__":
    main()
