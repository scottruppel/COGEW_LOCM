"""Build 32-channel RF spectral coupling graph (PRD §3.1)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse

from locm.connectome.laplacian import signed_laplacian
from locm.connectome.spectrum import compute_spectrum, save_spectrum
from locm.utils.logging import get_logger

logger = get_logger(__name__)


def build_rf_graph(
    n_channels: int = 32,
    freq_range: Tuple[float, float] = (2e9, 18e9),
    spectral_adjacency_weight: float = 1.0,
    next_neighbor_weight: float = 0.5,
    hardware_contention_weight: float = -0.5,
    threat_band_weight: float = 2.0,
    threat_center_ghz: float | None = 9.5,
    threat_bandwidth_ghz: float | None = 0.05,
) -> Tuple[sparse.spmatrix, np.ndarray, np.ndarray]:
    """
    Build symmetric sparse adjacency for 32-channel 2-18 GHz RF graph.

    Edges: spectral adjacency (i, i±1), next-neighbor (i, i±2),
    hardware contention (groups of 4 share DAC/ADC), optional threat-band grouping.

    Returns:
        W_rf: (n_channels, n_channels) sparse CSR
        channel_freqs: (n_channels,) center frequency per channel (Hz)
        channel_bandwidths: (n_channels,) bandwidth per channel (Hz)
    """
    f_lo, f_hi = freq_range
    bw = (f_hi - f_lo) / n_channels
    channel_freqs = np.linspace(f_lo + bw / 2, f_hi - bw / 2, n_channels)
    channel_bandwidths = np.full(n_channels, bw)

    row, col, data = [], [], []

    # Spectral adjacency: i, i±1
    for i in range(n_channels):
        for j in (i - 1, i + 1):
            if 0 <= j < n_channels:
                row.append(i)
                col.append(j)
                data.append(spectral_adjacency_weight)
    # Next-neighbor: i, i±2
    for i in range(n_channels):
        for j in (i - 2, i + 2):
            if 0 <= j < n_channels:
                row.append(i)
                col.append(j)
                data.append(next_neighbor_weight)

    # Hardware contention: groups of 4 share DAC/ADC (negative)
    for g in range(0, n_channels, 4):
        for i in range(g, min(g + 4, n_channels)):
            for j in range(g, min(g + 4, n_channels)):
                if i != j:
                    row.append(i)
                    col.append(j)
                    data.append(hardware_contention_weight)

    # Threat-band grouping: channels overlapping threat band get strong positive
    if threat_center_ghz is not None and threat_bandwidth_ghz is not None:
        fc = threat_center_ghz * 1e9
        bw_t = threat_bandwidth_ghz * 1e9
        for i in range(n_channels):
            c_lo = channel_freqs[i] - bw / 2
            c_hi = channel_freqs[i] + bw / 2
            t_lo = fc - bw_t / 2
            t_hi = fc + bw_t / 2
            if c_hi >= t_lo and c_lo <= t_hi:
                for j in range(n_channels):
                    if i != j:
                        c_lo_j = channel_freqs[j] - bw / 2
                        c_hi_j = channel_freqs[j] + bw / 2
                        if c_hi_j >= t_lo and c_lo_j <= t_hi:
                            row.append(i)
                            col.append(j)
                            data.append(threat_band_weight)

    W_rf = sparse.csr_matrix(
        (data, (row, col)), shape=(n_channels, n_channels), dtype=np.float64
    )
    # Symmetrize
    W_rf = (W_rf + W_rf.T) / 2.0
    logger.info(
        "RF graph: n=%s, nnz=%s, freq_range=[%s, %s] GHz",
        n_channels,
        W_rf.nnz,
        f_lo / 1e9,
        f_hi / 1e9,
    )
    return W_rf, channel_freqs, channel_bandwidths


def build_and_save_spectrum(
    W_rf: sparse.spmatrix,
    r: int,
    out_dir: str | None = None,
    metadata: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute signed Laplacian, top-r spectrum, optionally save. Returns U_r, Lambda_r."""
    L_conn = signed_laplacian(W_rf)
    n = L_conn.shape[0]
    r_use = min(r, n - 1)
    U_r, Lambda_r = compute_spectrum(L_conn, r_use, which="SM")
    if out_dir:
        save_spectrum(U_r, Lambda_r, out_dir, metadata=metadata or {})
    return U_r, Lambda_r
