# CogEW-LOCM Integration with SDR Orchestration Layer

This document describes how CogEW-LOCM connects to an existing SDR orchestration layer (e.g. CogEW_Sandbox or a future MCP-based spectral awareness stack).

## 1. Spectral Coupling Graph → SDR Channel Allocation

- **CogEW output**: A 32-node signed graph (adjacency `W_rf`) and its Laplacian spectrum (`U_r`, `Lambda_r`) in `data/cogew/`. Each node is a 500 MHz processing channel spanning 2–18 GHz.
- **Integration**: The orchestration layer should map graph nodes to physical SDR channel allocation (DAC/ADC pairs). The PRD assumes groups of 4 channels share a DAC/ADC (hardware contention edges). When allocating waveforms to channels, use the same node–channel mapping so that LOCM’s 32-dim action vector aligns with hardware channels.
- **Artifacts**: `data/cogew/U_r.npy`, `Lambda_r.npy`, `channel_freqs.npy`, `channel_bandwidths.npy`, `spectrum_manifest.json`.

## 2. Threat Characterization Vector → Observation

- **CogEW input**: A 64-dim observation vector (PRD §4.1): threat frequency estimate (4), mode one-hot (4), 32-channel PSD (32), time since mode change (1), effectiveness estimate (1), padding (22).
- **Integration**: In a full system, this vector is supplied by:
  - **ES / MCP spectral awareness module**: center frequency, bandwidth, PRF, PRI, and radar mode (search / TWS / STT / off).
  - **Front-end receiver**: 32-channel power spectral density.
  - **BDA / feedback**: current jamming effectiveness estimate.
- **Demo**: The threat model in `cogew_locm.threat_model` produces this observation from simulated ground truth; replace with live ES + receiver feeds at the integration point.

## 3. Channel Amplitude Output → Waveform Generator

- **CogEW output**: A 32-dim action vector `a(t)` (channel amplitudes in [-1, 1]) at each control step (e.g. 10 μs macro-step).
- **Integration**: The orchestration layer passes `a(t)` to the waveform synthesis block:
  - **Formula** (PRD §4.2):  
    `x_TX(t) = sum_v a_v(t) * g_v(t) * exp(j 2 π f_v t)`  
    where `g_v(t)` is the channel shaping (e.g. Gaussian for noise, time-delayed copy for DRFM).
- **Demo**: `cogew_locm.waveform_synthesis` provides power/PSD from amplitudes only; full RF synthesis is out of scope and belongs in the SDR waveform generator.

## 4. Config and Experiment Layout

- **Config**: `configs/cogew.yaml` holds graph, LOCM, threat, and training parameters. The same keys can be overridden by the orchestration layer (e.g. via Hydra or env).
- **Experiments**: `experiments/cogew_demo/` holds checkpoints (`bc_model.pkl`), `config.json`, `metrics.json`, and figures. This layout can be aligned with CogEW_Sandbox experiment manifests for unified reporting.

## 5. Summary Table

| CogEW artifact        | Consumer / integration point                    |
|-----------------------|-------------------------------------------------|
| 32-node graph, spectrum | SDR channel allocation, LOCM-EW controller init |
| 64-dim observation    | From ES + receiver + BDA (MCP spectral awareness) |
| 32-dim action         | To waveform generator (RF synthesis)            |
| configs/cogew.yaml    | Orchestrator overrides                          |
| experiments/cogew_demo | Metrics, checkpoints, figures                    |
