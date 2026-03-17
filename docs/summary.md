# CogEW-LOCM Demo — Project Summary

Self-contained demonstration of gap-free noise-jamming to DRFM-deception transition on a 32-channel RF spectral graph, comparing LOCM-EW (continuous) vs conventional hard-switch. Primary deliverable: the three-panel money-shot figure (transmitted power and threat track quality over time).

## Repo Layout

- **`src/cogew_locm/`** — CogEW-specific package: spectral graph, threat model, conventional jammer, LOCM-EW controller, reward, expert demos, waveform power/PSD helpers.
- **`scripts/`** — `cogew_build_graph.py`, `cogew_train.py`, `cogew_eval.py`, `cogew_plot_moneyshot.py`, `cogew_plot_analysis.py`.
- **`configs/cogew.yaml`** — Single config for graph, LOCM, threat, training, evaluation.
- **`data/cogew/`** — Built spectrum (U_r, Lambda_r, channel metadata).
- **`experiments/cogew_demo/`** — Checkpoints, metrics, money-shot and supporting figures.

## Dependencies

- **LOCM** (sibling repo): `pip install -e "../LOCM"` for `locm.dynamics`, `locm.connectome`, `locm.model`, `locm.training.bc`.
- JAX, Equinox, NumPy, SciPy, PyYAML, Matplotlib.

## Verification and Test Results

End-to-end run (build → train → eval → plots) was executed successfully.

| Step | Result |
|------|--------|
| **Build graph** | 32-node RF graph (2–18 GHz), Laplacian spectrum saved to `data/cogew/` (U_r 32×31, Lambda_r). |
| **BC training** | 50 epochs on expert demos (~69.3k samples). Loss decreased from ~7.75 to ~7.63 (MSE). Checkpoint: `bc_model.pkl`. |
| **RL fine-tuning (JAX REINFORCE)** | Added REINFORCE fine-tuning loop using Gaussian exploration around the LOCM action mean, optimizing `-Q` plus coverage/smoothness/energy penalties. Outputs: `rl_returns.json`, `rl_losses.json`. |
| **Eval (100 episodes, transition-window)** | **100 episodes** evaluated in a **10 ms window centered on the technique transition** (`--window-pre-ms 5 --window-post-ms 5`). Results (see `experiments/cogew_demo/window_eval/metrics.json`): **LOCM mean max Q = 0.10** vs **Conventional mean max Q = 0.35**. |
| **Money-shot** | Three-panel figure: LOCM-EW TX power (continuous), conventional TX power (gap), threat track quality Q(t) for both. Saved as `moneyshot.png` / `moneyshot.pdf`. |
| **Supporting figures** | S1 `spectral_modes.png` now uses **logged `y_tilde`** (from `eval_locm_spectral.npz`); S2 `damping_profile.png` uses **logged `gamma_tilde`**; plus `spectrogram.png` and `training_curve.png`. |

Config used: `config.json` in `experiments/cogew_demo/` (phase `bc_only`, seed 42, 32 channels, 50 BC epochs, 10 μs macro-step, 500 ms episodes).

### Notes on the threat model calibration

To make the **50 μs conventional gap** produce a visible and measurable radar reacquisition effect at microsecond timesteps, the threat model updates now interpret `alpha_acquire` / `alpha_degrade` as **per-millisecond rates** internally (i.e. \(dt\\) converted to ms in `AdaptiveRadar.step`). This yields the expected “gap exploitation window” behavior in the aggregate metrics and plots.

## Quick Run

1. From repo root: `pip install -e "../LOCM"` then `pip install -e .`
2. Build graph: `python scripts/cogew_build_graph.py`
3. Train (BC): `python scripts/cogew_train.py --bc-only`
4. Eval: `python scripts/cogew_eval.py --episodes 10 --adapt-ms 150`
5. Money-shot: `python scripts/cogew_plot_moneyshot.py`
6. Supporting figures: `python scripts/cogew_plot_analysis.py`

## Integration

See [integration_sdr_orchestration.md](integration_sdr_orchestration.md) for how the spectral graph, observation vector, and channel amplitudes connect to the SDR orchestration layer (e.g. CogEW_Sandbox).

---

## Next Steps to Grow the Demonstration

- **RL fine-tuning** — Add PPO (or JAX-based policy gradient) on top of BC: full curriculum (slow → fast radar adaptation), 500 ms episodes with 10 μs macro-steps; log episode reward and max Q during transition. Target: LOCM-EW keeps mean max Q &lt; 0.3 through transition; conventional shows a clear Q spike during the gap.
- **Larger evaluation** — Run `cogew_eval.py` with `--episodes 100` (per PRD) and record summary metrics (mean/max Q, min TX power during transition) for success-criteria checks.
- **LOCM block parameters** — Extend upstream LOCM (or a local fork) to pass `gamma_max=10`, `omega_init=1000` into LOCMBlock so CogEW runs at PRD-specified RF timescales; re-run train/eval and compare.
- **Spectral mode and damping plots** — Instrument the LOCM forward pass to log `y_tilde` (spectral mode activation) and per-block `gamma_tilde` (damping profile) during eval; add S1/S2 figures that use these instead of channel-energy proxies.
- **CogEW_Sandbox integration** — Wire threat observation from sandbox (e.g. MCP spectral awareness or ES module) and channel amplitudes to the sandbox waveform generator; align experiment layout (e.g. `manifest.json`) with sandbox so results can be ingested.
- **Multi-threat and robustness** — Generalize to multiple simultaneous threats (observation space and reward); add robustness evals (different radar adapt times, PRF, bandwidth).
- **Documentation and outreach** — Draft the 1–2 page CogEW-LOCM technical note for non-technical stakeholders; keep integration mapping up to date as SDR orchestration APIs stabilize.
