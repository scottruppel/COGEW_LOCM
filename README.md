# CogEW-LOCM Demo

Gap-free noise-to-DRFM technique transition demonstration on a 32-channel spectral coupling graph (PRD: `CogEW_LOCM_demo_PRD.md`). Compares **LOCM-EW** (continuous transition) vs **conventional hard-switch** (50 μs gap); primary deliverable is the three-panel money-shot figure.

## Setup

1. **LOCM dependency** (sibling repo):

   ```bash
   pip install -e "../LOCM"
   ```

2. **CogEW-LOCM** (from this repo):

   ```bash
   pip install -e .
   ```

   Or install from `requirements.txt` (includes `-e ../LOCM`):

   ```bash
   pip install -r requirements.txt
   ```

3. **Virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -e "../LOCM"
   pip install -e .
   ```

## Usage

| Step | Command | Output |
|------|---------|--------|
| Build graph | `python scripts/cogew_build_graph.py` | `data/cogew/` (U_r, Lambda_r, channel metadata) |
| Train (BC) | `python scripts/cogew_train.py --bc-only` | `experiments/cogew_demo/bc_model.pkl`, `bc_losses.json` |
| Train (BC + RL episodes) | `python scripts/cogew_train.py` | + `rl_rewards.json` |
| Eval | `python scripts/cogew_eval.py --episodes 10 --adapt-ms 150` | `eval_locm_ew.npz`, `eval_conventional.npz`, `metrics.json` |
| Money-shot figure | `python scripts/cogew_plot_moneyshot.py` | `moneyshot.png`, `moneyshot.pdf` |
| Supporting figures | `python scripts/cogew_plot_analysis.py` | `spectral_modes.png`, `spectrogram.png`, `training_curve.png` |

## Config

Edit `configs/cogew.yaml` for graph weights, LOCM/ threat/ training/ eval parameters.

## Integration with CogEW_Sandbox

See [docs/integration_sdr_orchestration.md](docs/integration_sdr_orchestration.md) for how the 32-channel graph, threat observation vector, and channel amplitude output map to the SDR orchestration layer.

## Project summary

See [docs/summary.md](docs/summary.md) for a short overview and quick-run steps.
