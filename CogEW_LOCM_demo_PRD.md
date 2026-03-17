# Product Requirements Document: CogEW-LOCM Demo
## Gap-Free Technique Transition Demonstration on SDR Orchestration Layer

**Version**: 0.1
**Author**: Scott Ruppel, Col, USAF (Ret.)
**Date**: 16 March 2026
**Status**: Build-Ready
**Codebase**: Extends `scottruppel/LOCM` — same spectral step, damping MLP, GLU mixing

---

## 1. Purpose

Build a self-contained demonstration that produces one figure:

> **Transmitted signal power and simulated threat radar track error over time, during a noise-jamming-to-DRFM-deception transition, comparing LOCM-EW (continuous) vs. conventional hard-switch.**

The figure should show:
- **Top panel**: LOCM-EW transmitted waveform power (continuous, no gap)
- **Middle panel**: Conventional system transmitted waveform power (gap during switch)
- **Bottom panel**: Threat radar track error — LOCM-EW stays above threshold; conventional dips to zero during the gap, allowing the radar to reacquire

This is the single artifact that validates the core LOCM-EW claim and provides the conversation-opener for MIT/LL engagement.

---

## 2. Scope

### In Scope
- 32-channel spectral coupling graph covering a representative 2-18 GHz band
- LOCM spectral dynamics (reusing `locm.dynamics.spectral_step`, `DampingMLP`, `LOCMBlock`)
- Simulated adaptive radar threat model (pulse-Doppler, track-while-scan)
- Two EW technique regimes: noise jamming and DRFM deception
- Conventional hard-switch baseline for comparison
- RL training loop (simplified: single-threat, single-transition scenario)
- The money-shot figure + supporting analysis plots
- Integration point with existing SDR orchestration layer

### Out of Scope
- Real RF hardware / SDR transmit
- Multiple simultaneous threats
- ES subsystem (threat characterization is provided as ground truth from simulation)
- FPGA implementation
- Classified threat models

---

## 3. Technical Architecture

### 3.1 Spectral Coupling Graph $\mathcal{G}_\text{RF}$

**32 nodes**, each representing a 500 MHz processing channel spanning 2-18 GHz:

```
Channel 0:  2.0 -  2.5 GHz
Channel 1:  2.5 -  3.0 GHz
...
Channel 31: 17.5 - 18.0 GHz
```

**Edge structure** (symmetric, sparse):

| Coupling Type | Rule | Weight Sign | Count |
|---|---|---|---|
| Spectral adjacency | Channels $i$ and $i\pm1$ | Positive (cooperative) | 62 edges |
| Spectral next-neighbor | Channels $i$ and $i\pm2$ | Weak positive | 60 edges |
| Hardware contention | Channels sharing same DAC/ADC pair (groups of 4) | Negative (competitive) | 48 edges |
| Threat-band grouping | Channels covering same threat radar bandwidth | Strong positive | Variable |

Total: ~170 edges on 32 nodes. Sparse enough that the full Laplacian eigensystem is trivially computable (32×32).

**Implementation**: Build as a `scipy.sparse` matrix following the same conventions as the Drosophila connectome pipeline. Reuse `signed_laplacian()` and `compute_spectrum()` directly — the math doesn't know it's an SDR instead of a brain.

### 3.2 LOCM-EW Core (Reused from LOCM Codebase)

Directly import and instantiate:

```python
from locm.dynamics.spectral_step import spectral_step
from locm.dynamics.damping_mlp import DampingMLP
from locm.dynamics.block import LOCMBlock  # after GLU fix to spectral-space
from locm.connectome.laplacian import signed_laplacian
from locm.connectome.spectrum import compute_spectrum
```

**Configuration differences from fly LOCM**:

| Parameter | Fly LOCM | CogEW-LOCM |
|---|---|---|
| N (nodes) | 139,246 | 32 |
| r (spectral modes) | 256-512 | 16-32 (can use full spectrum) |
| dt (timestep) | 0.01 s (control rate) | 1 μs (RF control rate) |
| obs_dim | ~200-700 (fly proprioception) | 64-128 (threat characterization vector) |
| action_dim | 59/12 (leg/wing actuators) | 32 (channel amplitudes) |
| gamma_max | 2.0 | 10.0 (faster transitions needed at μs scale) |
| omega range | ~1-10 (gait frequencies) | ~1e3-1e6 (RF timing scales) |
| n_blocks | 2 | 2 |

### 3.3 Simulated Adaptive Radar Threat

A simplified but physically meaningful threat model:

**Radar type**: Pulse-Doppler, track-while-scan

**State vector** $\mathbf{r}(t)$:
- Track estimate: range $\hat{R}$, range rate $\hat{\dot{R}}$, azimuth $\hat{\theta}$
- Track quality: $Q(t) \in [0, 1]$ — 1.0 = solid track, 0.0 = track lost
- Track covariance: $\mathbf{P}(t)$ — Kalman filter uncertainty

**Dynamics**:
```
If not jammed:
    Q(t+dt) = min(1.0, Q(t) + α_acquire * dt)     # track improves
    P(t+dt) = P(t) * (1 - β_acquire * dt)           # covariance shrinks

If jammed (effective):
    Q(t+dt) = max(0.0, Q(t) - α_degrade * dt)      # track degrades
    P(t+dt) = P(t) * (1 + β_degrade * dt)           # covariance grows

If jammed (ineffective / wrong technique):
    Q(t+dt) = min(1.0, Q(t) + α_acquire * 0.5 * dt) # partial acquisition
```

**Adaptive behavior**: The radar monitors its own track quality. When $Q$ drops below a threshold, it:
1. Changes PRF (pulse repetition frequency)
2. Shifts center frequency by ±Δf
3. Switches from TWS to STT (single-target track) mode

This adaptation is what makes the technique transition necessary — the jammer must shift from noise (effective against TWS) to DRFM deception (effective against STT).

**Jamming effectiveness model**:

| Technique | vs. TWS Mode | vs. STT Mode |
|---|---|---|
| Noise jamming | High (masks returns) | Low (STT burns through) |
| DRFM deception | Low (TWS rejects coherent copies) | High (pulls range gate) |
| Transition gap (no signal) | Zero (clean look) | Zero (clean look) |
| LOCM-EW mid-transition | Medium-High (hybrid waveform) | Medium (partial deception) |

The key: during a conventional hard-switch from noise → DRFM, the threat radar gets a clean window and $Q$ spikes. During LOCM-EW's continuous transition, the waveform is always producing *some* countermeasure effect, keeping $Q$ suppressed.

### 3.4 Conventional Baseline (Hard-Switch)

A simple state machine for comparison:

```python
class ConventionalJammer:
    def __init__(self):
        self.technique = "noise"
        self.transition_timer = 0
        self.transition_duration = 50e-6  # 50 μs gap
    
    def step(self, threat_state):
        if self.technique == "noise" and threat_state.mode == "STT":
            # Threat adapted — need to switch to DRFM
            self.technique = "transitioning"
            self.transition_timer = self.transition_duration
        
        if self.technique == "transitioning":
            self.transition_timer -= dt
            if self.transition_timer <= 0:
                self.technique = "drfm"
            return np.zeros(32)  # NO OUTPUT during transition
        
        if self.technique == "noise":
            return self.generate_noise()
        else:
            return self.generate_drfm(threat_state)
```

### 3.5 LOCM-EW Controller

Uses the LOCM architecture with the spectral coupling graph:

```python
class LOCMEWController:
    def __init__(self, U_r, Lambda_r, obs_dim, action_dim):
        self.model = LOCM(obs_dim, action_dim, U_r, Lambda_r, n_blocks=2)
        self.state = None  # (y_tilde, z_tilde) persistent across steps
    
    def step(self, threat_obs):
        action, self.state = self.model(threat_obs, self.state, dt=1e-6)
        return action  # 32-dim channel amplitudes — ALWAYS defined
```

The damping MLP learns to shift the oscillatory dynamics from the noise-jamming attractor to the DRFM-deception attractor when it observes the threat radar switching to STT mode. The transition is continuous — no gap.

---

## 4. Training Pipeline

### 4.1 Observation Space

The threat characterization vector $\mathbf{s}(t) \in \mathbb{R}^{64}$:

| Component | Dimensions | Source |
|---|---|---|
| Threat radar frequency estimate | 4 (center freq, bandwidth, PRF, PRI) | ES subsystem (simulated ground truth) |
| Threat radar mode indicator | 4 (one-hot: search, TWS, STT, off) | ES subsystem |
| Received signal features | 32 (power spectral density per channel) | Front-end receiver |
| Time since last mode change | 1 | Clock |
| Current jamming effectiveness estimate | 1 | BDA feedback |
| Padding/reserved | 22 | Zero |

### 4.2 Action Space

Channel amplitude vector $\mathbf{a}(t) \in \mathbb{R}^{32}$:
- Each element $a_v \in [-1, 1]$ represents the signed amplitude on channel $v$
- Positive: in-phase with received signal (deceptive)
- Negative: anti-phase (cancellation)
- Magnitude: power level relative to max

The waveform synthesis layer converts these to RF:
$$x_\text{TX}(t) = \sum_{v=0}^{31} a_v(t) \cdot g_v(t) \cdot e^{j2\pi f_v t}$$

where $g_v(t)$ is the channel's shaping function (Gaussian envelope for noise, time-delayed copy for DRFM).

### 4.3 Reward Function

```python
def compute_reward(threat_state, action, prev_action):
    # Primary: keep threat radar track quality LOW
    r_suppress = -threat_state.Q  # negative Q is good for us
    
    # Penalty: transition gaps (action near zero)
    power = np.sum(action ** 2)
    r_coverage = -max(0, 0.1 - power)  # penalize if total power drops below threshold
    
    # Penalty: excessive power (SWAP budget)
    r_energy = -0.01 * power
    
    # Penalty: abrupt changes (spectral splatter)
    r_smooth = -0.1 * np.sum((action - prev_action) ** 2)
    
    # Bonus: successful technique effect
    if threat_state.Q < 0.3:
        r_effect = 1.0  # track suppressed
    else:
        r_effect = 0.0
    
    return r_suppress + r_coverage + r_energy + r_smooth + r_effect
```

### 4.4 Training Procedure

**Phase A: Behavioral Cloning warm-start**

Generate expert demonstrations:
1. **Noise expert**: fixed noise jamming against TWS-mode radar → record (obs, action) pairs
2. **DRFM expert**: fixed DRFM deception against STT-mode radar → record (obs, action) pairs
3. **Transition expert**: hand-crafted linear interpolation between noise and DRFM over 100 μs window → record pairs

Train LOCM-EW via BC on the combined dataset. This initializes the model into the right region of parameter space — it knows what noise jamming and DRFM look like, and has a rough idea of how to blend them.

**Phase B: RL fine-tuning**

Run PPO against the simulated adaptive radar:
1. Episode starts with radar in TWS mode, jammer in noise regime
2. At random time (50-200 ms into episode), radar adapts to STT mode
3. Jammer must transition to DRFM deception while maintaining track suppression
4. Episode ends after 500 ms or if track quality exceeds 0.9 (radar locks on)

Curriculum:
- Stage 1: Radar adaptation is slow (α_acquire low) — easy to keep Q suppressed
- Stage 2: Radar adaptation is moderate
- Stage 3: Radar adaptation is fast (realistic) — requires rapid, gap-free transition

**Phase C: Conventional baseline**

Run the same episodes with `ConventionalJammer` (hard-switch with 50 μs gap). Record the same metrics. No training needed — the conventional jammer is deterministic.

---

## 5. The Money-Shot Figure

### 5.1 Specification

Three-panel time-series plot, 500 ms duration, with the radar mode transition at t = 150 ms:

**Panel 1: LOCM-EW Transmitted Power**
- X-axis: time (ms)
- Y-axis: total transmitted power (dBm relative)
- Color-coded by dominant technique: blue = noise, red = DRFM, purple = transitional blend
- Key feature: **continuous signal through the transition region** (purple zone smoothly connecting blue to red)

**Panel 2: Conventional Jammer Transmitted Power**
- Same axes
- Color-coded: blue = noise, red = DRFM, **white gap** during transition
- Key feature: **visible zero-power gap** at the technique switch point

**Panel 3: Threat Radar Track Quality**
- X-axis: time (ms)
- Y-axis: track quality Q(t), range [0, 1]
- Two lines: LOCM-EW (solid) vs. conventional (dashed)
- Horizontal line at Q = 0.3 (track suppression threshold)
- Key feature: conventional line **spikes above threshold during the gap**, LOCM-EW line **stays below threshold continuously**
- Shaded region: "exploitation window" where conventional system allows clean radar return

**Annotations**:
- Vertical dashed line at t = 150 ms: "Radar adapts: TWS → STT"
- Arrow pointing to conventional spike: "50 μs transition gap — radar reacquires"
- Arrow pointing to LOCM-EW suppression: "Continuous transition — no exploitation window"

### 5.2 Supporting Figures

**Figure S1: Spectral Mode Activation**
- Heatmap: spectral modes (y-axis) vs. time (x-axis), color = mode energy
- Shows which modes are active during noise jamming, which during DRFM, and how they smoothly cross-fade during transition
- Analogous to the gait-mode PSD plot from the fly demonstrator

**Figure S2: Liquid Damping Profile**
- 32-channel damping values $\gamma_v(t)$ over time
- Shows the damping network selectively suppressing noise-mode channels and amplifying DRFM-mode channels during transition

**Figure S3: Waveform Spectrogram**
- Frequency (y-axis) vs. time (x-axis), color = power spectral density
- LOCM-EW: smooth spectral evolution
- Conventional: sharp discontinuity at transition point

**Figure S4: Training Curve**
- Episode reward vs. training step for LOCM-EW
- Compare BC-only vs. BC + RL fine-tuning

---

## 6. Implementation Plan

### 6.1 File Structure (within LOCM repo)

```
src/locm/
├── cogew/                          # NEW: CogEW-specific code
│   ├── __init__.py
│   ├── spectral_graph.py           # Build 32-channel coupling graph
│   ├── threat_model.py             # Adaptive radar simulation
│   ├── conventional_jammer.py      # Hard-switch baseline
│   ├── locm_ew_controller.py       # LOCM-EW wrapper using LOCM model
│   ├── waveform_synthesis.py       # Channel amplitudes → RF-level signal
│   ├── reward.py                   # CogEW reward function
│   └── expert_demos.py             # Generate BC demonstrations
├── dynamics/                       # SHARED: unchanged from fly LOCM
│   ├── spectral_step.py            # (reused as-is)
│   ├── damping_mlp.py              # (reused as-is)
│   └── block.py                    # (reused after GLU fix)
├── model/
│   └── locm.py                     # (reused — just different dims)
└── ...

scripts/
├── cogew_build_graph.py            # Build and save spectral coupling graph
├── cogew_train.py                  # BC + RL training for CogEW-LOCM
├── cogew_eval.py                   # Run evaluation episodes, collect data
├── cogew_plot_moneyshot.py         # Generate the money-shot figure
└── cogew_plot_analysis.py          # Generate supporting figures S1-S4

configs/
├── cogew.yaml                      # CogEW-specific config
└── ...

experiments/
├── cogew_demo/                     # Output directory
│   ├── config.json
│   ├── metrics.json
│   ├── moneyshot.png               # THE FIGURE
│   ├── spectral_modes.png          # S1
│   ├── damping_profile.png         # S2
│   ├── spectrogram.png             # S3
│   └── training_curve.png          # S4
└── ...
```

### 6.2 Work Breakdown

#### Step 1: Spectral Coupling Graph (Day 1)

- [ ] Implement `cogew/spectral_graph.py`:
  - `build_rf_graph(n_channels=32, freq_range=(2e9, 18e9))` → sparse adjacency, channel metadata
  - Spectral adjacency, hardware contention, threat-band grouping edges
  - Return W_rf, channel_freqs, channel_bandwidths
- [ ] Compute Laplacian and full eigendecomposition (32×32 — instant)
- [ ] Save to `data/cogew/` using same `save_spectrum` utility
- [ ] Visualize graph topology and eigenmodes
- [ ] **Validation**: Eigenvalue 0 exists (connected graph); eigenvectors show frequency-band structure

#### Step 2: Threat Model (Day 1-2)

- [ ] Implement `cogew/threat_model.py`:
  - `AdaptiveRadar` class with state (Q, P, mode, freq, PRF)
  - `step(jammer_signal)` → updates track quality based on jamming effectiveness
  - Mode adaptation logic: TWS → STT when Q drops, with configurable adaptation speed
  - Configurable parameters: α_acquire, α_degrade, adaptation_threshold
- [ ] Implement `cogew/conventional_jammer.py`:
  - `ConventionalJammer` with hard noise/DRFM switch and configurable gap duration
- [ ] **Validation**: Run standalone simulation — radar acquires when unjammed, loses track when jammed, adapts after sustained jamming

#### Step 3: Expert Demonstrations (Day 2)

- [ ] Implement `cogew/expert_demos.py`:
  - Generate noise-jamming expert trajectories (radar in TWS)
  - Generate DRFM expert trajectories (radar in STT)
  - Generate interpolated transition trajectories
  - Save as npz: (obs_batch, action_batch) — same format as fly BC data
- [ ] **Validation**: Expert noise jamming keeps Q < 0.3 against TWS; expert DRFM keeps Q < 0.3 against STT

#### Step 4: LOCM-EW Controller + BC Training (Day 2-3)

- [ ] Implement `cogew/locm_ew_controller.py`:
  - Thin wrapper around `LOCM` model with CogEW-specific obs/action dims
  - Persistent state (y_tilde, z_tilde) across timesteps
- [ ] Implement `cogew/reward.py`
- [ ] BC training on expert demonstrations using existing `locm.training.bc`
- [ ] **Validation**: BC loss decreases; LOCM-EW produces recognizable noise-like output when threat is in TWS, DRFM-like output when threat is in STT

#### Step 5: RL Fine-Tuning (Day 3-4)

- [ ] Implement PPO training loop in `scripts/cogew_train.py`:
  - Episode: 500 ms simulated time, 1 μs steps = 500K steps per episode
  - (For tractability: use 10 μs macro-steps with 10x sub-stepping in threat model = 50K steps)
  - Curriculum: slow → moderate → fast radar adaptation
  - Log: episode reward, mean Q during transition, max Q during transition
- [ ] Train LOCM-EW until transition episodes show continuous Q suppression
- [ ] **Validation**: Training curve improves; transition episodes show no Q spike

#### Step 6: Evaluation + Figures (Day 4-5)

- [ ] Implement `scripts/cogew_eval.py`:
  - Run 100 evaluation episodes with LOCM-EW
  - Run 100 evaluation episodes with conventional jammer
  - Record full time series: action, threat state, Q, spectral mode activation, damping profile
  - Save to `experiments/cogew_demo/`
- [ ] Implement `scripts/cogew_plot_moneyshot.py`:
  - Three-panel figure per specification in Section 5.1
  - Use a single representative episode (or median across episodes)
  - Save as `moneyshot.png` and `moneyshot.pdf`
- [ ] Implement `scripts/cogew_plot_analysis.py`:
  - Figures S1-S4 per Section 5.2
- [ ] **Validation**: The money-shot figure shows the claimed behavior

#### Step 7: Integration Notes (Day 5)

- [ ] Document SDR orchestration layer integration points:
  - Where the spectral coupling graph connects to real SDR channel allocation
  - Where the threat characterization vector comes from (MCP spectral awareness module)
  - Where the channel amplitude output feeds the waveform generator
- [ ] Update `experiments/cogew_demo/config.json` with full reproducibility info
- [ ] Update `experiments/manifest.json`

### 6.3 Configuration

```yaml
# configs/cogew.yaml

# Spectral coupling graph
n_channels: 32
freq_range_ghz: [2.0, 18.0]
spectral_adjacency_weight: 1.0
hardware_contention_weight: -0.5
threat_band_weight: 2.0

# LOCM-EW model
r: 32                    # full spectrum for 32-node graph
n_blocks: 2
gamma_max: 10.0
omega_init: 1000.0        # base oscillator frequency
alpha: 0.1                # coupling strength

# Threat model
radar_freq_ghz: 9.5       # X-band threat
radar_bandwidth_mhz: 50
alpha_acquire: 5.0         # track acquisition rate (Q/s)
alpha_degrade: 10.0        # track degradation rate under jamming (Q/s)
adaptation_threshold: 0.3  # Q below which radar adapts mode
conventional_gap_us: 50    # hard-switch transition duration

# Training
bc_epochs: 50
bc_lr: 1e-3
bc_expert_episodes: 200
rl_episodes: 1000
rl_lr: 3e-4
rl_curriculum_stages: 3
episode_duration_ms: 500
macro_step_us: 10          # control rate for RL (sub-step threat model)

# Evaluation
eval_episodes: 100
```

---

## 7. Success Criteria

| Criterion | Metric | Threshold | Measurement |
|---|---|---|---|
| Gap-free transition | Max Q during LOCM-EW transition window | Q < 0.3 | Mean across eval episodes |
| Conventional gap visible | Max Q during conventional transition window | Q > 0.5 | Mean across eval episodes |
| Continuous power | Min transmitted power during LOCM-EW transition | > -3 dB from steady-state | Single representative episode |
| Conventional gap duration | Duration of zero-power interval | ≥ 50 μs | Configuration parameter |
| Training convergence | BC loss | < 0.1 (MSE) | After 50 epochs |
| RL improvement | Mean episode reward improvement over BC-only | > 20% | After curriculum training |
| Shared codebase validation | LOCM spectral_step, DampingMLP, LOCMBlock used without modification | Pass | Code inspection |

---

## 8. Deliverables

| Deliverable | Format | Purpose |
|---|---|---|
| `moneyshot.png/pdf` | Figure | The single artifact for LL conversation |
| `experiments/cogew_demo/` | Directory | Full reproducible experiment with configs, metrics, figures |
| `src/locm/cogew/` | Python package | CogEW-specific modules (graph, threat, reward) |
| CogEW-LOCM technical note | Markdown (1-2 pages) | Summary for non-technical stakeholders: problem, approach, result |
| Integration mapping | Markdown | How CogEW-LOCM maps to SDR orchestration layer modules |

---

## 9. Connection to Broader LOCM Program

This demonstration validates three things simultaneously:

1. **The LOCM architecture generalizes beyond locomotion.** The same `spectral_step`, `DampingMLP`, and `LOCMBlock` code that drives the fly demonstrator produces meaningful behavior in a completely different domain. The spectral coupling graph is the abstraction layer — swap the graph, change the domain.

2. **The smooth transition claim is real.** The money-shot figure is empirical evidence for the core LOCM-EW capability described in the application plan. It moves the argument from "this should work in principle" to "here's what it looks like."

3. **The SDR orchestration layer has a mathematical backbone.** The spectral coupling graph formalizes the relationships between SDR channels that your orchestration layer currently manages procedurally. LOCM-EW provides a principled, trainable alternative to hand-coded channel allocation and technique scheduling.

---

## 10. Timeline

| Day | Activity | Output |
|---|---|---|
| 1 | Spectral graph + threat model | `spectral_graph.py`, `threat_model.py`, validated standalone |
| 2 | Expert demos + LOCM-EW controller + BC | `expert_demos.py`, `locm_ew_controller.py`, BC-trained model |
| 3 | RL training (curriculum stages 1-2) | Partially trained LOCM-EW |
| 4 | RL training (stage 3) + evaluation runs | Trained model + evaluation data |
| 5 | Figures + documentation | `moneyshot.png`, supporting figures, technical note |

**Total: 5 working days to the money-shot figure.**
