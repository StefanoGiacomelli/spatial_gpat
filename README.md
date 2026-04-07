# GP-AT SELD — Technical Report

Spatial × Semantic Sound Event Localization and Detection  
Stefano Giacomelli — DISIM, University of L'Aquila

---

## Table of Contents

1. [Project Overview and Scope](#1-project-overview-and-scope)
2. [From Multi-ACCDOA / ADPIT to Separate Heads + tPIT](#2-from-multi-accdoa--adpit-to-separate-heads--tpit)
3. [Architecture Overview](#3-architecture-overview)
4. [Input Processing Modules](#4-input-processing-modules)
5. [Semantic Backbone — GP-AT (E-PANNs)](#5-semantic-backbone--gp-at-e-panns)
6. [Cross-Stitch Fusion (Optional)](#6-cross-stitch-fusion-optional)
7. [Spatial Processing Modules](#7-spatial-processing-modules)
8. [Track Smoothing](#8-track-smoothing)
9. [Output Heads](#9-output-heads)
10. [Loss Function](#10-loss-function)
11. [SELD Metrics](#11-seld-metrics)
12. [Training Pipeline — NAS Phase 1](#12-training-pipeline--nas-phase-1)
13. [Dataset Management](#13-dataset-management)
14. [Experiment Configuration](#14-experiment-configuration)

---

## 1. Project Overview and Scope

### SELD Task

Sound Event Localization and Detection (SELD) requires the system to simultaneously:

- **Detect** which sound event classes are active at each time frame.
- **Localize** each active source by estimating its Direction of Arrival (azimuth and elevation).
- Handle **multi-source polyphony**: multiple events of the same or different classes may be simultaneously active.

The prediction target at each frame `t` is a set of up to `N` active sources, each characterized by a class label `c ∈ {0, …, C-1}` and a Cartesian unit vector `(x, y, z)` pointing toward the source.

### GP-AT Approach

This project extends the **Generalised Pretrained Audio Tagger (GP-AT)** paradigm by coupling a pretrained audio tagging backbone (E-PANNs, fine-tunable) with a configurable spatial processing pipeline. The two branches exchange information via optional **CrossStitch bridges**:

- **Semantic branch**: processes mono audio through E-PANNs (pretrained on AudioSet, 527 classes) to produce class-identity features — strong class discriminability, no spatial information.
- **Spatial branch**: processes multichannel audio (FOA or MIC array) through a configurable front-end and feature stack to predict source activity and direction per track.
- **Cross-stitch fusion**: learnable per-channel coupling gates allow each branch to selectively absorb information from the other at two points in the pipeline (early and late).

### NAS Phase 1

The current development phase performs a **Neural Architecture Search (NAS) Phase 1**:

- **Shallow configurations**: each module (input, spatial early, spatial late, smoother) is instantiated with a single layer (`n_layers=1`).
- **20 full epochs**, no Early Stopping (`patience=0`) → equal-length evaluation across all configurations.
- **Goal**: identify which module combinations yield the most promising performance to guide deeper NAS in subsequent phases.

---

## 2. From Multi-ACCDOA / ADPIT to Separate Heads + tPIT

### Previous Approach: Multi-ACCDOA + ADPIT

The original design used a **Multi-ACCDOA** output representation:

```
p(b, t, n, c) = sigmoid(a) × normalize(d) ∈ R³
```

Activity was encoded in the vector norm: `‖p‖ > τ → active`, direction from `p / ‖p‖`.

The **ADPIT loss** (Auxiliary Duplicating Permutation Invariant Training) minimized MSE on this unified vector by enumerating all **surjective functions** from N prediction tracks to M active targets (not all N! permutations).

### Problems Identified

1. **Bias initialization bug**: `head_activity.bias = -2.0` → `sigmoid(−2) ≈ 0.12` → predictions never exceed the 0.5 threshold → guaranteed `ER = 1.0, F = 0.0` for the first epoch, regardless of other parameters.

2. **Gradient entanglement**: the combined vector loss couples activity and direction gradients — improving DOA may degrade SED and vice versa, with no independent control.

3. **Loss opacity**: a single scalar loss gave no insight into whether the model was failing on activity detection, localization, or both.

4. **ADPIT surjections**: with M=1 active target, all N tracks were pushed toward the same source. This inflated the SED recall artificially (all tracks active) while degrading precision.

### Current Approach: Separate Heads + tPIT

**Three fully independent heads** are now used:

| Head | Output | Shape | Loss |
|------|--------|-------|------|
| `head_sed` | Raw SED logits | `(B, T, N, C)` | BCE (with pos_weight) |
| `head_doa` | tanh-bounded DOA | `(B, T, N, C, 3)` | MSE |
| `head_semantic` | Class-presence logits | `(B, T_sem, C)` | BCE |

The **tPIT loss** (track-wise Permutation Invariant Training) operates on independent SED and DOA outputs:

- Standard **N! permutations** (not surjections): with M=1 source, exactly one track is assigned to it; the others are pushed toward silence.
- **Permutation selection** uses unweighted BCE + MSE to prevent pos_weight from dominating the matching.
- **Gradient** flows through weighted BCE (pos_weight correction) + MSE of the selected permutation.
- β controls the explicit SED/DOA balance.

---

## 3. Architecture Overview

### Full Data Flow

```
Input audio: (B, n_mics, T_samples) @ orig_sr (24 kHz)
    │
    ├─ resample to model_sr (32 kHz)
    │       (B, n_mics, T_32k)
    │
    ├──[mean channels]──► x_mono (B, T_32k)
    │                          │
    │                   SplitEPANNs.forward_early(x_mono)
    │                          │
    │                   h_early: (B, 256, T/8, F_mel/8)
    │
    └──────► Input Processor
                   │
              (B, C_in, T_mel, F_in)
                   │
            Spatial Early Stack   [ConvBlock ×1  or  ResNetBlock ×1]
                   │
              (B, C_e, T_mel, F_e)
                   │
         [Optional CrossStitchBridge #1 ↔ h_early]
                   │                    │
              (B, C_e, T_mel, F_e)   h_early'
                                        │
                                SplitEPANNs.forward_late(h_early')
                                        │
                                h_late: (B, 1024, T/32, F_mel/32)
                   │
            Spatial Late Stack    [FormantEnhancement ×1  or  TrackTransformer ×1]
                   │
              (B, C_l, T_mel, F_l)  or  (B, T_mel, N·d_model)
                   │
         [Optional CrossStitchBridge #2 ↔ h_late]
                   │
            Track Smoother        [BiGRU ×1  or  MHA-MGU ×1]
                   │
              (B, T_mel, D_s)
                   │
         ┌─────────┴──────────┐
         │                    │
    head_sed               head_doa
    Linear(D_s, N·C)       Linear(D_s, N·C·3)
         │                    │
  (B, T, N, C)          tanh(·)
  act_logits             (B, T, N, C, 3)
                         doa_tanh

    h_late → mean(F) → (B, T_sem, 1024)
    → fc1(1024→2048, pretrained) → ReLU
    → fc_audioset(2048→527, pretrained)
    → head_semantic(527→C, trainable, soft-prior init)
    → (B, T_sem, C)  out_semantic
```

### NAS Configuration Space

| Axis | Options |
|------|---------|
| Input | `spectral` (STFT), `gabor` (learnable Gabor) |
| Spatial Early | `convblock` (SELDnet), `resnet_block` (+ MSCA) |
| Spatial Late | `formant_enhancement` (IIR smoother), `track_transformer` (N Transformers) |
| Track Smoother | `bigru` (BiGRU, merge=mul), `mhamgu` (MHA-MGU) |

Total: 2 × 2 × 2 × 2 = **8 shallow_gab + 8 shallow_spec = 16 configurations**.

---

## 4. Input Processing Modules

### 4a. SpectralProcessor

Multichannel STFT front-end with optional DOA-explicit features.

**STFT per channel k:**

```
X_k(t, f) = Σ_n  x_k[n] · w[n − t·H] · exp(−j2πf·n / N_fft)
```

- Window: Hann, `N_fft = 512`, hop `H = 320` (10 ms @ 32 kHz), `F_out = N_fft/2 = 256` bins (DC removed).
- Base features (2·n_mics channels): magnitude `|X_k(t,f)|` and phase `∠X_k(t,f) ∈ [−π, π]`.

**Intensity Vectors (FOA, +3 channels):**

```
I_x = Re(X_W* · X_X),   I_y = Re(X_W* · X_Y),   I_z = Re(X_W* · X_Z)
```

W, X, Y, Z are the four FOA channels (omnidirectional + figure-of-eight). The IV vector `(I_x, I_y, I_z)` points toward the dominant sound source at each time-frequency bin.

**GCC-PHAT (MIC arrays, +n_mics·(n_mics−1)/2 channels per pair (i,j)):**

```
G_{ij}(f) = X_i(f) · X_j*(f) / |X_i(f) · X_j*(f)|        [unit-normalized cross-spectrum]
gcc_{ij}(τ) = IFFT(G_{ij})[τ],   τ ∈ [−τ_max, τ_max]
```

**Normalization**: independent `InstanceNorm2d` per feature group (magnitude, phase, IV, GCC) to handle their different value ranges.

---

### 4b. GaborFrontEnd

Learnable complex Gabor filter bank applied to raw multichannel waveforms.

**Learnable parameters per filter n:**

```
η_n = 0.5 · sigmoid(η_raw)       ∈ (0, 0.5)    [normalized center frequency, Nyquist = 0.5]
σ_n = exp(log_σ_raw)             > 0            [Gaussian bandwidth in samples]
```

**Complex Gabor kernel:**

```
g_n(t) = (1 / √(2πσ_n²)) · exp(−t² / (2σ_n²)) · exp(j2π η_n t)
```

Implemented as two real convolutions:

```
real(g_n)[t] = gauss(t, σ_n) · cos(2π η_n t)
imag(g_n)[t] = gauss(t, σ_n) · sin(2π η_n t)
```

with a Hamming window applied to reduce spectral leakage:

```
w[t] = 0.54 − 0.46 · cos(2πt / (L−1)),   t ∈ [0, L−1]
```

**Initialization**: center frequencies η_n spaced on the mel scale between `fmin` and `fmax`; σ_n inversely proportional to frequency (wider bandwidth for lower frequencies).

**Cross-spectra (FOA, n_mics = 4 channels W, X, Y, Z):**

Let `Z_k = R_k + j·I_k` be the Gabor output for channel k. The cross-spectrum with the reference channel (W, k=0) is:

```
CS_k = Z_k · conj(Z_0) = (R_k R_0 + I_k I_0) + j(I_k R_0 − R_k I_0)
```

Output features: `[|Z_0|, Re(CS_1), Im(CS_1), Re(CS_2), Im(CS_2), Re(CS_3), Im(CS_3)]`  
→ `n_spatial = 1 + 2·(n_mics − 1) = 7` channels.  
Output shape: `(B, n_spatial, T_out, n_filters)`, stride = 320.

---

## 5. Semantic Backbone — GP-AT (E-PANNs)

### 5a. Cnn14Pruned Architecture

E-PANNs is a pruned version of the CNN14 audio tagging model pretrained on AudioSet (527 classes, ≈2M clips).

**Input processing (frozen):**

```
x: (B, T_samples) @ 32 kHz
  → STFT: N_fft=1024, hop=320, Hann window
  → Log-Mel filterbank: 64 mel bins, fmin=50 Hz, fmax=14 kHz
  → LogMel(t, f) = log(max(Σ_f |X(t,f)|² · mel_filter_f, ε))
  → BN0: BatchNorm1d on mel axis
```

**Convolutional stack:**

| Block | Channels | MaxPool | Pruning |
|-------|----------|---------|---------|
| ConvBlock1 | 1 → 64 | 2×2 | none |
| ConvBlock2 | 64 → 128 | 2×2 | none |
| ConvBlock3 | 128 → 256 | 2×2 | none |
| ConvBlock4 | 256 → 512 | 2×2 | 50% |
| ConvBlock5 | 512 → 1024 | 2×2 | 50% |
| ConvBlock6 | 1024 → 2048 | 1×1 | 50% |

Each ConvBlock: `Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → ReLU → MaxPool → Dropout(0.2)`.

**Global pooling:**

```
f_pool = global_mean(T, F) + global_max(T, F)    element-wise sum → (B, C)
```

**Head:**

```
→ Dropout(0.5) → FC1: C → 2048, ReLU
→ Dropout(0.5) → FC_audioset: 2048 → 527, sigmoid
```

### 5b. SplitEPANNs — Per-Group Control

The pretrained model is split into three independently configurable parameter groups:

| Group | Layers | Intermediate output |
|-------|--------|---------------------|
| `early_group` | LogMel, BN0, ConvBlock1–3 | `h_early: (B, 256, T/8, F_mel/8)` |
| `late_group` | ConvBlock4–6 | `h_late: (B, 1024, T/32, F_mel/32)` |
| `head_group` | FC1, FC_audioset | used internally in NASModel |

**Per-group YAML config:**

```yaml
early_group:
  init: pretrained   # or 'random' to re-initialize
  frozen: false      # or true to freeze requires_grad
```

The intermediate hook `forward_early(x_mono) → h_early` and `forward_late(h_early') → h_late` allow the CrossStitch bridges to inject spatial information into the semantic branch between the two stages.

### 5c. Semantic Head in NASModel (EPANNs Splitted)

The semantic output head reuses the pretrained FC weights as a feature extractor and appends a trainable projection to the SELD dataset classes:

```
seq_sem = mean(F)(h_late) → (B, T_sem, 1024)
→ sem_fc1(1024 → 2048, pretrained) → ReLU_
→ sem_fc_audioset(2048 → 527, pretrained)
→ head_semantic(527 → n_cls, trainable)
→ out_semantic: (B, T_sem, n_cls)   raw class-presence logits
```

**Soft-prior initialization** of `head_semantic`:

```
W_semantic[c, as_idx] = 1 / |G_c|    for all as_idx ∈ G_c
```

where `G_c` is the set of AudioSet label indices mapped to SELD class c (from `configs/labels_map.json`). This structured initialization accelerates convergence by exploiting the semantic prior of the pretrained AudioSet model.

---

## 6. Cross-Stitch Fusion (Optional)

The CrossStitch bridges are disabled by default (`cs_early_enabled: false`, `cs_late_enabled: false`). When enabled, they implement a learnable bidirectional fusion between the spatial and semantic feature maps.

### 6a. MultiScaleChannelAttention (MSCA)

Channel-wise gating combining global and local context:

```
g(X) = GlobalAvgPool(X)               (B, C, 1, 1)
G(X) = BN(Conv2(ReLU(BN(Conv1(g(X))))))    global path (C → C/r → C, 1×1 conv)
L(X) = BN(Conv2(ReLU(BN(Conv1(X)))))       local path  (C → C/r → C, 1×1 conv)
Y = X ⊗ sigmoid(G(X) + L(X))
```

Bottleneck ratio r = 4 (default). Both paths use separate weights; no weight sharing.

### 6b. CrossStitching

Learnable per-channel 2×2 coupling matrix:

**Parameters**: `W ∈ R^(C×2×2)`, initialized Uniform(0.1, 0.9).

**Cascade update** (sequential, not symmetric):

```
a'_c = W[c, 0, 0] · a_c + W[c, 0, 1] · b_c          (uses original a, b)
b'_c = W[c, 1, 0] · a'_c + W[c, 1, 1] · b_c          (uses updated a', original b)
```

Note: this is a **cascade** (A→B in one step), not the symmetric mixing of Misra et al. (2016).

### 6c. CrossStitchBridge — 6-Step Pipeline

Input: `s: (B, C_s, T_s, F_s)` (spatial), `e: (B, C_e, T_sem, F_sem)` (semantic).

```
Step 1:  s_att  = MSCA(s)                                     (B, C_s, T_s, F_s)
Step 2:  s_proj = Conv1×1(s_att → C_b)                       (B, C_b, T_s, F_s)
Step 3:  e_proj = Conv1×1(e → C_b); align T_sem→T_s; broadcast F_s
         e_broad                                               (B, C_b, T_s, F_s)
Step 4:  [s_proj; e_broad] → joint MSCA → s_mid, e_mid       (B, C_b, T_s, F_s) each
Step 5:  s_cs, e_cs = CrossStitch(s_mid, e_mid)              (B, C_b, T_s, F_s) each
Step 6a: s_out = s + Dropout(Conv1×1(s_cs → C_s))            (B, C_s, T_s, F_s)
Step 6b: e_out = e + Dropout(Conv1×1(AvgPool_F(s_cs) → C_e)) (B, C_e, T_sem, F_sem)
```

**Design principle**: the spatial branch never collapses the frequency axis (F_s is preserved) because inter-frequency phase differences are essential DOA cues. The semantic branch is frequency-averaged before broadcast and after re-projection.

---

## 7. Spatial Processing Modules

### 7a. ConvBlock (SELDnet-style)

```
Conv2d(C_in → C_out, 3×3, padding='same')
  → BatchNorm2d → ReLU
  → MaxPool2d(kernel=(1, pool_size))       [frequency-only pooling, T preserved]
  → Dropout
```

Output shape: `(B, C_out, T_mel, F / pool_size)`.

### 7b. ResNetBlock (+ optional MSCA)

**Main path:**

```
Conv2d(3×3, stride) → BN → ReLU → Conv2d(3×3) → BN → [MSCA]
```

**Shortcut:** `Conv2d(1×1) + BN` if `C_in ≠ C_out` or `stride ≠ 1`.

**Output:** `ReLU(main_path + shortcut)`.

MSCA is applied before the residual addition (channel recalibration then add).

### 7c. Channel Schedule

For the spatial early stack, the channel progression through `n_layers` uses a bottleneck schedule:

| n_layers | Channel sequence |
|----------|-----------------|
| 1 | `[out_ch]` |
| 2 | `[2×out_ch, out_ch]` |
| ≥3 | `[3×out_ch, 2×out_ch, out_ch, …, out_ch]` |

### 7d. FormantEnhancementModule (IIR Frequency Smoother)

Learnable IIR filter along the frequency axis, designed to extract "formant traces" analogous to spectral envelope features.

**Learnable smoothing coefficient:** `s = sigmoid(s_raw) ∈ (0, 1)`

**Causal IIR forward pass along frequency dimension F:**

```
fr(t, f) = (1−s) · X(t, f) + s · fr(t, f−1)        [running average across freq bins]
dt(t, f) = X(t, f) − fr(t, f)                        [detail / residual signal]
```

**Output:**

```
Y = sigmoid(Conv_{3×1}(fr)) + tanh(Conv_{3×1}(dt))
```

where `Conv_{3×1}` is a 2D convolution with kernel `(3, 1)` — 3 frequency bins context, 1 time step context. Shape-preserving: output is `(B, C, T, F)`.

### 7e. TrackTransformer

N independent `TransformerEncoder` instances, all receiving the same frequency-pooled input. The track separation emerges during training via the PIT loss assignment.

**Preprocessing:**

```
(B, C, T, F) → mean(F) → (B, T, C)
```

**Per encoder k = 0, …, N−1:**

```
z_k = TransformerEncoder_k(z)     (B, T, d_model) — PyTorch standard encoder layer
```

Each `TransformerEncoderLayer` implements:

```
z' = LayerNorm(z + MHA(z, z, z))
z_out = LayerNorm(z' + FFN(z'))
```

**Sinusoidal positional encoding** (amplitude scaled by 0.1):

```
PE[pos, 2i]   = 0.1 · sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = 0.1 · cos(pos / 10000^(2i / d_model))
```

**Output stack:** `(B, T, N, d_model) → reshape → (B, T, N·d_model)`.

---

## 8. Track Smoothing

### 8a. BiGRU (merge_mode = 'mul')

Standard bidirectional GRU with element-wise product merge (output dimension = `hidden_size`, not `2·hidden_size`).

**GRU cell equations (per direction):**

```
r_t = σ(W_ir · x_t + W_hr · h_{t−1} + b_r)               reset gate
z_t = σ(W_iz · x_t + W_hz · h_{t−1} + b_z)               update gate
n_t = tanh(W_in · x_t + r_t ⊙ (W_hn · h_{t−1}) + b_n)   candidate
h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t−1}                   output
```

**Bidirectional merge:**

```
h_t = h_fwd_t ⊙ h_bwd_t          element-wise product (not concatenation)
```

Initialization: `weight_hh` orthogonal, `weight_ih` kaiming_uniform (sigmoid).

### 8b. MHA-MGU (Permutation-Invariant Track Smoother)

Multi-Head Attention Minimal Gated Unit — maintains `M_H` independent track state vectors updated at each timestep via soft assignment.

**Input:** `x_t ∈ R^(B, M_X, d_input)` — set of `M_X` detection features at time t.

**Step 1 — Input projection** (if `d_input ≠ d_model`):

```
x̃_t = Linear(d_input → d_model)(x_t)         (B, M_X, d_model)
```

**Step 2 — Key-Value set construction** (detections + previous track states):

```
KV_t = concat([x̃_t, h_{t−1}], dim=1)         (B, M_X + M_H, d_model)
```

**Step 3 — Multi-Head Attention** (soft assignment of detections to tracks):

```
context_t = Attention(Q=h_{t−1}, K=KV_t, V=KV_t)
           = softmax(Q · K^T / √d_k) · V       (B, M_H, d_model)
```

where `d_k = d_model / n_heads`.

**Step 4 — Minimal Gated Unit (MGU) state update:**

```
z_t  = σ(W_z · context_t)                     forget gate   (B, M_H, d_model)
h̃_t  = tanh(W_h · context_t)                  candidate     (B, M_H, d_model)
h_t  = (1 − z_t) ⊙ h_{t−1} + h̃_t             new state     (B, M_H, d_model)
```

**Output:** `H ∈ R^(B, T, M_H, d_model) → reshape → (B, T, M_H · d_model)`.

The permutation-invariance of `x_t` is guaranteed by the attention mechanism operating on an unordered set; track identity emerges from the persistent state `h_{t−1}`.

### 8c. mhamgu_expand Projection

When `spatial_late = FormantEnhancement`, the smoother input is `(B, T, C)` with no track structure (single detection per frame). Without intervention, all `M_H` track queries would attend to the same single input, making track assignment degenerate.

A dedicated learnable projection generates `M_H` distinct input vectors:

```
expand_out = Linear(C → M_H · d_model)(seq)   (B, T, M_H · d_model)
           → view(B, T, M_H, d_model)          distinct detection vectors per track
```

Initialization: `weight ~ Normal(0, 0.01)`, `bias = 0` (small init to avoid premature specialization).

---

## 9. Output Heads

### SED Head (`head_sed`)

```
act_logits = Linear(D_s, N·C)(spat_out)   → view(B, T, N, C)
```

- Initialization: `Xavier uniform` (weights), `zeros` (bias).
- At inference: `sigmoid(act_logits) > 0.5` → active track detection.

### DOA Head (`head_doa`)

```
doa_tanh = tanh(Linear(D_s, N·C·3)(spat_out))   → view(B, T, N, C, 3)
```

Output is bounded to `[−1, 1]³` (tanh). The targets are unit Cartesian vectors (norm = 1 when active, norm = 0 when silent), so MSE between `doa_tanh` and the target is well-scaled (max component error ≤ 2, max MSE ≤ 12).

- Initialization: `Xavier uniform` (weights), `zeros` (bias).

### Semantic Head (`head_semantic`)

```
seq_sem → sem_fc1(1024→2048, pretrained) → ReLU_
       → sem_fc_audioset(2048→527, pretrained)
       → head_semantic(527→C, trainable)
       → out_semantic: (B, T_sem, C)
```

Trained with BCE loss to predict binary class presence per frame. `T_sem < T_mel` (EPANNs ConvBlocks pool time by 32×).

---

## 10. Loss Function

### 10a. Total Loss

```
L_total = λ_s · L_tPIT + λ_e · L_sem
```

Default values: `λ_s = 0.9`, `λ_e = 0.1` (configurable via `training.lambda_spatial` / `training.lambda_semantic`).

### 10b. tPIT Loss — Permutation Selection

For each batch-time-class index `(b, t, c)` (flattened as BTC), all `P = N!` permutations of N tracks are evaluated. The **unweighted** cost for permutation π is:

```
C(π) = β · BCE_nopw(act_logits[π], a_target) + (1−β) · MSE(doa_tanh[π], d_target)
```

where:

```
a_target[b, t, n, c] = 1   iff  ‖target[b, t, n, c, :]‖₂ > TARGET_THRESH (= 0.5)
d_target[b, t, n, c, :] = target[b, t, n, c, :]    (unit vector or zero)
```

The **optimal permutation** (non-differentiable):

```
π*(b, t, c) = argmin_π C(π)
```

**Reason for unweighted BCE in selection**: using `pos_weight` (up to 20× for rare STARSS23 classes) in the selection cost would cause rare-class assignments to dominate, preventing the DOA term from contributing meaningfully to the track matching decision.

### 10c. tPIT Loss — Gradient

The gradient flows through the loss evaluated at the selected permutation:

```
L_tPIT = β · L_SED(π*) + (1−β) · L_DOA(π*)

L_SED(π*) = BCE_with_logits(act_logits[π*], a_target, pos_weight)
L_DOA(π*) = MSE(doa_tanh[π*], d_target)
```

`β = β_sed = 0.3` (configurable via `training.beta_sed`).

### 10d. pos_weight (Class-Imbalance Correction)

```
pos_weight[c] = neg_frames[c] / pos_frames[c]
```

Computed from the training set CSV annotations (counting unique `(clip, frame_100ms)` pairs per class). Capped at `pos_weight_max = 20.0` to prevent extreme gradient magnitudes for very rare classes.

Applied as a scaling factor in `binary_cross_entropy_with_logits`: rare classes receive proportionally larger gradient contributions.

### 10e. Semantic Presence Loss

```
presence[b, t, c] = 1   iff  ∃n:  ‖target[b, t, n, c, :]‖₂ > TARGET_THRESH

L_sem = BCE_with_logits(out_semantic, presence_downsampled, pos_weight)
```

`presence_downsampled` is obtained by nearest-neighbor downsampling of `tgt_spatial` from `T_mel` to `T_sem` (index selection, no interpolation).

### 10f. Effective Loss Weights

```
DOA:       λ_s × (1−β) = 0.9 × 0.7 = 0.63
SED:       λ_s ×  β    = 0.9 × 0.3 = 0.27
Semantic:  λ_e         =             0.10
```

---

## 11. SELD Metrics

All metrics are aligned with the DCASE 2023 baseline evaluation protocol.

### 11a. Activity Detection at Inference

```
p_active[b, t, n, c] = σ(act_logits[b, t, n, c]) > ACTIVE_THRESH    (= 0.5)
```

where `σ` is the sigmoid function. Note: this uses the independent SED head, not the norm of any combined vector.

### 11b. Angular Distance

```
d(a, b) = arccos( clip( (a · b) / (‖a‖₂ · ‖b‖₂), −1, 1 ) )     ∈ [0°, 180°]
```

The `clip` ensures numerical stability near ±1. Both input vectors need not be unit-normalized; the formula normalizes internally.

### 11c. Per-Frame Hungarian Matching (per class)

For each `(b, t, c)` with at least one active prediction or target:

1. Collect `P` active predicted vectors and `Q` active target vectors.
2. Compute `P × Q` pairwise angular distance matrix.
3. Solve optimal assignment: `row_ind, col_ind = linear_sum_assignment(cost)`.
4. Classify matched pairs:
   - **TP**: matched pair with `d ≤ angle_thresh` (= 20°)
   - **FP**: unmatched predictions + matched pairs with `d > 20°`
   - **FN**: unmatched targets

### 11d. LE_CD — Localization Error (Class-Dependent)

```
LE_CD = (1 / |C_TP|) · Σ_{c: tp_c > 0}  (angle_sum_c / tp_c)
```

- `angle_sum_c` = sum of angular errors for all TP pairs of class c
- `|C_TP|` = number of classes with at least 1 TP
- **Averaging excludes** classes with 0 TP (not penalized as 0° or 180°)
- Units: degrees; range [0°, 180°]; lower is better

### 11e. LR_CD — Localization Recall (Class-Dependent)

```
LR_CD = (1 / |C_ref|) · Σ_{c: (tp_c + fn_c) > 0}  tp_c / (tp_c + fn_c)
```

- `|C_ref|` = number of classes with at least 1 reference event
- **Averaging excludes** classes with 0 reference events
- Range [0, 1]; higher is better

### 11f. ER_20° — Error Rate (DCASE formula)

```
ER = max(FP_global, FN_global) / (TP_global + FN_global)

FP_global = Σ_c fp_c,   FN_global = Σ_c fn_c,   TP_global = Σ_c tp_c
```

Range [0, ∞) (practically [0, 2]); lower is better.

### 11g. F_20° — F-score (location-dependent)

```
F = TP_global / (TP_global + 0.5 · (FP_global + FN_global))
```

Range [0, 1]; higher is better.

### 11h. SELD_score (Combined)

```
SELD_score = (ER + (1 − F) + LE_CD/180 + (1 − LR_CD)) / 4
```

All four terms normalized to [0, 1]; range [0, 1]; lower is better.

### 11i. Note on Windowing and Double-Counting

When `hop_s < window_s`, overlapping windows cause each frame to be visited multiple times during validation/test. This leads to double-counting of TP/FP/FN statistics and slightly biased metric estimates. For DCASE-aligned evaluation, use `hop_s = window_s = 10s` (non-overlapping windows, no double-counting).

---

## 12. Training Pipeline — NAS Phase 1

### 12a. Optimizer

```
Adam(parameters, lr = 5×10⁻⁴)
```

No weight decay is configured (pure Adam). Only parameters with `requires_grad = True` are passed to the optimizer, respecting the per-group freeze settings of SplitEPANNs.

### 12b. Learning Rate Scheduler (CosineAnnealingLR)

```
lr(t) = η_min + 0.5 · (lr_0 − η_min) · (1 + cos(π · t / T_max))

T_max = max_epochs = 20
η_min = lr_0 × 0.1 = 5×10^-5
```

The LR decays smoothly from `5×10⁻⁴` to `5×10^-5` over 20 epochs, following a cosine curve. The scheduler is updated once per epoch (`interval = 'epoch'`).

### 12c. Gradient Management

**Gradient clipping** (L2 norm, applied per step):

```
‖g‖₂ ← min(‖g‖₂, 1.0)
```

**Gradient accumulation** (to simulate effective batch size = 16):

```
accum_steps = max(1, floor(16 / batch_size))
effective_batch = batch_size × accum_steps ≈ 16
```

Example: `batch_size = 4 → accum_steps = 4 → effective_batch = 16`.

### 12d. Early Stopping

Disabled by default (`patience = 0`). The rationale is that all NAS configurations must be evaluated for the full 20 epochs to enable a fair comparison. EarlyStopping can be re-enabled via `--patience N` (CLI) for production runs after the NAS phase.

### 12e. Checkpointing and Resume

**Checkpoint strategy:**

- `best.ckpt`: lowest `val_seld_score` observed (ModelCheckpoint, `mode=min`, `save_top_k=1`).
- `last.ckpt`: most recent completed epoch (used as resume target).

**Resume flow (interrupted experiment):**

```
Priority: last.ckpt → any *.ckpt (best) → fresh start
```

The `experiment_state.json` file records `{status, started_at, best_ckpt, metrics_file}` per experiment and is updated atomically (write to `.tmp`, then rename).

**Metrics log resume:** On resume, the existing metrics JSON is pre-loaded into `_epoch_metrics_log` before training continues, ensuring progressive accumulation of per-epoch results.

### 12f. Logging

**TensorBoard** (per-step and per-epoch):

| Key | Frequency | Description |
|-----|-----------|-------------|
| `train/loss` | step + epoch | Weighted total loss |
| `train/loss_tpit` | epoch | Spatial tPIT loss |
| `train/loss_bce` | epoch | Semantic BCE loss |
| `val/loss_tpit` | epoch | Validation tPIT loss |
| `val/loss_sed` | epoch | SED component (β-weighted BCE) |
| `val/loss_doa` | epoch | DOA component ((1−β)-weighted MSE) |
| `val/loss_bce` | epoch | Semantic BCE |
| `val/LE_CD` | epoch | Localization error (°) |
| `val/LR_CD` | epoch | Localization recall |
| `val/ER_20` | epoch | Error rate |
| `val/F_20` | epoch | F-score |
| `val/SELD_score` | epoch | Combined SELD metric |
| `val/sem_accuracy` | epoch | Semantic accuracy |
| `val/sem_f1` | epoch | Semantic F1 (macro) |
| `val/sem_auroc` | epoch | Semantic AUROC (macro) |

**JSON metrics log** (per-experiment, progressive):

```json
[
  {"epoch": 0, "val/SELD_score": 0.52, "val/LE_CD": 45.3, ...},
  {"epoch": 1, "val/SELD_score": 0.48, ...},
  ...
  {"epoch": "test", "test/SELD_score": 0.47, ...}
]
```

The confusion matrix for the semantic head is saved to a separate `{exp_name}_confusion_matrix.json` file at test time.

---

## 13. Dataset Management

### 13a. Multi-ACCDOA Target Format

The ground truth is stored as a **Multi-ACCDOA tensor**:

```
target: (T_mel, N_tracks, C_classes, 3)
```

For each event annotation `(frame_100ms, class_c, source_n, az_deg, el_deg)`:

```
az = az_deg × π/180,   el = el_deg × π/180

x = sin(az) · cos(el)
y = cos(az) · cos(el)
z = sin(el)

target[t_mel, n, c, :] = [x, y, z]     if source active (‖[x,y,z]‖₂ = 1)
target[t_mel, n, c, :] = [0, 0, 0]     if silent
```

Track slots are assigned in ascending `source_idx` order; sources beyond `N_tracks` are silently dropped.

### 13b. Mel-Frame to CSV-Frame Mapping

CSV annotations have 100 ms temporal resolution. The mapping from mel-frame index m to CSV frame:

```
t_s(m) = (start_at_model_sr + m · hop_length) / model_sr     [seconds]
t_csv(m) = floor(t_s(m) / 0.1)                               [CSV frame index, 0-indexed]
```

At `model_sr = 32000`, `hop_length = 320`: **10 mel-frames per 100 ms CSV frame**.

### 13c. Windowed Dataset

When `window_s > 0`:

```
window_samples = round(window_s × orig_sr)       [audio samples]
hop_samples    = round(hop_s × orig_sr)
T_mel_window   = 1 + floor(window_samples × model_sr / orig_sr / hop_length)
```

The flat item index is built by sliding a window over each clip:

```
items = [(clip_i, 0), (clip_i, hop_samples), (clip_i, 2·hop_samples), …]
```

The last window is zero-padded to `window_samples` if the clip ends early.

**DCASE-aligned evaluation**: `window_s = hop_s = 10s` → non-overlapping windows, no frame double-counting in metrics.

### 13d. STARSS23 Dataset

- **Classes**: 13 sound event classes
- **Native SR**: 24 kHz
- **Audio formats**: FOA (`foa_dev/`) and MIC (`mic_dev/`)
- **CSV format**: `frame_100ms, class_idx, source_idx, azimuth_deg, elevation_deg, distance_cm` (distance unused)
- **Split**: room-stratified random split with `val_fraction = 0.2`, seed = 42
  - Test: `dev-test-sony/` + `dev-test-tau/` (78 clips, fixed)
  - Train/Val: `dev-train-sony/` + `dev-train-tau/`, grouped by site-room, shuffled per room, 20% held out for validation

### 13e. TAU-NIGENS2021 Dataset

- **Classes**: 12 sound event classes
- **Native SR**: 24 kHz
- **CSV format**: `frame_100ms, class_idx, source_idx, azimuth_deg, elevation_deg` (5 columns)
- **Split**: official fold-based split:
  - Train: folds 1–4 (`dev-train/`, 400 clips)
  - Val: fold 5 (`dev-val/`, 100 clips)
  - Test: fold 6 (`dev-test/`, 100 clips)

### 13f. Data Augmentation (FOA)

Applied during training only, disabled by default (`augmentation.enabled: false`):

| Augmentation | Operation | Audio transform | Target transform |
|---|---|---|---|
| Azimuth rotation by θ ∈ U(0°, 360°) | FOA rotation | `X' = X cos θ − Y sin θ; Y' = X sin θ + Y cos θ; W, Z unchanged` | `x' = x cos θ − y sin θ; y' = x sin θ + y cos θ; z' = z` |
| Global gain `g = exp(U(log_min, log_max))` | Scale waveform | `wav *= g` | Target unchanged |
| Elevation flip | Mirror Z | `Z ← −Z; W, X, Y unchanged` | `z ← −z; x, y unchanged` |

---

## 14. Experiment Configuration

### 14a. YAML Hierarchy

```
configs/
├── datasets/
│   ├── STARSS23.yaml          # pos_weight_max, weighted_sampler, augmentation
│   └── TAU-NIGENS2021.yaml
└── experiments/foa/
    ├── shallow_spec/          # spectral input (8 configs)
    │   └── exp_spe_{cb,rn}_{fe,tt}_{bg,mg}.yaml
    └── shallow_gab/           # Gabor input (8 configs)
        └── exp_gab_{cb,rn}_{fe,tt}_{bg,mg}.yaml
```

### 14b. Experiment Naming Convention

```
exp_{INPUT}_{EARLY}_{LATE}_{SMOOTHER}

INPUT:   spe = spectral (STFT)
         gab = learnable Gabor
EARLY:   cb  = ConvBlock
         rn  = ResNetBlock (+ MSCA)
LATE:    fe  = FormantEnhancement
         tt  = TrackTransformer
SMOOTHER: bg = BiGRU (hidden=256)
          mg = MHA-MGU (d_model=128, n_heads=4)
```

### 14c. Key Configuration Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `n_tracks` | 2 | YAML (`heads.n_tracks`) — STARSS23 rarely has >2 simultaneous sources per class |
| `n_layers` | 1 | YAML (all spatial modules) — shallow NAS phase |
| `out_channels` | 64 | YAML (`spatial_early.out_channels`) |
| `hidden_size` | 256 | YAML (`track_smoothing_spatial.hidden_size`, BiGRU) |
| `d_model` | 128 | YAML (`track_smoothing_spatial.d_model`, MHA-MGU) |
| `n_heads` | 4 | YAML (`track_smoothing_spatial.n_heads`, MHA-MGU) |
| `cross_stitch` | disabled | YAML (`cs_early_enabled: false`, `cs_late_enabled: false`) |
| `seed` | 42 | YAML (`experiment.seed`) |

### 14d. Training Invocation

```bash
# Run all experiments in a config directory (40 configs)
python main.py --config-dir configs/experiments/foa/shallow_spec/ \
               --dataset STARSS23 --epochs 20 --batch-size 4

# Resume interrupted run
python main.py --exp exp_spe_cb_fe_mg --dataset STARSS23

# Single experiment with debug limit
python main.py --config-file configs/experiments/foa/shallow_spec/exp_spe_cb_fe_mg.yaml \
               --epochs 1 --limit-batches 20

# Check experiment status
python main.py --list
```

### 14e. Output Directory Layout

```
runs/{config_dir_name}/
├── experiment_state.json              # resume/skip state for all experiments
├── checkpoints/
│   └── {exp_name}/
│       ├── {epoch:02d}_{score:.4f}.ckpt   # best checkpoint (kept)
│       ├── last.ckpt                       # latest checkpoint (deleted on completion)
│       └── version_0/
│           └── events.out.tfevents.*       # TensorBoard logs
└── metrics/
    ├── {exp_name}.json                     # per-epoch + test metrics (progressive)
    └── {exp_name}_confusion_matrix.json    # semantic head confusion matrix (test only)
```
