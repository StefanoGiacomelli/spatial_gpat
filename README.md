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
15. [AT2SED — Overview and Motivation](#15-at2sed--overview-and-motivation)
16. [Label Space: AudioSet-Strong 456-Class Alignment](#16-label-space-audioset-strong-456-class-alignment)
17. [AT2SEDModel Architecture](#17-at2sedmodel-architecture)
18. [AudioSetStrongDataModule — Dataset Pipeline](#18-audiosetstrongdatamodule--dataset-pipeline)
19. [Loss Function and Metrics](#19-loss-function-and-metrics)
20. [Training Pipeline — AT2SED](#20-training-pipeline--at2sed)
21. [AT2SED Experiment Configuration](#21-at2sed-experiment-configuration)

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

---

## Part II — AT2SED: AudioSet-Strong SED Fine-Tuning

---

## 15. AT2SED — Overview and Motivation

### Task

AT2SED (AudioSet-Tagging to Sound Event Detection) is a dedicated fine-tuning pipeline that adapts the pretrained E-PANNs (Cnn14Pruned) backbone for **multi-label frame-level SED on AudioSet-Strong**. The spatial pipeline, CrossStitch bridges, track permutation and DOA estimation from Part I are all omitted — only the semantic backbone is used.

The primary research goal is to measure how much **classification quality** degrades as a function of **input window duration**, as a prerequisite for pseudo-real-time inference at short latencies.

### AudioSet-Strong vs AudioSet Standard

The pretrained E-PANNs model was trained on AudioSet Standard, which provides **clip-level** annotations (one label set per 10-second clip, no temporal resolution). AudioSet-Strong enhances a subset of those clips with **temporally precise** annotations: each event is labelled with a `start_time_seconds` and `end_time_seconds` within the clip, enabling proper frame-level SED evaluation.

### Relationship to GP-AT SELD (Part I)

| Aspect | GP-AT SELD (Part I) | AT2SED (Part II) |
|--------|---------------------|------------------|
| Backbone | SplitEPANNs (shared) | SplitEPANNs (shared) |
| Spatial branch | Yes (FOA/MIC) | No |
| Output | SED + DOA per track | Multi-label logits only |
| Target format | Multi-ACCDOA `(T, N, C, 3)` | Multi-hot `(C,)` |
| Loss | tPIT + semantic BCE | BCEWithLogitsLoss |
| Datasets | STARSS23, TAU-NIGENS2021 | AudioSet-Strong |
| Entry point | `main.py` | `main_sed_only.py` |

SplitEPANNs is instantiated identically in both pipelines; the per-group init/freeze configuration (YAML) applies to both without conflict. The two training workflows are fully independent.

---

## 16. Label Space: AudioSet-Strong 456-Class Alignment

### 16a. Class Set Comparison

AudioSet-Strong covers **456 MID classes**, while the pretrained E-PANNs checkpoint was trained on AudioSet Standard's **527 classes**. The two sets partially overlap:

| Category | Count |
|----------|-------|
| AudioSet Standard (pretrained backbone) | 527 |
| AudioSet-Strong | 456 |
| Shared (∩) | **382** (83.8 % of Strong) |
| Strong-only (not in pretrained) | 74 |
| Standard-only (not in Strong, unused) | 145 |

Naively mapping AudioSet-Strong labels onto the 527-class Standard space would silently discard ~22 % of all training events (the 74 Strong-only classes). The 456-class label map eliminates this loss.

### 16b. Label Map — `configs/datasets/audioset_strong_label_map.csv`

A dedicated CSV (456 rows, 5 columns) assigns a contiguous 0-based index to every AudioSet-Strong class:

```
index, mid, display_name, as_std_mid, as_std_display_name
0,     /m/09x0r, Speech,             /m/09x0r, Speech
1,     /m/05zppz, Male speech ...,   /m/05zppz, Male speech ...
...
381,   /m/07hvw1, ...,               /m/07hvw1, ...      ← last shared class
382,   /t/dd00144, Alert,            None, Strong-only   ← first Strong-only
...
455,   /m/01j423, Yawn,              None, Strong-only
```

**Row ordering:**

- **Indices 0–381**: 382 shared classes, sorted by their original AudioSet Standard index.
- **Indices 382–455**: 74 Strong-only classes, sorted alphabetically by `display_name`.

**Motivation for the ordering:** sorting shared classes by their Standard index preserves the exact semantic ordering of the pretrained `fc_audioset` rows, enabling a direct row-for-row weight copy (see §16c).

Generated once by `experiments_at2sed/build_strong_label_map.py` from:

- `mid_to_display_name.tsv` — AudioSet-Strong class names
- `class_labels_indices.csv` — AudioSet Standard 527-class index map

### 16c. Selective Weight Transfer — `SplitEPANNs._transfer_fc_audioset`

The pretrained checkpoint has `fc_audioset.weight: (527, 2048)`, one row per Standard class. When `n_cls = 456`, a shape mismatch is detected in `_load_checkpoint`. Rather than discarding the head entirely, a selective copy transfers 382 pretrained rows:

```
ckpt fc_audioset : (527, 2048)   pretrained, one row per AS-Standard class
new  fc_audioset : (456, 2048)   model to initialise

For each new_idx in [0, 455]:
    mid     = model_class_map[new_idx]
    old_idx = pretrained_class_map.get(mid)          None if Strong-only

    if old_idx is not None:
        fc_audioset.weight[new_idx] ← ckpt.weight[old_idx]
        fc_audioset.bias[new_idx]   ← ckpt.bias[old_idx]
    else:
        retain Xavier uniform initialisation            Strong-only class

Result: 382 rows transferred from pretrained, 74 Xavier-initialised
```

**Backward compatibility:** the NAS pipeline always calls `SplitEPANNs` with `n_cls = 527`; checkpoint shapes match → the existing `load_state_dict` code path runs unchanged. The new transfer logic is only reached inside the `shape_mismatch` branch.

### 16d. One-Shot Build Script

```bash
.venv/bin/python experiments_at2sed/build_strong_label_map.py
```

Reads the two metadata files and writes `configs/datasets/audioset_strong_label_map.csv`. Run once; output is committed to the repository.

---

## 17. AT2SEDModel Architecture

### 17a. Data Flow

```
x : (B, T_samples) @ 32 kHz
    │
    SplitEPANNs.forward_early(x)
    │   STFT (N_fft=1024, hop=320) → Log-Mel (64 bins)
    │   → BN0 → [SpecAugment, training only] → ConvBlock1–3
    ↓
h_early : (B, 256, T/8, F_mel/8)
    │
    SplitEPANNs.forward_late(h_early)
    │   ConvBlock4–6
    ↓
h_late : (B, 1024, T/32, F_mel/32)
    │
    mean(dim=F) → (B, 1024, T/32)
    max(dim=T)  +  mean(dim=T) → (B, 1024)    element-wise sum (Cnn14Pruned style)
    │
    Dropout(0.5)
    FC1:         1024 → 2048, ReLU
    Dropout(0.5)
    FC_audioset: 2048 → 456
    ↓
logits : (B, 456)   raw logits — BCEWithLogitsLoss at train, sigmoid at inference
```

### 17b. Per-Group Configuration

Identical to the NAS pipeline (§5b). All three parameter groups are configurable via the `semantic` YAML section:

| Group | Layers | Default (`at2sed_default.yaml`) |
|-------|--------|---------------------------------|
| `early_group` | LogMel, BN0, ConvBlock1–3 | `init: pretrained, frozen: false` |
| `late_group` | ConvBlock4–6 | `init: pretrained, frozen: false` |
| `head_group` | FC1, FC_audioset | `init: pretrained` — always unfrozen |

`head_group` has no `frozen` key: the classification stage is always trainable by design. Setting `init: pretrained` triggers the selective weight transfer (§16c); `init: random` applies Xavier uniform to the entire head, overriding any transfer.

### 17c. SpecAugmentation Guard for Short Windows

`forward_early` calls `bb.spec_augmenter(x)` during training. The torchlibrosa `SpecAugmentation` time-masking requires the mel-spectrogram time dimension to exceed the masking distance parameter. For windows shorter than ~1 s the tensor is too short and a `RuntimeError` is raised. The guard skips augmentation in that case:

```python
if self.training:
    try:
        x = bb.spec_augmenter(x)
    except RuntimeError:
        pass   # spectrogram too short for time/freq masking — skip augmentation
```

Training proceeds normally; augmentation is simply absent for short-window batches.

---

## 18. AudioSetStrongDataModule — Dataset Pipeline

### 18a. Annotation Format

**Train TSV** (4 columns):

```
segment_id            start_time_seconds  end_time_seconds  positive_labels
b0RFKhbpFJA_30000     0.0                 0.5               /m/09x0r,/m/07q4ntr
```

- `segment_id` = `{ytid}_{start_ms}` (start offset of the 10-second clip in the YouTube video)
- `positive_labels` = comma-separated MID list of all events active in `[start_s, end_s]`

**Eval TSV** (5 columns): adds `audio_filename` (not used by the dataloader; the ytid is extracted from `segment_id`).

### 18b. Clip ID → Audio File Mapping

AudioSet-Strong clip IDs embed the clip start offset (`{ytid}_{start_ms}`), but audio files are named `{ytid}.wav`. The dataset strips the `_\d+` suffix before looking up the file:

```
clip_id :   b0RFKhbpFJA_30000
audio   :   b0RFKhbpFJA.wav      search in audio_root_train, audio_root_unbalanced
```

File lookup searches across two directories:

| Directory | Contents |
|-----------|----------|
| `audio_root_train` (`balanced_train_segments/`) | ~20 k balanced-set clips |
| `audio_root_unbalanced` (`unbalanced_strong/`) | ~95,930 unbalanced clips |

### 18c. Unbalanced Audio Extraction

The unbalanced segments are distributed across 41 multi-part zip archives (`unbalanced_train_segments_part00.zip`, `.z01`, `.z02`, …). Python's `zipfile` module does not support split archives; extraction uses `7z`:

```bash
.venv/bin/python extract_unbalanced_strong.py   # or with --dry-run for inspection
```

```
Source : unbalanced_train_segments_part0*.zip  (41 archives)
Dest   : unbalanced_strong/
Files  : ~95,930 clips  (Y{ytid}.wav → {ytid}.wav after rename)
```

The script is resumable (skips already-extracted files) and supports `--parts N` to extract from only the first N archives.

### 18d. Label Binarisation

For each `(clip, window)` item, events with `[start_s, end_s)` overlapping the window are collected. Their MIDs are mapped to indices via the 456-class label map, producing a float32 multi-hot target:

```
y[c] = 1   if ∃ event with MID mapped to class c active in the window
y[c] = 0   otherwise
```

An overlap threshold of 0.0 s (any overlap) is used by default (`overlap_threshold: 0.0` in `AudioSetStrong.yaml`).

### 18e. Windowing

```
window_samples = round(window_s × 32000)
hop_samples    = round(hop_s × 32000)

items = [(clip_i, 0), (clip_i, hop_samples), (clip_i, 2·hop_samples), …]
```

Each item returns:

- `x : (T_samples,)` float32 @ 32 kHz — zero-padded if the clip is shorter than `window_s`
- `y : (456,)` float32 multi-hot label tensor

### 18f. pos_weight (Class-Imbalance Correction)

```
N_pos[c] = number of windows in training split with y[c] = 1
N_neg[c] = total windows − N_pos[c]

pos_weight[c] = clip(N_neg[c] / N_pos[c],  max = pos_weight_max)
pos_weight_max = 20.0   (default, configurable in AudioSetStrong.yaml)
```

Computed by `AudioSetStrongDataModule` after `setup('fit')` and injected into the loss via `model.set_pos_weight(pw)`. Rare classes receive proportionally larger gradient contributions during training.

### 18g. Dataset Split

| Split | Source | Size |
|-------|--------|------|
| Train | `strong_tsv_train` (80 %) | 80 % of TSV rows with a found audio file |
| Val | `strong_tsv_train` (20 %) | held-out; `val_fraction=0.2`, `val_seed=42` |
| Test | `strong_tsv_test` | `eval_segments` (fixed, official) |

---

## 19. Loss Function and Metrics

### 19a. SEDMultiLabelLoss

Binary cross-entropy with optional per-class positive weighting:

```
L = BCEWithLogitsLoss(logits, targets, pos_weight)    reduction = 'mean'

  = − (1 / B·C) Σ_{b,c} [ pw_c · y_{bc} · log σ(z_{bc})
                          + (1 − y_{bc}) · log(1 − σ(z_{bc})) ]
```

`pos_weight` is a registered buffer (follows the module to the correct device automatically). When `None`, all classes are weighted equally.

### 19b. SEDMultiLabelMetrics

Accumulated with `torchmetrics`. Metrics are split by whether they require a threshold:

| Metric | Type | Torchmetrics class | Notes |
|--------|------|--------------------|-------|
| `map_macro` | threshold-free | `MultilabelAveragePrecision` | **Primary metric** — checkpoint monitor |
| `auroc_macro` | threshold-free | `MultilabelAUROC` | Area under ROC curve |
| `f1_macro` | σ > thresh | `MultilabelF1Score(average='macro')` | |
| `f1_micro` | σ > thresh | `MultilabelF1Score(average='micro')` | |
| `precision_macro` | σ > thresh | `MultilabelPrecision(average='macro')` | |
| `recall_macro` | σ > thresh | `MultilabelRecall(average='macro')` | |
| `accuracy` | σ > thresh | `MultilabelAccuracy(average='micro')` | |
| `confusion_matrix` | σ > thresh | `MultilabelConfusionMatrix` | test only, saved as `.npy` |

Default threshold: 0.5 (configurable via `training.inference_threshold`).

mAP is threshold-free and is the standard AudioSet evaluation metric; it is robust to class imbalance and does not require calibrating a decision threshold, making it the primary criterion for checkpointing.

### 19c. Confusion Matrix

At test time, the `(456, 2, 2)` int64 confusion matrix is saved to `{run_dir}/confusion_matrix.npy`. It is not logged to TensorBoard (too large). Load with:

```python
import numpy as np
cm = np.load('runs/at2sed/at2sed_default/confusion_matrix.npy')
# cm[c] = [[TN, FP], [FN, TP]] for class c
```

---

## 20. Training Pipeline — AT2SED

### 20a. Optimizer

```
Adam(params_with_requires_grad, lr = lr_0)
```

Only parameters with `requires_grad = True` are passed to the optimizer, respecting the per-group freeze configuration.

### 20b. Learning Rate Scheduler (CosineAnnealingLR)

```
lr(t) = lr_min + 0.5 · (lr_0 − lr_min) · (1 + cos(π · t / T_max))

T_max  = epochs   (total training epochs for the run)
lr_0   = 1e-3     (CLI --lr; configurable per run)
lr_min = 1e-5     (training.lr_min in YAML; default 1e-5)
```

The LR decays smoothly from `1e-3` to `1e-5` over the full training budget. The scheduler is updated once per epoch (`interval = 'epoch'`).

### 20c. Gradient Management

Identical to the NAS pipeline (§12c):

**Gradient clipping** (L2 norm, per step):

```
‖g‖₂ ← min(‖g‖₂, 1.0)
```

**Gradient accumulation** (to target an effective batch size of 16):

```
accum_steps    = max(1, ⌊16 / batch_size⌋)
effective_batch = batch_size × accum_steps ≈ 16
```

Example: `batch_size = 32 → accum_steps = 1 → effective_batch = 32` (no accumulation).

### 20d. Checkpointing

```
filename   : {epoch:02d}_{val_map_macro:.4f}
monitor    : val_map_macro   (higher is better, mode = 'max')
save_top_k : 1               (only best checkpoint retained)
save_last  : True            → last.ckpt (deleted on successful completion)
```

### 20e. Resume Flow (Interrupted Experiment)

```
Priority: last.ckpt  →  any *.ckpt (alphabetical, last)  →  fresh start
```

`state.json` records:

```json
{
  "status":       "running | completed | pending",
  "started_at":   "2025-01-01T12:00:00",
  "best_ckpt":    "runs/at2sed/at2sed_default/checkpoints/05_0.3142.ckpt",
  "best_val_map": 0.3142,
  "completed_at": "2025-01-01T18:30:00"
}
```

Updated atomically (write `.tmp` → rename). The per-epoch metrics JSON is pre-loaded from `metrics.json` when resuming, ensuring progressive accumulation continues correctly.

### 20f. Smoke Test

```bash
# 5 batches per epoch, 1 epoch — status remains 'running' → auto-resume on next invocation
python main_sed_only.py --epochs 1 --limit-batches 5
```

### 20g. Logging

**TensorBoard** (port 6007, launched automatically):

| Key | Frequency | Description |
|-----|-----------|-------------|
| `train/loss` | step + epoch | BCE loss |
| `val/loss` | epoch | Validation BCE loss |
| `val/map_macro` | epoch | mAP macro (primary checkpoint metric) |
| `val/auroc_macro` | epoch | AUROC macro |
| `val/f1_macro`, `val/f1_micro` | epoch | F1 macro / micro |
| `val/precision_macro`, `val/recall_macro` | epoch | Precision / Recall macro |
| `val/accuracy` | epoch | Accuracy micro |
| `test/{metric}` | once (post-test) | Same metric set as validation |
| `lr/cosine` | epoch | Current learning rate |

**JSON metrics log** (per-run, progressive):

```json
[
  {"epoch": 0,    "val/map_macro": 0.12, "val/auroc_macro": 0.71, ...},
  {"epoch": 1,    "val/map_macro": 0.18, ...},
  ...
  {"epoch": "test", "test/map_macro": 0.31, "test/auroc_macro": 0.84, ...}
]
```

---

## 21. AT2SED Experiment Configuration

### 21a. YAML Hierarchy

```
configs/
├── datasets/
│   ├── AudioSetStrong.yaml              # data paths, pos_weight_max, augmentation
│   └── audioset_strong_label_map.csv   # 456-class label map (5 columns)
└── experiments_at2sed/
    └── at2sed_default.yaml             # model.n_cls, semantic groups, inference_threshold
```

### 21b. Config Merge Priority

```
CLI args  >  experiments_at2sed/{exp}.yaml  >  AudioSetStrong.yaml defaults
```

Sections merged: `data`, `augmentation`, `training`, `semantic`. Keys from the experiment YAML override matching keys from the dataset YAML. CLI arguments (`--lr`, `--epochs`, `--batch-size`, `--window-s`, `--hop-s`, `--num-workers`) override both YAML files.

### 21c. Key Configuration Parameters

| Parameter | Default value | Source |
|-----------|---------------|--------|
| `model.n_cls` | 456 | experiment YAML |
| `semantic.checkpoint` | `checkpoint_closeto_.44.pt` | experiment YAML |
| `semantic.early_group` | `{init: pretrained, frozen: false}` | experiment YAML |
| `semantic.late_group` | `{init: pretrained, frozen: false}` | experiment YAML |
| `semantic.head_group` | `{init: pretrained}` | experiment YAML |
| `training.inference_threshold` | 0.5 | experiment YAML |
| `training.lr_min` | 1e-5 | `configure_optimizers` default |
| `training.pos_weight_max` | 20.0 | AudioSetStrong.yaml |
| `data.val_fraction` | 0.2 | AudioSetStrong.yaml |
| `data.val_seed` | 42 | AudioSetStrong.yaml |
| `--window-s` | 0.5 | CLI default |
| `--hop-s` | 0.5 | CLI default |
| `--lr` | 1e-3 | CLI default |
| `--epochs` | 30 | CLI default |
| `--batch-size` | 32 | CLI default |

### 21d. Training Invocation

```bash
# Default run (window=0.5s, hop=0.5s, 30 epochs, lr=1e-3→1e-5)
python main_sed_only.py --exp at2sed_default --epochs 30

# Longer window sweep
python main_sed_only.py --window-s 2.0 --hop-s 2.0 --epochs 30

# Lower LR, more epochs
python main_sed_only.py --lr 5e-4 --epochs 50

# Smoke test (2 batches, 1 epoch — status remains 'running')
python main_sed_only.py --epochs 1 --limit-batches 2

# Print experiment status
python main_sed_only.py --list

# Force re-run of a completed experiment
python main_sed_only.py --exp at2sed_default --force
```

### 21e. Output Directory Layout

```text
runs/at2sed/{exp_name}/
├── state.json                                  # pending | running | completed
├── metrics.json                                # per-epoch + test metrics (progressive)
├── confusion_matrix.npy                        # (456, 2, 2) int64, test only
├── checkpoints/
│   ├── {epoch:02d}_{val_map_macro:.4f}.ckpt   # best checkpoint (retained)
│   └── last.ckpt                              # latest epoch (deleted on completion)
└── tb_logs/
    └── version_0/
        └── events.out.tfevents.*              # TensorBoard event files
```
