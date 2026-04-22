"""
spatial_modules — Spatial SELD Feature Extraction Toolkit
==========================================================
Public API for all spatial processing modules used in the NAS SELD pipeline.
Includes front-ends, convolutional blocks, recurrent layers, attention mechanisms,
cross-stitch adapters, and raw-waveform operators.

Importing from this package enables clean top-level access:

    from spatial_modules import ConvBlock, BiGRU, TrackTransformer

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

# ── Adapters ──────────────────────────────────────────────────────────────────
from .adapters import CrossStitchBridge

# ── SELDnet blocks ────────────────────────────────────────────────────────────
from .seldnet_modules import SpectralProcessor, ConvBlock, BiGRU

# ── Conformer-family blocks ───────────────────────────────────────────────────
from .conformer_modules import (
    MultiScaleChannelAttention,
    ResNetBlock,
    ConformerBlock,
    AttentiveStatisticsPooling,
)

# ── EIN-v2 blocks ─────────────────────────────────────────────────────────────
from .einv2_modules import (
    DoubleConv,
    CrossStitching,
    PositionalEncoder,
    TrackTransformer,
)

# ── PI-RNN block ──────────────────────────────────────────────────────────────
from .pirnn_modules import MHA_MGU

# ── Raw-waveform front-ends ───────────────────────────────────────────────────
from .raw_waveform_modules import (
    LearnableGaborFilterBank,
    GaborFrontEnd,
    FormantEnhancementModule,
)

# ── Neural-SRP encoder + spatial model ───────────────────────────────────────
from .neural_srp_modules import NeuralSRPEncoder, SpatialSRPModel

# ── HNet (Hungarian network for soft source assignment) ───────────────────────
from .hnet_modules import HNetGRU

__all__ = [
    # Adapters
    "CrossStitchBridge",
    # SELDnet
    "SpectralProcessor", "ConvBlock", "BiGRU",
    # Conformer-family
    "MultiScaleChannelAttention", "ResNetBlock",
    "ConformerBlock", "AttentiveStatisticsPooling",
    # EIN-v2
    "DoubleConv", "CrossStitching", "PositionalEncoder", "TrackTransformer",
    # PI-RNN
    "MHA_MGU",
    # Raw-waveform
    "LearnableGaborFilterBank", "GaborFrontEnd",
    "FormantEnhancementModule",
    # Neural-SRP
    "NeuralSRPEncoder", "SpatialSRPModel",
    # HNet
    "HNetGRU",
]
