"""
tests/test_spatial_modules.py
==============================
Profiling and architecture summary for every public class in ``spatial_modules``.

For each module and each batch size in {1, 2, 4, 8, 16} the script runs N_RUNS
forward passes and computes a full timing-statistics profile (mean, std, median,
skewness, kurtosis, p90).  GFLOPs are computed once per module via torchinfo.
A compact ╔══╗ box is printed per module; torchinfo.summary() follows inline.
All results are written to a timestamped CSV under ``results/``.

Usage
-----
    python tests/test_spatial_modules.py

Columns in CSV
--------------
module, total_params, trainable_params, memory_mb, gflops,
device, batch_size,
mean_ms, std_ms, median_ms, skewness, kurtosis, p90_ms

References
----------
Adavanne et al., "SELDnet", IEEE JSTSP 2019.
Cao et al., "EIN-v2", ICASSP 2021.
Diaz-Guerra et al., "PI-RNN", Forum Acusticum 2023.
Xue et al., "ResNet-Conformer + MSCA", WCSP 2023.
Vo & Han, "ResNet-Conformer DCASE", DCASE 2024.
He et al., "SoundDet", ICML 2021.
He & Markham, "SoundDoA", Interspeech 2022.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

import csv
import datetime
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml
from torchinfo import summary

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from spatial_modules import (
    CrossStitchBridge,
    SpectralProcessor, ConvBlock, BiGRU,
    MultiScaleChannelAttention, ResNetBlock, ConformerBlock, AttentiveStatisticsPooling,
    DoubleConv, CrossStitching, PositionalEncoder, TrackTransformer,
    MHA_MGU,
    LearnableGaborFilterBank, GaborFrontEnd, FormantEnhancementModule,
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

N_RUNS      = 20      # 20 runs → reliable median / skewness / kurtosis
N_WARMUP    = 2
BATCH_SIZES = [1, 2, 4, 8, 16]
BOX_WIDTH   = 66
CFG_DIR     = _ROOT / 'configs' / 'test_spatial_modules'
RESULTS_DIR = _ROOT / 'results'

# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════

def _load(name: str) -> dict:
    """Load a YAML config from CFG_DIR."""
    with open(CFG_DIR / f'{name}.yaml') as f:
        return yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════════
# Device detection
# ══════════════════════════════════════════════════════════════════════════════

_devices: List[str] = ['cpu']
if torch.cuda.is_available():
    _devices.append('cuda')
if torch.backends.mps.is_available():
    _devices.append('mps')

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_device(
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    device: str,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Move tensor or tuple of tensors to device."""
    if isinstance(inputs, tuple):
        return tuple(t.to(device) for t in inputs)
    return inputs.to(device)


def _call(
    model: torch.nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> Any:
    """Call model with single tensor or unpacked tuple."""
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _profile(
    model: torch.nn.Module,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    device: str,
    n_runs: int = N_RUNS,
    warmup: int = N_WARMUP,
) -> Dict[str, float]:
    """Return timing statistics dict over n_runs forward passes.

    Keys: mean_ms, std_ms, median_ms, skewness, kurtosis, p90_ms
    """
    with torch.no_grad():
        for _ in range(warmup):
            _call(model, inputs)
        times: List[float] = []
        for _ in range(n_runs):
            if device == 'cuda':
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _call(model, inputs)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                _call(model, inputs)
                times.append((time.perf_counter() - t0) * 1000.0)

    n  = len(times)
    mu = sum(times) / n
    var = sum((x - mu) ** 2 for x in times) / n
    std = math.sqrt(var)

    s_sorted = sorted(times)
    median   = (s_sorted[n // 2] if n % 2 == 1
                else (s_sorted[n // 2 - 1] + s_sorted[n // 2]) / 2.0)
    p90      = s_sorted[min(int(0.9 * n), n - 1)]

    denom3 = std ** 3 + 1e-12
    denom4 = std ** 4 + 1e-12
    skew   = (sum((x - mu) ** 3 for x in times) / n) / denom3
    kurt   = (sum((x - mu) ** 4 for x in times) / n) / denom4 - 3.0

    return dict(mean_ms=mu, std_ms=std, median_ms=median,
                skewness=skew, kurtosis=kurt, p90_ms=p90)


# ══════════════════════════════════════════════════════════════════════════════
# Box printer
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ms(mean: float, std: float) -> str:
    return f'{mean:.2f}+/-{std:.2f}ms'


def _box_line(text: str, width: int = BOX_WIDTH) -> str:
    return f'║  {text:<{width - 4}}║'


def _print_box(
    name: str,
    total_p: int,
    train_p: int,
    mem_mb: float,
    gflops: float,
    in_desc: str,
    out_desc: str,
    device_results: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    """Print a compact ╔══╗ profiling box for one module."""
    W = BOX_WIDTH
    sep   = '═' * W
    hline = f'╠{sep}╣'

    print(f'╔{sep}╗')

    # Header: name + params
    param_str = f'params: {total_p:,}'
    hdr = f'{name:<{W - 4 - len(param_str)}}{param_str}'
    print(_box_line(hdr))

    # Trainable + memory
    tr_str  = f'trainable: {train_p:,}'
    mem_str = f'memory: {mem_mb:.3f} MB'
    info = f'{tr_str:<{W - 4 - len(mem_str)}}{mem_str}'
    print(_box_line(info))

    # GFLOPs
    gf_str = f'GFLOPs (B=1): {gflops:.4f}'
    print(_box_line(gf_str))

    print(hline)

    # IO shapes
    print(_box_line(f'in  {in_desc}  ->  out  {out_desc}'))

    # Per-device batch scaling
    for dev, scaling in device_results.items():
        print(hline)
        print(_box_line(f'Batch scaling  [{dev}]  {N_RUNS} runs each'))

        entries = [f'B={B}  {_fmt_ms(s["mean_ms"], s["std_ms"])}'
                   for B, s in scaling.items()]

        line = ''
        for entry in entries:
            token = f'  {entry}' if line else entry
            if len(line) + len(token) <= W - 4:
                line += token
            else:
                print(_box_line(line))
                line = entry
        if line:
            print(_box_line(line))

    print(f'╚{sep}╝')


# ══════════════════════════════════════════════════════════════════════════════
# Module registry
# ══════════════════════════════════════════════════════════════════════════════

def _build_registry(cfgs: dict) -> List[dict]:
    """Build the list of ModuleSpec dicts from loaded configs."""
    ada  = cfgs['adapters']
    sel  = cfgs['seldnet']
    con  = cfgs['conformer']
    ein  = cfgs['einv2']
    pi   = cfgs['pirnn']
    raw  = cfgs['raw_waveform']

    # Shorthand for tensor factory
    def t(*shape):
        return lambda B, _dev=None: torch.randn(B, *shape)

    def tt(*shapes):
        """Tuple-of-tensors factory."""
        return lambda B, _dev=None: tuple(torch.randn(B, *s) for s in shapes)

    ih  = sel['input_handler']
    cb  = sel['conv_blocks']
    gru = sel['bigru']
    cfa = con['multi_scale_channel_attention']
    rb  = con['resnet_block']
    cf  = con['conformer']
    asp = con['attentive_statistics_pooling']
    dc  = ein['double_conv']
    cs  = ein['cross_stitch']
    tt_ = ein['track_transformer']
    pe  = ein['positional_encoding']
    mha = pi['mhamgu']
    gfb = raw['gabor_filter_bank']
    gfe = raw['gabor_front_end']
    fem = raw['formant_enhancement']
    adv = ada['cross_stitch_bridge']

    # Derived dims
    T_sp = ada['test']['T_spatial']
    F_sp = ada['test']['F_spatial']
    T_se = ada['test']['T_semantic']
    F_se = ada['test']['F_semantic']

    T_seld = 171   # floor((44100 - 512) / 256) + 1
    Fg     = 256   # n_fft // 2

    T_ein  = ein['test']['T_frames']   # 159
    F_ein  = ein['test']['mel_bands']  # 256
    d_ein  = ein['test']['d_model']    # 512

    T_pi   = pi['test']['T']           # 40
    M_pi   = pi['test']['M_X']         # 10
    d_pi   = mha['d_model']            # 128

    t_smp  = raw['test']['t_samples']  # 24000
    n_sp   = raw['test']['n_spatial']  # 7
    T_gab  = (t_smp - gfb['kernel_length']) // gfb['stride'] + 1   # ~39

    n_mics  = gfe['n_mics']            # 4
    nf_gab  = gfb['n_filters']         # 128

    T_con  = con['test']['time_frames']  # 250
    F_con  = con['test']['freq_bins']    # 64
    d_con  = con['test']['d_model']      # 256
    in_ch  = con['test']['in_channels']  # 32

    # in_size for BiGRU: out_channels × (256 / 8 / 8 / 2) = 64 × 2 = 128
    bigru_in = cb['out_channels'] * (Fg // 8 // 8 // 2)

    registry = [
        # ── Adapters ────────────────────────────────────────────────────────
        {
            'name':     'CrossStitchBridge',
            'cls':      CrossStitchBridge,
            'kwargs':   dict(c_spatial=adv['c_spatial'],
                             c_semantic=adv['c_semantic'],
                             c_bridge=adv['c_bridge']),
            'input_fn': tt((adv['c_spatial'], T_sp, F_sp),
                           (adv['c_semantic'], T_se, F_se)),
            'in_desc':  f'(B,{adv["c_spatial"]},{T_sp},{F_sp})+(B,{adv["c_semantic"]},{T_se},{F_se})',
            'out_desc': 'same shapes',
        },
        # ── SELDnet ─────────────────────────────────────────────────────────
        {
            'name':     'SpectralProcessor',
            'cls':      SpectralProcessor,
            'kwargs':   dict(n_fft=ih['n_fft'], hop_length=ih['hop_length']),
            'input_fn': t(n_mics, 44100),
            'in_desc':  f'(B,{n_mics},44100)',
            'out_desc': f'(B,{2*n_mics},{T_seld},{Fg})',
        },
        {
            'name':     'ConvBlock',
            'cls':      ConvBlock,
            'kwargs':   dict(in_channels=2 * n_mics,
                             out_channels=cb['out_channels'],
                             pool_size=cb['pool_sizes'][0]),
            'input_fn': t(2 * n_mics, T_seld, Fg),
            'in_desc':  f'(B,{2*n_mics},{T_seld},{Fg})',
            'out_desc': f'(B,{cb["out_channels"]},{T_seld},{Fg // cb["pool_sizes"][0]})',
        },
        {
            'name':     'BiGRU',
            'cls':      BiGRU,
            'kwargs':   dict(in_size=bigru_in,
                             hidden_size=gru['hidden_size'],
                             num_layers=gru['num_layers']),
            'input_fn': t(T_seld, bigru_in),
            'in_desc':  f'(B,{T_seld},{bigru_in})',
            'out_desc': f'(B,{T_seld},{gru["hidden_size"]})',
        },
        # ── Conformer-family ────────────────────────────────────────────────
        {
            'name':     'MultiScaleChannelAttention',
            'cls':      MultiScaleChannelAttention,
            'kwargs':   dict(n_channels=in_ch,
                             reduction_ratio=cfa['reduction_ratio']),
            'input_fn': t(in_ch, T_con, F_con),
            'in_desc':  f'(B,{in_ch},{T_con},{F_con})',
            'out_desc': 'same',
        },
        {
            'name':     'ResNetBlock',
            'cls':      ResNetBlock,
            'kwargs':   dict(in_channels=rb['in_channels'],
                             out_channels=rb['out_channels'],
                             stride=tuple(rb['stride']),
                             use_msca=rb['use_msca'],
                             msca_reduction=rb['msca_reduction']),
            'input_fn': t(rb['in_channels'], T_con, F_con),
            'in_desc':  f'(B,{rb["in_channels"]},{T_con},{F_con})',
            'out_desc': f'(B,{rb["out_channels"]},{T_con},{F_con})',
        },
        {
            'name':     'ConformerBlock',
            'cls':      ConformerBlock,
            'kwargs':   dict(d_model=cf['d_model'],
                             n_heads=cf['n_heads'],
                             ffn_expansion=cf['ffn_expansion'],
                             conv_kernel=cf['conv_kernel'],
                             dropout=cf['dropout'],
                             activation=cf['activation']),
            'input_fn': t(T_con, d_con),
            'in_desc':  f'(B,{T_con},{d_con})',
            'out_desc': 'same',
        },
        {
            'name':     'AttentiveStatisticsPooling',
            'cls':      AttentiveStatisticsPooling,
            'kwargs':   dict(d_input=asp['d_input']),
            'input_fn': t(T_con, d_con),
            'in_desc':  f'(B,{T_con},{d_con})',
            'out_desc': f'(B,{2 * asp["d_input"]})',
        },
        # ── EIN-v2 ──────────────────────────────────────────────────────────
        {
            'name':     'DoubleConv',
            'cls':      DoubleConv,
            'kwargs':   dict(in_channels=dc['in_channels'],
                             out_channels=dc['out_channels'],
                             kernel_size=tuple(dc['kernel_size']),
                             bias=dc['bias']),
            'input_fn': t(dc['in_channels'], T_ein, F_ein),
            'in_desc':  f'(B,{dc["in_channels"]},{T_ein},{F_ein})',
            'out_desc': f'(B,{dc["out_channels"]},{T_ein},{F_ein})',
        },
        {
            'name':     'CrossStitching',
            'cls':      CrossStitching,
            'kwargs':   dict(n_channels=cs['n_channels']),
            'input_fn': tt((cs['n_channels'], T_ein, F_ein),
                           (cs['n_channels'], T_ein, F_ein)),
            'in_desc':  f'2x(B,{cs["n_channels"]},{T_ein},{F_ein})',
            'out_desc': 'same shapes',
        },
        {
            'name':     'PositionalEncoder',
            'cls':      PositionalEncoder,
            'kwargs':   dict(d_model=pe['d_model'],
                             max_len=pe['max_len'],
                             pe_type=pe['pe_type'],
                             dropout=pe['dropout']),
            'input_fn': t(d_ein, T_ein),   # PositionalEncoder expects (B, C, T)
            'in_desc':  f'(B,{d_ein},{T_ein})',
            'out_desc': 'same',
        },
        {
            'name':     'TrackTransformer',
            'cls':      TrackTransformer,
            'kwargs':   dict(d_model=tt_['d_model'],
                             n_heads=tt_['n_heads'],
                             n_tracks=tt_['n_tracks'],
                             num_layers=tt_['num_layers'],
                             dim_feedforward=tt_['dim_feedforward'],
                             dropout=tt_['dropout'],
                             pe_enable=tt_['pe_enable'],
                             pe_type=tt_['pe_type'],
                             pe_dropout=tt_['pe_dropout']),
            'input_fn': t(T_ein, d_ein),
            'in_desc':  f'(B,{T_ein},{d_ein})',
            'out_desc': f'(B,{T_ein},{tt_["n_tracks"]},{d_ein})',
        },
        # ── PI-RNN ──────────────────────────────────────────────────────────
        {
            'name':     'MHA_MGU',
            'cls':      MHA_MGU,
            'kwargs':   dict(d_model=mha['d_model'],
                             n_heads=mha['n_heads'],
                             n_tracks=mha['n_tracks'],
                             d_input=mha.get('d_input')),
            'input_fn': t(T_pi, M_pi, d_pi),
            'in_desc':  f'(B,{T_pi},{M_pi},{d_pi})',
            'out_desc': f'(B,{T_pi},{mha["n_tracks"]},{d_pi})',
        },
        # ── Raw-waveform ─────────────────────────────────────────────────────
        {
            'name':     'LearnableGaborFilterBank',
            'cls':      LearnableGaborFilterBank,
            'kwargs':   dict(n_filters=gfb['n_filters'],
                             kernel_length=gfb['kernel_length'],
                             stride=gfb['stride'],
                             sr=gfb['sr'],
                             fmin=gfb['fmin'],
                             fmax=gfb['fmax']),
            'input_fn': t(t_smp),
            'in_desc':  f'(B,{t_smp})',
            'out_desc': f'(B,{2*nf_gab},{T_gab})',
        },
        {
            'name':     'GaborFrontEnd',
            'cls':      GaborFrontEnd,
            'kwargs':   dict(n_filters=gfe['n_filters'],
                             kernel_length=gfe['kernel_length'],
                             stride=gfe['stride'],
                             n_mics=gfe['n_mics'],
                             format=gfe['format'],
                             sr=gfe['sr'],
                             fmin=gfe['fmin'],
                             fmax=gfe['fmax']),
            'input_fn': t(n_mics, t_smp),
            'in_desc':  f'(B,{n_mics},{t_smp})',
            'out_desc': f'(B,{n_sp},{T_gab},{nf_gab})',
        },
        {
            'name':     'FormantEnhancementModule',
            'cls':      FormantEnhancementModule,
            'kwargs':   dict(n_channels=fem['n_channels'],
                             kernel_size=tuple(fem['kernel_size'])),
            'input_fn': t(fem['n_channels'], T_gab, nf_gab),
            'in_desc':  f'(B,{fem["n_channels"]},{T_gab},{nf_gab})',
            'out_desc': 'same',
        },
    ]
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Profile all spatial_modules classes and write results to CSV."""
    print(f'Devices: {_devices}')
    print(f'Batch sizes: {BATCH_SIZES}  |  Runs per size: {N_RUNS}  |  Warmup: {N_WARMUP}\n')

    cfgs     = {n: _load(n) for n in
                ['adapters', 'seldnet', 'conformer', 'einv2', 'pirnn', 'raw_waveform']}
    registry = _build_registry(cfgs)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    rows: List[dict] = []

    for spec in registry:
        model   = spec['cls'](**spec['kwargs']).eval()
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mem_mb  = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / 1024 ** 2

        # ── GFLOPs via torchinfo (B=1, cpu, silent) ───────────────────────
        inp_cpu  = _to_device(spec['input_fn'](1), 'cpu')
        inp_list = list(inp_cpu) if isinstance(inp_cpu, tuple) else [inp_cpu]
        _stats   = summary(
            model.cpu(), input_data=inp_list, device='cpu',
            verbose=0, col_names=('input_size', 'output_size', 'num_params'),
        )
        gflops = _stats.total_mult_adds * 2 / 1e9

        # ── Timing per device / batch size ────────────────────────────────
        device_results: Dict[str, Dict[int, Dict[str, float]]] = {}
        for dev in _devices:
            model_d = model.to(dev)
            scaling: Dict[int, Dict[str, float]] = {}
            for B in BATCH_SIZES:
                inp    = _to_device(spec['input_fn'](B), dev)
                stats  = _profile(model_d, inp, dev)
                scaling[B] = stats
                rows.append({
                    'module':           spec['name'],
                    'total_params':     total_p,
                    'trainable_params': train_p,
                    'memory_mb':        f'{mem_mb:.4f}',
                    'gflops':           f'{gflops:.4f}',
                    'device':           dev,
                    'batch_size':       B,
                    'mean_ms':          f'{stats["mean_ms"]:.4f}',
                    'std_ms':           f'{stats["std_ms"]:.4f}',
                    'median_ms':        f'{stats["median_ms"]:.4f}',
                    'skewness':         f'{stats["skewness"]:.4f}',
                    'kurtosis':         f'{stats["kurtosis"]:.4f}',
                    'p90_ms':           f'{stats["p90_ms"]:.4f}',
                })
            device_results[dev] = scaling
            model.to('cpu')

        # ── Print box ─────────────────────────────────────────────────────
        _print_box(
            spec['name'], total_p, train_p, mem_mb, gflops,
            spec['in_desc'], spec['out_desc'], device_results,
        )

        # ── Inline torchinfo summary (verbose, cpu, B=1) ──────────────────
        summary(
            model.cpu(), input_data=inp_list, device='cpu',
            verbose=1, col_names=('input_size', 'output_size', 'num_params'),
        )
        print()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f'spatial_modules_profile_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results -> {csv_path}')


if __name__ == '__main__':
    main()
