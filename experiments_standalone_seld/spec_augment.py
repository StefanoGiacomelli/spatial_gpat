import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Frequency and time masking applied to spectrogram features.

    Operates on (B, C_in, T, F): a single mask pattern is generated from the
    first channel and broadcast across all C_in channels to preserve cross-channel
    spatial coherence (important for FOA intensity-vector features).

    Masking is only applied when the module is in training mode (self.training).
    """

    def __init__(
        self,
        n_freq_masks:    int = 2,
        freq_mask_param: int = 20,
        n_time_masks:    int = 2,
        time_mask_param: int = 30,
    ):
        super().__init__()
        self.n_freq_masks    = n_freq_masks
        self.freq_mask_param = freq_mask_param
        self.n_time_masks    = n_time_masks
        self.time_mask_param = time_mask_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T, F)

        Returns:
            x with random frequency and time bands zeroed (training only).
        """
        if not self.training:
            return x

        B, C, T, F = x.shape
        mask = torch.ones(B, 1, T, F, device=x.device, dtype=x.dtype)

        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (B,), device=x.device)
            f0 = torch.randint(0, F, (B,), device=x.device)
            for b in range(B):
                f_end = min(f0[b].item() + f[b].item(), F)
                mask[b, 0, :, f0[b].item():f_end] = 0.0

        for _ in range(self.n_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (B,), device=x.device)
            t0 = torch.randint(0, T, (B,), device=x.device)
            for b in range(B):
                t_end = min(t0[b].item() + t[b].item(), T)
                mask[b, 0, t0[b].item():t_end, :] = 0.0

        return x * mask
