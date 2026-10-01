import torch
import torch.nn as nn


class AdaptiveFusionNetwork(nn.Module):
    """Adaptive fusion network F_omega (Eq. 8-9).

    Integrates the spatial and frequency expert outputs into the final
    synthesized image through adaptive weights:

        x_fused = pi_s * x_s + pi_f * x_f
        pi(j) = softmax_j(H_j([x_j, |x_s - x_f|]))

    where H_j is a learnable convolutional transformation
    (Conv3x3 -> ReLU -> Conv3x3) applied to the concatenation of the
    j-th expert output and the absolute difference map.
    """

    def __init__(self, in_channels=1, hidden=32):
        super().__init__()
        self.in_channels = in_channels

        # H_s and H_f: learnable convolutional transformations taking
        # [x_j, |x_s - x_f|] (2 * in_channels) and producing a logit map
        self.H_s = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        )
        self.H_f = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_s, x_f):
        """Fuse the spatial and frequency expert outputs.

        Args:
            x_s (torch.Tensor): Spatial expert output, (B, C, H, W).
            x_f (torch.Tensor): Frequency expert output (after IDCT),
                (B, C, H, W).

        Returns:
            torch.Tensor: Fused image, (B, C, H, W).
        """
        diff = torch.abs(x_s - x_f)
        logits = torch.cat([self.H_s(torch.cat([x_s, diff], dim=1)),
                            self.H_f(torch.cat([x_f, diff], dim=1))], dim=1)
        weights = torch.softmax(logits, dim=1)

        x_fused = weights[:, 0:1, ...] * x_s + weights[:, 1:2, ...] * x_f
        return x_fused
