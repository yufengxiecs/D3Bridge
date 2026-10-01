import torch

try:
    # Use scipy for DCT-II and IDCT when available
    from scipy.fftpack import dct, idct
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def dct2_torch(x):
    """2D DCT (type II, orthonormal) of a torch tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: DCT coefficients, same shape as the input.
    """
    if _HAS_SCIPY:
        device = x.device
        x_np = x.detach().cpu().numpy()
        X_np = _dct2_numpy(x_np)
        return torch.from_numpy(X_np).to(device)

    # Fallback: FFT-based orthonormal DCT-II applied separably along the
    # last two axes. Prefer scipy for exactness.
    out = _dct1d_torch(x, dim=-1)
    out = _dct1d_torch(out, dim=-2)
    return out


def idct2_torch(X):
    """2D inverse DCT of a torch tensor.

    Args:
        X (torch.Tensor): DCT coefficient tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Reconstructed image-domain tensor, same shape.
    """
    if _HAS_SCIPY:
        device = X.device
        X_np = X.detach().cpu().numpy()
        x_np = _idct2_numpy(X_np)
        return torch.from_numpy(x_np).to(device)

    # Fallback approximate inverse (not fully general; prefer scipy)
    out = _idct1d_torch(X, dim=-1)
    out = _idct1d_torch(out, dim=-2)
    return out


def _dct2_numpy(x_np):
    # x_np: (B, C, H, W) numpy array
    return dct(dct(x_np, axis=-1, norm='ortho'), axis=-2, norm='ortho')


def _idct2_numpy(X_np):
    return idct(idct(X_np, axis=-1, norm='ortho'), axis=-2, norm='ortho')


def _dct1d_torch(a, dim):
    # 1D orthonormal DCT-II along `dim` via the FFT trick
    N = a.size(dim)
    rev = a.flip(dims=[dim])
    a2 = torch.cat([a, rev], dim=dim)
    A = torch.fft.fft(a2, dim=dim)
    k = torch.arange(N, device=a.device, dtype=a.dtype)
    exp_factor = torch.exp(-1j * torch.pi * k / (2.0 * N))
    A_slice = A.narrow(dim, 0, N)
    return (A_slice * exp_factor).real * 2.0


def _idct1d_torch(A, dim):
    # 1D approximate inverse along `dim` (fallback path only)
    N = A.size(dim)
    k = torch.arange(N, device=A.device, dtype=A.dtype)
    exp_factor = torch.exp(1j * torch.pi * k / (2.0 * N))
    A_complex = A * exp_factor
    A2 = torch.cat([A_complex, torch.zeros_like(A_complex)], dim=dim)
    a2 = torch.fft.ifft(A2, dim=dim)
    return a2.narrow(dim, 0, N).real / 2.0
