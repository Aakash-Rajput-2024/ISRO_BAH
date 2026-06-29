"""Method 3 (ResU-Net) -- smoke tests.

Marked `ml`; skipped if torch is absent. Kept small and CPU-runnable: they prove
the architecture is wired correctly (forward shape) and can actually learn
(overfits a tiny batch), without a full training run or a GPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from methods.deep_resunet.model import ResUNet

pytestmark = pytest.mark.ml


def test_forward_shape():
    """Frame in -> coarse wavefront map out, with the expected shape."""
    model = ResUNet(target=32, base=8)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)
    assert torch.isfinite(y).all()
    assert float(y.detach().abs().max()) <= 1.0 + 1e-5  # tanh head


def test_overfits_tiny_batch():
    """A few Adam steps must drive the loss down -- the net can learn."""
    torch.manual_seed(0)
    model = ResUNet(target=16, base=8)
    x = torch.randn(4, 1, 64, 64)
    y = torch.tanh(torch.randn(4, 1, 16, 16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.6 * losses[0]
