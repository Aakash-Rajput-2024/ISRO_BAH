"""Temporal (N, T, D, D) dataset -- structure, correlation, windowing, no-leakage.

Generates a tiny dataset on a small Config (npix=64) so it runs quickly, then
checks: shapes; that the sequences are genuinely temporally correlated (lag-1
autocorrelation exceeds a longer lag); the windower's counts/shapes; and that
the by-instance train/val/test split is disjoint and complete (no leakage).
"""
import numpy as np
import pytest

from methods.modal_zernike import Config
from data.make_sequence_dataset import generate
from methods.common.sequence_dataset import (
    instance_split, load_sequences, split_from_manifest,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    cfg = Config(n_lenslets=8, pix_per_lenslet=8)   # npix=64, fast
    out = str(tmp_path_factory.mktemp("seq") / "ds")
    generate(cfg, out, n=4, t=8, n_layers=1, target_w=16, n_subharmonics=1, seed=0)
    return cfg, load_sequences(out)


def test_shapes(seq):
    cfg, data = seq
    assert data.frames.shape == (4, 8, 64, 64)
    assert data.coeffs.shape == (4, 8, cfg.n_modes)
    assert data.slopes.shape[:2] == (4, 8)
    assert data.wavefronts.shape == (4, 8, 16, 16)
    assert data.frames.dtype == np.uint8            # 8-bit detector
    assert data.manifest["kind"] == "sequence"


def test_temporally_correlated(seq):
    """Lag-1 autocorrelation must exceed a longer lag -- real time structure."""
    _, data = seq
    c = np.asarray(data.coeffs, float)
    c = c - c.mean(axis=1, keepdims=True)
    v0 = (c * c).sum(axis=2).mean()

    def ac(lag):
        a, b = c[:, :c.shape[1] - lag], c[:, lag:]
        return (a * b).sum(axis=2).mean() / v0

    assert ac(1) > ac(3)
    assert ac(1) > 0.2          # adjacent frames are clearly correlated


def test_windowing(seq):
    cfg, data = seq
    K, horizon = 3, 0
    wv = data.windows(K=K, horizon=horizon, label="coeffs")
    x, y = wv[0]
    assert x.shape == (K, 64, 64)
    assert y.shape == (cfg.n_modes,)
    assert x.max() <= 1.0 + 1e-6          # uint8 frames normalised to [0,1]
    # N instances x (T - K + 1 - horizon) windows each.
    assert len(wv) == 4 * (8 - K + 1 - horizon)


def test_split_no_leakage(seq):
    _, data = seq
    sp = split_from_manifest(data)
    allidx = sp["train"] + sp["val"] + sp["test"]
    assert sorted(allidx) == list(range(4))         # complete
    assert len(set(allidx)) == 4                     # disjoint

    s = instance_split(10, seed=0)
    union = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert union == set(range(10))
    assert not (set(s["train"]) & set(s["test"]))
