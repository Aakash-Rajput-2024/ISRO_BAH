"""Figure builders for the visual validation report.

Each function takes already-computed data and returns a matplotlib Figure, so
the heavy numerics live in scripts/validate_report.py (and the test suite) and
are never duplicated here. matplotlib uses the Agg backend (no display).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def crosstalk_heatmap(M: np.ndarray, modes) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(np.abs(M), cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Empirical mode transfer |R·pipeline|\n(ideal = identity)")
    ax.set_xlabel("injected Noll j"); ax.set_ylabel("recovered Noll j")
    ax.set_xticks(range(0, len(modes), 3)); ax.set_xticklabels(modes[::3])
    ax.set_yticks(range(0, len(modes), 3)); ax.set_yticklabels(modes[::3])
    fig.colorbar(im, ax=ax, label="|coefficient|")
    fig.tight_layout()
    return fig


def linearity(amps, recovered, mode_j, break_amp=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(amps, recovered, "o-", label=f"recovered (j={mode_j})")
    ax.plot(amps, amps, "k--", lw=1, label="ideal y=x")
    if break_amp is not None:
        ax.axvline(break_amp, color="crimson", ls=":", label="dynamic-range limit")
    ax.set_xlabel("injected amplitude [rad]"); ax.set_ylabel("recovered [rad]")
    ax.set_title("Reconstruction linearity & dynamic range")
    ax.legend(); fig.tight_layout()
    return fig


def residual_vs_snr(snr, residual) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(snr, residual, "o-")
    ax.set_xlabel("frame SNR"); ax.set_ylabel("residual wavefront RMS [rad]")
    ax.set_title("Reconstruction error vs detector SNR")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    return fig


def fitting_error_vs_modes(n_modes, residual) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(n_modes, residual, "s-")
    ax.set_xlabel("number of Zernike modes"); ax.set_ylabel("residual RMS [rad]")
    ax.set_title("Modal fitting error vs #modes")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


def r0_vs_truth(r0_true, r0_mean, r0_std) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(np.array(r0_true) * 1e3, np.array(r0_mean) * 1e3,
                yerr=np.array(r0_std) * 1e3, fmt="o", capsize=4, label="estimated")
    lo = min(r0_true); hi = max(r0_true)
    ax.plot([lo * 1e3, hi * 1e3], [lo * 1e3, hi * 1e3], "k--", label="truth")
    ax.set_xlabel("true r0 [mm]"); ax.set_ylabel("estimated r0 [mm]")
    ax.set_title("r0 recovery (mean ± std)")
    ax.legend(); fig.tight_layout()
    return fig


def r0_convergence(n_frames, r0_est, r0_true) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(n_frames, np.array(r0_est) * 1e3, "o-")
    ax.axhline(r0_true * 1e3, color="k", ls="--", label="truth")
    ax.set_xlabel("frames in ensemble"); ax.set_ylabel("estimated r0 [mm]")
    ax.set_title("r0 convergence vs ensemble size")
    ax.legend(); fig.tight_layout()
    return fig


def r0_vs_jstart(j_start, r0_est, r0_true) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(j_start, np.array(r0_est) * 1e3, "o-")
    ax.axhline(r0_true * 1e3, color="k", ls="--", label="truth")
    ax.set_xlabel("j_start (first mode in variance fit)")
    ax.set_ylabel("estimated r0 [mm]")
    ax.set_title("r0 bias vs j_start\n(low j_start hit by tip/tilt deficit)")
    ax.legend(); fig.tight_layout()
    return fig


def tau0_vs_wind(wind, tau0_meas, tau0_theory) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(wind, np.array(tau0_meas) * 1e3, "o-", label="measured (1/e autocorr.)")
    ax.plot(wind, np.array(tau0_theory) * 1e3, "s--", label="0.314 r0/v")
    ax.set_xlabel("wind speed [m/s]"); ax.set_ylabel("tau0 [ms]")
    ax.set_title("Coherence time vs wind speed")
    ax.legend(); fig.tight_layout()
    return fig


def autocorr_curves(lags, curves, labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for c, lab in zip(curves, labels):
        ax.plot(np.array(lags[:len(c)]) * 1e3, c, label=lab)
    ax.axhline(np.exp(-1), color="k", ls=":", label="1/e")
    ax.set_xlabel("lag [ms]"); ax.set_ylabel("normalised autocorrelation")
    ax.set_title("Temporal decorrelation (frozen flow)")
    ax.legend(); fig.tight_layout()
    return fig


def structure_function(r, measured, theory) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(np.array(r) * 1e3, measured, "o-", ms=3, label="measured")
    ax.loglog(np.array(r) * 1e3, theory, "k--", label=r"6.88$(r/r_0)^{5/3}$")
    ax.set_xlabel("separation r [mm]"); ax.set_ylabel(r"$D_\phi(r)$ [rad$^2$]")
    ax.set_title("Phase structure function vs Kolmogorov\n(low-freq deficit at large r)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    return fig


def radial_psd(f, psd, ref_slope=-11.0 / 3.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    good = np.isfinite(psd) & (psd > 0)
    ax.loglog(f[good], psd[good], "o-", ms=3, label="radial PSD")
    fref = f[good][1:]
    ref = psd[good][1] * (fref / fref[0]) ** ref_slope
    ax.loglog(fref, ref, "k--", label=r"$f^{-11/3}$")
    ax.set_xlabel("spatial frequency [cyc/m]"); ax.set_ylabel("PSD")
    ax.set_title("Radial PSD vs Kolmogorov slope")
    ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    return fig


def variance_spectrum(js, measured, theory) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(len(js))
    ax.bar(x - 0.2, measured, width=0.4, label="measured")
    ax.bar(x + 0.2, theory, width=0.4, label="Noll 1976", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(js)
    ax.set_xlabel("Noll index j"); ax.set_ylabel(r"variance [rad$^2$]")
    ax.set_title("Zernike variance spectrum vs Noll\n(j=2,3 suppressed by FFT deficit)")
    ax.legend(); fig.tight_layout()
    return fig


def wavefront_panels(frame, truth, recon, residual, mask) -> plt.Figure:
    """Generalised from scripts/demo.py: spots, truth, recon, residual."""
    t = np.where(mask, truth, np.nan)
    r = np.where(mask, recon, np.nan)
    res = np.where(mask, residual, np.nan)
    vmax = np.nanmax(np.abs(t))
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    ax[0].imshow(frame, cmap="inferno"); ax[0].set_title("SH-WFS frame")
    for a, dat, title in zip(ax[1:], (t, r, res),
                             ("input [rad]", "reconstructed [rad]", "residual [rad]")):
        a.imshow(dat, cmap="RdBu_r", vmin=-vmax, vmax=vmax); a.set_title(title)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    return fig
