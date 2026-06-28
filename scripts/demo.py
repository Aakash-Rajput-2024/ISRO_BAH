"""End-to-end demonstration of the synthetic SH-WFS pipeline.

Run:  python scripts/demo.py
Outputs figures to ./outputs/ and prints validation numbers to the console.

Stages shown:
  1. Generate a Kolmogorov phase screen with a known r0.
  2. Render a SH-WFS detector frame and centroid the spots.
  3. Reconstruct the wavefront (Zernike) and compare to the truth.
  4. Estimate r0 from an ensemble of frames.
  5. Estimate tau0 from a frozen-flow time series.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shwfs import Config, WFSPipeline
from shwfs.phasescreen import frozen_flow_sequence, kolmogorov_screen
from shwfs.simulate import flat_frame, render_frame
from shwfs.turbulence import estimate_r0, estimate_tau0

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(1)
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))

    D = cfg.pupil_diameter
    r0_true = D / 6.0  # D/r0 = 6 (moderately strong turbulence)
    print(f"pupil D = {D*1e3:.2f} mm | true r0 = {r0_true*1e3:.3f} mm | D/r0 = {D/r0_true:.1f}")
    print(f"valid sub-apertures: {pipe.geom.n_valid} | Zernike modes: {cfg.n_modes}")

    # --- 1-3: single frame round-trip --------------------------------------
    phase = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0_true, rng)
    phase = phase - phase[pipe.geom.pupil_mask].mean()  # drop piston
    frame = render_frame(phase, cfg, pipe.geom, read_noise=2.0, photon_noise=True, rng=rng)
    res = pipe.process(frame)

    mask = pipe.geom.pupil_mask
    truth = phase * mask
    recon = res.wavefront
    residual = (truth - recon) * mask
    rms_in = truth[mask].std()
    rms_res = residual[mask].std()
    print(f"input wavefront RMS  = {rms_in:.3f} rad")
    print(f"residual RMS         = {rms_res:.3f} rad  ({100*rms_res/rms_in:.1f}% of input)")
    print(f"max spot shift       = {np.abs(res.cx - pipe.ref_x).max():.2f} px")

    _plot_maps(frame, truth, recon, residual, mask, OUT)

    # --- 4: r0 from an ensemble --------------------------------------------
    n_frames = 200
    coeffs = np.empty((n_frames, cfg.n_modes))
    for k in range(n_frames):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0_true, rng)
        ph = ph - ph[mask].mean()
        coeffs[k] = pipe.process(render_frame(ph, cfg, pipe.geom, rng=rng)).coeffs
    r0_est = estimate_r0(coeffs, D)
    print(f"\nestimated r0 (n={n_frames}) = {r0_est*1e3:.3f} mm  "
          f"(true {r0_true*1e3:.3f} mm, error {100*(r0_est-r0_true)/r0_true:+.0f}%)")

    # --- 5: tau0 from frozen flow ------------------------------------------
    # The lab pupil is tiny (mm) so tau0 is ~tens of microseconds; sample finely
    # (~1 px of screen drift per frame) so the decorrelation is resolved.
    wind = 5.0
    dt = cfg.pupil_dx / wind        # ~1 px drift per frame
    seq = list(frozen_flow_sequence(cfg.npix, cfg.pupil_dx, r0_true, 120, wind, dt, rng))
    wfs = np.stack([pipe.process(render_frame(ph - ph[mask].mean(), cfg, pipe.geom, rng=rng)).coeffs
                    for ph in seq])
    tau0_meas = estimate_tau0(wfs, dt)
    tau0_theory = 0.314 * r0_true / wind
    print(f"estimated tau0       = {tau0_meas*1e3:.2f} ms  "
          f"(0.314 r0/v = {tau0_theory*1e3:.2f} ms, wind {wind} m/s)")

    print(f"\nfigures written to {OUT}/")


def _plot_maps(frame, truth, recon, residual, mask, out) -> None:
    truth = np.where(mask, truth, np.nan)
    recon = np.where(mask, recon, np.nan)
    residual = np.where(mask, residual, np.nan)
    vmax = np.nanmax(np.abs(truth))

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(frame, cmap="inferno"); ax[0].set_title("SH-WFS frame (spots)")
    ax[1].imshow(truth, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[1].set_title("input wavefront [rad]")
    ax[2].imshow(recon, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[2].set_title("reconstructed [rad]")
    ax[3].imshow(residual, cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[3].set_title("residual [rad]")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reconstruction.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
