"""Visual validation report for the SH-WFS pipeline.

Runs the synthetic-rigor characterisations, writes a figure per check to
outputs/validation/, and assembles outputs/validation/index.html with all
figures plus a pass/fail metrics table (measured value vs threshold).

    python scripts/validate_report.py
    ./run.sh validate
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.modal_zernike import Config, WFSPipeline
from methods.common.phasescreen import frozen_flow_sequence, kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame
from methods.common.turbulence import estimate_r0, estimate_tau0
from validations import metrics as m
from validations import plots
from validations.thresholds import TOL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

FIGS = []   # (filename, caption)
ROWS = []   # (check, measured, threshold, pass?)


def _save(fig, name, caption):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    FIGS.append((name, caption))


def _row(check, measured, threshold, ok):
    ROWS.append((check, measured, threshold, ok))


def _ensemble(pipe, r0, n, rng):
    cfg, mask = pipe.cfg, pipe.geom.pupil_mask
    co = np.empty((n, cfg.n_modes))
    for k in range(n):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0, rng)
        ph = ph - ph[mask].mean()
        co[k] = pipe.process(render_frame(ph, cfg, pipe.geom, rng=rng)).coeffs
    return co


def main() -> None:
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))
    geom, mask = pipe.geom, pipe.geom.pupil_mask
    D = cfg.pupil_diameter
    modes = list(range(2, cfg.n_modes + 2))
    rng = np.random.default_rng(2026)

    print("[1/12] mode-transfer / cross-talk")
    M = np.zeros((cfg.n_modes, cfg.n_modes))
    for k in range(cfg.n_modes):
        c = np.zeros(cfg.n_modes); c[k] = 0.2
        M[:, k] = pipe.process(render_frame(pipe.recon.wavefront_from_coeffs(c),
                                            cfg, geom)).coeffs / 0.2
    offdiag = np.abs(M - np.diag(np.diag(M))).max()
    _save(plots.crosstalk_heatmap(M, modes), "01_crosstalk.png",
          "Empirical mode-transfer matrix (ideal = identity).")
    _row("Cross-talk max off-diagonal", f"{offdiag:.3f}",
         f"< {TOL['crosstalk_offdiag']}", offdiag < TOL["crosstalk_offdiag"])

    print("[2/12] linearity / dynamic range")
    amps = np.linspace(0.05, 2.0, 16)
    rec = []
    for a in amps:
        c = np.zeros(cfg.n_modes); c[5 - 2] = a
        rec.append(pipe.process(render_frame(pipe.recon.wavefront_from_coeffs(c),
                                             cfg, geom)).coeffs[5 - 2])
    rec = np.array(rec)
    err = np.abs(rec - amps) / amps
    break_amp = amps[np.argmax(err > 0.1)] if (err > 0.1).any() else None
    _save(plots.linearity(amps, rec, 5, break_amp), "02_linearity.png",
          "Recovered vs injected amplitude; dynamic-range limit where error >10%.")

    print("[3/12] residual vs SNR")
    c = np.zeros(cfg.n_modes); c[5 - 2], c[8 - 2] = 0.6, -0.4
    truth = pipe.recon.wavefront_from_coeffs(c)
    snrs, resids = [], []
    for peak in (100, 300, 1000, 3000, 10000):
        rr, ss = [], []
        for _ in range(8):
            fr = render_frame(truth, cfg, geom, peak=peak, read_noise=5.0,
                              photon_noise=True, rng=rng)
            rr.append(m.residual_rms(truth, pipe.process(fr).wavefront, mask))
            ss.append(m.snr_of_frame(fr))
        snrs.append(np.mean(ss)); resids.append(np.mean(rr))
    _save(plots.residual_vs_snr(snrs, resids), "03_residual_vs_snr.png",
          "Reconstruction residual RMS falls as detector SNR rises.")

    print("[4/12] modal fitting error vs #modes")
    nmodes_list = [3, 6, 10, 15, 20, 28]
    fit_res = []
    base = Config()
    for nm in nmodes_list:
        c2 = Config(n_modes=nm)
        p2 = WFSPipeline(c2); p2.calibrate(flat_frame(c2, p2.geom))
        rr = []
        rng2 = np.random.default_rng(11)
        for _ in range(8):
            ph = kolmogorov_screen(base.npix, base.pupil_dx, D / 6.0, rng2)
            ph = (ph - ph[mask].mean()) * mask
            fr = render_frame(ph, c2, p2.geom, rng=rng2)
            rr.append(m.residual_rms(ph, p2.process(fr).wavefront, mask))
        fit_res.append(np.mean(rr))
    _save(plots.fitting_error_vs_modes(nmodes_list, fit_res), "04_fitting_error.png",
          "Residual RMS of a turbulent wavefront vs number of reconstructed modes.")

    print("[5/12] r0 vs truth")
    r0_true_list, r0_mean, r0_std = [], [], []
    for dr in (4.0, 6.0, 8.0, 12.0):
        r0t = D / dr
        ests = []
        for s in range(4):
            co = _ensemble(pipe, r0t, 80, np.random.default_rng(100 + s))
            ests.append(estimate_r0(co, D))
        r0_true_list.append(r0t); r0_mean.append(np.mean(ests)); r0_std.append(np.std(ests))
    _save(plots.r0_vs_truth(r0_true_list, r0_mean, r0_std), "05_r0_vs_truth.png",
          "Estimated r0 (mean ± std over ensembles) vs true r0.")
    worst = max(abs(me - tr) / tr for me, tr in zip(r0_mean, r0_true_list))
    _row("r0 worst relative error", f"{worst:.2f}", f"< {TOL['r0_rel']}",
         worst < TOL["r0_rel"])

    print("[6/12] r0 convergence")
    co_big = _ensemble(pipe, D / 6.0, 250, np.random.default_rng(5))
    ns = [20, 40, 80, 120, 180, 250]
    _save(plots.r0_convergence(ns, [estimate_r0(co_big[:n], D) for n in ns], D / 6.0),
          "06_r0_convergence.png", "r0 estimate stabilises as the ensemble grows.")

    print("[7/12] r0 vs j_start")
    js = [2, 3, 4, 5, 6]
    _save(plots.r0_vs_jstart(js, [estimate_r0(co_big, D, j_start=j) for j in js], D / 6.0),
          "07_r0_jstart.png", "r0 bias vs j_start; low j_start is hit by the tip/tilt deficit.")

    print("[8/12] tau0 & autocorrelation vs wind")
    winds = [3.0, 5.0, 8.0]
    tau_meas, tau_theory, curves, labels = [], [], [], []
    lags_ref = None
    for w in winds:
        rngw = np.random.default_rng(7)
        dt = cfg.pupil_dx / w
        seq = list(frozen_flow_sequence(cfg.npix, cfg.pupil_dx, D / 6.0, 120, w, dt, rngw))
        wfs = np.stack([pipe.process(render_frame(ph - ph[mask].mean(), cfg, geom,
                                                  rng=rngw)).coeffs for ph in seq])
        tau_meas.append(estimate_tau0(wfs, dt)); tau_theory.append(0.314 * (D / 6.0) / w)
        # autocorrelation curve for the plot
        x = wfs - wfs.mean(0, keepdims=True)
        var0 = (x * x).sum(1).mean()
        ml = len(wfs) // 2
        ac = np.array([(x[:len(x) - L] * x[L:]).sum(1).mean() / var0 for L in range(ml)])
        curves.append(ac); labels.append(f"v={w} m/s")
        lags_ref = np.arange(ml) * dt
    _save(plots.tau0_vs_wind(winds, tau_meas, tau_theory), "08_tau0_vs_wind.png",
          "Coherence time vs wind: measured 1/e decay and the 0.314 r0/v reference.")
    _save(plots.autocorr_curves(lags_ref, curves, labels), "09_autocorr.png",
          "Temporal decorrelation curves under frozen flow.")
    prod = np.array(tau_meas) * np.array(winds)
    inv = (prod.max() - prod.min()) / prod.mean()
    _row("tau0·wind invariance", f"{inv:.3f}", f"< {TOL['tau0_wind_rel']}",
         inv < TOL["tau0_wind_rel"])

    print("[9/12] structure function")
    acc = None
    rng3 = np.random.default_rng(0)
    for _ in range(8):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, D / 6.0, rng3)
        ph = ph - ph[mask].mean()
        r, d = m.structure_function(ph, cfg.pupil_dx, max_lag=cfg.npix // 2)
        acc = d if acc is None else acc + d
    d = acc / 8
    th = m.kolmogorov_structure_fn_theory(r, D / 6.0)
    _save(plots.structure_function(r, d, th), "10_structure_function.png",
          "Phase structure function vs Kolmogorov 6.88(r/r0)^(5/3).")
    slope = m.loglog_slope(r[2:cfg.npix // 16], d[2:cfg.npix // 16])
    _row("Structure-fn slope (mid-range)", f"{slope:.2f}",
         f"|·-{TOL['struct_fn_slope']:.2f}| < {TOL['struct_fn_slope_tol']}",
         abs(slope - TOL["struct_fn_slope"]) < TOL["struct_fn_slope_tol"])

    print("[10/12] radial PSD")
    ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, D / 6.0, np.random.default_rng(1))
    f, psd = m.radial_psd(ph, cfg.pupil_dx)
    _save(plots.radial_psd(f, psd), "11_radial_psd.png",
          "Radially-averaged PSD vs the Kolmogorov f^(-11/3) slope.")
    pslope = m.loglog_slope(f[3:cfg.npix // 4], psd[3:cfg.npix // 4])
    _row("PSD slope (mid-range)", f"{pslope:.2f}",
         f"|·-({TOL['psd_slope']:.2f})| < {TOL['psd_slope_tol']}",
         abs(pslope - TOL["psd_slope"]) < TOL["psd_slope_tol"])

    print("[11/12] Zernike variance spectrum")
    co = _ensemble(pipe, D / 6.0, 200, np.random.default_rng(0))
    meas = m.zernike_variance_spectrum(co)
    theory = m.noll_mode_variance_spectrum(modes, D, D / 6.0)
    _save(plots.variance_spectrum(modes, meas, theory), "12_variance_spectrum.png",
          "Per-mode coefficient variance vs Noll (1976); j=2,3 suppressed by the deficit.")
    mid = meas[2:].sum() / theory[2:].sum()
    _row("Mid-order captured fraction", f"{mid:.2f}",
         f"[{TOL['midorder_ratio_lo']}, {TOL['midorder_ratio_hi']}]",
         TOL["midorder_ratio_lo"] < mid < TOL["midorder_ratio_hi"])

    print("[12/12] wavefront panels")
    ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, D / 6.0, np.random.default_rng(1))
    ph = ph - ph[mask].mean()
    fr = render_frame(ph, cfg, geom, read_noise=2.0, photon_noise=True,
                      rng=np.random.default_rng(1))
    res = pipe.process(fr)
    _save(plots.wavefront_panels(fr, ph * mask, res.wavefront,
                                 (ph - res.wavefront) * mask, mask),
          "13_wavefront_panels.png",
          "Example: SH-WFS frame, input wavefront, reconstruction, residual.")

    _write_html(cfg)
    n_pass = sum(1 for *_, ok in ROWS if ok)
    print(f"\n{n_pass}/{len(ROWS)} checks passed")
    print(f"report: {os.path.join(OUT, 'index.html')}")


def _write_html(cfg) -> None:
    rows_html = "\n".join(
        f"<tr class='{'ok' if ok else 'fail'}'><td>{c}</td><td>{meas}</td>"
        f"<td>{thr}</td><td>{'PASS' if ok else 'FAIL'}</td></tr>"
        for c, meas, thr, ok in ROWS
    )
    figs_html = "\n".join(
        f"<figure><img src='{name}'/><figcaption>{cap}</figcaption></figure>"
        for name, cap in FIGS
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>SH-WFS validation report</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:2rem;max-width:1100px}}
 h1{{margin-bottom:0}} .sub{{color:#666;margin-top:.2rem}}
 table{{border-collapse:collapse;margin:1.5rem 0;width:100%}}
 td,th{{border:1px solid #ccc;padding:.4rem .7rem;text-align:left;font-size:.95rem}}
 tr.ok td:last-child{{color:#137a13;font-weight:600}}
 tr.fail td:last-child{{color:#c00;font-weight:600}}
 figure{{margin:1.5rem 0;border:1px solid #eee;border-radius:8px;padding:.8rem}}
 figure img{{max-width:100%;height:auto}} figcaption{{color:#444;margin-top:.4rem}}
</style></head><body>
<h1>SH-WFS validation report</h1>
<p class='sub'>Synthetic-rigor characterisation &mdash; n_lenslets={cfg.n_lenslets},
 valid sub-apertures via geometry, n_modes={cfg.n_modes}, D={cfg.pupil_diameter*1e3:.2f} mm.</p>
<h2>Acceptance checks</h2>
<table><tr><th>Check</th><th>Measured</th><th>Threshold</th><th>Status</th></tr>
{rows_html}</table>
<h2>Figures</h2>
{figs_html}
</body></html>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
