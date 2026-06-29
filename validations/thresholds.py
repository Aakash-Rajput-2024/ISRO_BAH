"""Canonical acceptance thresholds, shared by the test suite and the report.

Rationale for each value is documented in docs/VALIDATION.md. The plain FFT
phase screen under-represents low spatial frequencies (Lane et al. 1992), so
absolute structure-function / variance magnitudes are biased low; the
power-law SHAPE (slope) is validated instead of the magnitude.
"""

TOL = {
    "crosstalk_offdiag": 0.05,        # |off-diagonal| of empirical mode transfer
    "crosstalk_diag_lo": 0.95,        # per-mode gain (diagonal) lower bound
    "crosstalk_diag_hi": 1.05,        # per-mode gain upper bound
    "single_mode_recovery_rad": 0.03, # noiseless per-mode recovery
    "r0_rel": 0.40,                   # |r0_est - r0_true| / r0_true
    "left_inverse": 1e-9,             # |R@D - I|
    "struct_fn_slope": 5.0 / 3.0,     # Kolmogorov D_phi ~ r^(5/3)
    "struct_fn_slope_tol": 0.25,
    "psd_slope": -11.0 / 3.0,         # Kolmogorov PSD ~ f^(-11/3)
    "psd_slope_tol": 0.30,
    "tiptilt_deficit_ratio": 0.20,    # j=2,3 measured/Noll must be small (deficit)
    "midorder_ratio_lo": 0.50,        # j>=4 captured-variance / Noll band
    "midorder_ratio_hi": 1.10,
    "tau0_wind_rel": 0.05,            # tau0 * wind invariance across wind speeds
    "golden_centroid_px": 1e-6,       # noiseless centroid reproducibility
    "golden_coeff_rel": 1e-9,         # reconstruction vs stored R
}
