"""Wavefront-reconstruction methods for the SH-WFS pipeline.

`common/` holds method-agnostic infrastructure (system config, pupil geometry,
Zernike basis, phase-screen simulator, centroiding, turbulence statistics).
Each reconstruction approach is its own sub-package and reuses `common`:

  - `modal_zernike/`  -- Method 1: modal (Zernike) least-squares reconstruction.

Add future methods (e.g. zonal/Southwell, Karhunen-Loeve, deep learning) as
sibling sub-packages so the validation suite can compare them on equal footing.
"""
