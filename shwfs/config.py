"""System configuration for the Shack-Hartmann sensor.

All physical quantities are SI (metres, seconds). Replace the defaults with the
real lab numbers once they are available:
  - detector pixel size and frame resolution
  - MLA lenslet pitch, focal length, number of lenslets
  - pupil (turbulated beam) diameter
  - DM actuator count and inter-actuator coupling
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Microlens array (MLA) ---
    n_lenslets: int = 16            # lenslets across the pupil (square grid)
    lenslet_pitch: float = 300e-6   # centre-to-centre spacing of lenslets [m]
    focal_length: float = 0.02      # lenslet focal length [m]

    # --- Detector ---
    pix_per_lenslet: int = 24       # detector pixels spanning one lenslet
    pixel_size: float = 5.5e-6      # detector pixel pitch [m]

    # --- Source / pupil ---
    wavelength: float = 0.5e-6      # imaging wavelength [m]

    # --- Reconstruction ---
    n_modes: int = 20               # Zernike modes used (Noll j = 2 .. n_modes+1)
    illum_threshold: float = 0.5    # min illuminated fraction for a valid sub-aperture

    @property
    def pupil_diameter(self) -> float:
        """Diameter of the sampled pupil [m] = lenslet grid extent."""
        return self.n_lenslets * self.lenslet_pitch

    @property
    def npix(self) -> int:
        """Detector / pupil-sampling resolution along one axis [pixels]."""
        return self.n_lenslets * self.pix_per_lenslet

    @property
    def pupil_dx(self) -> float:
        """Pupil-plane sampling spacing [m/pixel]."""
        return self.pupil_diameter / self.npix

    @property
    def slope_scale(self) -> float:
        """Convert a spot shift [detector px] into a phase gradient [rad/m].

        Spot shift on the detector: dx_px = phase_slope * (f * lambda) / (2*pi * p)
        so the inverse is: phase_slope = dx_px * (2*pi * p) / (f * lambda).
        """
        return (2.0 * 3.141592653589793 * self.pixel_size) / (
            self.focal_length * self.wavelength
        )
