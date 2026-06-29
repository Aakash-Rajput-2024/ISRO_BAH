"""Validation framework: metrics, plots, oracle, references, and the report.

Submodules are imported on demand (kept out of this __init__ to avoid pulling
in matplotlib / optional libraries at package import):

  metrics      -- plot-free numeric kernels (shared by tests and the report)
  plots        -- matplotlib figure builders
  thresholds   -- canonical acceptance thresholds (TOL)
  oracle       -- golden-vector dump/load + comparator (C-port equivalence)
  references   -- lazy adapters to independent libraries (aotools / hcipy)
  realdata     -- real lab-frame ingest + sub-aperture grid auto-detection
"""
