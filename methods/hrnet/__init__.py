"""Method 4 -- HRNet-style temporal progressive-injection network (baseline).

Only the model is provided here; data flow / training / inference are wired
separately. See README.md for the I/O contract.

    from methods.hrnet import HRNetWavefront
"""

from .model import HRNetWavefront, BasicBlock, pick_device

__all__ = ["HRNetWavefront", "BasicBlock", "pick_device"]
