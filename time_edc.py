
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, "/workspace/GANs-for-1D-Signal")

from eddy_current_workflow.forward.edc_solver import edc_forward, ProbeSettings

sigma = np.full(51, 3e7)
mu = np.full(51, 1.0)
probe = ProbeSettings(frequency=1e6)

print("Timing edc_forward (n_quad=20)...")
t0 = time.time()
resp = edc_forward(sigma, mu, probe, n_quad=20)
t1 = time.time()
print(f"edc_forward took {t1 - t0:.4f} seconds")
print(f"Result: {resp.impedance_complex}")
