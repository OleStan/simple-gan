
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, "/workspace/GANs-for-1D-Signal")
sys.path.insert(0, "/workspace/GANs-for-1D-Signal/experiments/inverse_solver/dodd_gan")

from dodd_forward import dodd_forward, PROBE_DEFAULT

sigma = np.full(51, 3e7)
mu = np.full(51, 1.0)
probe = PROBE_DEFAULT

print("Timing dodd_forward...")
t0 = time.time()
resp = dodd_forward(sigma, mu, probe, integ_top_range=20)
t1 = time.time()
print(f"dodd_forward took {t1 - t0:.4f} seconds")
print(f"Result: {resp.impedance_complex}")
