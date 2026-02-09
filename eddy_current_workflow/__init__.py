"""
Eddy-Current Forward-Inverse Workflow Package

This package provides tools for:
1. Material profile generation and manipulation
2. Forward eddy-current response calculation
3. Inverse problem solving (EDC → material profiles)
4. Profile database storage and search
5. End-to-end pipelines

Author: Eddy-Current Research Team
Version: 0.1.0
"""

__version__ = "0.1.0"

from . import config
from . import profiles
from . import forward
from . import inverse
from . import database
from . import pipelines

__all__ = [
    'config',
    'profiles',
    'forward',
    'inverse',
    'database',
    'pipelines',
]
