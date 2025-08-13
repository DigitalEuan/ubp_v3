"""
Universal Binary Principle (UBP) Framework v3.0

A production-ready computational system implementing geometric coherence
and resonance-based computation across multiple physical realms.

Author: Euan Craig
Version: 3.0
Date: August 2025
"""

__version__ = "3.0.0"
__author__ = "Euan Craig"
__email__ = "euan@ubp.nz"  # Placeholder
__description__ = "Universal Binary Principle Computational Framework v3.0"

# Core module imports
from .core import UBPConstants
from .bitfield import Bitfield, OffBit
from .realms import PlatonicRealm, RealmManager
from .hex_dictionary import HexDictionary

__all__ = [
    'UBPConstants',
    'Bitfield',
    'OffBit',
    'PlatonicRealm',
    'RealmManager',
    'HexDictionary'
]

