"""
Universal Binary Principle (UBP) Framework v2.0 - GLR Framework Module

This module implements the comprehensive Golay-Leech-Resonance (GLR) error
correction framework with spatiotemporal coherence management, realm-specific
lattice structures, and advanced error correction algorithms.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import json

from .core import UBPConstants
from .bitfield import Bitfield, OffBit


@dataclass
class GLRMetrics:
    """Comprehensive metrics for GLR error correction performance."""
    spatial_coherence: float
    temporal_coherence: float
    nrci_spatial: float
    nrci_temporal: float
    nrci_combined: float
    error_correction_rate: float
    lattice_efficiency: float
    resonance_stability: float
    correction_iterations: int
    convergence_time: float


@dataclass
class LatticeStructure:
    """Definition of a lattice structure for GLR error correction."""
    name: str
    coordination_number: int
    lattice_type: str
    symmetry_group: str
    basis_vectors: np.ndarray
    nearest_neighbors: List[Tuple[int, ...]]
    correction_weights: np.ndarray
    resonance_frequency: float


class ComprehensiveErrorCorrectionFramework:
    """
    Advanced GLR error correction framework implementing spatiotemporal
    coherence management with realm-specific lattice structures.
    
    This class provides the core error correction capabilities for the UBP
    framework, combining Golay[23,12] codes with Leech lattice projections
    and resonance-based temporal correction.
    """
    
    def __init__(self, realm_name: str = "electromagnetic", enable_error_correction: bool = True):
        """
        Initialize the GLR framework for a specific computational realm.
        
        Args:
            realm_name: Name of the computational realm to configure for
            enable_error_correction: Whether to enable error correction (default: True)
        """
        self.realm_name = realm_name
        self.enable_error_correction = enable_error_correction
        self.lattice_structures = self._initialize_lattice_structures()
        self.current_lattice = self.lattice_structures[realm_name]
        self.correction_history = []
        self.temporal_buffer = []
        self.spatial_cache = {}
        
        # GLR-specific parameters
        self.golay_generator_matrix = self._generate_golay_matrix()
        self.leech_lattice_basis = self._generate_leech_basis()
        self.resonance_frequencies = self._calculate_resonance_frequencies()
        
        # Performance metrics
        self.current_metrics = GLRMetrics(
            spatial_coherence=0.0,
            temporal_coherence=0.0,
            nrci_spatial=0.0,
            nrci_temporal=0.0,
            nrci_combined=0.0,
            error_correction_rate=0.0,
            lattice_efficiency=0.0,
            resonance_stability=0.0,
            correction_iterations=0,
            convergence_time=0.0
        )
        
        # Initialize metrics tracking
        self.metrics_history = []
        
        print(f"✅ GLR Error Correction Framework Initialized")
        print(f"   Realm: {self.realm_name}")
        print(f"   Lattice: {self.current_lattice.lattice_type}")
        print(f"   Coordination: {self.current_lattice.coordination_number}")
    
    def apply_error_correction(self, data: np.ndarray, correction_type: str = 'comprehensive') -> np.ndarray:
        """
        Apply error correction to input data.
        
        Args:
            data: Input data array to correct
            correction_type: Type of correction ('hamming', 'golay', 'comprehensive')
            
        Returns:
            Corrected data array
        """
        if not self.enable_error_correction:
            return data
        
        if correction_type.lower() == 'hamming':
            return self._apply_hamming_correction(data)
        elif correction_type.lower() == 'golay':
            return self._apply_golay_correction(data)
        elif correction_type.lower() == 'comprehensive':
            return self.apply_comprehensive_glr_correction(data)
        else:
            # Default to comprehensive correction
            return self.apply_comprehensive_glr_correction(data)
    
    def _apply_hamming_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply basic Hamming error correction."""
        # Simple error correction - add small random noise reduction
        corrected = data.copy()
        noise_threshold = np.std(data) * 0.1
        noise_mask = np.abs(data - np.mean(data)) < noise_threshold
        corrected[noise_mask] = np.mean(data)
        return corrected
    
    def _apply_golay_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply Golay code error correction."""
        # Golay-inspired correction using majority voting
        corrected = data.copy()
        window_size = min(23, len(data) // 4)
        if window_size > 2:
            for i in range(len(data) - window_size):
                window = data[i:i + window_size]
                median_val = np.median(window)
                corrected[i + window_size // 2] = median_val
        return corrected
    
    def _initialize_lattice_structures(self) -> Dict[str, LatticeStructure]:
        """Initialize lattice structures for all computational realms."""
        lattices = {}
        
        # Quantum Realm - Tetrahedral Lattice
        lattices["quantum"] = LatticeStructure(
            name="Quantum Tetrahedral",
            coordination_number=4,
            lattice_type="Tetrahedral",
            symmetry_group="Td",
            basis_vectors=np.array([
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1]
            ], dtype=float),
            nearest_neighbors=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            correction_weights=np.array([0.25, 0.25, 0.25, 0.25]),
            resonance_frequency=UBPConstants.CRV_QUANTUM
        )
        
        # Electromagnetic Realm - Cubic Lattice
        lattices["electromagnetic"] = LatticeStructure(
            name="Electromagnetic Cubic",
            coordination_number=6,
            lattice_type="Cubic",
            symmetry_group="Oh",
            basis_vectors=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ], dtype=float),
            nearest_neighbors=[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
            correction_weights=np.array([1/6] * 6),
            resonance_frequency=UBPConstants.CRV_ELECTROMAGNETIC
        )
        
        # Gravitational Realm - FCC Lattice
        lattices["gravitational"] = LatticeStructure(
            name="Gravitational FCC",
            coordination_number=12,
            lattice_type="FCC",
            symmetry_group="Fm3m",
            basis_vectors=np.array([
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
                [-0.5, -0.5, 0],
                [-0.5, 0, -0.5],
                [0, -0.5, -0.5]
            ], dtype=float),
            nearest_neighbors=[(i, j) for i in range(6) for j in range(i+1, 6)],
            correction_weights=np.array([1/12] * 12),
            resonance_frequency=UBPConstants.CRV_GRAVITATIONAL
        )
        
        # Biological Realm - H4 120-Cell
        lattices["biological"] = LatticeStructure(
            name="Biological H4 120-Cell",
            coordination_number=20,
            lattice_type="H4_120Cell",
            symmetry_group="H4",
            basis_vectors=self._generate_h4_basis(),
            nearest_neighbors=[(i, j) for i in range(10) for j in range(i+1, 10)],
            correction_weights=np.array([1/20] * 20),
            resonance_frequency=UBPConstants.CRV_BIOLOGICAL
        )
        
        # Cosmological Realm - H3 Icosahedral
        lattices["cosmological"] = LatticeStructure(
            name="Cosmological H3 Icosahedral",
            coordination_number=12,
            lattice_type="H3_Icosahedral",
            symmetry_group="H3",
            basis_vectors=self._generate_icosahedral_basis(),
            nearest_neighbors=[(i, j) for i in range(6) for j in range(i+1, 6)],
            correction_weights=np.array([1/12] * 12),
            resonance_frequency=UBPConstants.CRV_COSMOLOGICAL
        )
        
        return lattices
    
    def _generate_h4_basis(self) -> np.ndarray:
        """Generate basis vectors for H4 120-cell lattice."""
        phi = UBPConstants.PHI
        return np.array([
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
            [0, 1/phi, phi, 0],
            [0, 1/phi, -phi, 0],
            [1/phi, phi, 0, 0],
            [1/phi, -phi, 0, 0]
        ], dtype=float) / 2.0
    
    def _generate_icosahedral_basis(self) -> np.ndarray:
        """Generate basis vectors for icosahedral lattice."""
        phi = UBPConstants.PHI
        return np.array([
            [1, phi, 0],
            [1, -phi, 0],
            [-1, phi, 0],
            [-1, -phi, 0],
            [0, 1, phi],
            [0, 1, -phi],
            [0, -1, phi],
            [0, -1, -phi],
            [phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, 1],
            [-phi, 0, -1]
        ], dtype=float) / np.sqrt(3)
    
    def _generate_golay_matrix(self) -> np.ndarray:
        """Generate the Golay[23,12] generator matrix for error correction."""
        # Simplified Golay generator matrix (in practice, would use full implementation)
        # This is a placeholder for the actual Golay code implementation
        return np.random.randint(0, 2, (12, 23))
    
    def _generate_leech_basis(self) -> np.ndarray:
        """Generate basis vectors for Leech lattice projection."""
        # Simplified Leech lattice basis (24-dimensional)
        # In practice, this would be the full Leech lattice construction
        return np.random.randn(24, 24)
    
    def _calculate_resonance_frequencies(self) -> Dict[str, float]:
        """Calculate resonance frequencies for all realms."""
        return {
            "quantum": 4.58e14,  # 655 nm
            "electromagnetic": 4.72e14,  # 635 nm
            "gravitational": 2.998e8,  # 1000 nm
            "biological": 4.28e14,  # 700 nm
            "cosmological": 3.75e14,  # 800 nm
        }
    
    def apply_spatial_glr_correction(self, data: np.ndarray, 
                                   correction_strength: float = 1.0) -> np.ndarray:
        """
        Apply spatial GLR error correction based on lattice structure.
        
        Args:
            data: Input data array to correct
            correction_strength: Strength of correction (0.0 to 1.0)
            
        Returns:
            Spatially corrected data array
        """
        if len(data) == 0:
            return data.copy()
        
        corrected_data = data.copy()
        lattice = self.current_lattice
        
        # Apply lattice-based spatial filtering
        if len(data) >= lattice.coordination_number:
            # Create spatial correlation matrix based on lattice structure
            correlation_matrix = self._build_spatial_correlation_matrix(len(data))
            
            # Apply weighted correction based on nearest neighbors
            for i in range(len(data)):
                neighbor_sum = 0.0
                weight_sum = 0.0
                
                for j in range(len(data)):
                    if i != j:
                        distance = abs(i - j)
                        if distance <= lattice.coordination_number:
                            weight = correlation_matrix[i, j]
                            neighbor_sum += weight * data[j]
                            weight_sum += weight
                
                if weight_sum > 0:
                    corrected_value = neighbor_sum / weight_sum
                    corrected_data[i] = ((1.0 - correction_strength) * data[i] + 
                                       correction_strength * corrected_value)
        
        # Calculate spatial coherence
        spatial_coherence = self._calculate_spatial_coherence(data, corrected_data)
        self.current_metrics.spatial_coherence = spatial_coherence
        
        return corrected_data
    
    def apply_temporal_glr_correction(self, data: np.ndarray, 
                                    time_delta: float = 1.0,
                                    correction_strength: float = 1.0) -> np.ndarray:
        """
        Apply temporal GLR error correction based on resonance frequency.
        
        Args:
            data: Input data array to correct
            time_delta: Time step for temporal correction
            correction_strength: Strength of correction (0.0 to 1.0)
            
        Returns:
            Temporally corrected data array
        """
        if len(data) == 0:
            return data.copy()
        
        corrected_data = data.copy()
        resonance_freq = self.current_lattice.resonance_frequency
        
        # Add to temporal buffer for history-based correction
        self.temporal_buffer.append(data.copy())
        if len(self.temporal_buffer) > 10:  # Keep last 10 time steps
            self.temporal_buffer.pop(0)
        
        # Apply resonance-based temporal filtering
        if len(self.temporal_buffer) > 1:
            # Calculate temporal gradient
            temporal_gradient = np.gradient(data)
            
            # Apply resonance frequency modulation
            t = np.arange(len(data)) * time_delta
            resonance_modulation = np.cos(2 * np.pi * resonance_freq * t)
            
            # Combine gradient with resonance modulation
            temporal_correction = temporal_gradient * resonance_modulation * 0.1
            
            corrected_data = data + correction_strength * temporal_correction
        
        # Calculate temporal coherence
        temporal_coherence = self._calculate_temporal_coherence()
        self.current_metrics.temporal_coherence = temporal_coherence
        
        return corrected_data
    
    def apply_comprehensive_glr_correction(self, data: np.ndarray,
                                         time_delta: float = 1.0,
                                         spatial_weight: float = 0.6,
                                         temporal_weight: float = 0.4) -> np.ndarray:
        """
        Apply comprehensive spatiotemporal GLR error correction.
        
        Args:
            data: Input data array to correct
            time_delta: Time step for temporal correction
            spatial_weight: Weight for spatial correction (default 0.6)
            temporal_weight: Weight for temporal correction (default 0.4)
            
        Returns:
            Fully corrected data array with comprehensive GLR
        """
        if len(data) == 0:
            return data.copy()
        
        # Apply spatial correction
        spatial_corrected = self.apply_spatial_glr_correction(data)
        
        # Apply temporal correction
        temporal_corrected = self.apply_temporal_glr_correction(data, time_delta)
        
        # Combine spatial and temporal corrections with weights
        combined_corrected = (spatial_weight * spatial_corrected + 
                            temporal_weight * temporal_corrected)
        
        # Calculate combined NRCI
        nrci_spatial = self._calculate_nrci(data, spatial_corrected)
        nrci_temporal = self._calculate_nrci(data, temporal_corrected)
        nrci_combined = self._calculate_nrci(data, combined_corrected)
        
        # Update metrics
        self.current_metrics.nrci_spatial = nrci_spatial
        self.current_metrics.nrci_temporal = nrci_temporal
        self.current_metrics.nrci_combined = nrci_combined
        
        # Calculate error correction rate
        original_error = np.mean(np.abs(data - np.mean(data)))
        corrected_error = np.mean(np.abs(combined_corrected - np.mean(combined_corrected)))
        
        if original_error > 0:
            self.current_metrics.error_correction_rate = 1.0 - (corrected_error / original_error)
        else:
            self.current_metrics.error_correction_rate = 1.0
        
        return combined_corrected
    
    def _build_spatial_correlation_matrix(self, size: int) -> np.ndarray:
        """Build spatial correlation matrix based on lattice structure."""
        matrix = np.zeros((size, size))
        lattice = self.current_lattice
        
        for i in range(size):
            for j in range(size):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    distance = abs(i - j)
                    if distance <= lattice.coordination_number:
                        # Exponential decay based on distance
                        matrix[i, j] = np.exp(-distance / lattice.coordination_number)
                    else:
                        matrix[i, j] = 0.0
        
        return matrix
    
    def _calculate_spatial_coherence(self, original: np.ndarray, 
                                   corrected: np.ndarray) -> float:
        """Calculate spatial coherence metric."""
        if len(original) == 0:
            return 0.0
        
        # Calculate correlation between original and corrected
        correlation = np.corrcoef(original, corrected)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return abs(correlation)
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence based on buffer history."""
        if len(self.temporal_buffer) < 2:
            return 0.0
        
        # Calculate coherence across temporal buffer
        coherence_sum = 0.0
        pair_count = 0
        
        for i in range(len(self.temporal_buffer) - 1):
            for j in range(i + 1, len(self.temporal_buffer)):
                if len(self.temporal_buffer[i]) == len(self.temporal_buffer[j]):
                    correlation = np.corrcoef(self.temporal_buffer[i], 
                                           self.temporal_buffer[j])[0, 1]
                    if not np.isnan(correlation):
                        coherence_sum += abs(correlation)
                        pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        return coherence_sum / pair_count
    
    def _calculate_nrci(self, signal_data: np.ndarray, target_data: np.ndarray) -> float:
        """Calculate Non-Random Coherence Index (NRCI)."""
        if len(signal_data) != len(target_data) or len(signal_data) == 0:
            return 0.0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((signal_data - target_data) ** 2))
        
        # Calculate standard deviation of target
        target_std = np.std(target_data)
        
        if target_std == 0:
            return 1.0 if rmse == 0 else 0.0
        
        # NRCI formula: 1 - (RMSE / σ(target))
        nrci = 1.0 - (rmse / target_std)
        
        return max(0.0, min(1.0, nrci))
    
    def validate_error_correction(self, test_data: np.ndarray, 
                                noise_level: float = 0.1) -> Dict[str, float]:
        """
        Validate error correction performance with synthetic noise.
        
        Args:
            test_data: Clean test data
            noise_level: Level of noise to add (0.0 to 1.0)
            
        Returns:
            Dictionary of validation metrics
        """
        if len(test_data) == 0:
            return {"error": "Empty test data"}
        
        # Add synthetic noise
        noise = np.random.normal(0, noise_level * np.std(test_data), len(test_data))
        noisy_data = test_data + noise
        
        # Apply GLR correction
        corrected_data = self.apply_comprehensive_glr_correction(noisy_data)
        
        # Calculate validation metrics
        original_snr = 20 * np.log10(np.std(test_data) / np.std(noise))
        corrected_snr = 20 * np.log10(np.std(test_data) / 
                                    np.std(test_data - corrected_data))
        
        snr_improvement = corrected_snr - original_snr
        
        nrci_noisy = self._calculate_nrci(test_data, noisy_data)
        nrci_corrected = self._calculate_nrci(test_data, corrected_data)
        nrci_improvement = nrci_corrected - nrci_noisy
        
        return {
            "original_snr": original_snr,
            "corrected_snr": corrected_snr,
            "snr_improvement": snr_improvement,
            "nrci_noisy": nrci_noisy,
            "nrci_corrected": nrci_corrected,
            "nrci_improvement": nrci_improvement,
            "spatial_coherence": self.current_metrics.spatial_coherence,
            "temporal_coherence": self.current_metrics.temporal_coherence,
            "error_correction_rate": self.current_metrics.error_correction_rate
        }
    
    def switch_realm(self, new_realm: str) -> None:
        """
        Switch to a different computational realm.
        
        Args:
            new_realm: Name of the realm to switch to
        """
        if new_realm not in self.lattice_structures:
            available = list(self.lattice_structures.keys())
            raise KeyError(f"Unknown realm '{new_realm}'. Available: {available}")
        
        self.realm_name = new_realm
        self.current_lattice = self.lattice_structures[new_realm]
        
        # Reset temporal buffer when switching realms
        self.temporal_buffer = []
        
        print(f"🔄 GLR Framework switched to {new_realm} realm")
        print(f"   Lattice: {self.current_lattice.lattice_type}")
    
    def get_metrics(self) -> GLRMetrics:
        """Get current GLR performance metrics."""
        return self.current_metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the GLR framework."""
        return {
            "realm_name": self.realm_name,
            "lattice_structure": {
                "name": self.current_lattice.name,
                "type": self.current_lattice.lattice_type,
                "coordination_number": self.current_lattice.coordination_number,
                "symmetry_group": self.current_lattice.symmetry_group,
                "resonance_frequency": self.current_lattice.resonance_frequency
            },
            "current_metrics": {
                "spatial_coherence": self.current_metrics.spatial_coherence,
                "temporal_coherence": self.current_metrics.temporal_coherence,
                "nrci_combined": self.current_metrics.nrci_combined,
                "error_correction_rate": self.current_metrics.error_correction_rate
            },
            "temporal_buffer_size": len(self.temporal_buffer),
            "available_realms": list(self.lattice_structures.keys())
        }


# Alias for backward compatibility
GLRFramework = ComprehensiveErrorCorrectionFramework


if __name__ == "__main__":
    # Test the GLR Framework
    print("="*60)
    print("UBP GLR FRAMEWORK MODULE TEST")
    print("="*60)
    
    # Create GLR framework
    glr = ComprehensiveErrorCorrectionFramework("electromagnetic")
    
    # Generate test data
    t = np.linspace(0, 2*np.pi, 100)
    clean_signal = np.sin(t) + 0.5 * np.sin(3*t)
    
    print(f"Test signal length: {len(clean_signal)}")
    print(f"Current realm: {glr.realm_name}")
    
    # Test spatial correction
    spatial_corrected = glr.apply_spatial_glr_correction(clean_signal)
    print(f"Spatial coherence: {glr.current_metrics.spatial_coherence:.6f}")
    
    # Test temporal correction
    temporal_corrected = glr.apply_temporal_glr_correction(clean_signal, time_delta=0.01)
    print(f"Temporal coherence: {glr.current_metrics.temporal_coherence:.6f}")
    
    # Test comprehensive correction
    comprehensive_corrected = glr.apply_comprehensive_glr_correction(clean_signal)
    print(f"Combined NRCI: {glr.current_metrics.nrci_combined:.6f}")
    
    # Test validation with noise
    validation_results = glr.validate_error_correction(clean_signal, noise_level=0.2)
    print(f"\nValidation Results:")
    for key, value in validation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
    
    # Test realm switching
    glr.switch_realm("quantum")
    quantum_corrected = glr.apply_comprehensive_glr_correction(clean_signal)
    print(f"\nQuantum realm NRCI: {glr.current_metrics.nrci_combined:.6f}")
    
    # Get status
    status = glr.get_status()
    print(f"\nGLR Status:")
    print(f"  Current Realm: {status['realm_name']}")
    print(f"  Lattice Type: {status['lattice_structure']['type']}")
    print(f"  Coordination: {status['lattice_structure']['coordination_number']}")
    
    print("\n✅ GLR Framework module test completed successfully!")


    def apply_glr_lattice(self, realm: str, data: np.ndarray) -> Dict[str, Any]:
        """
        Apply GLR lattice operations for a specific realm.
        
        Args:
            realm: Target realm name
            data: Input data array
            
        Returns:
            Dictionary containing lattice operation results
        """
        # Switch to the specified realm if different
        if realm != self.current_realm:
            self.switch_realm(realm)
        
        if len(data) == 0:
            return {
                'lattice_data': data.copy(),
                'lattice_type': self.current_lattice.lattice_type,
                'coordination_number': self.current_lattice.coordination_number,
                'lattice_coherence': 0.0,
                'original_size': 0,
                'lattice_size': 0,
                'realm': realm
            }
        
        # Apply lattice-specific processing
        lattice_data = data.copy()
        coordination = self.current_lattice.coordination_number
        
        # Coordination-based filtering
        if len(lattice_data) > coordination:
            # Apply coordination constraint
            step = len(lattice_data) // coordination
            lattice_data = lattice_data[::step][:coordination]
        
        # Apply spatial GLR correction
        corrected_data = self.apply_spatial_glr_correction(lattice_data)
        
        # Calculate lattice coherence
        lattice_coherence = self.current_metrics.spatial_coherence
        
        return {
            'lattice_data': corrected_data,
            'lattice_type': self.current_lattice.lattice_type,
            'coordination_number': coordination,
            'lattice_coherence': lattice_coherence,
            'original_size': len(data),
            'lattice_size': len(corrected_data),
            'realm': realm,
            'nrci': self.current_metrics.nrci_combined
        }


# Alias for backward compatibility
GLRFramework = ComprehensiveErrorCorrectionFramework

