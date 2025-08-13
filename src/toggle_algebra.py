"""
Universal Binary Principle (UBP) Framework v2.0 - Toggle Algebra Module

This module implements the comprehensive Toggle Algebra operations engine,
providing both basic Boolean operations and advanced physics-inspired
operations for OffBit manipulation within the UBP framework.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import math
from scipy.special import factorial
from scipy.optimize import minimize_scalar

from .core import UBPConstants
from .bitfield import Bitfield, OffBit


@dataclass
class ToggleOperationResult:
    """Result of a toggle algebra operation."""
    result_value: int
    operation_type: str
    input_values: List[int]
    coherence_change: float
    energy_delta: float
    execution_time: float
    nrci_score: float = 0.0  # Add missing nrci_score field


@dataclass
class ToggleAlgebraMetrics:
    """Performance metrics for toggle algebra operations."""
    total_operations: int
    successful_operations: int
    average_coherence: float
    total_energy_change: float
    operation_distribution: Dict[str, int]
    average_execution_time: float
    resonance_stability: float


class ToggleAlgebra:
    """
    Comprehensive Toggle Algebra engine for UBP OffBit operations.
    
    This class provides the fundamental bit-level operations that drive
    all dynamic processes in the UBP framework, from basic Boolean logic
    to advanced physics-inspired operations.
    """
    
    def __init__(self, bitfield_instance: Optional[Bitfield] = None, glr_framework=None):
        """
        Initialize the Toggle Algebra engine.
        
        Args:
            bitfield_instance: Optional Bitfield instance for operations
            glr_framework: Optional GLR framework for error correction
        """
        self.bitfield = bitfield_instance
        self.glr_framework = glr_framework
        self.operation_history = []
        self.metrics = ToggleAlgebraMetrics(
            total_operations=0,
            successful_operations=0,
            average_coherence=0.0,
            total_energy_change=0.0,
            operation_distribution={},
            average_execution_time=0.0,
            resonance_stability=1.0
        )
        
        # Operation registry
        self.operations = {
            # Basic Boolean Operations
            'AND': self.and_operation,
            'OR': self.or_operation,
            'XOR': self.xor_operation,
            'NOT': self.not_operation,
            'NAND': self.nand_operation,
            'NOR': self.nor_operation,
            
            # Advanced Physics-Inspired Operations
            'RESONANCE': self.resonance_operation,
            'ENTANGLEMENT': self.entanglement_operation,
            'SUPERPOSITION': self.superposition_operation,
            'SPIN_TRANSITION': self.spin_transition_operation,
            'HYBRID_PROM': self.hybrid_prom_operation,
            
            # Electromagnetic Operations (WGE)
            'NONLINEAR_MAXWELL': self.nonlinear_maxwell_operation,
            'LORENTZ_FORCE': self.lorentz_force_operation,
            'WEYL_METRIC': self.weyl_metric_operation,
            
            # Rune Protocol Operations
            'GLYPH_QUANTIFY': self.glyph_quantify_operation,
            'GLYPH_CORRELATE': self.glyph_correlate_operation,
            'GLYPH_SELF_REFERENCE': self.glyph_self_reference_operation
        }
        
        print("✅ UBP Toggle Algebra Engine Initialized")
        print(f"   Available Operations: {len(self.operations)}")
        print(f"   Bitfield Connected: {'Yes' if bitfield_instance else 'No'}")
    
    def _record_operation(self, result: ToggleOperationResult) -> None:
        """Record an operation in the history and update metrics."""
        self.operation_history.append(result)
        
        # Update metrics
        self.metrics.total_operations += 1
        if result.result_value is not None:
            self.metrics.successful_operations += 1
        
        # Update operation distribution
        op_type = result.operation_type
        if op_type not in self.metrics.operation_distribution:
            self.metrics.operation_distribution[op_type] = 0
        self.metrics.operation_distribution[op_type] += 1
        
        # Update running averages
        self.metrics.total_energy_change += result.energy_delta
        
        if self.metrics.successful_operations > 0:
            total_coherence = sum(op.coherence_change for op in self.operation_history)
            self.metrics.average_coherence = total_coherence / self.metrics.successful_operations
            
            total_time = sum(op.execution_time for op in self.operation_history)
            self.metrics.average_execution_time = total_time / self.metrics.successful_operations
    
    # ========================================================================
    # BASIC BOOLEAN OPERATIONS
    # ========================================================================
    
    def and_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform AND operation: min(b_i, b_j)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with AND result
        """
        import time
        start_time = time.time()
        
        # Extract activation layers for the operation
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        # Perform AND on activation layers
        result_activation = min(activation_i, activation_j)
        
        # Create result OffBit preserving other layers from b_i
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        # Calculate coherence change
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="AND",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,  # Energy proportional to coherence
            execution_time=execution_time
        )
    
    def or_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform OR operation: max(b_i, b_j)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with OR result
        """
        import time
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        result_activation = max(activation_i, activation_j)
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="OR",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time
        )
    
    def xor_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform XOR operation: |b_i - b_j|
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with XOR result
        """
        import time
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        result_activation = abs(activation_i - activation_j)
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="XOR",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time
        )
    
    def not_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform NOT operation: invert activation layer
        
        Args:
            b_i: OffBit value to invert
            
        Returns:
            ToggleOperationResult with NOT result
        """
        import time
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        result_activation = 63 - activation_i  # Invert 6-bit value
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="NOT",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time
        )
    
    def nand_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """Perform NAND operation: NOT(AND(b_i, b_j))"""
        and_result = self.and_operation(b_i, b_j)
        not_result = self.not_operation(and_result.result_value)
        
        return ToggleOperationResult(
            result_value=not_result.result_value,
            operation_type="NAND",
            input_values=[b_i, b_j],
            coherence_change=not_result.coherence_change,
            energy_delta=not_result.energy_delta,
            execution_time=and_result.execution_time + not_result.execution_time
        )
    
    def nor_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """Perform NOR operation: NOT(OR(b_i, b_j))"""
        or_result = self.or_operation(b_i, b_j)
        not_result = self.not_operation(or_result.result_value)
        
        return ToggleOperationResult(
            result_value=not_result.result_value,
            operation_type="NOR",
            input_values=[b_i, b_j],
            coherence_change=not_result.coherence_change,
            energy_delta=not_result.energy_delta,
            execution_time=or_result.execution_time + not_result.execution_time
        )
    
    # ========================================================================
    # ADVANCED PHYSICS-INSPIRED OPERATIONS
    # ========================================================================
    
    def resonance_operation(self, b_i: int, time: float, frequency: float, 
                          **kwargs) -> ToggleOperationResult:
        """
        Perform resonance operation: b_i * exp(-0.0002 * (time * frequency)^2)
        
        Args:
            b_i: OffBit value
            time: Time parameter
            frequency: Resonance frequency
            
        Returns:
            ToggleOperationResult with resonance result
        """
        import time as time_module
        start_time = time_module.time()
        
        # Calculate resonance decay factor
        decay_factor = np.exp(-0.0002 * (time * frequency) ** 2)
        
        # Apply to information layer (resonance affects information content)
        info_layer = OffBit.get_information_layer(b_i)
        new_info = int(info_layer * decay_factor) % 64  # Keep within 6-bit range
        
        result = OffBit.set_information_layer(b_i, new_info)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time_module.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="RESONANCE",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=decay_factor * 0.2,  # Energy based on resonance strength
            execution_time=execution_time
        )
    
    def entanglement_operation(self, b_i: int, b_j: int, coherence_factor: float,
                             **kwargs) -> ToggleOperationResult:
        """
        Perform entanglement operation: b_i * b_j * coherence_factor
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            coherence_factor: Entanglement coherence (0.0 to 1.0)
            
        Returns:
            ToggleOperationResult with entangled OffBits
        """
        import time
        start_time = time.time()
        
        # Extract information layers for entanglement
        info_i = OffBit.get_information_layer(b_i)
        info_j = OffBit.get_information_layer(b_j)
        
        # Entangle information layers
        entangled_info_i = int((info_i * (1 - coherence_factor) + 
                               info_j * coherence_factor)) % 64
        entangled_info_j = int((info_j * (1 - coherence_factor) + 
                               info_i * coherence_factor)) % 64
        
        result_i = OffBit.set_information_layer(b_i, entangled_info_i)
        result_j = OffBit.set_information_layer(b_j, entangled_info_j)
        
        # Calculate coherence change for both OffBits
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = (OffBit.calculate_coherence(result_i) + OffBit.calculate_coherence(result_j)) / 2
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        # Return the first entangled OffBit (in practice, both would be updated)
        return ToggleOperationResult(
            result_value=result_i,
            operation_type="ENTANGLEMENT",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_factor * 0.3,  # Energy based on entanglement strength
            execution_time=execution_time
        )
    
    def superposition_operation(self, states: List[int], weights: List[float],
                              **kwargs) -> ToggleOperationResult:
        """
        Perform superposition operation: Σ(states * weights)
        
        Args:
            states: List of OffBit states
            weights: List of weights (must sum to 1.0)
            
        Returns:
            ToggleOperationResult with superposition result
        """
        import time
        start_time = time.time()
        
        if len(states) != len(weights):
            raise ValueError("States and weights must have the same length")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        # Calculate weighted superposition for each layer
        result_layers = {'reality': 0, 'information': 0, 'activation': 0, 'unactivated': 0}
        
        for state, weight in zip(states, weights):
            layers = OffBit.get_all_layers(state)
            for layer_name, layer_value in layers.items():
                result_layers[layer_name] += weight * layer_value
        
        # Convert to integers and clamp to 6-bit range
        for layer_name in result_layers:
            result_layers[layer_name] = int(result_layers[layer_name]) % 64
        
        result = OffBit.create_offbit(**result_layers)
        
        # Calculate average coherence change
        coherence_before = sum(OffBit.calculate_coherence(state) for state in states) / len(states)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="SUPERPOSITION",
            input_values=states,
            coherence_change=coherence_change,
            energy_delta=sum(weights) * 0.25,  # Energy based on superposition complexity
            execution_time=execution_time
        )
    
    def spin_transition_operation(self, b_i: int, spin_probability: float,
                                **kwargs) -> ToggleOperationResult:
        """
        Perform spin transition operation: b_i * ln(1/p_s)
        
        Args:
            b_i: OffBit value
            spin_probability: Probability of spin transition
            
        Returns:
            ToggleOperationResult with spin transition result
        """
        import time
        start_time = time.time()
        
        # Ensure probability is not zero to avoid log(inf)
        spin_probability = max(spin_probability, 1e-9)
        log_factor = np.log(1 / spin_probability)
        
        # Apply to reality layer (spin affects fundamental state)
        reality_layer = OffBit.get_reality_layer(b_i)
        new_reality = int(reality_layer * log_factor) % 64
        
        result = OffBit.set_reality_layer(b_i, new_reality)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="SPIN_TRANSITION",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=log_factor * 0.15,  # Energy based on transition magnitude
            execution_time=execution_time
        )
    
    def hybrid_prom_operation(self, b_i: int, b_j: int, time: float, 
                            frequency: float, **kwargs) -> ToggleOperationResult:
        """
        Perform hybrid PROM operation: |b_i - b_j| * exp(-0.0002 * d^2)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            time: Time parameter
            frequency: Frequency parameter
            
        Returns:
            ToggleOperationResult with hybrid PROM result
        """
        import time as time_module
        start_time = time_module.time()
        
        # Combine XOR with resonance decay
        xor_result = self.xor_operation(b_i, b_j)
        resonance_result = self.resonance_operation(xor_result.result_value, time, frequency)
        
        execution_time = time_module.time() - start_time
        
        return ToggleOperationResult(
            result_value=resonance_result.result_value,
            operation_type="HYBRID_PROM",
            input_values=[b_i, b_j],
            coherence_change=resonance_result.coherence_change,
            energy_delta=xor_result.energy_delta + resonance_result.energy_delta,
            execution_time=execution_time
        )
    
    # ========================================================================
    # ELECTROMAGNETIC OPERATIONS (WGE)
    # ========================================================================
    
    def nonlinear_maxwell_operation(self, b_i: int, field_strength: float,
                                   **kwargs) -> ToggleOperationResult:
        """
        Perform nonlinear Maxwell operation for electromagnetic dynamics.
        
        Args:
            b_i: OffBit value
            field_strength: Electromagnetic field strength
            
        Returns:
            ToggleOperationResult with Maxwell field result
        """
        import time
        start_time = time.time()
        
        # Apply nonlinear electromagnetic field effects
        activation = OffBit.get_activation_layer(b_i)
        
        # Nonlinear field response
        field_response = activation * (1 + 0.1 * field_strength * np.sin(field_strength))
        new_activation = int(field_response) % 64
        
        result = OffBit.set_activation_layer(b_i, new_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="NONLINEAR_MAXWELL",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=field_strength * 0.2,
            execution_time=execution_time
        )
    
    def lorentz_force_operation(self, b_i: int, velocity: float, 
                              magnetic_field: float, **kwargs) -> ToggleOperationResult:
        """
        Perform Lorentz force operation for charged particle dynamics.
        
        Args:
            b_i: OffBit value
            velocity: Particle velocity
            magnetic_field: Magnetic field strength
            
        Returns:
            ToggleOperationResult with Lorentz force result
        """
        import time
        start_time = time.time()
        
        # Calculate Lorentz force effect
        force_magnitude = velocity * magnetic_field
        
        # Apply force to reality layer (affects fundamental motion)
        reality = OffBit.get_reality_layer(b_i)
        new_reality = int(reality + force_magnitude * 10) % 64
        
        result = OffBit.set_reality_layer(b_i, new_reality)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="LORENTZ_FORCE",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=force_magnitude * 0.1,
            execution_time=execution_time
        )
    
    def weyl_metric_operation(self, b_i: int, metric_factor: float,
                            **kwargs) -> ToggleOperationResult:
        """
        Perform Weyl metric operation for geometric electromagnetic effects.
        
        Args:
            b_i: OffBit value
            metric_factor: Weyl metric scaling factor
            
        Returns:
            ToggleOperationResult with Weyl metric result
        """
        import time
        start_time = time.time()
        
        # Apply Weyl metric transformation
        all_layers = OffBit.get_all_layers(b_i)
        
        # Scale all layers by metric factor
        for layer_name, layer_value in all_layers.items():
            all_layers[layer_name] = int(layer_value * metric_factor) % 64
        
        result = OffBit.create_offbit(**all_layers)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="WEYL_METRIC",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=abs(metric_factor - 1.0) * 0.3,
            execution_time=execution_time
        )
    
    # ========================================================================
    # RUNE PROTOCOL OPERATIONS
    # ========================================================================
    
    def glyph_quantify_operation(self, glyph_state: List[int], **kwargs) -> ToggleOperationResult:
        """
        Perform glyph quantification: Q(G, state) = Σ G_i(state)
        
        Args:
            glyph_state: List of OffBit values representing glyph state
            
        Returns:
            ToggleOperationResult with quantified glyph
        """
        import time
        start_time = time.time()
        
        if not glyph_state:
            raise ValueError("Glyph state cannot be empty")
        
        # Quantify glyph by summing activation layers
        total_activation = 0
        for offbit in glyph_state:
            total_activation += OffBit.get_activation_layer(offbit)
        
        # Create result OffBit with quantified activation
        quantified_activation = total_activation % 64
        result = OffBit.create_offbit(activation=quantified_activation)
        
        # Calculate average coherence
        avg_coherence_before = sum(OffBit.calculate_coherence(ob) for ob in glyph_state) / len(glyph_state)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - avg_coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_QUANTIFY",
            input_values=glyph_state,
            coherence_change=coherence_change,
            energy_delta=len(glyph_state) * 0.1,
            execution_time=execution_time
        )
    
    def glyph_correlate_operation(self, glyph_i: List[int], glyph_j: List[int],
                                **kwargs) -> ToggleOperationResult:
        """
        Perform glyph correlation: C(G, R_i, R_j) = P(R_i) * P(R_j) / P(R_i ∩ R_j)
        
        Args:
            glyph_i: First glyph state
            glyph_j: Second glyph state
            
        Returns:
            ToggleOperationResult with correlation result
        """
        import time
        start_time = time.time()
        
        if not glyph_i or not glyph_j:
            raise ValueError("Glyph states cannot be empty")
        
        # Calculate glyph probabilities (normalized activation sums)
        prob_i = sum(OffBit.get_activation_layer(ob) for ob in glyph_i) / (len(glyph_i) * 63)
        prob_j = sum(OffBit.get_activation_layer(ob) for ob in glyph_j) / (len(glyph_j) * 63)
        
        # Calculate intersection probability (simplified as minimum)
        prob_intersection = min(prob_i, prob_j)
        
        # Avoid division by zero
        if prob_intersection == 0:
            correlation = 0.0
        else:
            correlation = (prob_i * prob_j) / prob_intersection
        
        # Create result OffBit with correlation value
        correlation_activation = int(correlation * 63) % 64
        result = OffBit.create_offbit(activation=correlation_activation)
        
        coherence_change = correlation * 0.1  # Correlation contributes to coherence
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_CORRELATE",
            input_values=glyph_i + glyph_j,
            coherence_change=coherence_change,
            energy_delta=correlation * 0.2,
            execution_time=execution_time
        )
    
    def glyph_self_reference_operation(self, glyph_state: List[int], 
                                     recursion_depth: int = 1,
                                     **kwargs) -> ToggleOperationResult:
        """
        Perform glyph self-reference with recursive feedback.
        
        Args:
            glyph_state: Glyph state for self-reference
            recursion_depth: Depth of recursive self-reference
            
        Returns:
            ToggleOperationResult with self-referenced glyph
        """
        import time
        start_time = time.time()
        
        if not glyph_state:
            raise ValueError("Glyph state cannot be empty")
        
        current_state = glyph_state.copy()
        
        # Apply recursive self-reference
        for depth in range(recursion_depth):
            # Self-reference by applying glyph to itself
            quantify_result = self.glyph_quantify_operation(current_state)
            
            # Update state with self-reference
            for i in range(len(current_state)):
                # Blend original with self-reference
                original_activation = OffBit.get_activation_layer(current_state[i])
                reference_activation = OffBit.get_activation_layer(quantify_result.result_value)
                
                blended_activation = (original_activation + reference_activation) // 2
                current_state[i] = OffBit.set_activation_layer(current_state[i], blended_activation)
        
        # Final result is the first OffBit of the self-referenced state
        result = current_state[0] if current_state else 0
        
        # Calculate coherence change
        avg_coherence_before = sum(OffBit.calculate_coherence(ob) for ob in glyph_state) / len(glyph_state)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - avg_coherence_before
        
        execution_time = time.time() - start_time
        
        return ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_SELF_REFERENCE",
            input_values=glyph_state,
            coherence_change=coherence_change,
            energy_delta=recursion_depth * 0.3,
            execution_time=execution_time
        )
    
    # ========================================================================
    # HIGH-LEVEL OPERATION INTERFACE
    # ========================================================================
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> ToggleOperationResult:
        """
        Execute a named toggle algebra operation.
        
        Args:
            operation_name: Name of the operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            ToggleOperationResult with operation result
        """
        if operation_name not in self.operations:
            available = list(self.operations.keys())
            raise KeyError(f"Unknown operation '{operation_name}'. Available: {available}")
        
        operation_func = self.operations[operation_name]
        result = operation_func(*args, **kwargs)
        
        self._record_operation(result)
        return result
    
    def batch_execute(self, operations: List[Tuple[str, tuple, dict]]) -> List[ToggleOperationResult]:
        """
        Execute a batch of toggle algebra operations.
        
        Args:
            operations: List of (operation_name, args, kwargs) tuples
            
        Returns:
            List of ToggleOperationResults
        """
        results = []
        
        for operation_name, args, kwargs in operations:
            try:
                result = self.execute_operation(operation_name, *args, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = ToggleOperationResult(
                    result_value=None,
                    operation_type=operation_name,
                    input_values=list(args),
                    coherence_change=0.0,
                    energy_delta=0.0,
                    execution_time=0.0
                )
                results.append(error_result)
        
        return results
    
    def get_metrics(self) -> ToggleAlgebraMetrics:
        """Get current toggle algebra performance metrics."""
        return self.metrics
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[ToggleOperationResult]:
        """
        Get operation history.
        
        Args:
            limit: Maximum number of recent operations to return
            
        Returns:
            List of recent ToggleOperationResults
        """
        if limit is None:
            return self.operation_history.copy()
        else:
            return self.operation_history[-limit:]
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics and operation history."""
        self.operation_history = []
        self.metrics = ToggleAlgebraMetrics(
            total_operations=0,
            successful_operations=0,
            average_coherence=0.0,
            total_energy_change=0.0,
            operation_distribution={},
            average_execution_time=0.0,
            resonance_stability=1.0
        )
        print("🔄 Toggle Algebra metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the Toggle Algebra engine."""
        return {
            "available_operations": list(self.operations.keys()),
            "total_operations": self.metrics.total_operations,
            "successful_operations": self.metrics.successful_operations,
            "success_rate": (self.metrics.successful_operations / max(1, self.metrics.total_operations)),
            "average_coherence": self.metrics.average_coherence,
            "total_energy_change": self.metrics.total_energy_change,
            "average_execution_time": self.metrics.average_execution_time,
            "operation_distribution": self.metrics.operation_distribution,
            "bitfield_connected": self.bitfield is not None
        }


if __name__ == "__main__":
    # Test the Toggle Algebra engine
    print("="*60)
    print("UBP TOGGLE ALGEBRA MODULE TEST")
    print("="*60)
    
    # Create Toggle Algebra engine
    ta = ToggleAlgebra()
    
    # Create test OffBits
    offbit1 = OffBit.create_offbit(reality=15, information=31, activation=7, unactivated=3)
    offbit2 = OffBit.create_offbit(reality=8, information=16, activation=12, unactivated=5)
    
    print(f"Test OffBit 1: {offbit1:032b}")
    print(f"Test OffBit 2: {offbit2:032b}")
    
    # Test basic operations
    print("\n--- Basic Operations ---")
    and_result = ta.execute_operation("AND", offbit1, offbit2)
    print(f"AND result: {and_result.result_value:032b}, coherence change: {and_result.coherence_change:.6f}")
    
    xor_result = ta.execute_operation("XOR", offbit1, offbit2)
    print(f"XOR result: {xor_result.result_value:032b}, coherence change: {xor_result.coherence_change:.6f}")
    
    # Test advanced operations
    print("\n--- Advanced Operations ---")
    resonance_result = ta.execute_operation("RESONANCE", offbit1, time=0.001, frequency=1e6)
    print(f"Resonance result: {resonance_result.result_value:032b}, energy delta: {resonance_result.energy_delta:.6f}")
    
    entanglement_result = ta.execute_operation("ENTANGLEMENT", offbit1, offbit2, coherence_factor=0.8)
    print(f"Entanglement result: {entanglement_result.result_value:032b}, coherence change: {entanglement_result.coherence_change:.6f}")
    
    # Test superposition
    states = [offbit1, offbit2]
    weights = [0.6, 0.4]
    superposition_result = ta.execute_operation("SUPERPOSITION", states, weights)
    print(f"Superposition result: {superposition_result.result_value:032b}")
    
    # Test Rune Protocol operations
    print("\n--- Rune Protocol Operations ---")
    glyph_state = [offbit1, offbit2]
    quantify_result = ta.execute_operation("GLYPH_QUANTIFY", glyph_state)
    print(f"Glyph quantify result: {quantify_result.result_value:032b}")
    
    # Test batch execution
    print("\n--- Batch Execution ---")
    batch_ops = [
        ("AND", (offbit1, offbit2), {}),
        ("OR", (offbit1, offbit2), {}),
        ("XOR", (offbit1, offbit2), {})
    ]
    batch_results = ta.batch_execute(batch_ops)
    print(f"Batch executed {len(batch_results)} operations")
    
    # Get metrics
    metrics = ta.get_metrics()
    print(f"\n--- Performance Metrics ---")
    print(f"Total operations: {metrics.total_operations}")
    print(f"Success rate: {metrics.successful_operations / max(1, metrics.total_operations):.3f}")
    print(f"Average coherence: {metrics.average_coherence:.6f}")
    print(f"Average execution time: {metrics.average_execution_time:.6f}s")
    
    # Get status
    status = ta.get_status()
    print(f"\nToggle Algebra Status:")
    print(f"  Available operations: {len(status['available_operations'])}")
    print(f"  Success rate: {status['success_rate']:.3f}")
    
    print("\n✅ Toggle Algebra module test completed successfully!")

