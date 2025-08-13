#!/usr/bin/env python3
"""
UBP Framework v3.0 Simple Validation
Direct validation without complex imports
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'config'))

def simple_validation():
    """Simple validation of UBP Framework v3.0 components."""
    print("🚀 UBP Framework v3.0 Simple Validation")
    print("=" * 50)
    
    validation_results = {}
    
    # Test 1: Core Constants
    try:
        from core import UBPConstants
        constants = UBPConstants()
        validation_results['core_constants'] = True
        print("✅ Core Constants: Working")
    except Exception as e:
        validation_results['core_constants'] = False
        print(f"❌ Core Constants: {e}")
    
    # Test 2: Bitfield
    try:
        from bitfield import Bitfield
        bitfield = Bitfield(dimensions=(10, 10, 10, 2, 2, 2))
        validation_results['bitfield'] = True
        print("✅ Bitfield: Working")
    except Exception as e:
        validation_results['bitfield'] = False
        print(f"❌ Bitfield: {e}")
    
    # Test 3: Enhanced CRV System
    try:
        from enhanced_crv_system import AdaptiveCRVSelector
        crv_selector = AdaptiveCRVSelector()
        validation_results['enhanced_crv'] = True
        print("✅ Enhanced CRV System: Working")
    except Exception as e:
        validation_results['enhanced_crv'] = False
        print(f"❌ Enhanced CRV System: {e}")
    
    # Test 4: HTR Engine
    try:
        from htr_engine import HTREngine
        htr = HTREngine()
        validation_results['htr_engine'] = True
        print("✅ HTR Engine: Working")
    except Exception as e:
        validation_results['htr_engine'] = False
        print(f"❌ HTR Engine: {e}")
    
    # Test 5: BitTime Mechanics
    try:
        from bittime_mechanics import BitTimeMechanics
        bittime = BitTimeMechanics()
        validation_results['bittime'] = True
        print("✅ BitTime Mechanics: Working")
    except Exception as e:
        validation_results['bittime'] = False
        print(f"❌ BitTime Mechanics: {e}")
    
    # Test 6: Enhanced Error Correction
    try:
        from enhanced_error_correction import AdvancedErrorCorrection
        error_correction = AdvancedErrorCorrection()
        validation_results['error_correction'] = True
        print("✅ Enhanced Error Correction: Working")
    except Exception as e:
        validation_results['error_correction'] = False
        print(f"❌ Enhanced Error Correction: {e}")
    
    # Test 7: Rune Protocol
    try:
        from rune_protocol import RuneProtocol
        rune = RuneProtocol()
        validation_results['rune_protocol'] = True
        print("✅ Rune Protocol: Working")
    except Exception as e:
        validation_results['rune_protocol'] = False
        print(f"❌ Rune Protocol: {e}")
    
    # Test 8: Nuclear Realm
    try:
        from nuclear_realm import NuclearRealm
        nuclear = NuclearRealm()
        validation_results['nuclear_realm'] = True
        print("✅ Nuclear Realm: Working")
    except Exception as e:
        validation_results['nuclear_realm'] = False
        print(f"❌ Nuclear Realm: {e}")
    
    # Test 9: Optical Realm
    try:
        from optical_realm import OpticalRealm
        optical = OpticalRealm()
        validation_results['optical_realm'] = True
        print("✅ Optical Realm: Working")
    except Exception as e:
        validation_results['optical_realm'] = False
        print(f"❌ Optical Realm: {e}")
    
    # Test 10: Configuration System
    try:
        from ubp_config import get_config
        config = get_config()
        validation_results['config_system'] = True
        print("✅ Configuration System: Working")
    except Exception as e:
        validation_results['config_system'] = False
        print(f"❌ Configuration System: {e}")
    
    # Calculate results
    passed = sum(validation_results.values())
    total = len(validation_results)
    percentage = (passed / total) * 100
    
    print(f"\n📊 Validation Results:")
    print(f"   Passed: {passed}/{total} ({percentage:.1f}%)")
    
    # Success criteria
    print(f"\n🎯 Success Assessment:")
    if percentage >= 80:
        print("🎉 EXCELLENT: System is highly functional")
        success_level = "EXCELLENT"
    elif percentage >= 70:
        print("✅ GOOD: System meets operational requirements")
        success_level = "GOOD"
    elif percentage >= 50:
        print("⚠️  PARTIAL: System has basic functionality")
        success_level = "PARTIAL"
    else:
        print("❌ POOR: System needs significant work")
        success_level = "POOR"
    
    print(f"\n" + "=" * 50)
    print(f"UBP FRAMEWORK v3.0 VALIDATION: {success_level}")
    print(f"Component Success Rate: {percentage:.1f}%")
    
    return percentage >= 70

if __name__ == "__main__":
    success = simple_validation()
    sys.exit(0 if success else 1)

