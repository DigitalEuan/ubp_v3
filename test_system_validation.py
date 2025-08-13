#!/usr/bin/env python3
"""
UBP Framework v3.0 System Validation Test
Simple test to validate the complete system operation
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_system_validation():
    """Test complete UBP Framework v3.0 system validation."""
    print("🚀 UBP Framework v3.0 System Validation Test")
    print("=" * 60)
    
    try:
        # Import the framework
        from ubp_framework_v3 import create_ubp_framework_v3
        
        # Initialize the framework
        print("📋 Initializing UBP Framework v3.0...")
        ubp = create_ubp_framework_v3('development')
        print("✅ Framework initialized successfully")
        
        # Run system diagnostics
        print("\n🔍 Running system diagnostics...")
        diagnostics = ubp.run_system_diagnostics()
        
        # Display results
        print(f"\n📊 System Diagnostics Results:")
        print(f"   Version: {diagnostics['system_version']}")
        print(f"   Overall Health: {diagnostics['overall_health']}")
        
        # Component tests
        component_tests = diagnostics['component_tests']
        passed_components = sum(component_tests.values())
        total_components = len(component_tests)
        component_percentage = (passed_components / total_components) * 100
        
        print(f"\n🔧 Component Tests: {passed_components}/{total_components} ({component_percentage:.1f}%)")
        for component, status in component_tests.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")
        
        # Performance tests
        performance_tests = diagnostics['performance_tests']
        passed_performance = sum(performance_tests.values())
        total_performance = len(performance_tests)
        performance_percentage = (passed_performance / total_performance) * 100
        
        print(f"\n⚡ Performance Tests: {passed_performance}/{total_performance} ({performance_percentage:.1f}%)")
        for test, status in performance_tests.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {test}")
        
        # Integration tests
        integration_tests = diagnostics['integration_tests']
        passed_integration = sum(integration_tests.values())
        total_integration = len(integration_tests)
        integration_percentage = (passed_integration / total_integration) * 100
        
        print(f"\n🔗 Integration Tests: {passed_integration}/{total_integration} ({integration_percentage:.1f}%)")
        for test, status in integration_tests.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {test}")
        
        # Overall assessment
        print(f"\n🎯 Overall System Assessment:")
        print(f"   Component Validation: {component_percentage:.1f}% (Target: >70%)")
        print(f"   Performance Quality: {performance_percentage:.1f}%")
        print(f"   Integration Quality: {integration_percentage:.1f}%")
        print(f"   System Health: {diagnostics['overall_health'].upper()}")
        
        # Success criteria
        success_criteria = [
            ("Component Validation", component_percentage >= 70.0),
            ("Performance Quality", performance_percentage >= 50.0),
            ("Integration Quality", integration_percentage >= 50.0),
            ("System Health", diagnostics['overall_health'] == 'good')
        ]
        
        print(f"\n✅ Success Criteria:")
        all_passed = True
        for criterion, passed in success_criteria:
            status_icon = "✅" if passed else "❌"
            print(f"   {status_icon} {criterion}")
            if not passed:
                all_passed = False
        
        # Final result
        print(f"\n" + "=" * 60)
        if all_passed:
            print("🎉 UBP FRAMEWORK v3.0 VALIDATION: COMPLETE SUCCESS!")
            print("   System is ready for production deployment")
            return True
        else:
            print("⚠️  UBP FRAMEWORK v3.0 VALIDATION: PARTIAL SUCCESS")
            print("   System is functional but some criteria not met")
            return False
            
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_validation()
    sys.exit(0 if success else 1)

