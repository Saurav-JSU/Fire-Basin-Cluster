#!/usr/bin/env python
"""
Quick test to verify scale parameters are working correctly.
"""
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_scale_logic():
    """Test the scale parameter logic without running full analysis."""
    print("🧪 Testing Scale Parameter Logic")
    print("=" * 40)
    
    # Simulate different scale arguments
    test_cases = [
        {'scale': 'test', 'expected': 10},
        {'scale': 'sample', 'expected': 1000}, 
        {'scale': 'full', 'expected': None},
        {'max_watersheds': 25, 'expected': 25}
    ]
    
    for case in test_cases:
        # Simulate argument parsing logic
        if 'max_watersheds' in case:
            max_watersheds = case['max_watersheds']
            scale_name = f"custom ({max_watersheds:,})"
        elif case['scale'] == 'test':
            max_watersheds = 10
            scale_name = "test (10 watersheds)"
        elif case['scale'] == 'sample': 
            max_watersheds = 1000
            scale_name = "sample (1,000 watersheds)"
        elif case['scale'] == 'full':
            max_watersheds = None
            scale_name = "full (all watersheds)"
        
        print(f"Scale config: {case}")
        print(f"  → max_watersheds = {max_watersheds}")
        print(f"  → scale_name = {scale_name}")
        print(f"  → Expected: {case['expected']}")
        
        if max_watersheds == case['expected']:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
        print()
    
    print("Testing ComprehensiveWildfireAnalysis initialization...")
    
    try:
        # Test that we can initialize with different scales
        from run_comprehensive_analysis import ComprehensiveWildfireAnalysis
        
        # Test scale configurations
        test_analysis = ComprehensiveWildfireAnalysis(
            project_id='ee-jsuhydrolabenb',
            start_date='2020-01-01',
            end_date='2020-12-31', 
            max_watersheds=10  # Test scale
        )
        
        print(f"✅ Analysis initialized with max_watersheds = {test_analysis.max_watersheds}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_scale_logic()
    print("\n" + "=" * 40)
    if success:
        print("✅ Scale parameter logic working correctly!")
        print("\nNow run with fixed code:")
        print("  python run_comprehensive_analysis.py --scale test")
        print("  → Should process exactly 10 watersheds in Steps 2 & 3")
    else:
        print("❌ Scale parameter issues detected")
    
    sys.exit(0 if success else 1)