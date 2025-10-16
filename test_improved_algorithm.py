#!/usr/bin/env python3
"""
IMPROVED ALGORITHM DYNAMIC SCENARIO TEST
=======================================

Test the new optimized adaptive algorithm against the dynamic traffic scenarios
to measure performance improvements.
"""

import sys
import os
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')

def run_improved_algorithm_test():
    """Run the dynamic scenario test with the improved algorithm"""
    
    print("🚀 TESTING IMPROVED ADAPTIVE ALGORITHM")
    print("=" * 60)
    print("📊 Running dynamic scenario with optimized controller...")
    print("   • Dynamic adaptation intervals (15-50s)")
    print("   • Traffic-aware change limits (6-35%)")
    print("   • Enhanced traffic categorization")
    print("   • Critical situation detection")
    print()
    
    # Change to src directory and run the test
    os.chdir('src')
    
    try:
        # Import and run the dynamic scenario test  
        from dynamic_scenario_test import DynamicScenarioTestSrc
        
        # Initialize test
        test = DynamicScenarioTestSrc()
        
        print("🎯 PHASE 1: Running Normal Mode (Baseline)")
        print("-" * 45)
        test.run_normal_mode_simulation()
        
        print("\n🚀 PHASE 2: Running Improved Adaptive Mode")
        print("-" * 50)
        test.run_adaptive_mode_simulation()
        
        print("\n📊 PHASE 3: Analyzing Results and Creating Graphs")
        print("-" * 55)
        test.analyze_and_visualize_results()
        
        print("\n🎉 IMPROVED ALGORITHM TEST COMPLETED!")
        print("=" * 60)
        print("📂 Results saved in: src/scenario_results/")
        print("📊 Check the new performance graphs and summary report")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running improved algorithm test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Return to original directory
        os.chdir('..')

def compare_results():
    """Compare old vs new algorithm results"""
    
    print("\n🔍 COMPARING ALGORITHM PERFORMANCE")
    print("=" * 40)
    
    # Check if we have both old and new results
    old_results_file = "src/scenario_results/mixed_vehicle_summary_report.txt"
    new_results_file = "src/scenario_results/mixed_vehicle_summary_report.txt"
    
    if os.path.exists(old_results_file):
        print("📊 Performance comparison will be available after test completion")
        print("🔍 Key metrics to watch:")
        print("   • Waiting time improvements per phase")
        print("   • Adaptation count and effectiveness")
        print("   • Overall performance ratio changes")
        print("   • Critical situation handling")
    else:
        print("ℹ️  No previous results found for comparison")
        print("   This will establish the baseline for future comparisons")

def create_performance_summary():
    """Create a summary of expected vs actual improvements"""
    
    summary_text = """
IMPROVED ADAPTIVE ALGORITHM TEST SUMMARY
========================================

Expected Improvements:
---------------------
• Light Traffic: From -6641% to -1000% to -2000% (major improvement)
• Heavy North: From -5404% to -500% to -1000% (significant improvement)  
• Heavy East: From -8002% to -800% to -1500% (major improvement)
• Rush Hour: From -72.7% to +10% to +30% (becomes beneficial)

Key Algorithm Changes:
---------------------
1. Dynamic Adaptation Intervals:
   - CRITICAL situations: 15s response time
   - URGENT traffic: 20s response time
   - NORMAL traffic: 30s response time  
   - LIGHT traffic: 40s response time
   - MINIMAL traffic: 50s response time

2. Traffic-Aware Change Limits:
   - CRITICAL: 35% change allowed
   - URGENT: 25% change allowed
   - NORMAL: 18% change allowed
   - LIGHT: 12% change allowed
   - MINIMAL: 6% change allowed

3. Enhanced Traffic Detection:
   - Multi-factor categorization (vehicles + waiting time + speed)
   - Critical situation detection (>45s wait, >15 vehicles, stopped traffic)
   - Urgency assessment for appropriate response levels

4. Smart Timing Optimization:
   - Urgency-based weight adjustments
   - Speed-based traffic prioritization
   - Maximum timing limits to prevent excessive green phases
   - Minimum safety intervals for critical overrides

Performance Monitoring:
----------------------
- Track adaptation success rates
- Monitor actual improvement ratios
- Log critical situation responses
- Compare against baseline normal mode

Test Results:
------------
[Results will be populated after test completion]
"""
    
    with open("improved_algorithm_test_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print("📄 Test summary template created: improved_algorithm_test_summary.txt")

def main():
    """Main function to run the improved algorithm test"""
    
    print("🧪 IMPROVED ADAPTIVE ALGORITHM TESTING SUITE")
    print("=" * 65)
    
    # Create performance summary template
    create_performance_summary()
    
    # Run the actual test
    print("\n🚀 STARTING COMPREHENSIVE DYNAMIC SCENARIO TEST...")
    success = run_improved_algorithm_test()
    
    if success:
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        compare_results()
        
        print("\n📈 NEXT STEPS:")
        print("1. Review the performance graphs in src/scenario_results/")
        print("2. Check the summary report for detailed metrics") 
        print("3. Compare waiting times across all 6 traffic phases")
        print("4. Analyze adaptation frequency and effectiveness")
        print("5. Look for critical situation handling improvements")
        
        print("\n🎯 EXPECTED OUTCOMES:")
        print("• 40-70% reduction in waiting times during light/moderate traffic")
        print("• 60-90% improvement during heavy traffic scenarios")
        print("• Better responsiveness to critical traffic situations")
        print("• More appropriate adaptation timing and frequency")
        
    else:
        print("\n❌ TEST FAILED - Check error messages above")
        print("🔧 Troubleshooting steps:")
        print("1. Ensure SUMO is properly installed and in PATH")
        print("2. Check that all required modules are in src/ directory")
        print("3. Verify file permissions for results directory")

if __name__ == "__main__":
    main()