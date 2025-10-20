"""
Enhanced Traffic Management Complete Runner
==========================================

This script runs the complete enhanced 12-hour simulation and generates
all analysis including separate graph files and percentage improvements.
"""

import os
import sys
from enhanced_traffic_management_12hour import Enhanced12HourTrafficSimulation
from enhanced_analysis_generator import TrafficAnalysisGenerator

def run_complete_enhanced_simulation():
    """Run the complete enhanced traffic management analysis."""
    
    print("🚀 STARTING COMPLETE ENHANCED 12-HOUR TRAFFIC MANAGEMENT ANALYSIS")
    print("=" * 120)
    print("This comprehensive analysis includes:")
    print("   ✅ 12-hour simulation with 12 realistic traffic phases")
    print("   ✅ Enhanced adaptive algorithm with sophisticated edge decisions")
    print("   ✅ Detailed statistical analysis and performance metrics")
    print("   ✅ Individual graph files for each analysis type")
    print("   ✅ Comprehensive percentage improvement analysis")
    print("   ✅ Professional analysis report")
    print()
    print("⏱️ Expected completion time: 3-5 minutes")
    print()
    
    try:
        # Step 1: Run enhanced 12-hour simulation
        print("STEP 1: Running Enhanced 12-Hour Simulation")
        print("-" * 60)
        
        simulator = Enhanced12HourTrafficSimulation()
        simulation_results = simulator.run_enhanced_simulation()
        
        results_file = os.path.join(simulator.results_dir, "enhanced_12hour_complete.json")
        
        print(f"✅ Simulation complete! Results saved to: {results_file}")
        print()
        
        # Step 2: Generate comprehensive analysis
        print("STEP 2: Generating Comprehensive Analysis and Visualizations")
        print("-" * 70)
        
        analyzer = TrafficAnalysisGenerator(simulator.results_dir)
        analysis_results = analyzer.generate_all_analysis(results_file)
        
        print()
        print("STEP 3: Summary of Generated Files")
        print("-" * 40)
        
        print("📊 Individual Graph Files:")
        for i, graph_file in enumerate(analysis_results['graph_files'], 1):
            graph_name = os.path.basename(graph_file)
            print(f"   {i}. {graph_name}")
        
        print(f"\\n📋 Analysis Report:")
        print(f"   → {os.path.basename(analysis_results['report_file'])}")
        
        print(f"\\n📁 All files location:")
        print(f"   → {simulator.results_dir}")
        
        # Display key results
        print()
        print("STEP 4: Key Performance Results")
        print("-" * 35)
        
        stats = analysis_results['statistical_analysis']
        phase_results = analysis_results['phase_analysis']
        
        print(f"🏆 OVERALL PERFORMANCE:")
        print(f"   ⏱️ Waiting Time Improvement: {stats['improvements']['waiting_time']['mean']:+.1f}%")
        print(f"   🚗 Throughput Improvement: {stats['improvements']['throughput']['mean']:+.1f}%")
        print(f"   🏃 Speed Improvement: {stats['improvements']['speed']['mean']:+.1f}%")
        print(f"   📊 Phase Victories: {phase_results['adaptive_wins']}/{phase_results['total_phases']}")
        
        if stats['improvements']['waiting_time']['mean'] > 0:
            print(f"\\n🎉 SUCCESS! Adaptive mode shows consistent improvement!")
        
        print(f"\\n✅ ENHANCED 12-HOUR ANALYSIS COMPLETE!")
        print(f"📈 Check the individual graphs for detailed insights!")
        print(f"📋 Review the comprehensive report for full analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_enhanced_simulation()
    
    if success:
        print("\\n🎯 All analysis completed successfully!")
        print("Check the 'enhanced_12hour_results' folder for all generated files.")
    else:
        print("\\n⚠️ Analysis encountered errors. Please check the output above.")
        sys.exit(1)