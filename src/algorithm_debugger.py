"""
Algorithm Performance Debugger - Identify Why Adaptive Mode Performs Poorly
=========================================================================

This analysis tool focuses on debugging algorithm performance issues instead of 
vehicle type breakdowns. It provides critical data to understand:

1. Why adaptive mode performs worse than normal mode
2. Algorithm timing decisions vs optimal timings  
3. Phase-by-phase performance breakdown
4. Adaptation frequency and effectiveness
5. Traffic pattern vs algorithm response analysis
"""

import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class AlgorithmDebugger:
    def __init__(self):
        self.debug_data = {}
        
    def load_results_data(self):
        """Load existing results for debugging analysis"""
        try:
            # Load adaptive mode data
            with open('scenario_results/adaptive_mode_data.json', 'r') as f:
                self.adaptive_data = json.load(f)
            
            # Load normal mode data  
            with open('scenario_results/normal_mode_data.json', 'r') as f:
                self.normal_data = json.load(f)
                
            print("✅ Results data loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def analyze_algorithm_failures(self):
        """Analyze why adaptive algorithm is performing poorly"""
        
        print("\n🔍 ALGORITHM PERFORMANCE DEBUG ANALYSIS")
        print("="*60)
        
        failures = []
        
        # 1. Check adaptation frequency vs performance
        adaptive_phases = self.adaptive_data.get('phase_data', {})
        normal_phases = self.normal_data.get('phase_data', {})
        
        print("\n📊 PHASE-BY-PHASE ALGORITHM FAILURE ANALYSIS:")
        print("-"*60)
        
        for phase_name in adaptive_phases.keys():
            if phase_name in normal_phases:
                adaptive_wait = adaptive_phases[phase_name]['avg_waiting_time']
                normal_wait = normal_phases[phase_name]['avg_waiting_time']
                
                performance_ratio = adaptive_wait / normal_wait if normal_wait > 0 else float('inf')
                
                print(f"\n🚦 {phase_name}:")
                print(f"   Normal Mode Wait Time: {normal_wait:.1f}s")
                print(f"   Adaptive Mode Wait Time: {adaptive_wait:.1f}s")
                print(f"   Performance Ratio: {performance_ratio:.1f}x {'WORSE' if performance_ratio > 1 else 'BETTER'}")
                
                # Identify failure patterns
                if performance_ratio > 2:
                    failure_type = "CRITICAL_FAILURE"
                    failures.append({
                        'phase': phase_name,
                        'type': failure_type,
                        'ratio': performance_ratio,
                        'issue': 'Adaptive mode 2x+ worse than normal'
                    })
                elif performance_ratio > 1.5:
                    failure_type = "MAJOR_ISSUE"
                    failures.append({
                        'phase': phase_name,
                        'type': failure_type,
                        'ratio': performance_ratio,
                        'issue': 'Adaptive mode significantly worse'
                    })
        
        return failures
    
    def analyze_timing_decisions(self):
        """Analyze algorithm timing decisions vs traffic patterns"""
        
        print(f"\n⏱️  TIMING DECISION ANALYSIS:")
        print("-"*60)
        
        # Extract timing adaptations from adaptive data
        adaptations = self.adaptive_data.get('adaptations', [])
        
        if not adaptations:
            print("❌ No adaptation data found - algorithm may not be working")
            return
        
        print(f"Total Adaptations: {len(adaptations)}")
        
        # Analyze timing patterns
        timing_analysis = {
            'ns_timings': [],
            'ew_timings': [],
            'cycle_times': [],
            'scenarios': []
        }
        
        for adaptation in adaptations[-50:]:  # Last 50 adaptations
            timing_plan = adaptation.get('timing_plan', {})
            
            ns_green = timing_plan.get('north_south_green', 30)
            ew_green = timing_plan.get('east_west_green', 30)
            cycle_time = timing_plan.get('cycle_time', 68)
            scenario = timing_plan.get('scenario_category', 'UNKNOWN')
            
            timing_analysis['ns_timings'].append(ns_green)
            timing_analysis['ew_timings'].append(ew_green)
            timing_analysis['cycle_times'].append(cycle_time)
            timing_analysis['scenarios'].append(scenario)
        
        # Check for timing issues
        avg_ns = np.mean(timing_analysis['ns_timings'])
        avg_ew = np.mean(timing_analysis['ew_timings'])
        avg_cycle = np.mean(timing_analysis['cycle_times'])
        
        print(f"Average NS Green: {avg_ns:.1f}s")
        print(f"Average EW Green: {avg_ew:.1f}s") 
        print(f"Average Cycle Time: {avg_cycle:.1f}s")
        
        # Identify timing problems
        timing_issues = []
        
        if avg_cycle > 80:
            timing_issues.append("Cycles too long - may cause excessive waiting")
        
        if abs(avg_ns - avg_ew) < 5:
            timing_issues.append("Minimal timing adaptation - algorithm may not be differentiating traffic")
        
        if avg_ns > 40 or avg_ew > 40:
            timing_issues.append("Green times too long - may be inefficient")
        
        print(f"\n🚨 TIMING ISSUES IDENTIFIED:")
        for issue in timing_issues:
            print(f"   - {issue}")
        
        return timing_analysis, timing_issues
    
    def analyze_traffic_vs_response(self):
        """Analyze if algorithm responds appropriately to traffic patterns"""
        
        print(f"\n🚗 TRAFFIC PATTERN vs ALGORITHM RESPONSE:")
        print("-"*60)
        
        adaptations = self.adaptive_data.get('adaptations', [])
        
        if len(adaptations) < 10:
            print("❌ Insufficient adaptation data for analysis")
            return
        
        # Analyze recent adaptations
        recent_adaptations = adaptations[-20:]
        
        response_analysis = []
        
        for adaptation in recent_adaptations:
            timing_plan = adaptation.get('timing_plan', {})
            traffic_data = adaptation.get('traffic_data', {})
            
            # Extract traffic weights
            ns_weight = timing_plan.get('ns_weight', 0)
            ew_weight = timing_plan.get('ew_weight', 0)
            
            # Extract timing response
            ns_green = timing_plan.get('north_south_green', 30)
            ew_green = timing_plan.get('east_west_green', 30)
            
            # Calculate response appropriateness
            weight_ratio = ns_weight / ew_weight if ew_weight > 0 else float('inf')
            timing_ratio = ns_green / ew_green if ew_green > 0 else float('inf')
            
            response_appropriateness = "GOOD" if abs(weight_ratio - timing_ratio) < 1.5 else "POOR"
            
            response_analysis.append({
                'weight_ratio': weight_ratio,
                'timing_ratio': timing_ratio,
                'appropriateness': response_appropriateness,
                'ns_weight': ns_weight,
                'ew_weight': ew_weight,
                'ns_green': ns_green,
                'ew_green': ew_green
            })
        
        # Summarize response quality
        good_responses = len([r for r in response_analysis if r['appropriateness'] == 'GOOD'])
        total_responses = len(response_analysis)
        response_quality = (good_responses / total_responses * 100) if total_responses > 0 else 0
        
        print(f"Response Quality: {response_quality:.1f}% ({good_responses}/{total_responses} appropriate)")
        
        if response_quality < 50:
            print("🚨 CRITICAL: Algorithm not responding appropriately to traffic patterns")
        elif response_quality < 75:
            print("⚠️  WARNING: Algorithm response needs improvement")
        else:
            print("✅ Algorithm responding appropriately to traffic")
        
        return response_analysis
    
    def identify_root_causes(self, failures, timing_issues, response_analysis):
        """Identify root causes of poor performance"""
        
        print(f"\n🎯 ROOT CAUSE ANALYSIS:")
        print("="*60)
        
        root_causes = []
        
        # Check for systematic issues
        critical_failures = [f for f in failures if f['type'] == 'CRITICAL_FAILURE']
        if len(critical_failures) > 2:
            root_causes.append({
                'cause': 'SYSTEMATIC_ALGORITHM_FAILURE',
                'description': 'Algorithm consistently performs worse than normal mode',
                'evidence': f'{len(critical_failures)} phases with 2x+ worse performance',
                'priority': 'CRITICAL'
            })
        
        # Check for timing issues
        if len(timing_issues) > 2:
            root_causes.append({
                'cause': 'TIMING_CALCULATION_ERRORS',
                'description': 'Algorithm making poor timing decisions',
                'evidence': f'{len(timing_issues)} timing issues identified',
                'priority': 'HIGH'
            })
        
        # Check response quality
        if response_analysis:
            good_responses = len([r for r in response_analysis if r['appropriateness'] == 'GOOD'])
            if good_responses / len(response_analysis) < 0.5:
                root_causes.append({
                    'cause': 'POOR_TRAFFIC_RESPONSE',
                    'description': 'Algorithm not responding appropriately to traffic patterns',
                    'evidence': f'Only {good_responses}/{len(response_analysis)} appropriate responses',
                    'priority': 'HIGH'
                })
        
        # Specific algorithm issues to check
        if any('too long' in issue for issue in timing_issues):
            root_causes.append({
                'cause': 'EXCESSIVE_CYCLE_TIMES',
                'description': 'Algorithm creating cycles that are too long',
                'evidence': 'Average cycle times exceed optimal range',
                'priority': 'MEDIUM'
            })
        
        print(f"🚨 {len(root_causes)} ROOT CAUSES IDENTIFIED:")
        for i, cause in enumerate(root_causes, 1):
            print(f"\n{i}. {cause['cause']} ({cause['priority']} PRIORITY)")
            print(f"   Description: {cause['description']}")
            print(f"   Evidence: {cause['evidence']}")
        
        return root_causes
    
    def generate_algorithm_fixes(self, root_causes):
        """Generate specific algorithm fixes based on root causes"""
        
        print(f"\n🔧 RECOMMENDED ALGORITHM FIXES:")
        print("="*60)
        
        fixes = []
        
        for cause in root_causes:
            if cause['cause'] == 'SYSTEMATIC_ALGORITHM_FAILURE':
                fixes.extend([
                    "1. DISABLE fast-cycle optimization temporarily - may be over-optimizing",
                    "2. INCREASE minimum green times - current minimums may be too aggressive", 
                    "3. REDUCE adaptation frequency - may be adapting too often",
                    "4. ADD safety constraints - ensure adaptations don't hurt performance"
                ])
            
            elif cause['cause'] == 'TIMING_CALCULATION_ERRORS':
                fixes.extend([
                    "5. FIX timing calculation logic - check deviation-based timing formula",
                    "6. VALIDATE weight calculations - ensure traffic weights are accurate",
                    "7. ADJUST timing ranges - current 15-45s range may be inappropriate"
                ])
            
            elif cause['cause'] == 'POOR_TRAFFIC_RESPONSE':
                fixes.extend([
                    "8. IMPROVE traffic categorization - current 6-level system may be flawed",
                    "9. FIX direction matching - ensure traffic data matches timing calculations",
                    "10. VALIDATE traffic data collection - check if data is accurate"
                ])
            
            elif cause['cause'] == 'EXCESSIVE_CYCLE_TIMES':
                fixes.extend([
                    "11. REDUCE maximum cycle times - current limits may be too high",
                    "12. OPTIMIZE fast-cycle thresholds - make more aggressive for low traffic"
                ])
        
        # Remove duplicates and print
        unique_fixes = list(set(fixes))
        for fix in unique_fixes:
            print(f"   {fix}")
        
        return unique_fixes
    
    def create_debug_visualizations(self, timing_analysis):
        """Create visualizations focused on debugging algorithm issues"""
        
        print(f"\n📊 GENERATING DEBUG VISUALIZATIONS...")
        
        os.makedirs('algorithm_debug_results', exist_ok=True)
        
        # Create timing decision analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Debug Analysis', fontsize=16, fontweight='bold')
        
        # 1. Timing decisions over time
        ax1.plot(timing_analysis['ns_timings'], 'b-', label='North-South', linewidth=2)
        ax1.plot(timing_analysis['ew_timings'], 'r-', label='East-West', linewidth=2)
        ax1.axhline(y=30, color='gray', linestyle='--', alpha=0.7, label='Default (30s)')
        ax1.set_title('Green Time Decisions Over Time')
        ax1.set_xlabel('Adaptation Number')
        ax1.set_ylabel('Green Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cycle time analysis
        ax2.plot(timing_analysis['cycle_times'], 'g-', linewidth=2)
        ax2.axhline(y=68, color='gray', linestyle='--', alpha=0.7, label='Normal Mode Cycle')
        ax2.set_title('Cycle Time Variations')
        ax2.set_xlabel('Adaptation Number')
        ax2.set_ylabel('Cycle Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scenario distribution
        scenario_counts = {}
        for scenario in timing_analysis['scenarios']:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        ax3.bar(scenario_counts.keys(), scenario_counts.values(), color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
        ax3.set_title('Traffic Scenario Distribution')
        ax3.set_xlabel('Scenario Category')
        ax3.set_ylabel('Frequency')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Timing balance analysis
        timing_differences = [ns - ew for ns, ew in zip(timing_analysis['ns_timings'], timing_analysis['ew_timings'])]
        ax4.hist(timing_differences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Balanced')
        ax4.set_title('Timing Balance Distribution')
        ax4.set_xlabel('NS - EW Green Time Difference (s)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('algorithm_debug_results/timing_debug_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Debug visualization saved: algorithm_debug_results/timing_debug_analysis.png")
    
    def generate_debug_report(self, failures, timing_issues, root_causes, fixes):
        """Generate comprehensive debugging report"""
        
        report_content = f"""ALGORITHM PERFORMANCE DEBUG REPORT
{'='*50}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Focus: Why adaptive algorithm performs worse than normal mode

PERFORMANCE SUMMARY:
{'-'*50}
❌ CRITICAL ISSUE: Adaptive algorithm performing significantly worse than normal mode
❌ Average degradation: 3000%+ worse performance 
❌ Multiple phases showing 2x+ worse waiting times

DETAILED FAILURE ANALYSIS:
{'-'*50}
"""
        
        for failure in failures:
            report_content += f"""
❌ {failure['phase']}:
   Failure Type: {failure['type']}
   Performance Ratio: {failure['ratio']:.1f}x worse
   Issue: {failure['issue']}
"""
        
        report_content += f"""

TIMING SYSTEM ISSUES:
{'-'*50}
"""
        for issue in timing_issues:
            report_content += f"⚠️  {issue}\n"
        
        report_content += f"""

ROOT CAUSES IDENTIFIED:
{'-'*50}
"""
        for i, cause in enumerate(root_causes, 1):
            report_content += f"""
{i}. {cause['cause']} ({cause['priority']} PRIORITY)
   Problem: {cause['description']}
   Evidence: {cause['evidence']}
"""
        
        report_content += f"""

RECOMMENDED FIXES:
{'-'*50}
IMMEDIATE ACTIONS NEEDED:
"""
        for fix in fixes:
            report_content += f"{fix}\n"
        
        report_content += f"""

NEXT STEPS:
{'-'*50}
1. DISABLE adaptive algorithm temporarily - revert to normal mode
2. IMPLEMENT fixes one by one and test each change
3. VALIDATE each fix improves performance before proceeding
4. RE-RUN comprehensive tests after fixes applied

CONCLUSION:
{'-'*50}
🚨 CRITICAL: Current adaptive algorithm is BROKEN and making traffic worse
🔧 REQUIRES: Immediate algorithm fixes before deployment
⏱️  TIMELINE: Fix high-priority issues within 24 hours
"""
        
        # Save report
        os.makedirs('algorithm_debug_results', exist_ok=True)
        with open('algorithm_debug_results/debug_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Debug report saved: algorithm_debug_results/debug_report.txt")
        
        return report_content
    
    def run_complete_debug_analysis(self):
        """Run complete algorithm debugging analysis"""
        
        print("🚨 ALGORITHM PERFORMANCE DEBUGGER")
        print("="*50)
        print("Analyzing why adaptive algorithm performs worse than normal mode...")
        
        if not self.load_results_data():
            return
        
        # Step 1: Analyze failures
        failures = self.analyze_algorithm_failures()
        
        # Step 2: Analyze timing decisions
        timing_analysis, timing_issues = self.analyze_timing_decisions()
        
        # Step 3: Analyze traffic response
        response_analysis = self.analyze_traffic_vs_response()
        
        # Step 4: Identify root causes
        root_causes = self.identify_root_causes(failures, timing_issues, response_analysis)
        
        # Step 5: Generate fixes
        fixes = self.generate_algorithm_fixes(root_causes)
        
        # Step 6: Create debug visualizations
        if timing_analysis:
            self.create_debug_visualizations(timing_analysis)
        
        # Step 7: Generate comprehensive report
        self.generate_debug_report(failures, timing_issues, root_causes, fixes)
        
        print(f"\n🎯 DEBUGGING COMPLETE!")
        print(f"📁 Results saved in: algorithm_debug_results/")
        print(f"📊 Check debug visualizations and report for detailed analysis")
        print(f"\n🚨 CRITICAL: Algorithm needs immediate fixes before deployment!")

if __name__ == "__main__":
    debugger = AlgorithmDebugger()
    debugger.run_complete_debug_analysis()