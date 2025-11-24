"""
Simple Algorithm Performance Debugger
====================================

Analyzes why adaptive algorithm performs worse than normal mode.
Focuses on key metrics and provides actionable debugging information.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_and_analyze_results():
    """Load results and provide debugging analysis"""
    
    print("🚨 ALGORITHM PERFORMANCE ANALYSIS")
    print("="*50)
    
    try:
        # Load data files
        with open('dynamic_scenario_results/adaptive_mode_data.json', 'r') as f:
            adaptive_data = json.load(f)
        
        with open('dynamic_scenario_results/normal_mode_data.json', 'r') as f:
            normal_data = json.load(f)
        
        print("✅ Data loaded successfully")
        
        # Extract key metrics
        adaptive_waits = [d['avg_waiting_time'] for d in adaptive_data if d['avg_waiting_time'] > 0]
        normal_waits = [d['avg_waiting_time'] for d in normal_data if d['avg_waiting_time'] > 0]
        
        adaptive_adaptations = [d['adaptations'] for d in adaptive_data]
        
        # Calculate performance comparison
        avg_adaptive_wait = np.mean(adaptive_waits) if adaptive_waits else 0
        avg_normal_wait = np.mean(normal_waits) if normal_waits else 0
        
        performance_ratio = avg_adaptive_wait / avg_normal_wait if avg_normal_wait > 0 else float('inf')
        total_adaptations = max(adaptive_adaptations) if adaptive_adaptations else 0
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print("-"*30)
        print(f"Normal Mode Avg Wait: {avg_normal_wait:.1f}s")
        print(f"Adaptive Mode Avg Wait: {avg_adaptive_wait:.1f}s")
        print(f"Performance Ratio: {performance_ratio:.1f}x")
        print(f"Total Adaptations: {total_adaptations}")
        
        # Identify the problem
        print(f"\n🚨 PROBLEM ANALYSIS:")
        print("-"*30)
        
        if performance_ratio > 3:
            print("❌ CRITICAL FAILURE: Adaptive algorithm making traffic 3x+ worse!")
            print("❌ Root cause: Algorithm is fundamentally broken")
        elif performance_ratio > 2:
            print("❌ MAJOR ISSUE: Adaptive algorithm making traffic 2x+ worse!")
            print("❌ Root cause: Algorithm timing decisions are harmful")
        elif performance_ratio > 1.2:
            print("⚠️  MINOR ISSUE: Adaptive algorithm slightly worse than normal")
        else:
            print("✅ Algorithm performing as expected")
        
        # Analyze adaptation patterns
        print(f"\n⏱️  ADAPTATION ANALYSIS:")
        print("-"*30)
        
        if total_adaptations > 500:
            print(f"✅ High adaptation rate: {total_adaptations} adaptations")
            print("   Algorithm is actively adapting to traffic")
        else:
            print(f"⚠️  Low adaptation rate: {total_adaptations} adaptations")
            print("   Algorithm may not be detecting traffic changes")
        
        # Look at waiting time trends
        adaptive_times = [d['time'] for d in adaptive_data[-100:]]  # Last 100 data points
        adaptive_recent_waits = [d['avg_waiting_time'] for d in adaptive_data[-100:]]
        normal_recent_waits = [d['avg_waiting_time'] for d in normal_data[-100:]]
        
        avg_recent_adaptive = np.mean([w for w in adaptive_recent_waits if w > 0])
        avg_recent_normal = np.mean([w for w in normal_recent_waits if w > 0])
        
        print(f"\n📈 RECENT PERFORMANCE (Last 100 data points):")
        print("-"*30)
        print(f"Recent Normal Wait: {avg_recent_normal:.1f}s")
        print(f"Recent Adaptive Wait: {avg_recent_adaptive:.1f}s")
        print(f"Recent Ratio: {(avg_recent_adaptive/avg_recent_normal):.1f}x")
        
        # Generate specific diagnosis
        print(f"\n🎯 ALGORITHM DIAGNOSIS:")
        print("="*50)
        
        diagnoses = []
        
        if avg_adaptive_wait > 100:
            diagnoses.append("🚨 CRITICAL: Waiting times over 100s indicate severe congestion")
            diagnoses.append("   → Algorithm may be creating traffic jams instead of resolving them")
        
        if performance_ratio > 5:
            diagnoses.append("🚨 CRITICAL: 5x+ worse performance indicates fundamental algorithm failure")
            diagnoses.append("   → Algorithm timing logic is severely flawed")
        
        if total_adaptations > 700:
            diagnoses.append("⚠️  WARNING: Very high adaptation frequency may indicate instability")
            diagnoses.append("   → Algorithm may be over-adapting and causing oscillations")
        
        if avg_adaptive_wait > avg_normal_wait * 2:
            diagnoses.append("🚨 MAJOR: Adaptive algorithm making traffic significantly worse")
            diagnoses.append("   → Current algorithm should be disabled immediately")
        
        # Print diagnoses
        for diagnosis in diagnoses:
            print(diagnosis)
        
        if not diagnoses:
            print("✅ No critical issues detected in algorithm performance")
        
        # Generate immediate action items
        print(f"\n🔧 IMMEDIATE ACTIONS REQUIRED:")
        print("="*50)
        
        if performance_ratio > 3:
            print("1. 🚨 DISABLE adaptive algorithm immediately")
            print("2. 🚨 Revert to normal mode for all intersections")
            print("3. 🔍 Debug timing calculation logic")
            print("4. 🧪 Test individual algorithm components")
            print("5. 🚧 Do NOT deploy until performance improves")
        elif performance_ratio > 1.5:
            print("1. ⚠️  Suspend adaptive algorithm deployment")
            print("2. 🔍 Review timing parameters and thresholds")
            print("3. 🧪 Test with reduced adaptation frequency")
            print("4. 📊 Monitor performance closely")
        else:
            print("✅ Continue monitoring algorithm performance")
        
        # Create simple debugging visualization
        create_debug_plot(adaptive_data, normal_data)
        
        # Generate simple report
        generate_simple_report(avg_normal_wait, avg_adaptive_wait, performance_ratio, total_adaptations, diagnoses)
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

def create_debug_plot(adaptive_data, normal_data):
    """Create simple debugging visualization"""
    
    print(f"\n📊 Creating debug visualization...")
    
    os.makedirs('algorithm_debug_results', exist_ok=True)
    
    # Extract data for plotting
    adaptive_times = [d['time'] for d in adaptive_data[::10]]  # Every 10th point
    adaptive_waits = [d['avg_waiting_time'] for d in adaptive_data[::10]]
    adaptive_adaptations = [d['adaptations'] for d in adaptive_data[::10]]
    
    normal_times = [d['time'] for d in normal_data[::10]]
    normal_waits = [d['avg_waiting_time'] for d in normal_data[::10]]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Algorithm Performance Debug Analysis', fontsize=16, fontweight='bold')
    
    # Waiting time comparison
    ax1.plot(normal_times, normal_waits, 'g-', label='Normal Mode', linewidth=2, alpha=0.8)
    ax1.plot(adaptive_times, adaptive_waits, 'r-', label='Adaptive Mode', linewidth=2, alpha=0.8)
    ax1.set_title('Waiting Time Comparison: Normal vs Adaptive Mode')
    ax1.set_xlabel('Simulation Time (seconds)')
    ax1.set_ylabel('Average Waiting Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add performance annotations
    avg_normal = np.mean([w for w in normal_waits if w > 0])
    avg_adaptive = np.mean([w for w in adaptive_waits if w > 0])
    ratio = avg_adaptive / avg_normal if avg_normal > 0 else 1
    
    ax1.axhline(y=avg_normal, color='green', linestyle='--', alpha=0.7, label=f'Normal Avg: {avg_normal:.1f}s')
    ax1.axhline(y=avg_adaptive, color='red', linestyle='--', alpha=0.7, label=f'Adaptive Avg: {avg_adaptive:.1f}s')
    
    # Add problem indicator
    if ratio > 2:
        ax1.text(0.02, 0.95, f'🚨 PROBLEM: {ratio:.1f}x WORSE PERFORMANCE', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                verticalalignment='top', color='white')
    
    # Adaptations over time
    ax2.plot(adaptive_times, adaptive_adaptations, 'b-', linewidth=2)
    ax2.set_title('Algorithm Adaptations Over Time')
    ax2.set_xlabel('Simulation Time (seconds)')
    ax2.set_ylabel('Total Adaptations')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_debug_results/performance_debug.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Debug plot saved: algorithm_debug_results/performance_debug.png")

def generate_simple_report(normal_wait, adaptive_wait, ratio, adaptations, diagnoses):
    """Generate simple debugging report"""
    
    report = f"""ALGORITHM PERFORMANCE DEBUG REPORT
{'='*50}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE SUMMARY:
{'-'*30}
Normal Mode Average Wait Time: {normal_wait:.1f} seconds
Adaptive Mode Average Wait Time: {adaptive_wait:.1f} seconds
Performance Ratio: {ratio:.1f}x ({'WORSE' if ratio > 1 else 'BETTER'})
Total Algorithm Adaptations: {adaptations}

PROBLEM SEVERITY:
{'-'*30}
"""
    
    if ratio > 5:
        report += "🚨 CRITICAL FAILURE - Algorithm making traffic 5x+ worse\n"
    elif ratio > 3:
        report += "🚨 MAJOR FAILURE - Algorithm making traffic 3x+ worse\n"
    elif ratio > 2:
        report += "❌ SIGNIFICANT ISSUE - Algorithm making traffic 2x+ worse\n"
    elif ratio > 1.5:
        report += "⚠️  MODERATE ISSUE - Algorithm making traffic worse\n"
    elif ratio > 1.1:
        report += "⚠️  MINOR ISSUE - Algorithm slightly worse than normal\n"
    else:
        report += "✅ ACCEPTABLE PERFORMANCE\n"
    
    report += f"\nDIAGNOSES:\n{'-'*30}\n"
    for diagnosis in diagnoses:
        report += f"{diagnosis}\n"
    
    report += f"""

IMMEDIATE ACTIONS:
{'-'*30}
"""
    
    if ratio > 3:
        report += """🚨 CRITICAL ACTIONS:
1. DISABLE adaptive algorithm immediately
2. Revert all intersections to normal mode
3. Debug algorithm timing calculations
4. Fix before any deployment

"""
    elif ratio > 1.5:
        report += """⚠️  REQUIRED ACTIONS:
1. Suspend adaptive algorithm deployment
2. Review and fix timing parameters
3. Test with safer settings
4. Monitor performance closely

"""
    else:
        report += """✅ MONITORING ACTIONS:
1. Continue performance monitoring
2. Look for improvement opportunities
3. Gradual optimization

"""
    
    report += f"""
CONCLUSION:
{'-'*30}
"""
    
    if ratio > 2:
        report += "🚨 The adaptive algorithm is BROKEN and making traffic significantly worse.\n"
        report += "🚧 DO NOT DEPLOY this algorithm until major fixes are implemented.\n"
        report += "🔧 Focus on debugging timing calculation and adaptation logic.\n"
    elif ratio > 1.2:
        report += "⚠️  The adaptive algorithm needs improvement before deployment.\n"
        report += "🔧 Focus on fine-tuning parameters and testing thoroughly.\n"
    else:
        report += "✅ Algorithm performance is acceptable for continued testing.\n"
    
    # Save report
    with open('algorithm_debug_results/simple_debug_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Debug report saved: algorithm_debug_results/simple_debug_report.txt")

if __name__ == "__main__":
    load_and_analyze_results()