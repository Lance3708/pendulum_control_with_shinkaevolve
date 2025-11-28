#!/usr/bin/env python3
"""
Simple test script to evaluate the new initial controller performance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from initial import run_simulation
from evaluate import aggregate_metrics

def test_initial_performance():
    """Test the initial controller performance after adjustments"""
    print("=" * 60)
    print("Testing New Adjusted Controller Performance")
    print("=" * 60)

    # Run simulation
    states, forces = run_simulation(seed=42)

    print(f"Simulation completed: {len(states)} steps")
    print(f"Control force sequence length: {len(forces)}")

    # Calculate metrics
    metrics = aggregate_metrics([(states, forces)], '')

    # Print detailed results
    print(f"\n🏆 Final Score: {metrics['combined_score']:.2f} / 7300 points")
    print(f"   Percentage of total: {(metrics['combined_score']/7300)*100:.1f}%")

    print(f"\n📊 Detailed Score Breakdown:")
    print(f"   Base Stability: {metrics['public']['base_score']:.2f} points")
    print(f"   Time Efficiency: {metrics['public']['time_bonus']:.2f} points")
    print(f"   Energy Efficiency: {metrics['public']['energy_bonus']:.2f} points")
    print(f"   Success Bonus: {metrics['public']['success_bonus']:.2f} points")

    print(f"\n⚡ Key Performance Metrics:")
    print(f"   Stabilization Time: {metrics['public']['stabilization_time']} steps ({metrics['public']['stabilization_ratio']*100:.1f}%)")
    print(f"   Average Energy Consumption: {metrics['public']['avg_energy_per_step']:.4f}")
    print(f"   Total Energy: {metrics['public']['total_energy']:.2f}")
    print(f"   Final Angle Error: {metrics['public']['final_theta_error']:.4f} rad ({np.rad2deg(metrics['public']['final_theta_error']):.2f}°)")
    print(f"   Final Position Error: {metrics['public']['final_x_error']:.4f} m")

    # Physical statistics
    theta = states[:, 1]
    x = states[:, 0]
    print(f"\n📏 Physical Statistics:")
    print(f"   Maximum Angle Deviation: {np.max(np.abs(theta)):.3f} rad ({np.rad2deg(np.max(np.abs(theta))):.1f}°)")
    print(f"   Maximum Position Deviation: {np.max(np.abs(x)):.3f} m")
    print(f"   Maximum Control Force: {np.max(np.abs(forces)):.1f} N")
    print(f"   Control Force Std Dev: {np.std(forces):.2f} N")

    # Analysis
    print(f"\n🎯 Performance Analysis:")
    if metrics['combined_score'] < 3000:
        print("   ❌ Initial controller performance is low - ample room for evolution")
    elif metrics['combined_score'] < 4500:
        print("   ⚠️  Initial controller performance is moderate - good room for evolution")
    elif metrics['combined_score'] < 6000:
        print("   ✅ Initial controller performance is good - moderate room for evolution")
    else:
        print("   ⚠️  Initial controller performance is too high - consider increasing difficulty")

    # Check if stabilization achieved
    if metrics['public']['stabilization_time'] < len(states):
        print(f"   ✅ Successfully stabilized in {metrics['public']['stabilization_time']} steps")
    else:
        print("   ❌ Failed to stabilize - controller needs improvement")

    return metrics

if __name__ == "__main__":
    test_initial_performance()