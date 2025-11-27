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
    print("测试新的调整后控制器性能")
    print("=" * 60)

    # Run simulation
    states, forces = run_simulation(seed=42)

    print(f"仿真完成: {len(states)} 步")
    print(f"控制力序列长度: {len(forces)}")

    # Calculate metrics
    metrics = aggregate_metrics([(states, forces)], '')

    # Print detailed results
    print(f"\n🏆 最终得分: {metrics['combined_score']:.2f} / 7300 分")
    print(f"   占总分比例: {(metrics['combined_score']/7300)*100:.1f}%")

    print(f"\n📊 详细得分构成:")
    print(f"   基础稳定性: {metrics['public']['base_score']:.2f} 分")
    print(f"   时间效率: {metrics['public']['time_bonus']:.2f} 分")
    print(f"   能量效率: {metrics['public']['energy_bonus']:.2f} 分")
    print(f"   成功奖励: {metrics['public']['success_bonus']:.2f} 分")

    print(f"\n⚡ 关键性能指标:")
    print(f"   稳定时间: {metrics['public']['stabilization_time']} 步 ({metrics['public']['stabilization_ratio']*100:.1f}%)")
    print(f"   平均能耗: {metrics['public']['avg_energy_per_step']:.4f}")
    print(f"   总能耗: {metrics['public']['total_energy']:.2f}")
    print(f"   最终角度误差: {metrics['public']['final_theta_error']:.4f} rad ({np.rad2deg(metrics['public']['final_theta_error']):.2f}°)")
    print(f"   最终位置误差: {metrics['public']['final_x_error']:.4f} m")

    # Physical statistics
    theta = states[:, 1]
    x = states[:, 0]
    print(f"\n📏 物理统计:")
    print(f"   最大角度偏差: {np.max(np.abs(theta)):.3f} rad ({np.rad2deg(np.max(np.abs(theta))):.1f}°)")
    print(f"   最大位置偏差: {np.max(np.abs(x)):.3f} m")
    print(f"   最大控制力: {np.max(np.abs(forces)):.1f} N")
    print(f"   控制力标准差: {np.std(forces):.2f} N")

    # Analysis
    print(f"\n🎯 性能分析:")
    if metrics['combined_score'] < 3000:
        print("   ❌ 初始控制器性能偏低 - 进化空间充足")
    elif metrics['combined_score'] < 4500:
        print("   ⚠️  初始控制器性能中等 - 有良好进化空间")
    elif metrics['combined_score'] < 6000:
        print("   ✅ 初始控制器性能良好 - 进化空间适中")
    else:
        print("   ⚠️  初始控制器性能过高 - 考虑进一步增加难度")

    # Check if stabilization achieved
    if metrics['public']['stabilization_time'] < len(states):
        print(f"   ✅ 成功稳定，耗时 {metrics['public']['stabilization_time']} 步")
    else:
        print("   ❌ 未能稳定 - 控制器需要改进")

    return metrics

if __name__ == "__main__":
    test_initial_performance()