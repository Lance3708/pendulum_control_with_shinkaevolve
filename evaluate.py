from shinka.core import run_shinka_eval
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

def validate_pendulum(run_output: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, Optional[str]]:
    """
    Validates the simulation output with STRICTER criteria.
    
    Args:
        run_output: tuple (states, forces)
    Returns:
        (is_valid, error_msg)
    """
    states, forces = run_output
    
    # 1. Check for NaNs
    if np.any(np.isnan(states)):
        return False, "Simulation crashed (NaNs detected)"
        
    # 2. Cart limits: +/- 10.0m
    # Standard lab tracks are often limited.
    cart_pos = states[:, 0]
    if np.any(np.abs(cart_pos) > 10.0):
        return False, "Cart moved out of bounds (>10.0m)"
        
    # 3. Angle limits: Must not fall over completely
    # If the pole exceeds 1.5 radian (~86 degrees), it's considered a failure for this stabilization task.
    # Relaxed from 1.0 rad to 1.5 rad to accommodate harder physics parameters
    theta = states[:, 1]
    if np.any(np.abs(theta) > 1.5):
        return False, "Pole fell over (> 1.5 rad)"
        
    return True, None

def aggregate_metrics(results: List[Tuple[np.ndarray, np.ndarray]], results_dir: str) -> Dict[str, Any]:
    """
    Computes metrics for the Single Pendulum Stabilization.
    
    Goal: Perfect stabilization with MINIMAL time and energy (核心目标).
    
    Scoring Philosophy (Rebalanced for Evolution):
    - Base Score: ~800-1200 (basic stability, reduced)
    - Time Bonus: up to +4500 (exponential reward for speed)
    - Energy Bonus: up to +4000 (exponential reward for efficiency)  
    - Success Bonus: up to +1000 (milestone for stability)
    Total Maximum: ~10500 points
    
    Expected Scores:
    - Basic PID (initial.py): ~2000-3000 分
    - Optimized Controller: ~7000-9000 分
    - Near-Perfect: ~9500-10000 分
    
    核心：时间和能量的非线性奖励机制，越极限越高分！
    设计重点：留有足够的进化改进空间
    """
    states, forces = results[0]
    
    if len(states) > len(forces):
        states = states[1:]
    
    # Unpack state: [x, theta, dx, dtheta]
    x = states[:, 0]
    theta = states[:, 1]
    dx = states[:, 2]
    dtheta = states[:, 3]
    
    # Normalize angles
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    
    total_steps = len(states)
    
    # ========== 1. 基础稳定性得分 (Base Stability Score) ==========
    # 目标: 500-800 分（进一步降低，为进化留出更多空间）

    # 1.1 精度奖励 (Theta) - 权重 0.3（进一步降低）
    r_theta = np.exp(-25.0 * np.square(theta))  # 更陡峭的函数，精度要求更高

    # 1.2 位置奖励 (X) - 权重 0.1（进一步降低）
    r_x = np.exp(-0.8 * np.square(x))  # 稍微更严格的位置要求

    # 1.3 稳定性奖励 (Velocities) - 权重 0.05（进一步降低）
    r_stability = np.exp(-0.15 * (np.square(dx) + np.square(dtheta)))  # 更严格要求

    # 步进奖励（进一步降低系数）
    step_rewards = (0.3 * r_theta) + (0.1 * r_x) + (0.05 * r_stability)
    base_score = np.sum(step_rewards)
    
    # ========== 2. 时间效率奖励 (Time Efficiency Bonus) ==========
    # 目标: 最高 +3000 分（降低总分，但提高难度）
    # 核心机制：更严格的稳定定义 + 更陡峭的指数函数
    # 新的物理参数下，快速稳定极其困难

    # 更严格的"稳定状态"定义: theta < 0.08 rad (4.6°), x < 1.0 m
    stable_mask = (np.abs(theta) < 0.08) & (np.abs(x) < 1.0) & \
                  (np.abs(dx) < 0.3) & (np.abs(dtheta) < 0.3)

    # 找到首次连续稳定80步的起始点（增加稳定要求）
    stabilization_time = total_steps
    stable_window = 80

    if np.any(stable_mask):
        for i in range(len(stable_mask) - stable_window):
            if np.all(stable_mask[i:i+stable_window]):
                stabilization_time = i
                break

    # 时间比例 (越小越好)
    time_ratio = stabilization_time / total_steps

    # 使用更陡峭的指数函数，让超快稳定才值得高分
    # 公式: 3000 * exp(-8 * time_ratio^2)
    # 在10%时间内稳定: ~2000分 (极难)
    # 在30%时间内稳定: ~800分 (优秀)
    # 在50%时间内稳定: ~200分 (一般)
    # 在80%时间内稳定: ~15分 (较差)
    time_bonus = 3000.0 * np.exp(-8.0 * np.square(time_ratio))
    
    # ========== 3. 能量效率奖励 (Energy Efficiency Bonus) ==========
    # 目标: 最高 +2500 分（降低总分，提高难度）
    # 核心机制：更陡峭的能量惩罚，让极低能耗才值得高分
    # 新的物理参数（高摩擦）下，节能控制极其困难

    # 计算归一化能量消耗
    u_norm = forces / 100.0  # 归一化到 [-1, 1]
    total_energy = np.sum(np.square(u_norm))

    # 平均能量比 (每步的平均力平方)
    avg_energy_per_step = total_energy / total_steps

    # 使用更严苛的指数函数，只有极低能耗才值得高分
    # 公式: 2500 * exp(-25 * avg_energy^1.8)
    # 平均能量0.02: ~2000分 (极难，需要完美控制)
    # 平均能量0.05: ~800分 (优秀，需要精细控制)
    # 平均能量0.10: ~200分 (一般，保守控制)
    # 平均能量0.20: ~10分 (较差，浪费能量)
    energy_bonus = 2500.0 * np.exp(-25.0 * np.power(avg_energy_per_step, 1.8))
    
    # ========== 4. 成功奖励 (Success Bonus) ==========
    # 目标: 最高 +800 分（降低总分，更严格的标准）

    last_100_theta = np.abs(theta[-100:])
    last_100_x = np.abs(x[-100:])

    success_bonus = 0.0

    # 极严格标准（完美稳定，极难达到）
    if (np.all(last_100_theta < 0.03) and np.all(last_100_x < 0.8)):
        success_bonus = 800.0
    # 严格标准（高精度稳定）
    elif (np.all(last_100_theta < 0.06) and np.all(last_100_x < 1.2)):
        success_bonus = 400.0
    # 宽松标准（基本稳定）
    elif (np.all(last_100_theta < 0.1) and np.all(last_100_x < 2.0)):
        success_bonus = 150.0
    
    # ========== 5. 惩罚项 (Penalties) ==========
    
    # 5.1 控制震荡惩罚（鼓励平滑控制）
    force_changes = np.abs(np.diff(forces))
    oscillation_penalty = -1.0 * np.sum(force_changes > 50)
    
    # 5.2 角度震荡惩罚（NEW - 鼓励角度稳定）
    # 计算角度的变化率（角速度的变化）
    theta_changes = np.abs(np.diff(theta))
    # 惩罚剧烈的角度震荡（角度变化超过0.1 rad/step）
    angle_oscillation_penalty = -2.0 * np.sum(theta_changes > 0.1)
    # 额外惩罚稳定后仍然震荡（如果已经稳定）
    if stabilization_time < total_steps:
        stable_theta_changes = np.abs(np.diff(theta[stabilization_time:]))
        angle_oscillation_penalty -= 5.0 * np.sum(stable_theta_changes > 0.05)
    
    # 5.3 超出边界严重惩罚
    boundary_penalty = 0.0
    if np.any(np.abs(x) > 5.0):
        boundary_penalty -= 1000.0
    if np.any(np.abs(theta) > np.pi/2):
        boundary_penalty -= 1000.0
    
    # 5.3 未稳定惩罚（如果从未达到稳定状态）
    if stabilization_time >= total_steps:
        unstable_penalty = -500.0
    else:
        unstable_penalty = 0.0
    
    # ========== 最终得分 ==========
    final_score = (
        base_score +
        time_bonus +
        energy_bonus +
        success_bonus +
        oscillation_penalty +
        angle_oscillation_penalty +
        boundary_penalty +
        unstable_penalty
    )
    
    # 确保分数非负
    final_score = max(0.0, final_score)
    
    # ========== 计算额外统计指标 ==========
    
    # 稳定后的平均误差
    if stabilization_time < total_steps:
        stable_theta_error = np.mean(np.abs(theta[stabilization_time:]))
        stable_x_error = np.mean(np.abs(x[stabilization_time:]))
    else:
        stable_theta_error = np.mean(np.abs(theta))
        stable_x_error = np.mean(np.abs(x))
    
    # ========== 返回详细指标 ==========
    return {
        "combined_score": float(final_score),
        "public": {
            # 总分
            "score": float(final_score),
            
            # 分数组成
            "base_score": float(base_score),
            "time_bonus": float(time_bonus),
            "energy_bonus": float(energy_bonus),
            "success_bonus": float(success_bonus),
            
            # 核心指标
            "stabilization_time": int(stabilization_time),
            "stabilization_ratio": float(time_ratio),
            "avg_energy_per_step": float(avg_energy_per_step),
            "total_energy": float(total_energy),
            
            # 质量指标
            "mean_step_reward": float(np.mean(step_rewards)),
            "final_theta_error": float(np.abs(theta[-1])),
            "final_x_error": float(np.abs(x[-1])),
            "stable_theta_error": float(stable_theta_error),
            "stable_x_error": float(stable_x_error),
        },
        "private": {
            # 惩罚详情
            "oscillation_penalty": float(oscillation_penalty),
            "angle_oscillation_penalty": float(angle_oscillation_penalty),
            "boundary_penalty": float(boundary_penalty),
            "unstable_penalty": float(unstable_penalty),
            
            # 调试信息
            "max_theta": float(np.max(np.abs(theta))),
            "max_x": float(np.max(np.abs(x))),
            "max_force": float(np.max(np.abs(forces))),
            "force_std": float(np.std(forces)),
        }
    }


def get_pendulum_kwargs(run_index: int) -> Dict[str, Any]:
    return {"seed": 42 + run_index}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    def _aggregator_with_context(results):
        return aggregate_metrics(results, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_simulation",
        num_runs=1,
        get_experiment_kwargs=get_pendulum_kwargs,
        validate_fn=validate_pendulum,
        aggregate_metrics_fn=_aggregator_with_context
    )
    
    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pendulum Liter evaluator")
    parser.add_argument("--program_path", type=str, help="Path to program to evaluate")
    parser.add_argument("--results_dir", type=str, help="Dir to save results")
    args = parser.parse_args()
    
    main(args.program_path, args.results_dir)
