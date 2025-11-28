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
    
    Goal: Perfect stabilization with MINIMAL time and energy (core objective).
    
    Scoring Philosophy (Rebalanced for Evolution):
    - Base Score: ~800-1200 (basic stability, reduced)
    - Time Bonus: up to +4500 (exponential reward for speed)
    - Energy Bonus: up to +4000 (exponential reward for efficiency)  
    - Success Bonus: up to +1000 (milestone for stability)
    Total Maximum: ~10500 points
    
    Expected Scores:
    - Basic PID (initial.py): ~2000-3000 points
    - Optimized Controller: ~7000-9000 points
    - Near-Perfect: ~9500-10000 points
    
    Core: Nonlinear reward mechanism for time and energy - more extreme = higher score!
    Design focus: Leave sufficient room for evolutionary improvement
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
    
    # ========== 1. Base Stability Score ==========
    # Target: 500-800 points (further reduced to leave more room for evolution)

    # 1.1 Precision reward (Theta) - weight 0.3 (further reduced)
    r_theta = np.exp(-25.0 * np.square(theta))  # Steeper function, higher precision requirement

    # 1.2 Position reward (X) - weight 0.1 (further reduced)
    r_x = np.exp(-0.8 * np.square(x))  # Slightly stricter position requirement

    # 1.3 Stability reward (Velocities) - weight 0.05 (further reduced)
    r_stability = np.exp(-0.15 * (np.square(dx) + np.square(dtheta)))  # Stricter requirement

    # Step rewards (further reduced coefficients)
    step_rewards = (0.3 * r_theta) + (0.1 * r_x) + (0.05 * r_stability)
    base_score = np.sum(step_rewards)
    
    # ========== 2. Time Efficiency Bonus ==========
    # Target: up to +3000 points (reduced total, but increased difficulty)
    # Core mechanism: Stricter stability definition + steeper exponential function
    # With new physical parameters, fast stabilization is extremely difficult

    # Stricter "stable state" definition: theta < 0.08 rad (4.6°), x < 1.0 m
    stable_mask = (np.abs(theta) < 0.08) & (np.abs(x) < 1.0) & \
                  (np.abs(dx) < 0.3) & (np.abs(dtheta) < 0.3)

    # Find the first point of continuous stability for 80 steps (increased stability requirement)
    stabilization_time = total_steps
    stable_window = 80

    if np.any(stable_mask):
        for i in range(len(stable_mask) - stable_window):
            if np.all(stable_mask[i:i+stable_window]):
                stabilization_time = i
                break

    # Time ratio (lower is better)
    time_ratio = stabilization_time / total_steps

    # Use steeper exponential function - only ultra-fast stabilization deserves high score
    # Formula: 3000 * exp(-8 * time_ratio^2)
    # Stabilize within 10% time: ~2000 points (extremely difficult)
    # Stabilize within 30% time: ~800 points (excellent)
    # Stabilize within 50% time: ~200 points (average)
    # Stabilize within 80% time: ~15 points (poor)
    time_bonus = 3000.0 * np.exp(-8.0 * np.square(time_ratio))
    
    # ========== 3. Energy Efficiency Bonus ==========
    # Target: up to +2500 points (reduced total, increased difficulty)
    # Core mechanism: Steeper energy penalty - only extremely low energy consumption deserves high score
    # With new physical parameters (high friction), energy-efficient control is extremely difficult

    # Calculate normalized energy consumption
    u_norm = forces / 100.0  # Normalize to [-1, 1]
    total_energy = np.sum(np.square(u_norm))

    # Average energy ratio (average force squared per step)
    avg_energy_per_step = total_energy / total_steps

    # Use stricter exponential function - only extremely low energy consumption deserves high score
    # Formula: 2500 * exp(-25 * avg_energy^1.8)
    # Average energy 0.02: ~2000 points (extremely difficult, requires perfect control)
    # Average energy 0.05: ~800 points (excellent, requires fine control)
    # Average energy 0.10: ~200 points (average, conservative control)
    # Average energy 0.20: ~10 points (poor, energy waste)
    energy_bonus = 2500.0 * np.exp(-25.0 * np.power(avg_energy_per_step, 1.8))
    
    # ========== 4. Success Bonus ==========
    # Target: up to +800 points (reduced total, stricter standards)

    last_100_theta = np.abs(theta[-100:])
    last_100_x = np.abs(x[-100:])

    success_bonus = 0.0

    # Extremely strict standard (perfect stability, extremely difficult to achieve)
    if (np.all(last_100_theta < 0.03) and np.all(last_100_x < 0.8)):
        success_bonus = 800.0
    # Strict standard (high-precision stability)
    elif (np.all(last_100_theta < 0.06) and np.all(last_100_x < 1.2)):
        success_bonus = 400.0
    # Lenient standard (basic stability)
    elif (np.all(last_100_theta < 0.1) and np.all(last_100_x < 2.0)):
        success_bonus = 150.0
    
    # ========== 5. Penalties ==========
    
    # 5.1 Control oscillation penalty (encourage smooth control)
    force_changes = np.abs(np.diff(forces))
    oscillation_penalty = -1.0 * np.sum(force_changes > 50)
    
    # 5.2 Angle oscillation penalty (NEW - encourage angle stability)
    # Calculate rate of angle change (change in angular velocity)
    theta_changes = np.abs(np.diff(theta))
    # Penalize severe angle oscillations (angle change exceeds 0.1 rad/step)
    angle_oscillation_penalty = -2.0 * np.sum(theta_changes > 0.1)
    # Additional penalty for oscillations after stabilization (if already stable)
    if stabilization_time < total_steps:
        stable_theta_changes = np.abs(np.diff(theta[stabilization_time:]))
        angle_oscillation_penalty -= 5.0 * np.sum(stable_theta_changes > 0.05)
    
    # 5.3 Severe boundary violation penalty
    boundary_penalty = 0.0
    if np.any(np.abs(x) > 5.0):
        boundary_penalty -= 1000.0
    if np.any(np.abs(theta) > np.pi/2):
        boundary_penalty -= 1000.0
    
    # 5.4 Unstable penalty (if never achieved stable state)
    if stabilization_time >= total_steps:
        unstable_penalty = -500.0
    else:
        unstable_penalty = 0.0
    
    # ========== Final Score ==========
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
    
    # Ensure score is non-negative
    final_score = max(0.0, final_score)
    
    # ========== Calculate Additional Statistics ==========
    
    # Average error after stabilization
    if stabilization_time < total_steps:
        stable_theta_error = np.mean(np.abs(theta[stabilization_time:]))
        stable_x_error = np.mean(np.abs(x[stabilization_time:]))
    else:
        stable_theta_error = np.mean(np.abs(theta))
        stable_x_error = np.mean(np.abs(x))
    
    # ========== Return Detailed Metrics ==========
    return {
        "combined_score": float(final_score),
        "public": {
            # Total score
            "score": float(final_score),
            
            # Score breakdown
            "base_score": float(base_score),
            "time_bonus": float(time_bonus),
            "energy_bonus": float(energy_bonus),
            "success_bonus": float(success_bonus),
            
            # Core metrics
            "stabilization_time": int(stabilization_time),
            "stabilization_ratio": float(time_ratio),
            "avg_energy_per_step": float(avg_energy_per_step),
            "total_energy": float(total_energy),
            
            # Quality metrics
            "mean_step_reward": float(np.mean(step_rewards)),
            "final_theta_error": float(np.abs(theta[-1])),
            "final_x_error": float(np.abs(x[-1])),
            "stable_theta_error": float(stable_theta_error),
            "stable_x_error": float(stable_x_error),
        },
        "private": {
            # Penalty details
            "oscillation_penalty": float(oscillation_penalty),
            "angle_oscillation_penalty": float(angle_oscillation_penalty),
            "boundary_penalty": float(boundary_penalty),
            "unstable_penalty": float(unstable_penalty),
            
            # Debug information
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
