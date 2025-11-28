import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display, clear_output
import sys
import os
import importlib.util
import initial
from evaluate import aggregate_metrics

def load_agent(path_or_gen):
    """
    Loads the agent module.
    Args:
        path_or_gen: Either a direct path to a .py file (e.g., 'initial.py')
                     or a generation folder name (e.g., 'gen_7').
    """
    if path_or_gen.endswith(".py"):
        path = path_or_gen
    else:
        results_dir = "results_pendulum_liter"
        candidate_path = f"{results_dir}/{path_or_gen}/main.py"
        
        if os.path.exists(candidate_path):
            path = candidate_path
        elif os.path.exists(f"{path_or_gen}/main.py"):
            path = f"{path_or_gen}/main.py"
        elif os.path.exists("initial.py"):
            print(f"Warning: Could not find '{candidate_path}'. Falling back to 'initial.py'")
            path = "initial.py"
        else:
            raise FileNotFoundError(f"Could not find agent file for '{path_or_gen}'")

    print(f"Loading agent from: {path}")
    spec = importlib.util.spec_from_file_location("agent_module", path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    return agent_module

def run_simulation_with_agent(agent_module, max_steps=1000):
    """
    Runs the simulation using the agent's own run_simulation function.
    """
    if hasattr(agent_module, "run_simulation"):
        print("Using agent's internal run_simulation()...")
        return agent_module.run_simulation()
    else:
        print("Agent does not have run_simulation, falling back to default loop...")
        # Fallback loop
        state = np.array([0.0, 0.05, 0.0, 0.0]) # Standard initial state
        
        states = [state]
        forces = []
        
        for _ in range(max_steps):
            force = agent_module.get_control_action(state)
            force = np.clip(force, -100.0, 100.0)
            
            next_state = initial.simulate_pendulum_step(state, force, initial.DT)
            
            states.append(next_state)
            forces.append(force)
            state = next_state
            
            if np.any(np.isnan(state)):
                break
                
        return np.array(states), np.array(forces)

def visualize_pendulum_enhanced(states, forces, energy_ylim=50000, speed_factor=1.0):
    """
    Visualizes the single pendulum trajectory live.
    """
    dt = initial.DT
    l_pole = initial.L_POLE
    
    times = np.arange(len(states)) * dt
    
    if len(forces) < len(states):
        forces = np.append(forces, [0] * (len(states) - len(forces)))
    
    # --- Metrics ---
    energy_consumption = np.cumsum(np.square(forces)) * dt
    
    # Setup Figure - adjust to square animation area
    plt.ioff()
    fig = plt.figure(figsize=(16, 9))  # Keep overall 16:9, but adjust internal layout
    # Use width_ratios to make left side nearly square (9 units wide for 9 units height)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])
    
    # Animation Axis - square area, adapted for 2m pole
    ax_anim = fig.add_subplot(gs[:, 0])
    ax_anim.set_xlim(-4, 4)  # Increase range to accommodate longer pole
    ax_anim.set_ylim(-1, 3)  # Adjust Y-axis to show full motion of 2m pole
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.grid(True)
    ax_anim.set_title("Single Pendulum Simulation (2.5m pole)")
    
    cart_w, cart_h = 0.5, 0.3
    cart_patch = Rectangle((0, 0), cart_w, cart_h, color='black')
    ax_anim.add_patch(cart_patch)
    line_rod, = ax_anim.plot([], [], 'o-', lw=3, color='blue')
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes)
    
    # Plot Axes
    
    # 1. Angle
    ax_angle = fig.add_subplot(gs[0, 1])
    ax_angle.set_title("Pole Angle (rad)")
    ax_angle.set_ylabel("Angle")
    line_th, = ax_angle.plot([], [], label='Theta', color='purple')
    ax_angle.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax_angle.grid(True)
    
    valid_states = states[np.isfinite(states).all(axis=1)]
    if len(valid_states) > 0:
        th_max_abs = np.max(np.abs(valid_states[:, 1]))
        ax_angle.set_ylim(-max(0.2, th_max_abs*1.1), max(0.2, th_max_abs*1.1))
    
    # 2. Energy
    ax_energy = fig.add_subplot(gs[1, 1], sharex=ax_angle)
    ax_energy.set_title("Cumulative Energy Consumed")
    ax_energy.set_ylabel("Effort")
    line_energy, = ax_energy.plot([], [], color='red', lw=2)
    ax_energy.grid(True)
    ax_energy.set_ylim(0, energy_ylim)
    
    # 3. Force
    ax_force = fig.add_subplot(gs[2, 1], sharex=ax_angle)
    ax_force.set_title("Control Force")
    ax_force.set_ylabel("Force (N)")
    ax_force.set_xlabel("Time (s)")
    line_force, = ax_force.plot([], [], color='blue', lw=2)
    ax_force.set_ylim(-110, 110)
    ax_force.grid(True)
    
    for ax in [ax_angle, ax_energy, ax_force]:
        ax.set_xlim(0, times[-1])
    
    step_size = 5
    
    try:
        for i in range(0, len(states), step_size):
            s = states[i]
            x0 = s[0]
            th = s[1]
            
            x1 = x0 + l_pole * np.sin(th)
            y1 = 0 + l_pole * np.cos(th)
            
            cart_patch.set_xy((x0 - cart_w/2, -cart_h/2))
            line_rod.set_data([x0, x1], [0, y1])
            
            # Camera follow
            ax_anim.set_xlim(x0 - 3, x0 + 3)
            
            time_text.set_text(f"t={times[i]:.2f}s")
            
            current_times = times[:i+1]
            line_th.set_data(current_times, states[:i+1, 1])
            line_energy.set_data(current_times, energy_consumption[:i+1])
            line_force.set_data(current_times, forces[:i+1])
            
            clear_output(wait=True)
            display(fig)
            
    except KeyboardInterrupt:
        pass
    finally:
        clear_output(wait=True)
        display(fig)
        plt.close(fig)

def print_score(states, forces):
    """
    Calculate and print score using evaluate.py logic
    
    Args:
        states: simulation states array
        forces: control forces array
    
    Returns:
        dict: score metrics
    """
    # Call the existing aggregate_metrics function
    metrics = aggregate_metrics([(states, forces)], results_dir='')
    
    # Print formatted output
    print("\n" + "=" * 70)
    print("  🏆 PERFORMANCE SCORE (Updated Scoring System)")
    print("=" * 70)
    print(f"Final Score:         {metrics['combined_score']:10.2f} / 10000")
    
    print("\n📊 Score Breakdown:")
    print(f"  Base Score:        {metrics['public']['base_score']:10.2f}  (stability quality)")
    print(f"  Time Bonus:        {metrics['public']['time_bonus']:10.2f}  ⚡ (stabilized at step {metrics['public']['stabilization_time']})")
    print(f"  Energy Bonus:      {metrics['public']['energy_bonus']:10.2f}  💪 (avg energy: {metrics['public']['avg_energy_per_step']:.4f})")
    print(f"  Success Bonus:     {metrics['public']['success_bonus']:10.2f}  ✓")
    
    # 显示惩罚项（如果有）
    if 'private' in metrics:
        penalties = []
        if metrics['private'].get('oscillation_penalty', 0) < 0:
            penalties.append(f"  Oscillation:       {metrics['private']['oscillation_penalty']:10.2f}  (control smoothness)")
        if metrics['private'].get('angle_oscillation_penalty', 0) < 0:
            penalties.append(f"  Angle Oscillation: {metrics['private']['angle_oscillation_penalty']:10.2f}  (angle stability)")
        if metrics['private'].get('boundary_penalty', 0) < 0:
            penalties.append(f"  Boundary:          {metrics['private']['boundary_penalty']:10.2f}  (safety violations)")
        if metrics['private'].get('unstable_penalty', 0) < 0:
            penalties.append(f"  Unstable:          {metrics['private']['unstable_penalty']:10.2f}  (never stabilized)")
        
        if penalties:
            print("\n⚠️  Penalties:")
            for penalty in penalties:
                print(penalty)
    
    # Display key performance metrics
    print("\n📈 Key Performance Metrics:")
    print(f"  Stabilization Ratio:  {metrics['public']['stabilization_ratio']:6.2%}  (lower is better)")
    print(f"  Total Energy:         {metrics['public']['total_energy']:10.2f}")
    print(f"  Final Theta Error:    {metrics['public']['final_theta_error']:10.4f} rad ({np.rad2deg(metrics['public']['final_theta_error']):.2f}°)")
    print(f"  Final X Error:        {metrics['public']['final_x_error']:10.4f} m")
    
    # Display average error after stabilization (new)
    if 'stable_theta_error' in metrics['public']:
        print(f"  Stable Theta Error:   {metrics['public']['stable_theta_error']:10.4f} rad (avg after stabilization)")
        print(f"  Stable X Error:       {metrics['public']['stable_x_error']:10.4f} m (avg after stabilization)")
    
    print("=" * 70 + "\n")
    
    return metrics


