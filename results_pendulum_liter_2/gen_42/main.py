import numpy as np

# --- Physics Constants ---
M_CART = 1.0       # Mass of the cart (kg)
M_POLE = 0.35      # Mass of the pole (kg) - 更重，大幅增加控制难度
L_POLE = 2.5       # Total length of the pole (m) - 更长，极不稳定
L_COM = L_POLE / 2 # Length to center of mass (m)
G = 9.81           # Gravity (m/s^2)
FRICTION_CART = 0.35 # Coefficient of friction for cart - 高摩擦，更多能量损失
FRICTION_JOINT = 0.25 # Coefficient of friction for joint - 高关节摩擦
DT = 0.02          # Time step (s)
MAX_STEPS = 1000   # 20 seconds simulation

def simulate_pendulum_step(state, force, dt):
    """
    Simulates one time step of the Single Inverted Pendulum.

    State vector: [x, theta, dx, dtheta]
    - x: Cart position (m)
    - theta: Pole angle (rad), 0 is upright
    - dx: Cart velocity (m/s)
    - dtheta: Pole angular velocity (rad/s)

    Args:
        state: numpy array of shape (4,)
        force: scalar float, force applied to the cart (N)
        dt: float, time step (s)

    Returns:
        next_state: numpy array of shape (4,)
    """
    x, theta, dx, dtheta = state

    # Precompute trig terms
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Equations of Motion (Non-linear)
    # derived from Lagrangian dynamics

    # Total mass
    M_total = M_CART + M_POLE

    # Friction forces
    f_cart = -FRICTION_CART * dx
    f_joint = -FRICTION_JOINT * dtheta

    # Denominator for solving linear system of accelerations
    # Derived from solving the system:
    # 1) (M+m)x_dd + (ml cos)theta_dd = F + f_cart + ml*theta_d^2*sin
    # 2) (ml cos)x_dd + (ml^2)theta_dd = mgl sin + f_joint

    temp = (force + f_cart + M_POLE * L_COM * dtheta**2 * sin_theta) / M_total

    theta_acc = (G * sin_theta - cos_theta * temp + f_joint / (M_POLE * L_COM)) / \
                (L_COM * (4.0/3.0 - M_POLE * cos_theta**2 / M_total))

    x_acc = temp - (M_POLE * L_COM * theta_acc * cos_theta) / M_total

    # Euler integration
    next_x = x + dx * dt
    next_theta = theta + dtheta * dt
    next_dx = dx + x_acc * dt
    next_dtheta = dtheta + theta_acc * dt

    return np.array([next_x, next_theta, next_dx, next_dtheta])


# EVOLVE-BLOCK-START
class Controller:
    """
    Three-Phase Predictive Controller with Energy Optimization
    
    Phase 1: Emergency Catch - Predictive positioning to catch falling pole
    Phase 2: Energy-Guided Swing - Direction-aware damping and energy control  
    Phase 3: Precision Hold - High-gain stabilization with adaptive integral
    """

    def __init__(self):
        # System parameters for physics calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT
        
        # Natural frequency for normalization
        self.omega_n = np.sqrt(G / L_COM)
        
        # Integral control for position
        self.integral_x = 0.0
        
        # Energy tracking
        self.last_energy = 0.0
        
        # Phase detection smoothing
        self.phase_history = []

    def calculate_energy(self, theta, dtheta, dx):
        """Calculate total mechanical energy of the system"""
        # Potential energy (pole)
        E_pot = self.m * self.g * self.l * (1 - np.cos(theta))
        # Kinetic energy (pole rotation)
        E_kin_pole = 0.5 * self.m * (self.l * dtheta)**2
        # Kinetic energy (cart + pole translation)
        E_kin_cart = 0.5 * (self.M + self.m) * dx**2
        return E_pot + E_kin_pole + E_kin_cart

    def get_action(self, state):
        """Three-phase control with energy optimization"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Calculate current energy
        current_energy = self.calculate_energy(theta, dtheta, dx)
        energy_change = current_energy - self.last_energy
        self.last_energy = current_energy

        # --- Phase Detection ---
        
        # Emergency: large angle AND moving away from vertical OR high energy
        is_emergency = (abs(theta) > 0.6 and theta * dtheta > 0.05) or \
                      (abs(theta) > 0.8 and current_energy > 2.0)
        
        # Precision hold: very close to equilibrium with low velocities
        is_settling = abs(theta) < 0.08 and abs(dtheta) < 0.15 and abs(dx) < 0.2
        
        # Default to swing phase if neither emergency nor settling
        is_swing = not (is_emergency or is_settling)
        
        # Track phase for smoothing
        self.phase_history.append(1 if is_emergency else (2 if is_swing else 3))
        if len(self.phase_history) > 5:
            self.phase_history.pop(0)
        
        # Smooth phase transitions
        phase_mode = max(set(self.phase_history), key=self.phase_history.count) if self.phase_history else 2

        # --- Phase-Specific Control Laws ---
        
        if phase_mode == 1:  # Emergency Catch
            # Predictive cart positioning: anticipate where pole will be
            predict_time = 0.12  # 6 steps ahead
            theta_pred = theta + dtheta * predict_time
            x_target = self.l * np.sin(theta_pred) * 1.2  # Overshoot slightly for catch
            
            # Velocity target matches pole motion
            dx_target = self.l * dtheta * np.cos(theta)
            
            # Aggressive PD control to reach target quickly
            Kp_emergency = 45.0
            Kd_emergency = 18.0
            force = Kp_emergency * (x_target - x) + Kd_emergency * (dx_target - dx)
            
            # Add energy if pole is falling fast
            if abs(dtheta) > 0.8:
                energy_boost = 12.0 * np.sign(theta) * np.tanh(2.0 * abs(dtheta))
                force += energy_boost
            
            # Decay integrator
            self.integral_x *= 0.8

        elif phase_mode == 3:  # Precision Hold
            # High-gain stabilization near equilibrium
            predict_time = 0.08  # 4 steps ahead
            theta_future = theta + dtheta * predict_time
            x_future = x + dx * predict_time
            
            # Update integral term with adaptive gain
            stability = np.exp(-15.0 * (theta**2 + 0.3 * dtheta**2))
            if stability > 0.7:
                self.integral_x += x * DT * stability
                self.integral_x = np.clip(self.integral_x, -1.0, 1.0)
            else:
                self.integral_x *= 0.98
            
            # Adaptive integral gain based on proximity to equilibrium
            K_i_adaptive = 1.2 * stability
            
            # High-gain state feedback
            force = (85.0 * theta_future + 
                     22.0 * dtheta + 
                     6.5 * x_future + 
                     9.0 * dx + 
                     K_i_adaptive * self.integral_x)

        else:  # Swing Phase (phase_mode == 2)
            # Energy-guided control with direction-aware damping
            target_energy = self.m * self.g * self.l * 0.1  # Small target energy
            
            # Energy correction term
            if current_energy < target_energy:
                # Add energy to help swing up
                energy_add = 5.0 * np.sign(theta) * (target_energy - current_energy)
            else:
                # Remove excess energy
                energy_add = -3.0 * np.sign(dtheta) * (current_energy - target_energy)
            
            # Base swing control
            predict_time = 0.15  # 7-8 steps ahead
            theta_future = theta + dtheta * predict_time
            
            # Direction-aware damping: only damp when moving away from vertical
            if theta * dtheta > 0:  # Moving away
                damping = -8.0 * dtheta * np.tanh(4.0 * abs(theta))
            else:  # Moving toward vertical - assist
                damping = 2.0 * dtheta * np.tanh(2.0 * abs(theta))
            
            # Position correction with friction compensation
            pos_correction = -4.0 * x - 6.0 * dx
            
            force = (65.0 * theta_future + 
                     15.0 * dtheta + 
                     pos_correction + 
                     damping + 
                     energy_add)
            
            # Decay integrator
            self.integral_x *= 0.9

        return float(force)

# EVOLVE-BLOCK-END

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
    return float(np.clip(force, -100.0, 100.0))

def run_simulation(seed=None):
    """
    Runs the simulation loop.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initial state: 1.02 rad (~58 degrees)
    # 更大初始角度配合更重更长的杆子，极具挑战性
    state = np.array([0.0, 1.02, 0.0, 0.0])

    states = [state]
    forces = []

    for _ in range(MAX_STEPS):
        force = get_control_action(state)
        # Clip force to realistic limits
        force = np.clip(force, -100.0, 100.0)

        next_state = simulate_pendulum_step(state, force, DT)

        states.append(next_state)
        forces.append(force)

        state = next_state

        # # Early termination checks
        # if np.any(np.isnan(state)):
        #     break
        # # Fail fast if it falls over (> 1.0 rad, matching evaluate.py)
        # if abs(state[1]) > 1.0:
        #     break

    return np.array(states), np.array(forces)