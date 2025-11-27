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
class SlidingModeEnergyController:
    """
    Hybrid Sliding Mode Control with Energy Shaping for Inverted Pendulum.
    
    Combines:
    1. Energy-based swing-up/stabilization mode switching
    2. Sliding mode control for robustness
    3. Adaptive gain scheduling for efficiency
    4. Friction compensation
    """
    
    def __init__(self):
        # Physical parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.mu_cart = FRICTION_CART
        self.mu_joint = FRICTION_JOINT
        
        # Moment of inertia of pole about pivot
        self.I = (1/3) * self.m * L_POLE**2
        
        # Energy at upright equilibrium (potential energy reference)
        self.E_target = self.m * self.g * self.l
        
        # Sliding mode parameters - tuned for this challenging setup
        self.lambda_theta = 12.0  # Sliding surface slope for angle
        self.lambda_x = 3.5       # Sliding surface slope for position
        self.eta = 45.0           # Reaching law gain
        self.phi = 0.08           # Boundary layer thickness for anti-chattering
        
        # Gain scheduling parameters
        self.k_theta = 85.0       # Angle error gain
        self.k_x = 18.0           # Position error gain
        self.k_dx = 12.0          # Velocity damping gain
        self.k_dtheta = 8.0       # Angular velocity damping
        
        # Energy shaping parameters
        self.k_energy = 2.5       # Energy injection/dissipation rate
        
    def normalize_angle(self, theta):
        """Normalize angle to [-pi, pi]"""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def compute_energy(self, theta, dtheta):
        """Compute total mechanical energy of the pendulum relative to upright"""
        # Kinetic energy of pole rotation
        KE = 0.5 * self.I * dtheta**2
        # Potential energy (0 at upright, negative when hanging)
        PE = self.m * self.g * self.l * (np.cos(theta) - 1)
        return KE + PE + self.E_target  # Shift so upright equilibrium has E = E_target
    
    def smooth_sign(self, s):
        """Smooth approximation of sign function to reduce chattering"""
        return np.tanh(s / self.phi)
    
    def get_action(self, state):
        x, theta, dx, dtheta = state
        theta = self.normalize_angle(theta)
        
        # Compute current energy
        E_current = self.compute_energy(theta, dtheta)
        E_error = E_current - self.E_target
        
        # Determine control mode based on angle magnitude
        abs_theta = abs(theta)
        
        if abs_theta > 0.5:
            # Far from upright - use energy-based swing-up with SMC
            force = self._energy_swing_control(theta, dtheta, dx, x, E_error)
        elif abs_theta > 0.15:
            # Transition zone - blend energy and stabilization
            alpha = (abs_theta - 0.15) / 0.35  # 0 to 1
            f_energy = self._energy_swing_control(theta, dtheta, dx, x, E_error)
            f_stab = self._sliding_mode_stabilize(x, theta, dx, dtheta)
            force = alpha * f_energy + (1 - alpha) * f_stab
        else:
            # Near upright - pure stabilization with SMC
            force = self._sliding_mode_stabilize(x, theta, dx, dtheta)
        
        # Add friction compensation
        force += self._friction_compensation(dx, dtheta, theta)
        
        return float(np.clip(force, -100.0, 100.0))
    
    def _energy_swing_control(self, theta, dtheta, dx, x, E_error):
        """Energy-based control for swing-up phase"""
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Energy pumping: inject energy when below target, dissipate when above
        # Direction based on angular velocity and angle
        energy_direction = np.sign(dtheta * cos_theta)
        
        # Adaptive energy gain based on how far from target energy
        energy_gain = self.k_energy * np.tanh(abs(E_error) / self.E_target)
        
        # Energy control component
        f_energy = -energy_gain * energy_direction * np.sign(E_error)
        
        # Add position regulation to prevent cart runaway
        f_pos = -self.k_x * 0.5 * x - self.k_dx * 0.3 * dx
        
        # Scale based on angle - less position control when swinging
        pos_scale = 1.0 - 0.7 * abs(cos_theta)
        
        return 25.0 * f_energy + pos_scale * f_pos
    
    def _sliding_mode_stabilize(self, x, theta, dx, dtheta):
        """Sliding mode control for stabilization near upright"""
        # Define sliding surfaces
        # s1: for angle stabilization
        s_theta = dtheta + self.lambda_theta * theta
        
        # s2: for position regulation (coupled with angle)
        s_x = dx + self.lambda_x * x
        
        # Combined sliding surface with priority on angle
        abs_theta = abs(theta)
        theta_weight = 1.0 + 2.0 * abs_theta  # More weight on angle when larger
        x_weight = 0.3 + 0.7 * np.exp(-10 * abs_theta)  # Less weight on x when angle large
        
        # Equivalent control (based on linearized dynamics)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Compute desired accelerations
        theta_acc_des = -self.lambda_theta * dtheta - self.k_theta * theta
        x_acc_des = -self.lambda_x * dx - self.k_x * x
        
        # Feedforward based on dynamics
        M_total = self.M + self.m
        denom = self.l * (4.0/3.0 - self.m * cos_theta**2 / M_total)
        
        # Approximate equivalent control
        f_eq = (M_total * x_acc_des + 
                self.m * self.l * theta_acc_des * cos_theta -
                self.m * self.l * dtheta**2 * sin_theta)
        
        # Switching control for robustness
        s_combined = theta_weight * s_theta + x_weight * s_x * 0.1
        f_switch = -self.eta * self.smooth_sign(s_combined)
        
        # Adaptive gain based on distance from equilibrium
        dist = np.sqrt(theta**2 + 0.01 * x**2)
        adaptive_gain = 1.0 + 1.5 * dist
        
        # State feedback component
        f_fb = (-self.k_theta * theta - 
                self.k_dtheta * dtheta - 
                self.k_x * x * 0.5 - 
                self.k_dx * dx * 0.4)
        
        return adaptive_gain * (0.3 * f_eq + 0.4 * f_switch + 0.3 * f_fb)
    
    def _friction_compensation(self, dx, dtheta, theta):
        """Compensate for system friction to improve efficiency"""
        # Cart friction compensation
        f_cart_comp = self.mu_cart * np.sign(dx) * 0.5
        
        # Joint friction effect on cart (through dynamics coupling)
        cos_theta = np.cos(theta)
        f_joint_effect = self.mu_joint * np.sign(dtheta) * cos_theta * 0.3
        
        return f_cart_comp + f_joint_effect

# Initialize controller  
controller = SlidingModeEnergyController()

def get_control_action(state):
    return float(controller.get_action(state))
# EVOLVE-BLOCK-END

def run_simulation(seed=None):
    """
    Runs the simulation loop.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initial state: 0.4 rad (~23 degrees)
    # 更大初始角度配合更重更长的杆子，极具挑战性
    state = np.array([0.0, 0.9, 0.0, 0.0])

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