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
    Adaptive Hybrid Controller with Phase-Based Control Strategies
    
    Implements three distinct control modes:
    1. Energy-based swing-up for large angles
    2. Blended transition control for medium angles
    3. LQR + integral action for stabilization
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        self.denom0 = self.l * (4.0 / 3.0 - self.m / self.Mtot)
        
        # Integral action variables
        self.integral_x = 0.0
        self.integral_theta = 0.0
        
        # Anti-windup limits
        self.integral_limit = 5.0
        
        # Compute natural frequency for energy calculations
        self.omega_n = np.sqrt(self.g / self.l)
        
        # Pre-compute LQR gains for stabilization mode
        self._compute_lqr_gains()

    def _compute_lqr_gains(self):
        """Compute LQR gains for the stabilization mode"""
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[3, 1] = self.g / self.denom0
        A[3, 2] = FRICTION_CART / (self.Mtot * self.denom0)
        A[3, 3] = -FRICTION_JOINT / (self.m * self.l * self.denom0)
        A[2, 1] = -(self.m * self.l / self.Mtot) * A[3, 1]
        A[2, 2] = -FRICTION_CART / self.Mtot - (self.m * self.l / self.Mtot) * A[3, 2]
        A[2, 3] = FRICTION_JOINT / (self.Mtot * self.denom0)

        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / self.Mtot + (self.m * self.l) / (self.Mtot**2 * self.denom0)
        B[3, 0] = -1.0 / (self.Mtot * self.denom0)

        Q = np.diag([5.0, 50.0, 0.7, 3.5])  # Tuned for faster response
        R = np.array([[1.0]])

        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        self.K_lqr = np.linalg.inv(R) @ B.T @ P

    def _normalize_angle(self, theta):
        """Robust angle normalization using arctan2"""
        return np.arctan2(np.sin(theta), np.cos(theta))

    def _compute_pendulum_energy(self, theta, dtheta):
        """Compute total mechanical energy of the pendulum"""
        T = 0.5 * self.m * (self.l**2) * (dtheta**2)  # Kinetic energy
        V = self.m * self.g * self.l * (1 - np.cos(theta))  # Potential energy
        return T + V

    def _swing_up_control(self, state):
        """Energy-based swing-up control for large angles"""
        x, theta, dx, dtheta = state
        
        # Compute energy and desired energy (at upright position)
        current_energy = self._compute_pendulum_energy(theta, dtheta)
        desired_energy = 2 * self.m * self.g * self.l  # Energy at inverted position
        
        # Energy difference
        energy_error = desired_energy - current_energy
        
        # Control law: proportional to energy error and angular velocity
        # Sign ensures we add energy when needed
        force = 20.0 * np.sign(dtheta * np.cos(theta)) * np.tanh(energy_error / 10.0)
        
        return force

    def _transition_control(self, state):
        """Blended control during transition from swing-up to stabilization"""
        x, theta, dx, dtheta = state
        
        # Blend swing-up and LQR based on angle
        blend_factor = np.tanh(5.0 * (abs(theta) - 0.3) / 0.2)  # Sharp transition around 0.3 rad
        blend_factor = 0.5 * (1 + blend_factor)  # Map to [0, 1]
        
        # Get both control actions
        swing_force = self._swing_up_control(state)
        lqr_force = self._lqr_control(state)
        
        # Blend them
        force = blend_factor * lqr_force + (1 - blend_factor) * swing_force
        
        return force

    def _lqr_control(self, state):
        """Standard LQR control with integral action"""
        x, theta, dx, dtheta = state
        
        # Update integrals with anti-windup
        self.integral_x += x * DT
        self.integral_theta += theta * DT
        
        # Anti-windup protection
        self.integral_x = np.clip(self.integral_x, -self.integral_limit, self.integral_limit)
        self.integral_theta = np.clip(self.integral_theta, -self.integral_limit, self.integral_limit)
        
        # Extended state vector with integrals
        state_vec = np.array([x, theta, dx, dtheta, self.integral_x, self.integral_theta])
        
        # Extended LQR gain matrix (adding integral action)
        K_ext = np.zeros((1, 6))
        K_ext[0, :4] = self.K_lqr[0, :]
        
        # Add integral gains (tuned for zero steady-state error)
        K_ext[0, 4] = 0.8  # Integral gain for position
        K_ext[0, 5] = 2.5  # Integral gain for angle
        
        force = -K_ext @ state_vec
        
        return force[0]

    def _apply_nonlinear_modifications(self, force, state):
        """Apply nonlinear modifications to control force"""
        x, theta, dx, dtheta = state
        
        # Velocity-dependent damping augmentation
        vel_damping = 0.3 * np.tanh(abs(dx) - 0.5) * dx
        force -= vel_damping
        
        # Control authority scaling based on angle
        authority_scaling = 1.0 - 0.3 * np.tanh(10.0 * abs(theta))
        force *= authority_scaling
        
        return force

    def get_action(self, state):
        """Main control function selecting appropriate control strategy"""
        # Normalize angle for all computations
        x, theta, dx, dtheta = state
        theta = self._normalize_angle(theta)
        normalized_state = np.array([x, theta, dx, dtheta])
        
        # Select control mode based on angle magnitude
        abs_theta = abs(theta)
        
        if abs_theta > 0.5:
            # Swing-up mode for large angles
            force = self._swing_up_control(normalized_state)
        elif abs_theta > 0.1:
            # Transition mode for medium angles
            force = self._transition_control(normalized_state)
        else:
            # Stabilization mode for small angles
            force = self._lqr_control(normalized_state)
            
            # Apply additional nonlinear modifications near equilibrium
            force = self._apply_nonlinear_modifications(force, normalized_state)
        
        # Final clipping to actuator limits
        force = np.clip(force, -100.0, 100.0)
        
        return float(force)

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
    return float(np.clip(force, -100.0, 100.0))
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