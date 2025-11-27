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
    Precision Return Damper v2: Physics-Informed Phase Control

    Handles three distinct operational phases:
    1. Emergency Recovery (large angles): Aggressive swing-up with energy management
    2. Mid-Swing Transition: Directional damping to prevent overshoot
    3. Precision Hold (small angles): Fine positioning with adaptive integral control
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        self.denom0 = self.l * (4.0 / 3.0 - self.m / self.Mtot)
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT

        # Build base A matrix for LQR
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[3, 1] = self.g / self.denom0
        A[3, 2] = self.b_c / (self.Mtot * self.denom0)
        A[3, 3] = -self.b_j / (self.m * self.l * self.denom0)
        A[2, 1] = -(self.m * self.l / self.Mtot) * A[3, 1]
        A[2, 2] = -self.b_c / self.Mtot - (self.m * self.l / self.Mtot) * A[3, 2]
        A[2, 3] = self.b_j / (self.Mtot * self.denom0)

        # B matrix
        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / self.Mtot + (self.m * self.l) / (self.Mtot**2 * self.denom0)
        B[3, 0] = -1.0 / (self.Mtot * self.denom0)

        # Q and R matrices for LQR
        Q = np.diag([4.5, 44.0, 0.6, 3.2])
        Q[1, 3] = 0.8
        Q[3, 1] = 0.8
        R = np.array([[1.0]])

        self.A = A
        self.B = B
        self.K = self.solve_lqr(A, B, Q, R)

        # Integral control parameters
        self.integral_x = 0.0
        self.K_i = 0.85

        # Natural frequency for normalized calculations
        self.omega_n = np.sqrt(G / L_COM)

    def solve_lqr(self, A, B, Q, R):
        """Solve continuous-time LQR"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """Phase-aware control with physics-informed compensation"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        state_vec = np.array([x, theta, dx, dtheta])

        # Base LQR force
        base_force = -self.K @ state_vec

        # Adaptive gain scheduling
        pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.6))
        vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 1.0))
        adaptive_gain = pos_gain * vel_gain

        force = base_force * adaptive_gain

        # --- Phase 1: Emergency Recovery (Large Angles) ---
        if abs(theta) > 0.8:
            # Energy-based swing-up with directional awareness
            swing_activation = np.tanh(6.0 * (abs(theta) - 0.8))
            # More aggressive when actually falling away from vertical
            falling_factor = 1.0 + np.tanh(2.0 * theta * dtheta / (L_COM * self.omega_n))
            u_swing = 9.0 * swing_activation * np.sign(theta) * falling_factor
            force = force + u_swing

            # Add some damping to prevent excessive oscillation during swing-up
            force = force - 0.3 * dx * np.tanh(3.0 * abs(theta))

        # --- Phase 2: Mid-Swing Transition ---
        elif abs(theta) > 0.25:
            # Enhanced directional damping with phase-plane awareness
            # Only damp when moving away from vertical (theta*dtheta > 0) and use smooth activation
            mid_swing_activation = np.exp(-4.0 * (abs(theta) - 0.5)**2)
            direction_factor = 0.5 * (1.0 + np.tanh(3.0 * theta * dtheta))  # 1 when moving away, 0 when returning
            if direction_factor > 0.6:  # Effectively when moving away
                K_d_midswing = 4.0  # Slightly increased for better momentum control
                midswing_damp = -K_d_midswing * dtheta * mid_swing_activation * direction_factor
                force = force + midswing_damp

            # Adaptive cart damping to suppress non-essential movement
            cart_damping = -0.4 * dx * np.tanh(4.0 * abs(theta))
            force = force + cart_damping

            # Enhanced predictive compensation with explicit centrifugal isolation
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            f_cart = -self.b_c * dx
            f_joint = -self.b_j * dtheta

            # Explicit centrifugal force isolation for clearer physics
            centrifugal_force = self.m * self.l * dtheta**2 * sin_theta
            cart_force_component = force[0] + f_cart
            temp_est = (cart_force_component + centrifugal_force) / self.Mtot

            denominator = self.l * (4.0/3.0 - self.m * cos_theta**2 / self.Mtot)
            theta_acc_est = (self.g * sin_theta - cos_theta * temp_est + f_joint / (self.m * self.l)) / denominator

            dtheta_pred = dtheta + theta_acc_est * 0.04  # Predict 2 timesteps ahead

            # Enhanced divergence metric with returning awareness
            pred_divergence = theta * dtheta_pred / (self.omega_n * L_COM)
            # Reduce correction when naturally returning to vertical
            returning_mask = 0.5 * (1.0 - np.tanh(4.0 * theta * dtheta))
            predictive_correction = -0.12 * pred_divergence * np.tanh(3.0 * abs(theta)) * returning_mask
            force = force + predictive_correction

        # --- Phase 3: Precision Hold (Small Angles) ---
        else:
            # Enhanced integral control with exponential gain scheduling
            # More aggressive decay near equilibrium to prevent overshoot
            K_i_adaptive = 0.85 * np.exp(-3.0 * (abs(theta) + 0.8*abs(dtheta)))

            # Dual-gated integral accumulation for better stability
            angle_condition = abs(theta) < 0.08
            velocity_condition = abs(dtheta) < 0.3
            if angle_condition and velocity_condition:
                self.integral_x += x * DT
                self.integral_x = np.clip(self.integral_x, -0.8, 0.8)  # Tighter bounds
            else:
                self.integral_x *= 0.98  # Slower decay to preserve correction

            integral_force = K_i_adaptive * self.integral_x
            force = force + integral_force

            # Precision damping tuned for minimal energy usage
            force = force - 0.3 * dx - 0.15 * dtheta

        # Adaptive cart damping that scales with system energy state
        energy_state = abs(theta) + 0.3 * abs(dtheta) + 0.1 * abs(dx)
        cart_damping_factor = 0.1 + 0.05 * np.tanh(2.0 * energy_state)
        force = force - cart_damping_factor * dx

        return float(force[0])

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