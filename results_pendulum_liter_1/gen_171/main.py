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
    Hybrid Adaptive Integral LQR with Enhanced Damping and Robust Integral Control
    Combines proven gain scheduling with sophisticated integral action and enhanced
    angular velocity damping for superior stabilization performance.
    """

    def __init__(self):
        # System parameters
        m = M_POLE
        M = M_CART
        l = L_COM
        g = G
        Mtot = M + m
        denom0 = l * (4.0 / 3.0 - m / Mtot)
        b_c = FRICTION_CART
        b_j = FRICTION_JOINT

        # Physically accurate linearized A matrix with corrected joint friction
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0

        # theta_acc row (3) - using corrected joint friction term matching simulator
        A[3, 1] = g / denom0
        A[3, 2] = b_c / (Mtot * denom0)
        A[3, 3] = -b_j / (m * l * denom0)  # Matches simulator: f_joint/(m*l) / denom0

        # x_acc row (2)
        A[2, 1] = -(m * l / Mtot) * A[3, 1]
        A[2, 2] = -b_c / Mtot - (m * l / Mtot) * A[3, 2]
        A[2, 3] = b_j / (Mtot * denom0)

        # B matrix
        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        B[3, 0] = -1.0 / (Mtot * denom0)

        # Proven optimal LQR weights (Q[3]=3.2 achieves best performance)
        Q = np.diag([4.5, 44.0, 0.6, 3.2])
        R = np.array([[1.0]])

        # Solve LQR gains
        self.K = self.solve_lqr(A, B, Q, R)

        # Integral control parameters - slightly increased for faster cart centering
        self.integral_x = 0.0
        self.K_i = 0.85  # Integral gain for cart position

        # Natural frequency for normalized falling severity
        self.omega_n = np.sqrt(G / L_COM)

    def solve_lqr(self, A, B, Q, R):
        """Solve continuous-time LQR"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """Adaptive LQR with integral action and robust swing-up assist"""
        x, theta, dx, dtheta = state

        # Robust angle normalization using arctan2
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        state_vec = np.array([x, theta, dx, dtheta])
        base_force = -self.K @ state_vec

        # Enhanced gain scheduling with steeper transitions for faster response
        pos_gain = 1.0 + 0.6 * np.tanh(6.0 * max(0.0, abs(theta) - 0.55))
        vel_gain = 1.0 + 0.4 * np.tanh(5.0 * max(0.0, abs(dtheta) - 0.8))

        # Combined multiplicative gain
        adaptive_gain = pos_gain * vel_gain

        force = base_force * adaptive_gain

        # Predictive momentum-based swing-up assist with energy awareness
        if abs(theta) > 0.75:
            # Calculate current mechanical energy of pendulum
            E_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
            E_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
            E_current = E_kinetic + E_potential

            # Energy required to reach upright unstable equilibrium
            E_target = 2 * M_POLE * G * L_COM  # Potential energy at top

            # Energy deficit ratio (0 = full energy, 1 = no energy)
            energy_deficit = max(0.0, 1.0 - E_current / E_target)

            # Enhanced activation based on angle and energy deficit
            swing_activation = np.tanh(7.0 * (abs(theta) - 0.75)) * (0.7 + 0.3 * energy_deficit)

            # Predictive momentum compensation using estimated angular acceleration
            # Estimate angular acceleration from friction-aware dynamics
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            Mtot = M_CART + M_POLE
            denom = L_COM * (4.0/3.0 - M_POLE * cos_t**2 / Mtot)
            theta_acc_est = (G * sin_t - cos_t * (M_POLE * L_COM * dtheta**2 * sin_t) / Mtot) / denom
            # Add friction effects
            theta_acc_est -= (FRICTION_JOINT * dtheta) / (M_POLE * L_COM**2)

            # Predict future angular velocity
            dtheta_pred = dtheta + theta_acc_est * DT

            # Momentum-based falling factor with prediction
            momentum_factor = theta * dtheta_pred
            natural_freq = np.sqrt(G / L_COM)
            falling_factor = 1.0 + np.tanh(2.5 * momentum_factor / (L_COM * natural_freq))

            u_swing = 9.0 * swing_activation * np.sign(theta) * falling_factor
            force = force + u_swing

        # Velocity-gated integral action for better cart centering
        angle_gate = np.tanh(15.0 * (0.12 - abs(theta)))
        velocity_gate = np.tanh(10.0 * (1.2 - abs(dtheta)))
        integral_gate = angle_gate * velocity_gate

        if integral_gate > 0.15:  # Tighter threshold for more precise activation
            self.integral_x += x * DT
        else:
            # Adaptive leaky integration based on angle magnitude
            leak_rate = 0.92 + 0.06 * np.exp(-8.0 * abs(theta))
            self.integral_x *= leak_rate

        integral_force = self.K_i * integral_gate * self.integral_x
        force = force + integral_force

        # Enhanced cart velocity damping when near equilibrium for better settling
        if abs(theta) < 0.2:
            # Quadratic damping for smoother response near target
            cart_damp = -0.12 * dx * abs(dx) * (0.2 - abs(theta)) / 0.2
            force = force + cart_damp

        return float(force[0])

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
    # Apply force limits
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