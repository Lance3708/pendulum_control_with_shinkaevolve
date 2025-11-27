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
    Adaptive Integral LQR Controller with Velocity-Aware Gain Scheduling
    Combines physically accurate friction modeling with intelligent gain adaptation
    and integral action for ultra-fast stabilization while maintaining energy efficiency.
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

        # Enhanced LQR weights with increased angular velocity damping for faster settling
        Q = np.diag([4.5, 44.0, 0.6, 3.4])
        R = np.array([[1.0]])

        # Solve LQR gains
        self.K = self.solve_lqr(A, B, Q, R)

        # Integral control parameters
        self.integral_x = 0.0
        self.integral_momentum = 0.0  # Track angular momentum deficit
        self.K_i = 0.8  # Integral gain for cart position
        self.K_mom = 0.015  # Momentum integral gain

        # Natural frequency for normalized falling severity
        self.omega_n = np.sqrt(G / L_COM)

    def solve_lqr(self, A, B, Q, R):
        """Solve continuous-time LQR"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """Adaptive LQR with integral action and additive swing-up assist for large angles"""
        x, theta, dx, dtheta = state

        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        state_vec = np.array([x, theta, dx, dtheta])
        base_force = -self.K @ state_vec

        # Proven optimal gain scheduling from best performer
        pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.6))
        vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 1.0))

        # Combined multiplicative gain
        adaptive_gain = pos_gain * vel_gain

        force = base_force * adaptive_gain

        # Enhanced swing-up assist using mechanical energy deficit with predictive compensation
        if abs(theta) > 0.8:
            # Calculate current mechanical energy of pendulum
            E_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
            E_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
            E_current = E_kinetic + E_potential

            # Energy required to reach upright unstable equilibrium
            E_target = 2 * M_POLE * G * L_COM  # Potential energy at top

            # Energy deficit ratio (0 = full energy, 1 = no energy)
            energy_deficit = max(0.0, 1.0 - E_current / E_target)

            # Activation based on angle and energy deficit
            swing_activation = np.tanh(6.0 * (abs(theta) - 0.8)) * energy_deficit

            # Directional assist based on pole angle and falling momentum
            natural_freq = np.sqrt(G / L_COM)
            falling_factor = 1.0 + np.tanh(2.0 * theta * dtheta / (L_COM * natural_freq))

            # Predictive momentum compensation with friction-aware dynamics
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            Mtot = M_CART + M_POLE
            denom0 = L_COM * (4.0/3.0 - M_POLE * cos_t**2 / Mtot)
            temp_pred = (M_POLE * L_COM * dtheta**2 * sin_t) / Mtot
            # Include friction terms in prediction
            theta_acc_pred = (G * sin_t - cos_t * temp_pred + FRICTION_JOINT * dtheta / (M_POLE * L_COM)) / denom0
            theta_pred = theta + dtheta * DT + 0.5 * theta_acc_pred * DT**2
            predictive_boost = 1.0 + 0.15 * np.tanh(3.0 * (abs(theta_pred) - abs(theta)))

            u_swing = 8.0 * swing_activation * np.sign(theta) * falling_factor * predictive_boost
            force = force + u_swing

        # Gaussian-shaped mid-swing damping for critical transition zone
        if 0.3 < abs(theta) < 0.7:
            # Gaussian activation centered at 0.5 rad with sigma=0.2
            gaussian_activation = np.exp(-0.5 * ((abs(theta) - 0.5) / 0.2)**2)
            # Damping proportional to angular velocity and activation
            midswing_damp = -2.5 * gaussian_activation * dtheta
            force = force + midswing_damp

        # Dual-gated integral action for cart position and momentum
        # Only activates when pole is nearly upright AND moving slowly
        angle_gate = np.tanh(12.0 * (0.1 - abs(theta)))
        velocity_gate = np.tanh(8.0 * (1.0 - abs(dtheta)))
        integral_gate = angle_gate * velocity_gate

        if integral_gate > 0.1:  # Only update integrals when gate is significant
            self.integral_x += x * DT
            # Momentum integral: track angular velocity deficit
            momentum_error = -dtheta  # We want dtheta -> 0
            self.integral_momentum += momentum_error * DT
            self.integral_momentum = np.clip(self.integral_momentum, -0.5, 0.5)
        else:
            # Dynamic decay: stronger when angle is large, gentler near equilibrium
            decay_rate = 0.90 + 0.08 * np.exp(-10.0 * abs(theta))
            self.integral_x *= decay_rate
            self.integral_momentum *= 0.9  # Leaky decay for momentum integral

        integral_force = self.K_i * integral_gate * self.integral_x
        momentum_force = self.K_mom * integral_gate * self.integral_momentum * np.sign(theta + 1e-8)
        force = force + integral_force + momentum_force

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