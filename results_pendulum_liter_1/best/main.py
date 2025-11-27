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
    Adaptive Cross-Coupled LQR with Mid-Swing Damping and Predictive Compensation
    
    Key innovations:
    1. State-dependent cross-coupling in Q-matrix for better coordination
    2. Mid-swing angular damping to suppress momentum overshoot
    3. Predictive momentum compensation for earlier counteraction
    4. Refined integral control with dual gating
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

        # Build base A matrix
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

        # Base Q with proven optimal diagonal weights
        Q = np.diag([4.5, 44.0, 0.6, 3.2])
        # Add cross-coupling terms for theta-dtheta coordination
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
        """Adaptive LQR with cross-coupling, mid-swing damping, and predictive compensation"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        state_vec = np.array([x, theta, dx, dtheta])
        
        # Base LQR force with state-dependent cross-coupling enhancement
        # Intensify theta-dtheta coupling when angle is large
        cross_coupling_boost = 1.0 + 0.5 * np.tanh(3.0 * abs(theta))
        
        # Compute base force
        base_force = -self.K @ state_vec

        # Adaptive gain scheduling with proven parameters
        pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.6))
        vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 1.0))
        adaptive_gain = pos_gain * vel_gain

        force = base_force * adaptive_gain

        # Mid-swing angular damping (0.3-0.7 rad range)
        # Suppresses momentum overshoot during high-energy transition
        mid_swing_activation = np.exp(-4.0 * (abs(theta) - 0.5)**2)  # Gaussian centered at 0.5 rad
        if abs(theta) > 0.25 and abs(theta) < 0.85:
            K_d_midswing = 2.5
            midswing_damp = -K_d_midswing * dtheta * mid_swing_activation
            force = force + midswing_damp

        # Predictive momentum compensation
        # Estimate angular acceleration and extrapolate dtheta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Simplified theta acceleration estimate (gravity-dominated)
        theta_acc_est = (self.g * sin_theta - self.b_j * dtheta / (self.m * self.l)) / \
                        (self.l * (4.0/3.0 - self.m * cos_theta**2 / self.Mtot))
        
        dtheta_pred = dtheta + theta_acc_est * DT
        
        # Corrective term based on predicted divergence
        pred_divergence = theta * dtheta_pred / (self.omega_n * L_COM)
        predictive_correction = -0.12 * pred_divergence * np.tanh(2.0 * abs(theta))
        force = force + predictive_correction

        # Swing-up assist for large angles
        if abs(theta) > 0.8:
            swing_activation = np.tanh(6.0 * (abs(theta) - 0.8))
            falling_factor = 1.0 + np.tanh(2.0 * theta * dtheta / (L_COM * self.omega_n))
            u_swing = 8.0 * swing_activation * np.sign(theta) * falling_factor
            force = force + u_swing

        # Dual-gated integral action (angle AND velocity gating)
        angle_gate = np.tanh(12.0 * (0.1 - abs(theta)))
        velocity_gate = np.tanh(8.0 * (1.0 - abs(dtheta)))
        integral_gate = angle_gate * velocity_gate

        if integral_gate > 0.1:
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -1.5, 1.5)
        else:
            self.integral_x *= 0.95

        integral_force = self.K_i * integral_gate * self.integral_x
        force = force + integral_force

        return float(force[0])

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