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
    Discrete-Time LQR with Integral Control and Error-Mode-Weighted Adaptation
    Combines discrete-time control theory with integral action for perfect stabilization
    and physics-informed gain adaptation for ultra-fast convergence.
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

        # Continuous-time state-space matrices (4 states)
        A_cont = np.zeros((4, 4))
        A_cont[0, 2] = 1.0
        A_cont[1, 3] = 1.0

        # theta_acc row (3) - matches simulator physics exactly
        A_cont[3, 1] = g / denom0
        A_cont[3, 2] = b_c / (Mtot * denom0)
        A_cont[3, 3] = -b_j / (m * l * denom0)

        # x_acc row (2)
        A_cont[2, 1] = -(m * l / Mtot) * A_cont[3, 1]
        A_cont[2, 2] = -b_c / Mtot - (m * l / Mtot) * A_cont[3, 2]
        A_cont[2, 3] = b_j / (Mtot * denom0)

        # B matrix
        B_cont = np.zeros((4, 1))
        B_cont[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        B_cont[3, 0] = -1.0 / (Mtot * denom0)

        # Convert to discrete-time using matrix exponential (zero-order hold)
        from scipy.linalg import expm
        M_cont = np.zeros((5, 5))
        M_cont[:4, :4] = A_cont
        M_cont[:4, 4:] = B_cont
        M_disc = expm(M_cont * DT)
        
        A_disc = M_disc[:4, :4]
        B_disc = M_disc[:4, 4:]

        # Optimized LQR weights for discrete-time system
        # Based on analysis of best-performing previous controllers
        Q = np.diag([4.5, 44.0, 0.6, 3.2])  # Tuned for fast stabilization
        R = np.array([[1.0]])

        # Solve discrete-time LQR
        self.K = self.solve_dlqr(A_disc, B_disc, Q, R)
        
        # Integral gain for cart position
        self.integral_x = 0.0
        self.Ki = -0.8  # Tuned integral gain

    def solve_dlqr(self, A, B, Q, R):
        """Solve discrete-time LQR using algebraic Riccati equation"""
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def get_action(self, state):
        """Discrete-time LQR control with integral action and smooth gain adaptation"""
        x, theta, dx, dtheta = state

        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Update integral term for cart position
        self.integral_x += x * DT

        # Base LQR control
        state_vec = np.array([x, theta, dx, dtheta])
        base_force = -self.K @ state_vec

        # Add integral action for cart position
        integral_force = self.Ki * self.integral_x

        # Smooth error-mode-weighted blending
        # Physics-informed: prioritize angle control when angle is large,
        # prioritize velocity damping when angular velocity is high
        theta_weight = 1.0 / (1.0 + np.exp(-8.0 * (abs(theta) - 0.3 * abs(dtheta))))
        
        # Adaptive gain: stronger control during large deviations
        adaptive_gain = 1.0 + 0.4 * theta_weight * np.tanh(2.0 * abs(theta))

        # Combine all components
        force = (base_force + integral_force) * adaptive_gain

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