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
    Aggressive LQR Controller with State-Dependent Gain Scheduling.

    Features:
    1. Higher Q weights for faster convergence and better precision
    2. Lower R penalty to allow more aggressive control
    3. Gain scheduling: smoother transition between recovery and fine-tuning phases
    4. Optimized for the challenging physical setup (heavy, long pole with high friction)
    """

    def __init__(self):
        # System parameters
        m = M_POLE  # Pole mass
        M = M_CART  # Cart mass
        l = L_COM   # Distance to center of mass
        g = G       # Gravity

        # Linearized state-space model around upright equilibrium
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m+M)*g/(M*l), 0, 0]
        ])

        B = np.array([
            [0],
            [0],
            [1/M],
            [-1/(M*l)]
        ])

        # Aggressive LQR weights for fast stabilization
        # Q: High weights on theta and dtheta for tight angle control
        Q_aggressive = np.diag([15.0, 180.0, 2.0, 25.0])  # [x, theta, dx, dtheta]
        R_aggressive = np.array([[0.3]])  # Lower penalty allows stronger control

        # Conservative LQR weights for large angle recovery (more stable)
        Q_conservative = np.diag([5.0, 60.0, 0.5, 8.0])
        R_conservative = np.array([[0.8]])

        # Solve both LQR gains
        self.K_aggressive = self.solve_lqr(A, B, Q_aggressive, R_aggressive)
        self.K_conservative = self.solve_lqr(A, B, Q_conservative, R_conservative)

    def solve_lqr(self, A, B, Q, R):
        """Solve continuous-time LQR problem"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """LQR control with smooth gain scheduling based on angle magnitude"""
        x, theta, dx, dtheta = state

        # Angle wrapping to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        state_vec = np.array([x, theta, dx, dtheta])

        abs_theta = abs(theta)

        # Smooth blending using sigmoid-like function
        # Transition zone: 0.15 rad to 0.5 rad
        # Below 0.15 rad: fully aggressive (fine-tuning)
        # Above 0.5 rad: fully conservative (recovery)
        theta_low = 0.15
        theta_high = 0.5

        if abs_theta <= theta_low:
            alpha = 1.0  # Fully aggressive
        elif abs_theta >= theta_high:
            alpha = 0.0  # Fully conservative
        else:
            # Smooth cosine interpolation
            t = (abs_theta - theta_low) / (theta_high - theta_low)
            alpha = 0.5 * (1 + np.cos(np.pi * t))

        # Blend gains
        K_blended = alpha * self.K_aggressive + (1 - alpha) * self.K_conservative

        force = -K_blended @ state_vec
        return float(force[0])
=======

# Initialize controller
controller = Controller()

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