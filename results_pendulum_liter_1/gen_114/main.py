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
import numpy as np
from scipy.linalg import solve_continuous_are, dlqr
from scipy.signal import cont2discrete

class Controller:
    """
    Feedback Linearization + Discrete LQR + Gated Integral Controller
    
    Novel decoupling control: computes optimal desired theta_ddot from discrete LQR,
    then inverts exact nonlinear dynamics to achieve it precisely.
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT
        self.Mtot = self.M + self.m
        self.DT = 0.02
        denom0 = self.l * (4.0 / 3.0 - self.m / self.Mtot)

        # Continuous-time A and B matrices (for linearization reference)
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[3, 1] = self.g / denom0
        A[3, 2] = self.b_c / (self.Mtot * denom0)
        A[3, 3] = -self.b_j / (self.m * self.l * denom0)
        A[2, 1] = -(self.m * self.l / self.Mtot) * A[3, 1]
        A[2, 2] = -self.b_c / self.Mtot - (self.m * self.l / self.Mtot) * A[3, 2]
        A[2, 3] = self.b_j / (self.Mtot * denom0)

        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / self.Mtot + (self.m * self.l) / (self.Mtot**2 * denom0)
        B[3, 0] = -1.0 / (self.Mtot * denom0)

        # Store linear theta row for prediction
        self.A_theta_row = A[3, :].copy()
        self.B_theta = B[3, 0]

        # Discrete-time LQR (ZOH matching simulator)
        Ad, Bd, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), self.DT, method='zoh')

        # Optimized Q with refined dtheta weight
        Q = np.diag([4.5, 44.0, 0.6, 3.28])
        R = np.array([[1.0]])

        # Discrete LQR gains
        self.K, _, _ = dlqr(Ad, Bd, Q, R)

        # Integral control parameters
        self.integral_x = 0.0
        self.K_i = 8.0  # Tuned for zero steady-state x error
        self.integral_limit = 2.0

    def get_action(self, state):
        x, theta, dx, dtheta = state

        # Robust arctan2 normalization (rec 1)
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        state_vec = np.array([x, theta, dx, dtheta])

        # Proven adaptive gain scheduling
        pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.6))
        vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 1.0))
        adaptive_gain = pos_gain * vel_gain

        # Boosted linear LQR force
        u_lin = -self.K @ state_vec * adaptive_gain

        # Desired theta_ddot from linear closed-loop prediction
        desired_tt = self.A_theta_row @ state_vec + self.B_theta * u_lin

        # Nonlinear inverse dynamics to achieve desired_tt exactly
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        f_c = -self.b_c * dx
        f_j = -self.b_j * dtheta
        denom_th = self.l * (4.0 / 3.0 - self.m * cos_th**2 / self.Mtot)
        cent = self.m * self.l * dtheta**2 * sin_th

        if abs(cos_th) < 1e-3 or abs(denom_th) < 1e-6:
            # Fallback near singularity
            force = float(u_lin[0])
        else:
            temp = (self.g * sin_th + f_j / (self.m * self.l) - desired_tt * denom_th) / cos_th
            force = self.Mtot * temp - f_c - cent

        # Soft-switched integral on x (rec 5)
        gate = np.tanh(12.0 * (0.1 - abs(theta)))
        self.integral_x += gate * x * self.DT
        self.integral_x = np.tanh(self.integral_x / self.integral_limit) * self.integral_limit
        force += self.K_i * gate * self.integral_x

        return float(np.clip(force, -100.0, 100.0))


# Initialize controller
controller = Controller()

def get_control_action(state):
    return controller.get_action(state)
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