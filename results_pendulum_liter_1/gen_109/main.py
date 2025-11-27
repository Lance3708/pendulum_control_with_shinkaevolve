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
    Phase-Predictive LQR Controller with Discrete-Time Alignment
    Uses ZOH discretization and phase-aware control switching for optimal performance.
    """

    def __init__(self):
        from scipy.linalg import solve_discrete_are
        from scipy.signal import cont2discrete
        
        m = M_POLE
        M = M_CART
        l = L_COM
        g = G
        Mtot = M + m
        denom0 = l * (4.0 / 3.0 - m / Mtot)
        b_c = FRICTION_CART
        b_j = FRICTION_JOINT

        # Continuous-time A matrix with friction
        Ac = np.zeros((4, 4))
        Ac[0, 2] = 1.0
        Ac[1, 3] = 1.0
        Ac[3, 1] = g / denom0
        Ac[3, 2] = b_c / (Mtot * denom0)
        Ac[3, 3] = -b_j / (m * l * denom0)
        Ac[2, 1] = -(m * l / Mtot) * Ac[3, 1]
        Ac[2, 2] = -b_c / Mtot - (m * l / Mtot) * Ac[3, 2]
        Ac[2, 3] = b_j / (Mtot * denom0)

        # Continuous-time B matrix
        Bc = np.zeros((4, 1))
        Bc[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        Bc[3, 0] = -1.0 / (Mtot * denom0)

        # Discretize using ZOH for alignment with Euler integration
        Cc = np.eye(4)
        Dc = np.zeros((4, 1))
        Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, Cc, Dc), DT, method='zoh')

        # Discrete-time LQR weights - tuned for faster convergence
        Q = np.diag([4.8, 46.0, 0.65, 3.4])
        R = np.array([[1.0]])

        # Solve discrete-time Riccati equation
        P = solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        
        # Store matrices for prediction
        self.Ad = Ad
        self.Bd = Bd
        
        # Natural frequency for normalized swing-up
        self.omega_n = np.sqrt(G / L_COM)
        
        # Integral state for precision phase
        self.integral_x = 0.0
        self.k_i = 0.025

    def get_action(self, state):
        x, theta, dx, dtheta = state

        # Robust angle normalization using arctan2
        theta_norm = np.arctan2(np.sin(theta), np.cos(theta))
        abs_theta = abs(theta_norm)

        state_vec = np.array([x, theta_norm, dx, dtheta])
        
        # Base LQR force
        base_force = -self.K @ state_vec

        # Phase detection
        if abs_theta > 0.5:
            # RECOVERY PHASE: Aggressive swing-up with predictive correction
            
            # Predict next state
            predicted = self.Ad @ state_vec + self.Bd.flatten() * base_force
            
            # Predictive correction: counter predicted overshoot
            pred_correction = -0.15 * (predicted[1] - theta_norm) * np.sign(theta_norm)
            
            # Normalized angular momentum for physically consistent scaling
            angular_momentum = theta_norm * dtheta
            normalized_momentum = angular_momentum / (L_COM * self.omega_n)
            falling_severity = 1.0 + 0.6 * np.tanh(2.5 * normalized_momentum)
            
            # Swing-up assist with smooth activation
            swing_activation = np.tanh(5.0 * (abs_theta - 0.5))
            u_swing = 10.0 * swing_activation * np.sign(theta_norm) * falling_severity
            
            # Velocity boost for rapid recovery
            vel_boost = 1.0 + 0.4 * np.tanh(3.0 * max(0.0, abs(dtheta) - 0.8))
            
            force = base_force * vel_boost + u_swing + pred_correction
            
            # Reset integral during recovery
            self.integral_x = 0.0
            
        elif abs_theta > 0.1:
            # TRANSITION PHASE: Smooth blend between recovery and precision
            
            blend = (abs_theta - 0.1) / 0.4  # 0 at 0.1 rad, 1 at 0.5 rad
            
            # Mild gain boost that fades as we approach precision zone
            adaptive_gain = 1.0 + 0.25 * blend * np.tanh(4.0 * abs_theta)
            
            force = base_force * adaptive_gain
            
            # Gradual integral buildup
            self.integral_x *= 0.95  # Decay during transition
            
        else:
            # PRECISION PHASE: High-accuracy LQR with integral action
            
            # Soft-switched integral action for zero steady-state error
            integral_gate = np.tanh(15.0 * (0.1 - abs_theta))
            self.integral_x += x * DT * integral_gate
            self.integral_x = np.clip(self.integral_x, -2.0, 2.0)  # Anti-windup
            
            # Precision force with integral correction
            force = base_force + self.k_i * self.integral_x

        return float(np.clip(force[0] if hasattr(force, '__len__') else force, -100.0, 100.0))

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