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
class HighPerformanceLQRController:
    """
    High-performance LQR controller with optimized gains for extreme pendulum dynamics.
    
    Structural improvements:
    - Modular design with separate model building and gain computation
    - Configurable cost weights via central config
    - Clean state preprocessing pipeline
    - Optimized Q/R for aggressive stabilization and precision
    """
    
    def __init__(self):
        # Centralized configuration for easy tuning
        self.config = {
            'Q_weights': [15.0, 35.0, 2.0, 4.0],  # [x, theta, dx, dtheta]
            'R_weight': 0.8                        # Control effort penalty
        }
        
        A, B = build_linear_model()
        self.K = self._solve_lqr(A, B)
    
    def _solve_lqr(self, A, B):
        """Solve continuous-time LQR problem using configured weights."""
        from scipy.linalg import solve_continuous_are
        Q = np.diag(self.config['Q_weights'])
        R = np.array([[self.config['R_weight']]])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    
    def preprocess_state(self, state):
        """Normalize angle to [-pi, pi] range."""
        x, theta, dx, dtheta = state
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return np.array([x, theta, dx, dtheta])
    
    def compute_control(self, state_vector):
        """Compute control action using LQR law: u = -Kx."""
        return -self.K @ state_vector
    
    def get_action(self, state):
        """Main control interface."""
        processed_state = self.preprocess_state(state)
        force = self.compute_control(processed_state)
        return float(force[0])

# Initialize optimized controller
controller = HighPerformanceLQRController()

def get_control_action(state):
    """External API for control action."""
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