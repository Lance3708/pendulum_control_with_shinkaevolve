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
    Phase-Aware Rule-Based Controller

    Key innovations:
    1. Distinct control strategies for different operational phases
    2. Momentum prediction for proactive control
    3. Energy-efficient minimal-force corrections
    4. Friction-aware compensation
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT

        # State variables for memory-based control
        self.prev_theta = 0.0
        self.prev_dtheta = 0.0
        self.integral_x = 0.0
        self.integral_theta = 0.0

    def get_action(self, state):
        """Phase-aware rule-based control with predictive elements"""
        x, theta, dx, dtheta = state

        # Normalize angle to [-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Calculate angular acceleration estimate for prediction
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Estimate angular acceleration considering gravity and friction
        theta_acc = (self.g * sin_theta - self.b_j * dtheta / (self.m * self.l)) / self.l

        # Predict future state (0.1 seconds ahead)
        pred_dt = 0.1
        pred_theta = theta + dtheta * pred_dt + 0.5 * theta_acc * pred_dt**2
        pred_dtheta = dtheta + theta_acc * pred_dt

        # Calculate how fast the pole is falling
        falling_speed = abs(theta * dtheta)

        # Determine which phase we're in and apply appropriate strategy
        if abs(theta) > 0.7:  # Emergency phase - large angle recovery
            force = self._emergency_control(theta, dtheta, pred_theta, x, dx)
        elif abs(theta) > 0.3:  # Transition phase - swing through vertical
            force = self._transition_control(theta, dtheta, pred_theta, x, dx)
        elif abs(theta) > 0.1:  # Fine positioning phase - getting close to vertical
            force = self._fine_position_control(theta, dtheta, x, dx)
        else:  # Precision holding phase - maintaining balance
            force = self._precision_hold(theta, dtheta, x, dx, pred_theta)

        # Apply friction compensation
        friction_compensation = self.b_c * dx
        force += friction_compensation

        # Update memory variables
        self.prev_theta = theta
        self.prev_dtheta = dtheta

        return float(np.clip(force, -100.0, 100.0))

    def _emergency_control(self, theta, dtheta, pred_theta, x, dx):
        """Aggressive swing-up when pole is far from vertical"""
        # Push in direction of fall with magnitude based on urgency
        urgency = abs(theta) * (1 + abs(dtheta))
        force = 25.0 * np.sign(theta) * urgency

        # Counteract predicted overshoot
        if abs(pred_theta) > abs(theta):
            force -= 10.0 * np.sign(pred_theta) * (abs(pred_theta) - abs(theta))

        # Move cart toward center to prepare for balance
        if abs(x) > 0.5:
            force -= 5.0 * np.sign(x)
        else:
            force -= 2.0 * x  # Gentle centering

        return force

    def _transition_control(self, theta, dtheta, pred_theta, x, dx):
        """Manage momentum during swing-through"""
        # Predictive damping - counteract predicted motion
        pred_force = -8.0 * pred_theta - 3.0 * dtheta

        # Coordinate cart position with pole angle
        coord_force = -10.0 * theta - 2.0 * x - 1.0 * dx

        # Blend based on how close we are to vertical
        blend = (abs(theta) - 0.3) / 0.4  # 1 when at 0.7, 0 when at 0.3
        force = blend * pred_force + (1 - blend) * coord_force

        return force

    def _fine_position_control(self, theta, dtheta, x, dx):
        """Coordinated control to bring system to rest at origin"""
        # Balance pole while centering cart
        pole_force = -50.0 * theta - 10.0 * dtheta
        cart_force = -10.0 * x - 3.0 * dx

        # Blend forces based on relative importance
        force = pole_force + cart_force

        return force

    def _precision_hold(self, theta, dtheta, x, dx, pred_theta):
        """Minimal energy corrections to maintain balance"""
        # Very gentle proportional control
        pole_force = -30.0 * theta - 8.0 * dtheta
        cart_force = -8.0 * x - 2.0 * dx

        # Predictive correction for tiny deviations
        pred_correction = -5.0 * pred_theta
        force = pole_force + cart_force + pred_correction

        # Integral action for persistent errors (very gentle)
        self.integral_theta += theta * DT
        self.integral_x += x * DT

        # Limit integral windup
        self.integral_theta = np.clip(self.integral_theta, -0.5, 0.5)
        self.integral_x = np.clip(self.integral_x, -0.5, 0.5)

        force += 0.5 * self.integral_theta + 0.3 * self.integral_x

        return force

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