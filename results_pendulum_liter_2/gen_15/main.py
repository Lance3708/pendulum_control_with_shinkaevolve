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
    Phase-Based Controller with Distinct Control Strategies

    Key innovations:
    1. Four distinct phases with completely different control strategies
    2. No matrix operations or optimal control theory
    3. Simple, interpretable rules for each phase
    4. Sharp transitions between phases (no blending)
    """

    def __init__(self):
        # System parameters for calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m

        # Natural frequency for normalized calculations
        self.omega_n = np.sqrt(G / L_COM)

        # Integral control for precision phase
        self.integral_x = 0.0
        self.integral_theta = 0.0

        # Previous state for prediction
        self.prev_state = None

    def get_action(self, state):
        """Phase-based control with distinct strategies for each situation"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Determine current phase
        abs_theta = abs(theta)
        abs_dtheta = abs(dtheta)

        # Phase 1: Emergency (rod falling fast)
        if abs_theta > 0.7:
            return self.emergency_response(x, theta, dx, dtheta)

        # Phase 2: Convergence (getting close to vertical)
        elif abs_theta > 0.2:
            return self.convergence_response(x, theta, dx, dtheta)

        # Phase 3: Stabilization (nearly balanced)
        elif abs_theta > 0.05 or abs_dtheta > 0.1:
            return self.stabilization_response(x, theta, dx, dtheta)

        # Phase 4: Precision Hold (nearly perfect)
        else:
            return self.precision_hold_response(x, theta, dx, dtheta)

    def emergency_response(self, x, theta, dx, dtheta):
        """Aggressive swing-up to quickly get rod vertical"""
        # Push cart toward falling direction with urgency
        urgency = min(3.0, abs(theta) / 0.3)  # Scale with angle

        # Base force: push toward where rod is falling
        base_force = 15.0 * urgency * np.sign(theta)

        # Add energy injection based on angular velocity
        # If rod is falling in the right direction, add more energy
        if theta * dtheta > 0:  # Falling away from vertical
            energy_boost = 8.0 * np.sign(theta)
        else:  # Falling toward vertical
            energy_boost = 3.0 * np.sign(theta)

        # Cart position correction (secondary concern during emergency)
        pos_correction = -2.0 * np.tanh(x / 2.0)

        force = base_force + energy_boost + pos_correction

        return float(np.clip(force, -100.0, 100.0))

    def convergence_response(self, x, theta, dx, dtheta):
        """Controlled guidance to vertical position"""
        # Predict where rod will be in 0.1 seconds
        theta_pred = theta + dtheta * 0.1

        # Push cart to intercept predicted rod position
        intercept_force = 12.0 * np.sign(theta_pred) * (1.0 + abs(theta) / 0.5)

        # Dampen angular velocity
        damping_force = -4.0 * dtheta

        # Gentle cart position correction
        pos_correction = -3.0 * np.tanh(x / 1.5)

        # Cart velocity damping
        vel_damping = -1.5 * dx

        force = intercept_force + damping_force + pos_correction + vel_damping

        return float(np.clip(force, -100.0, 100.0))

    def stabilization_response(self, x, theta, dx, dtheta):
        """Fine-tune position and angle"""
        # Primary goal: keep rod vertical
        angle_correction = -25.0 * theta - 8.0 * dtheta

        # Secondary goal: center cart
        pos_correction = -8.0 * x - 3.0 * dx

        # Predictive component: anticipate where rod is going
        pred_component = -5.0 * theta * dtheta

        force = angle_correction + pos_correction + pred_component

        # Update integral for position drift
        self.integral_x += x * DT
        self.integral_x = np.clip(self.integral_x, -0.5, 0.5)
        integral_force = -2.0 * self.integral_x

        force += integral_force

        return float(np.clip(force, -100.0, 100.0))

    def precision_hold_response(self, x, theta, dx, dtheta):
        """Minimize drift and maintain perfect balance"""
        # Very gentle angle correction
        angle_correction = -15.0 * theta - 5.0 * dtheta

        # Strong position correction to eliminate drift
        pos_correction = -12.0 * x - 4.0 * dx

        # Update integrals for both position and angle
        self.integral_x += x * DT
        self.integral_theta += theta * DT

        # Clip integrals to prevent windup
        self.integral_x = np.clip(self.integral_x, -0.3, 0.3)
        self.integral_theta = np.clip(self.integral_theta, -0.05, 0.05)

        # Integral corrections
        integral_force = -3.0 * self.integral_x - 20.0 * self.integral_theta

        force = angle_correction + pos_correction + integral_force

        return float(np.clip(force, -100.0, 100.0))

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