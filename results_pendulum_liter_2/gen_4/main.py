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
    Three Phase Controller - Inspired by human balancing reflexes
    PHASE 1: Emergency (save the falling pole)
    PHASE 2: Guidance (predict and intercept)
    PHASE 3: Precision (gentle corrections)

    Philosophy: Different problems need different solutions. Be decisive.
    """

    def __init__(self):
        # Phase thresholds
        self.EMERGENCY_ANGLE = 0.8    # rad (~45 degrees)
        self.EMERGENCY_OMEGA = 0.5    # rad/s
        self.GUIDANCE_ANGLE = 0.3     # rad (~17 degrees)
        self.GUIDANCE_OMEGA = 0.8     # rad/s
        self.PRECISION_ANGLE = 0.1    # rad (~5 degrees)
        self.PRECISION_SPEED = 0.3    # m/s

        # Emergency phase gains - strong and fast
        self.K_emergency_pos = 80.0   # Position feedback gain
        self.K_emergency_vel = 40.0   # Velocity feedback gain

        # Guidance phase gains - predictive
        self.K_predictive = 35.0      # Predictive momentum matching
        self.K_centering = 15.0       # Centering tendency

        # Precision phase gains - gentle
        self.Kp_angle = 25.0          # Proportional angle control
        self.Kp_omega = 12.0          # Proportional angular velocity
        self.Kp_position = 8.0        # Proportional position control
        self.Kp_velocity = 5.0        # Proportional velocity control

        # Internal state
        self.previous_theta = 0.0
        self.previous_x = 0.0

    def get_action(self, state):
        x, theta, dx, dtheta = state

        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Phase determination - CRISP boundaries
        is_emergency = (abs(theta) > self.EMERGENCY_ANGLE or
                       abs(dtheta) > self.EMERGENCY_OMEGA)

        is_guidance = (abs(theta) > self.GUIDANCE_ANGLE and
                      abs(dtheta) <= self.GUIDANCE_OMEGA and
                      abs(dx) < 1.0)

        is_precision = (abs(theta) <= self.PRECISION_ANGLE and
                       abs(dx) <= self.PRECISION_SPEED)

        # EMERGENCY PHASE: Save the pole at all costs
        if is_emergency:
            # Focus on angular stabilization, ignore position
            force = (-np.sign(theta) * self.K_emergency_pos * abs(theta)**1.2
                    - np.sign(dtheta) * self.K_emergency_vel * abs(dtheta)**1.1)

        # GUIDANCE PHASE: Predict and intercept
        elif is_guidance:
            # Predict where pole will cross zero and move cart there
            time_to_vertical = abs(theta / dtheta) if abs(dtheta) > 0.1 else 0.5

            # Predict cart movement needed - lead the target
            prediction_distance = 0.8 * time_to_vertical * abs(dtheta) * np.sign(theta)
            centering_force = -self.K_centering * x

            # Main control: push in direction of predicted fall
            predictive_force = self.K_predictive * prediction_distance * np.sign(dtheta)

            force = predictive_force + centering_force

        # PRECISION PHASE: Gentle corrections
        else:
            # Light P-control on all states
            force_angle = -self.Kp_angle * theta
            force_omega = -self.Kp_omega * dtheta
            force_position = -self.Kp_position * x
            force_velocity = -self.Kp_velocity * dx

            force = (force_angle + force_omega + force_position + force_velocity)

        return float(force)

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