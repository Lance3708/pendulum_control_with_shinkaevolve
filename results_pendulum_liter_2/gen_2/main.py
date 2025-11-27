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
    Emergency-Phase Controller: Different strategies for different situations

    Philosophy: When the rod is falling fast, push HARD toward where it's falling.
    When it's nearly balanced, use gentle correction. Be decisive, not timid.
    """

    def __init__(self):
        # Control parameters for different phases
        self.emergency_angle_threshold = 0.4  # rad (~23 degrees)
        self.precision_angle_threshold = 0.1   # rad (~6 degrees)

        # Emergency mode: push toward falling direction
        self.emergency_gain = 80.0

        # Guidance mode: predictive compensation
        self.guidance_gain = 45.0

        # Precision mode: gentle correction
        self.precision_pos_gain = 25.0
        self.precision_angle_gain = 35.0
        self.precision_vel_gain = 8.0
        self.precision_angvel_gain = 12.0

        # Integral control for precision mode
        self.integral_x = 0.0
        self.integral_gain = 0.4

        # System parameters for prediction
        self.g = G
        self.l = L_COM

    def get_action(self, state):
        """Phase-based control: emergency, guidance, or precision mode"""
        x, theta, dx, dtheta = state

        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Calculate urgency (how critical the situation is)
        angle_magnitude = abs(theta)
        angular_speed = abs(dtheta)
        urgency = angle_magnitude + 0.3 * angular_speed

        # PHASE 1: EMERGENCY MODE - rod is falling fast
        if urgency > 0.5 or angle_magnitude > self.emergency_angle_threshold:
            # Push HARD toward where the rod is falling
            # If theta positive (falling right), push right (positive force)
            emergency_force = self.emergency_gain * np.sign(theta) * min(urgency, 1.5)

            # Also counteract cart drift during emergency
            cart_correction = -12.0 * x - 4.0 * dx

            force = emergency_force + cart_correction

        # PHASE 2: GUIDANCE MODE - moderate angles, bring toward vertical
        elif angle_magnitude > self.precision_angle_threshold:
            # Predictive compensation: where will the rod be in 0.1 seconds?
            # Simple gravity-dominated prediction
            theta_pred = theta + dtheta * 0.1 + 0.5 * (self.g/self.l) * np.sin(theta) * (0.1**2)

            # Push to counteract predicted fall
            guidance_force = self.guidance_gain * theta_pred

            # Cart positioning
            cart_force = -15.0 * x - 6.0 * dx

            # Angular velocity damping
            angvel_damp = -8.0 * dtheta

            force = guidance_force + cart_force + angvel_damp

        # PHASE 3: PRECISION MODE - nearly balanced, gentle correction
        else:
            # Gentle PD control with integral action
            pos_force = -self.precision_pos_gain * x
            angle_force = -self.precision_angle_gain * theta
            vel_force = -self.precision_vel_gain * dx
            angvel_force = -self.precision_angvel_gain * dtheta

            # Integral control for cart position (only when nearly balanced)
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -2.0, 2.0)
            integral_force = -self.integral_gain * self.integral_x

            force = pos_force + angle_force + vel_force + angvel_force + integral_force

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