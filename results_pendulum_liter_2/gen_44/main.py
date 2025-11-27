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
    Phase-Based Rule Controller

    Philosophy: "Different phases need fundamentally different control strategies"

    Four distinct phases with completely different control logic:
    1. EMERGENCY: Aggressive cart movement to catch falling pole
    2. RECOVERY: Guided energy management to bring pole vertical
    3. BALANCING: Precise position control while maintaining balance
    4. HOLD: Minimal intervention to maintain perfect state
    """

    def __init__(self):
        # System parameters for physics calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m

        # Position integral for final centering
        self.integral_x = 0.0

        # Phase tracking
        self.last_phase = "UNKNOWN"

    def get_action(self, state):
        """Phase-based control with simple, interpretable rules"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Determine current phase based on state
        if abs(theta) > 0.7:
            current_phase = "EMERGENCY"
        elif abs(theta) > 0.3:
            current_phase = "RECOVERY"
        elif abs(theta) > 0.05 or abs(dtheta) > 0.5:
            current_phase = "BALANCING"
        else:
            current_phase = "HOLD"

        # Apply phase-specific control strategy
        if current_phase == "EMERGENCY":
            # EMERGENCY: Save the pole from falling!
            # Move cart aggressively toward where pole is falling
            urgency = min(3.0, abs(theta) / 0.3)  # Scale urgency with angle
            target_x = x + 2.0 * np.sign(theta) * urgency  # Move in falling direction
            force = 15.0 * (target_x - x) - 5.0 * dx  # PD control to target

        elif current_phase == "RECOVERY":
            # RECOVERY: Guide pole back to vertical with energy management
            # Push cart to create torque that brings pole up
            if theta * dtheta > 0:
                # Pole moving away from vertical - resist
                force = -8.0 * np.sign(theta) * abs(theta) - 2.0 * dx
            else:
                # Pole moving toward vertical - assist
                force = 12.0 * np.sign(theta) * abs(theta) - 1.5 * dx

        elif current_phase == "BALANCING":
            # BALANCING: Focus on position control while maintaining balance
            # Primary goal: center the cart
            position_force = -4.0 * x - 2.0 * dx

            # Secondary goal: keep pole vertical
            angle_force = -25.0 * theta - 8.0 * dtheta

            # Blend them, prioritizing angle when it's larger
            angle_weight = min(1.0, abs(theta) / 0.1)
            force = (1.0 - angle_weight) * position_force + angle_weight * angle_force

            # Update integral for position centering
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -1.0, 1.0)
            force -= 1.5 * self.integral_x

        else:  # HOLD phase
            # HOLD: Minimal intervention to maintain perfect state
            # Very gentle corrections only
            force = -1.0 * x - 0.5 * dx - 5.0 * theta - 2.0 * dtheta

            # Slowly decay integral to prevent windup
            self.integral_x *= 0.99
            force -= 0.5 * self.integral_x

        # Phase transition smoothing
        if current_phase != self.last_phase:
            # Smooth transitions to avoid jerky behavior
            force *= 0.7
        self.last_phase = current_phase

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