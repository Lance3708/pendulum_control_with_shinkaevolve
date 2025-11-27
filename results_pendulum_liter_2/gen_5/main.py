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
    Cascade Resonance Controller

    A physics-first approach that separates concerns into a hierarchy:
    1. Strategic Layer: Where should the cart be? -> Calculates target lean angle
    2. Tactical Layer: Keep the rod at that angle. -> Calculates force

    This avoids complex matrices in favor of intuitive physical relationships.
    It uses non-linear gains to be gentle when stable (saving energy)
    and aggressive when falling (saving the system).
    """

    def __init__(self):
        # No complex pre-computation needed
        pass

    def get_action(self, state):
        x, theta, dx, dtheta = state

        # --- 1. STRATEGIC PHASE: POSITION CONTROL ---
        # We manipulate the cart position by requesting a specific rod angle.
        # To move Left (decrease x), we must lean the rod Left (negative theta).

        # Gains tuned for the 2.5m heavy rod (slow dynamics)
        K_pos_p = 0.10  # Desired angle (rad) per meter of position error
        K_pos_d = 0.15  # Desired angle (rad) per m/s of velocity

        # Calculate the "virtual" setpoint for the rod angle
        target_theta = -(K_pos_p * x + K_pos_d * dx)

        # Safety Saturation:
        # Never request a lean angle > 0.20 rad (approx 11 deg) just for positioning.
        # Stability is the priority.
        target_theta = np.clip(target_theta, -0.20, 0.20)

        # --- 2. EMERGENCY OVERRIDE ---
        # If the rod is dangerously tilted, abandon position control.
        # Focus 100% on upright stability.
        if abs(theta) > 0.35:
            target_theta = 0.0

        # --- 3. TACTICAL PHASE: ATTITUDE CONTROL ---
        # Drive the rod angle to the target_theta using a specialized PD loop.

        error = theta - target_theta

        # Base Gains:
        # Kp needs to be high to overcome the heavy rod's gravity torque.
        # Gravity torque ~ mgL*sin(theta). For 0.35kg, 2.5m, it's significant.
        Kp = 85.0
        Kd = 20.0

        # Adaptive Stiffness (Non-linear Gain):
        # When error is large, stiffen the spring to "catch" the rod.
        # When error is small, relax to save energy and reduce jitter.
        urgency = 1.0 + 4.0 * abs(error)**1.5

        # Dynamic Damping:
        # Scale D gain with sqrt of stiffness to maintain constant damping ratio.
        # This prevents oscillations when stiffness increases.
        damping_boost = np.sqrt(urgency)

        # Calculate Control Force
        force = (Kp * urgency) * error + (Kd * damping_boost) * dtheta

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