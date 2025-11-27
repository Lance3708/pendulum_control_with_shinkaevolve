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
    A controller based on distinct operational phases, inspired by human reflexes.
    It switches between strategies depending on the system's state of stability.
    This avoids a single, complex control law in favor of simple, decisive rules
    for different situations.

    The three phases are:
    1. Emergency Catch: When the pole is at a large angle or falling fast. The sole
       priority is to arrest the fall, ignoring the cart's position. This is a
       powerful, non-linear response.
    2. Damping & Centering: When the pole is relatively stable but the system has
       excess energy (velocity). This phase focuses on bleeding off that energy
       while gently guiding the cart to the center.
    3. Precision Hold: When the system is near the target state. The focus shifts
       to fine, energy-efficient adjustments to counteract friction and drift,
       holding the system perfectly stable.
    """
    def __init__(self):
        """No complex initialization needed for this rule-based approach."""
        pass

    def get_action(self, state):
        """
        Calculates the control force based on the current phase.
        """
        x, theta, dx, dtheta = state

        # --- Phase Definition ---
        # An "instability metric" determines the current phase. It's a heuristic
        # combining angle (potential for falling) and angular velocity (kinetic
        # energy of the fall). The coefficients are tuned based on the pole's
        # physical properties to define sensible phase boundaries.
        instability_metric = abs(theta) + 0.3 * abs(dtheta)

        # --- Phase 1: Emergency Catch ---
        # Triggered by large angles or high angular velocity.
        if instability_metric > 0.3:
            # The primary goal is to reverse the fall. This is a powerful PD
            # controller focused only on the angle.
            # Force = P_gain * theta + D_gain * dtheta
            # Gains are high to generate a strong, immediate counter-torque.
            force = 85.0 * theta + 25.0 * dtheta

            # Add a non-linear "kicker" term. When theta is large, the situation
            # is more desperate, so we add an exponentially growing force to
            # ensure recovery. This mimics a human's panic response.
            kicker = 15.0 * theta * (abs(theta) ** 1.5)
            force += kicker

        # --- Phase 2: Damping & Centering ---
        # Active when the pole is under control but the system isn't settled.
        elif instability_metric > 0.025:
            # The control law now incorporates all state variables.
            # It's a combination of two PD controllers: one for the angle,
            # one for the cart's position.

            # Angle control is still dominant, but with reduced gains.
            angle_correction = 55.0 * theta + 22.0 * dtheta

            # Position control is introduced to start moving the cart to the
            # center. Its gains are chosen to be subordinate to angle control,
            # ensuring that centering efforts don't destabilize the pole.
            position_correction = 3.5 * x + 4.5 * dx

            force = angle_correction - position_correction

        # --- Phase 3: Precision Hold ---
        # The final phase for when the system is close to the desired state.
        else:
            # Gains are further refined for stability and energy efficiency.

            # Angle correction is gentle, providing just enough force to
            # counteract drift.
            angle_correction = 45.0 * theta + 20.0 * dtheta

            # Position correction becomes more assertive relative to angle
            # correction, as the main task now is to lock the cart at x=0.
            # The dx gain is higher to strongly damp any cart motion.
            position_correction = 6.0 * x + 8.0 * dx

            force = angle_correction - position_correction

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