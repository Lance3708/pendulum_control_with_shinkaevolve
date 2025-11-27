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
    Phase-Based Reflex Controller

    Philosophy: "React like a human balancing a broomstick"
    - Emergency: Move cart toward falling direction FAST
    - Recovery: Proportional response with damping
    - Precision: Gentle fine-tuning with position correction

    No matrices, no complex math - just physics intuition.
    """

    def __init__(self):
        # Physical constants for scaling
        self.l = L_COM
        self.g = G
        self.omega_n = np.sqrt(G / L_COM)  # Natural frequency ~2.8 rad/s

        # Integral term for position drift
        self.integral_x = 0.0

        # Previous state for derivative estimation
        self.prev_theta = None
        self.prev_dtheta = None

    def get_action(self, state):
        """Phase-based control with distinct strategies"""
        x, theta, dx, dtheta = state

        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        abs_theta = abs(theta)

        # Detect if rod is "falling" (angle and velocity same sign = getting worse)
        falling = theta * dtheta > 0

        # Urgency metric: how bad is the situation?
        urgency = abs_theta + 0.3 * abs(dtheta) / self.omega_n

        # ============ PHASE 1: EMERGENCY (|theta| > 0.5 rad, ~29 degrees) ============
        if abs_theta > 0.5:
            # Core insight: Push cart toward where rod is falling
            # The harder it's falling, the harder we push

            # Base response: proportional to angle
            K_theta_emergency = 55.0
            force = K_theta_emergency * theta

            # Velocity boost: if falling fast, push even harder
            K_dtheta_emergency = 12.0
            force += K_dtheta_emergency * dtheta

            # Extra boost when actively falling (same sign)
            if falling:
                boost = 1.0 + 0.8 * min(abs(dtheta) / 2.0, 1.5)
                force *= boost

            # Slight cart velocity damping to prevent runaway
            force -= 0.8 * dx

        # ============ PHASE 2: RECOVERY (0.15 < |theta| < 0.5 rad) ============
        elif abs_theta > 0.15:
            # Moderate response with good damping
            K_theta_recovery = 48.0
            K_dtheta_recovery = 14.0
            K_x_recovery = 3.5
            K_dx_recovery = 4.0

            force = K_theta_recovery * theta + K_dtheta_recovery * dtheta
            force -= K_x_recovery * x + K_dx_recovery * dx

            # Predictive term: where will theta be in ~0.1s?
            theta_pred = theta + dtheta * 0.1
            if abs(theta_pred) > abs_theta:  # Getting worse
                force += 5.0 * (theta_pred - theta)

        # ============ PHASE 3: PRECISION (|theta| < 0.15 rad, ~8.6 degrees) ============
        else:
            # Fine control - balance angle and position
            K_theta_precision = 42.0
            K_dtheta_precision = 16.0
            K_x_precision = 5.5
            K_dx_precision = 6.0

            force = K_theta_precision * theta + K_dtheta_precision * dtheta
            force -= K_x_precision * x + K_dx_precision * dx

            # Integral action for position (only when nearly balanced)
            if abs_theta < 0.08 and abs(dtheta) < 0.5:
                self.integral_x += x * DT
                self.integral_x = np.clip(self.integral_x, -2.0, 2.0)
                force -= 1.2 * self.integral_x
            else:
                # Decay integral when not in precision mode
                self.integral_x *= 0.92

        # ============ GLOBAL CORRECTIONS ============

        # Anti-overshoot: if rod is swinging back (opposite signs), reduce force
        if not falling and abs_theta > 0.1:
            # Rod is recovering on its own, don't over-correct
            damping_factor = 0.7 + 0.3 * abs_theta / 0.5
            force *= damping_factor

        # Cart boundary awareness
        if abs(x) > 3.0:
            # Gently push back toward center
            boundary_force = -2.0 * np.sign(x) * (abs(x) - 3.0)
            # But only if it won't destabilize the rod
            if abs_theta < 0.3:
                force += boundary_force

        # Store for next iteration
        self.prev_theta = theta
        self.prev_dtheta = dtheta

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