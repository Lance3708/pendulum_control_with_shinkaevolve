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
    Reflex Cascade Controller (Challenge F + A)

    Philosophy:
    Instead of solving equations, we implement a hierarchy of biological-style reflexes.
    1. SURVIVAL (High Priority): If rod is falling, generate strong force to catch it.
       Ignore position.
    2. BALANCE (Medium Priority): If rod is safe, guide it to vertical with damping.
    3. CENTERING (Low Priority): If balanced, gently nudge cart to 0.

    Physics-aware adjustments:
    - High-friction compensation: Static boost to overcome stickiness.
    - Long-pole inertia handling: Strong angular velocity feedback.
    """

    def __init__(self):
        # Heuristic Gains (No matrix math)
        # Tuned for M_POLE=0.35, L_POLE=2.5 (High inertia)
        self.Kp_ang = 105.0
        self.Kd_ang = 42.0

        self.Kp_pos = 3.5
        self.Kd_pos = 7.0

        self.integ_x = 0.0

    def get_action(self, state):
        x, theta, dx, dtheta = state

        # 1. Assess Urgency (The "Phase")
        # How close are we to failure?
        # Combined metric of angle and angular speed
        urgency = abs(theta) + 0.4 * abs(dtheta)

        # Stability score: 1.0 = perfect, 0.0 = chaos
        # Used to blend between "Survival" and "Precision" modes
        stability = max(0.0, 1.0 - 2.5 * urgency)

        # 2. Survival Reflex (Rod Control)
        # Push cart to get under the rod.
        # Nonlinear boost: If falling fast/far, react disproportionately stronger
        panic_boost = 1.0 + 2.0 * urgency**2
        rod_force = panic_boost * (self.Kp_ang * theta + self.Kd_ang * dtheta)

        # 3. Centering Reflex (Cart Control)
        # Push cart to x=0.
        # Negated because x>0 requires Force<0 to return.
        # Integral term leaks away if unstable to prevent windup during recovery
        if stability > 0.6:
            self.integ_x += x * 0.02 # DT is 0.02
            self.integ_x = np.clip(self.integ_x, -3.0, 3.0)
        else:
            self.integ_x *= 0.92

        pos_force = -(self.Kp_pos * x + self.Kd_pos * dx + 1.5 * self.integ_x)

        # 4. Reflex Blending
        # As stability drops, position control is cut to focus on rod survival
        # Rod force is always active (it's the primary constraint)
        total_force = rod_force + pos_force * (stability**3)

        # 5. Friction Injection
        # The cart has high friction (0.35). Small forces do nothing.
        # If we command a force, ensure it's enough to actually slide the cart
        # or help overcome the drag.
        if abs(dx) < 0.2 and abs(total_force) > 0.5:
             # Add a "kick" in the direction of intended force
             total_force += 4.0 * np.sign(total_force)

        return float(total_force)

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