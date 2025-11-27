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
    Simple Reactive Controller with Distinct Modes

    Philosophy: Simple reactive rules with distinct modes:
    1. Panic-save: When rod is falling dangerously
    2. Swing-up: When rod needs energy to reach upright
    3. Approach: When rod is near vertical but not stable
    4. Precise-hold: When rod is nearly balanced

    No LQR, no matrices, just intuitive physics responses.
    """

    def __init__(self):
        # System parameters for intuitive calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G

        # Natural frequency of pendulum
        self.omega_n = np.sqrt(G / L_COM)

        # Energy reference for upright position
        self.E_upright = M_POLE * G * L_COM  # Potential energy at upright

        # Previous state for change detection
        self.prev_state = None

        # Integral term for position correction
        self.integral_x = 0.0
        self.K_i = 0.8

    def get_action(self, state):
        """Simple reactive controller with distinct modes"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Initialize force
        force = 0.0

        # Mode 1: Panic-save - when rod is falling dangerously
        if abs(theta) > 0.7:
            # Calculate how urgently we need to react
            urgency = min(8.0, abs(theta) * 4.0)

            # If rod is falling away from upright, push hard in falling direction
            if theta * dtheta > 0:
                # Push cart under the falling rod
                force = -np.sign(theta) * min(80.0, 20.0 * urgency)
            else:
                # Rod is falling toward upright, help it gently
                force = -np.sign(theta) * np.sign(dtheta) * 15.0 * urgency

            # Always add cart position correction in panic mode
            force += -x * 2.0 - dx * 1.2

        # Mode 2: Swing-up - when rod needs energy to reach upright
        elif abs(theta) > 0.3:
            # Calculate current energy
            E_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
            E_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
            E_total = E_kinetic + E_potential

            # Energy deficit
            energy_deficit = self.E_upright - E_total

            # Push in direction of rod movement to add energy
            if energy_deficit > 0:
                force = np.sign(dtheta) * min(12.0, energy_deficit * 2.0)
            else:
                # Dampen if too much energy
                force = -np.sign(dtheta) * 5.0

            # Cart position correction
            force += -x * 1.5 - dx * 0.8

        # Mode 3: Approach - when rod is near vertical but not stable
        elif abs(theta) > 0.1 or abs(dtheta) > 0.2:
            # Calculate energy deficit
            E_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
            E_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
            E_total = E_kinetic + E_potential
            energy_deficit = self.E_upright - E_total

            # Simple PD control for angle
            force = -theta * 15.0 - dtheta * 3.0

            # Energy shaping - add just enough energy to reach upright
            if energy_deficit > 0.1:
                energy_boost = np.sign(dtheta) * min(5.0, energy_deficit * 0.8)
                force += energy_boost

            # Cart position correction
            force += -x * 2.0 - dx * 1.0

        # Mode 4: Precise-hold - when rod is nearly balanced
        else:
            # Very gentle angle correction to save energy
            force = -theta * 5.0 - dtheta * 0.8

            # Precise position control with adaptive integral action
            # Only integrate when very close to target
            if abs(x) < 0.5 and abs(dx) < 0.5:
                self.integral_x += x * DT
                self.integral_x = np.clip(self.integral_x, -1.0, 1.0)
            else:
                # Decay integral when not close to target
                self.integral_x *= 0.98

            # Adaptive position gains - stronger when farther from center
            pos_gain = 2.0 + abs(x) * 2.0
            vel_gain = 1.0 + abs(dx) * 0.5

            force += -x * pos_gain - dx * vel_gain + self.K_i * self.integral_x

        # Always apply some damping to cart velocity
        force -= dx * 0.5

        # Limit force
        force = np.clip(force, -100.0, 100.0)

        # Update previous state
        self.prev_state = state.copy()

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