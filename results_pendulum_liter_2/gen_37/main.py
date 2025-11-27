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
    Phase-Driven Intuitive Controller

    This controller mimics human balancing intuition through distinct phases:
    1. Emergency Recovery: Aggressive cart positioning under falling pole
    2. Swing Control: Energy-managed swing with predictive positioning
    3. Precision Hold: Fine-tuned centering with minimal energy

    No complex math - just physics-inspired rules and phase transitions.
    """

    def __init__(self):
        # System parameters for physics calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m

        # Controller state
        self.integral_x = 0.0
        self.prev_theta = 0.0
        self.energy_history = []

        # Natural frequency for timing
        self.omega_n = np.sqrt(G / L_COM)

    def get_action(self, state):
        """Phase-driven intuitive control"""
        x, theta, dx, dtheta = state

        # Normalize angle
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Calculate system energy for phase decisions
        kinetic_energy = 0.5 * self.M * dx**2 + 0.5 * self.m * (dx**2 + (self.l * dtheta)**2 + 2 * dx * self.l * dtheta * np.cos(theta))
        potential_energy = self.m * self.g * self.l * (1 - np.cos(theta))
        total_energy = kinetic_energy + potential_energy

        # Track energy trend
        self.energy_history.append(total_energy)
        if len(self.energy_history) > 10:
            self.energy_history.pop(0)
        energy_trend = total_energy - self.energy_history[0] if len(self.energy_history) > 1 else 0

        # --- PHASE DETECTION ---

        # Emergency: Pole falling away from vertical with high energy
        is_emergency = abs(theta) > 0.4 and theta * dtheta > 0.1 and total_energy > 15.0

        # Precision: Very close to target state with low energy
        is_precision = abs(theta) < 0.08 and abs(dtheta) < 0.2 and abs(dx) < 0.2 and total_energy < 5.0

        # --- PHASE-BASED CONTROL ---

        if is_emergency:
            # PHASE 1: Emergency Recovery - Get cart under pole NOW
            target_x = self.l * np.sin(theta)  # Position under pole's COM
            target_dx = self.l * dtheta * np.cos(theta)  # Match pole's horizontal velocity

            # Aggressive PD control to reach target
            force = 80.0 * (target_x - x) + 30.0 * (target_dx - dx)

            # Reset integral during emergency
            self.integral_x *= 0.8

        elif is_precision:
            # PHASE 3: Precision Hold - Minimal energy centering
            predict_time = 0.08  # Look ahead slightly
            theta_future = theta + dtheta * predict_time
            x_future = x + dx * predict_time

            # Update integral for position correction
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -2.0, 2.0)

            # Gentle, precise control
            force = (120.0 * theta_future +    # Strong angle correction
                     28.0 * dtheta +           # Angular damping
                     6.0 * x_future +          # Position correction
                     9.0 * dx +                # Velocity damping
                     2.5 * self.integral_x)    # Integral action

        else:
            # PHASE 2: Swing Control - Energy management with prediction
            predict_time = 0.12
            theta_future = theta + dtheta * predict_time

            # Predict where pole will be and position cart accordingly
            future_com_x = self.l * np.sin(theta_future)
            cart_lead_distance = 0.3 * future_com_x  # Lead the pole slightly

            # Energy-aware damping
            if energy_trend > 0:  # Gaining energy, need more damping
                damping_factor = 1.5
            else:  # Losing energy, less damping
                damping_factor = 0.8

            # Balanced control
            pole_force = 100.0 * theta_future + 22.0 * dtheta * damping_factor
            cart_force = 8.0 * (cart_lead_distance - x) + 12.0 * dx

            force = pole_force + cart_force

            # Slow integral decay
            self.integral_x *= 0.98

        # Direction-aware mid-swing damping (when returning to vertical)
        if 0.25 < abs(theta) < 0.85 and theta * dtheta < 0:
            midswing_damp = 5.0 * dtheta * np.exp(-3.0 * (abs(theta) - 0.5)**2)
            force += midswing_damp

        # Store previous theta for trend analysis
        self.prev_theta = theta

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