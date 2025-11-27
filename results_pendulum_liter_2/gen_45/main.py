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
    An intuitive, three-phase controller based on human-like reflexes.
    
    Philosophy: Apply distinct, decisive strategies based on the situation,
    just as a person would, rather than a single blended mathematical formula.
    
    Phases:
    1. Emergency Catch: If the pole is falling far out, aggressively move the
       cart under its center of mass to arrest the fall. This is a pure reflex.
    2. Swing Damping: Once the immediate danger is over, focus on bleeding
       energy from the system to calm the pole's swing, while starting to
       guide the cart to the center.
    3. Precision Hold: When near vertical, switch to a high-precision mode
       that uses an integral term to eliminate steady-state error and hold
       the system perfectly still, fighting friction.
    """

    def __init__(self):
        """Initializes controller state variables."""
        # Integral term for the Precision Hold phase to eliminate steady-state error.
        self.integral_x = 0.0

    def get_action(self, state):
        """Calculates the control force based on the current phase."""
        x, theta, dx, dtheta = state

        # --- Phase Detection Logic ---
        # A simple but effective way to determine the system's state of urgency.

        # EMERGENCY: Pole is at a significant angle AND actively falling away from vertical.
        # This requires immediate, powerful intervention.
        # The threshold 0.25 rad is chosen to be sensitive but not overly twitchy.
        is_emergency = abs(theta) > 0.25 and (theta * dtheta) >= 0

        # HOLD: The system is very close to the desired state. Time for fine-tuning.
        is_holding = abs(theta) < 0.035 and abs(dtheta) < 0.08 and abs(dx) < 0.1

        # --- Control Law Selection ---

        if is_emergency:
            # --- PHASE 1: Emergency Catch ---
            # Goal: Get the cart under the pole's center of mass (COM) IMMEDIATELY.
            self.integral_x = 0  # Reset integrator during emergencies.

            # Calculate the target position for the cart to be directly under the COM.
            # An "overshoot" factor is added, proportional to the fall speed. This is
            # like a human shoving their hand past the falling point to reverse momentum.
            overshoot = 0.20 * dtheta
            target_x = L_COM * np.sin(theta) + overshoot
            
            # The cart should also try to match the horizontal velocity of the COM.
            target_dx = L_COM * dtheta * np.cos(theta)
            
            # A powerful PD controller drives the cart to this dynamic target.
            # Gains are high to generate large forces for rapid movement.
            force = 90.0 * (target_x - x) + 35.0 * (target_dx - dx)

        elif is_holding:
            # --- PHASE 3: Precision Hold ---
            # Goal: Maintain stability with high precision, eliminating drift.
            
            # Update integral term to counteract static friction and small biases.
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -4.0, 4.0) # Anti-windup.

            # A high-gain state feedback controller fine-tuned for small perturbations.
            # Using a short-term prediction makes the controller proactive.
            predict_time = 0.06 # Predict ~3 steps ahead
            theta_future = theta + dtheta * predict_time
            
            force = (115.0 * theta_future +      # Strong proportional angle control
                     28.0 * dtheta +           # Strong angular velocity damping
                     7.0 * x +                 # Firm proportional position control
                     10.0 * dx +               # Firm velocity damping
                     3.5 * self.integral_x)    # Integral action to correct steady-state error

        else: # Default is Swing Damping
            # --- PHASE 2: Swing Damping ---
            # Goal: Guide the pole to vertical and the cart to center, bleeding energy.
            self.integral_x *= 0.92  # Slowly decay the integral term when not in use.
            
            # Use a slightly longer prediction horizon to handle larger, slower swings.
            predict_time = 0.1 # Predict 5 steps ahead
            theta_future = theta + dtheta * predict_time
            
            # A balanced state feedback controller. Pole stability is the priority.
            pole_force = 105.0 * theta_future + 26.0 * dtheta
            cart_force = 5.0 * x + 8.0 * dx
            
            force = pole_force + cart_force

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