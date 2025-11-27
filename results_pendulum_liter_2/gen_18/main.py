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
    Three-Phase Predictive Control
    
    This controller uses a human-intuition-based, three-phase state machine. Each 
    phase represents a distinct tactical goal and employs a tailored control law,
    creating a decisive, non-linear strategy.
    
    1. Emergency Catch: When the pole is falling away, it aggressively moves the
       cart under the pole's center of mass to arrest the fall.
    2. Corrective Swing: A predictive PD-like controller to guide the system
       toward equilibrium, prioritizing pole stability.
    3. Precision Hold: A high-gain, integral-action controller to eliminate
       small errors and drift with minimal energy when near the target state.
    """

    def __init__(self):
        """Initializes the controller state."""
        # Integral term for the Precision Hold phase to eliminate steady-state position error.
        self.integral_x = 0.0

    def get_action(self, state):
        """
        Calculates the control force based on the current state and phase.
        """
        x, theta, dx, dtheta = state

        # Normalize angle to prevent issues with wraparound
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # --- Phase Detection Logic ---
        
        # Condition for Emergency: far from vertical and actively falling away.
        is_emergency = abs(theta) > 0.4 and (theta * dtheta) > 0.05

        # Condition for Precision Hold: very close to the target state.
        is_settling = abs(theta) < 0.05 and abs(dtheta) < 0.1 and abs(dx) < 0.1

        # --- Control Law Selection ---

        if is_emergency:
            # --- PHASE 1: Emergency Catch ---
            # Goal: Get the cart under the pole's center of mass (COM) immediately.
            
            # Target cart position is under the COM.
            x_target = L_COM * np.sin(theta)
            # Target cart velocity should match the horizontal sway of the COM.
            dx_target = L_COM * dtheta * np.cos(theta)
            
            # A powerful PD controller to drive the cart to its target state.
            force = 70.0 * (x_target - x) + 25.0 * (dx_target - dx)

            # Decay the integrator when not in the hold phase.
            self.integral_x *= 0.85

        elif is_settling:
            # --- PHASE 3: Precision Hold ---
            # Goal: Maintain stability with high precision and minimal energy.
            
            # Use a short-term prediction to be proactive.
            predict_time = 0.1  # 5 steps ahead
            theta_future = theta + dtheta * predict_time
            x_future = x + dx * predict_time
            
            # Update integral term for position error.
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -2.5, 2.5) # Clip to prevent windup
            
            # A high-gain state feedback controller fine-tuned for small perturbations.
            # The integral term is crucial for overcoming static friction.
            force = (110.0 * theta_future +   # Strong proportional angle control
                     25.0 * dtheta +          # Strong angular velocity damping
                     5.0 * x_future +         # Gentle proportional position control
                     8.0 * dx +               # Gentle velocity damping
                     3.0 * self.integral_x)   # Integral action to correct steady-state error

        else:
            # --- PHASE 2: Corrective Swing (Default) ---
            # Goal: Guide the pole to vertical and the cart to center.
            
            # Use a slightly longer prediction horizon to handle larger swings.
            predict_time = 0.15 # 7-8 steps ahead
            theta_future = theta + dtheta * predict_time
            
            # A balanced state feedback controller, prioritizing pole stability.
            pole_control = 95.0 * theta_future + 24.0 * dtheta
            cart_control = 7.0 * x + 10.0 * dx
            
            force = pole_control + cart_control
            
            # Decay the integrator when not in the hold phase.
            self.integral_x *= 0.85

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