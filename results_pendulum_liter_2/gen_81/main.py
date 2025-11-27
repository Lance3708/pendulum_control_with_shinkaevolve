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
def get_control_action(state):
    """
    Reflex-Based Triphasic Controller
    
    Philosophy: Three distinct, non-blended behaviors:
    1. EMERGENCY: Rod falling fast -> push directly under it
    2. RECOVERY: Passing through vertical -> predict swing and pre-position
    3. BALANCE: Nearly upright -> tiny corrections only
    
    No matrices, no LQR, no optimal control. Just fast reflexes.
    """
    x, theta, dx, dtheta = state
    
    # Normalize angle to [-pi, pi]
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    abs_theta = abs(theta)
    abs_dtheta = abs(dtheta)
    
    # === PHASE 1: EMERGENCY RESPONSE ===
    # Rod is falling hard - act immediately
    if abs_theta > 0.6 or abs_dtheta > 1.0:
        # Push cart in direction of fall, scaled by urgency
        # More tilt OR more speed = stronger push
        urgency = abs_theta * 0.7 + abs_dtheta * 0.3
        direction = np.sign(theta)  # Which way is it falling?
        
        # Strong, immediate force to get under the rod
        force = direction * (40.0 + urgency * 60.0)
        return float(np.clip(force, -100.0, 100.0))
    
    # === PHASE 2: RECOVERY GUIDANCE ===
    # Rod is swinging through vertical - guide it gently
    if abs_theta > 0.15:
        # Predict where it's going and move cart there
        predicted_angle = theta + dtheta * DT * 3  # Look 60ms ahead
        desired_cart_offset = -predicted_angle * 1.2  # Lead the pendulum
        
        # Proportional response to desired position
        force = -18.0 * (x - desired_cart_offset)
        
        # Reduce cart velocity to avoid overshoot
        force -= 8.0 * dx
        
        return float(np.clip(force, -100.0, 100.0))
    
    # === PHASE 3: PRECISION HOLD ===
    # Nearly balanced - only correct drift
    # Use integral term ONLY for persistent position error
    if not hasattr(get_control_action, 'integral'):
        get_control_action.integral = 0.0
    
    # Only integrate when very stable
    if abs_dtheta < 0.4 and abs_theta < 0.2:
        get_control_action.integral += x * DT
        # Windup protection
        get_control_action.integral = np.clip(get_control_action.integral, -1.0, 1.0)
    else:
        # Reset faster if disturbed
        get_control_action.integral *= 0.9
    
    # Tiny correction based on accumulated offset
    force = -12.0 * x - 6.0 * dx + 8.0 * get_control_action.integral
    
    return float(np.clip(force, -100.0, 100.0))
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