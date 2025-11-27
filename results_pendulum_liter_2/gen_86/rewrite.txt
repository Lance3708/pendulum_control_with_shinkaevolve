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
    Reflex-Based Predictive Controller
    
    Three distinct operational modes with simple, intuitive rules:
    1. EMERGENCY: When rod is falling fast, move cart aggressively under it
    2. PREDICTIVE: When rod is swinging, position cart where it will be
    3. GENTLE: When nearly balanced, make tiny corrections
    
    No matrices, no optimization - just reflexes and prediction
    """
    
    def __init__(self):
        # Physical constants for intuitive calculations
        self.pole_length = L_POLE
        self.gravity = G
        self.cart_mass = M_CART
        self.pole_mass = M_POLE
        
        # Memory for predictive calculations
        self.last_theta = 0.0
        self.last_dtheta = 0.0
        self.integral_error = 0.0
        
        # Simple tuning parameters
        self.emergency_threshold = 0.7  # radians
        self.balanced_threshold = 0.15  # radians
        self.max_force = 100.0
        
    def get_action(self, state):
        """Simple reflex-based control with prediction"""
        x, theta, dx, dtheta = state
        
        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        abs_theta = abs(theta)
        
        # Calculate rate of change for prediction
        theta_change = theta - self.last_theta
        dtheta_change = dtheta - self.last_dtheta
        
        # Update memory
        self.last_theta = theta
        self.last_dtheta = dtheta
        
        # --- MODE 1: EMERGENCY REFLEX (rod falling fast) ---
        if abs_theta > self.emergency_threshold or abs(dtheta) > 2.0:
            # Move cart aggressively under the falling rod
            # Push in direction rod is leaning, scaled by how fast it's falling
            urgency = min(3.0, abs_theta + 0.5 * abs(dtheta))
            direction = np.sign(theta)
            
            # Base emergency force
            emergency_force = direction * urgency * 15.0
            
            # Add extra push if rod is accelerating away from vertical
            if theta * dtheta > 0:  # Falling away from vertical
                emergency_force *= 1.5
                
            # Damp cart movement to prevent overshoot
            cart_damping = -dx * 2.0
            
            force = emergency_force + cart_damping
            
        # --- MODE 2: PREDICTIVE POSITIONING (rod swinging) ---
        elif abs_theta > self.balanced_threshold:
            # Predict where rod will be in near future
            # Simple linear prediction: where will theta be in 0.1 seconds?
            future_theta = theta + dtheta * 0.1
            
            # Position cart where the rod will need support
            # If rod will be leaning left, move cart left to catch it
            target_position = -future_theta * self.pole_length * 0.3
            
            # Calculate force to move toward target position
            position_error = target_position - x
            velocity_error = -dx  # Want to slow down as we approach
            
            predictive_force = position_error * 8.0 + velocity_error * 3.0
            
            # Add damping based on rod's angular velocity
            # Only damp when rod is moving away from vertical
            if theta * dtheta > 0:
                angular_damping = -dtheta * 2.0
                predictive_force += angular_damping
                
            force = predictive_force
            
        # --- MODE 3: GENTLE BALANCING (nearly vertical) ---
        else:
            # Very gentle corrections when nearly balanced
            # Focus on keeping rod vertical and cart centered
            
            # Tiny corrections for rod angle
            angle_correction = -theta * 5.0
            
            # Tiny corrections for rod rotation
            rotation_correction = -dtheta * 1.5
            
            # Gentle centering of cart
            position_correction = -x * 0.8
            velocity_correction = -dx * 0.5
            
            # Integral action for final centering
            if abs_theta < 0.05 and abs(dtheta) < 0.2:
                self.integral_error += x * DT
                # Limit integral to prevent windup
                self.integral_error = np.clip(self.integral_error, -2.0, 2.0)
                integral_correction = -self.integral_error * 0.3
            else:
                # Decay integral when not in perfect balance
                self.integral_error *= 0.95
                integral_correction = 0
                
            force = angle_correction + rotation_correction + position_correction + velocity_correction + integral_correction
            
        # Universal safety limits
        force = np.clip(force, -self.max_force, self.max_force)
        
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