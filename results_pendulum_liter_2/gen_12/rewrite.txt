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
    Rule-based phase controller for inverted pendulum
    
    Core philosophy: Three distinct modes for three distinct challenges.
    
    1. Emergency mode: Save the rod from falling with decisive action
    2. Guide mode: Lead the rod to vertical with anticipation
    3. Hold mode: Maintain perfect balance with gentle corrections
    
    Each mode uses simple physical intuition, not complex math.
    """
    
    def __init__(self):
        # System parameters for calculations
        self.rod_mass = M_POLE
        self.cart_mass = M_CART
        self.rod_length = L_COM
        self.gravity = G
        
        # Mode thresholds
        self.emergency_angle = 0.7  # radians
        self.hold_angle = 0.2       # radians
        
        # Control strengths for each mode
        self.emergency_strength = 16.0
        self.guide_strength = 8.0
        self.hold_strength = 4.0
        
        # Look-ahead time for predictions (seconds)
        self.look_ahead = 0.08
        
        # For final position holding
        self.position_error_memory = 0.0
        self.hold_gain = 0.6
        
        # Natural frequency of the system
        self.natural_freq = np.sqrt(self.gravity / self.rod_length)
    
    def predict_rod_fall(self, angle, angle_speed, time_ahead=None):
        """Predict where the rod will be after time_ahead time"""
        if time_ahead is None:
            time_ahead = self.look_ahead
            
        # Simplified physics: constant acceleration due to gravity
        accel = self.gravity * np.sin(angle) / self.rod_length
        
        # Future angle and speed
        future_angle = angle + angle_speed * time_ahead + 0.5 * accel * time_ahead**2
        future_speed = angle_speed + accel * time_ahead
        
        return future_angle, future_speed
    
    def emergency_mode(self, state):
        """Emergency response - save the rod from falling"""
        cart_pos, rod_angle, cart_speed, rod_speed = state
        
        # Normalize angle
        rod_angle = ((rod_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Predict where rod will be
        future_angle, future_speed = self.predict_rod_fall(rod_angle, rod_speed)
        
        # If rod is moving toward vertical, help it along
        if rod_angle * rod_speed < 0:
            # Calculate where cart should be to catch the rod
            # We want the cart under the rod when it reaches vertical
            target_cart_pos = -self.rod_length * np.sin(future_angle) * 1.3
            
            # Push cart toward target position
            pos_error = target_cart_pos - cart_pos
            speed_error = 0 - cart_speed
            
            force = self.emergency_strength * (pos_error + 0.5 * speed_error)
        else:
            # Rod is falling away from vertical - strong counter-action
            # Push opposite to the direction of fall
            if rod_angle > 0:
                force = -self.emergency_strength * 2.0
            else:
                force = self.emergency_strength * 2.0
        
        return force
    
    def guide_mode(self, state):
        """Guide the rod toward vertical position"""
        cart_pos, rod_angle, cart_speed, rod_speed = state
        
        # Predict rod position
        future_angle, future_speed = self.predict_rod_fall(rod_angle, rod_speed, self.look_ahead * 0.7)
        
        # Target cart position: slightly ahead of where the rod will be
        target_cart_pos = -self.rod_length * np.sin(future_angle) * 1.2
        
        # Push cart toward target
        pos_error = target_cart_pos - cart_pos
        speed_error = 0 - cart_speed
        
        force = self.guide_strength * (pos_error + 0.4 * speed_error)
        
        # Add damping to reduce rod rotation
        if abs(rod_angle) > 0.3:
            force -= 2.0 * rod_speed
        
        return force
    
    def hold_mode(self, state):
        """Hold the rod perfectly vertical with cart centered"""
        cart_pos, rod_angle, cart_speed, rod_speed = state
        
        # Small angle: target is under the rod
        target_cart_pos = -self.rod_length * rod_angle
        
        # Add compensation for rod movement
        target_cart_pos -= 0.3 * rod_speed / self.natural_freq
        
        # Position and velocity errors
        pos_error = target_cart_pos - cart_pos
        speed_error = 0 - cart_speed
        
        # Gentle corrective force
        force = self.hold_strength * (pos_error + 0.3 * speed_error)
        
        # Damp rod rotation
        force -= 1.5 * rod_speed
        
        # Only use integral action when very stable
        stability_measure = abs(rod_angle) + 0.1 * abs(rod_speed)
        
        if stability_measure < 0.15:
            self.position_error_memory += cart_pos * 0.02
            self.position_error_memory = np.clip(self.position_error_memory, -2.0, 2.0)
            
            force -= self.hold_gain * self.position_error_memory
        else:
            # Decay memory when not stable
            self.position_error_memory *= 0.95
        
        return force
    
    def get_action(self, state):
        """Choose the right action based on current situation"""
        cart_pos, rod_angle, cart_speed, rod_speed = state
        
        # Normalize angle
        rod_angle = ((rod_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # Choose control mode based on rod angle
        if abs(rod_angle) > self.emergency_angle:
            # Rod is in danger of falling - emergency mode
            force = self.emergency_mode(state)
        elif abs(rod_angle) > self.hold_angle:
            # Rod is approaching vertical - guide mode
            force = self.guide_mode(state)
        else:
            # Rod is nearly vertical - hold mode
            force = self.hold_mode(state)
        
        # Reduce force if system has lots of energy (to prevent overshoot)
        rod_kinetic_energy = 0.5 * self.rod_mass * (self.rod_length * rod_speed)**2
        rod_potential_energy = self.rod_mass * self.gravity * self.rod_length * (1 - np.cos(rod_angle))
        
        # Normalize energy compared to a reference (45 degrees)
        reference_energy = self.rod_mass * self.gravity * self.rod_length * (1 - np.cos(0.8))
        
        # Energy reduction factor
        total_energy = rod_kinetic_energy + rod_potential_energy
        energy_factor = min(1.0, reference_energy / (total_energy + 1e-6))
        
        # Apply energy modulation, but don't make force too weak
        force *= max(0.4, energy_factor)
        
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