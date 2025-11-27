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
    Reflex-Based Stabilizer: A human-like approach to balancing.
    
    Philosophy: "Predict where the pole is going, then react quickly to any surprises."
    """
    
    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT
        
        # For predictive control
        self.predicted_theta = 0.0
        self.predicted_dtheta = 0.0
        
        # For energy management
        self.energy_history = []
        
    def predict_state(self, theta, dtheta, dx, force):
        """Predict state one timestep ahead using physics model"""
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        # Friction forces
        f_cart = -self.b_c * dx
        f_joint = -self.b_j * dtheta
        
        # Accelerations
        temp = (force + f_cart + self.m * self.l * dtheta**2 * sin_t) / self.Mtot
        denom = self.l * (4.0/3.0 - self.m * cos_t**2 / self.Mtot)
        theta_acc = (self.g * sin_t - cos_t * temp + f_joint / (self.m * self.l)) / denom
        x_acc = temp - (self.m * self.l * theta_acc * cos_t) / self.Mtot
        
        # Predictions
        pred_theta = theta + dtheta * DT
        pred_dtheta = dtheta + theta_acc * DT
        pred_dx = dx + x_acc * DT
        
        return pred_theta, pred_dtheta, pred_dx
    
    def calculate_kinetic_energy(self, dx, dtheta):
        """Calculate total kinetic energy of the system"""
        cart_ke = 0.5 * self.M * dx**2
        pole_ke = 0.5 * self.m * (dx**2 + (self.l * dtheta)**2)
        return cart_ke + pole_ke
    
    def calculate_potential_energy(self, theta):
        """Calculate potential energy of the pole"""
        height = self.l * np.cos(theta)
        return self.m * self.g * (self.l - height)
    
    def get_action(self, state):
        """Main control logic"""
        x, theta, dx, dtheta = state
        
        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # --- PREDICTIVE CONTROL ---
        # Predict where the pole will be in one timestep
        pred_theta, pred_dtheta, pred_dx = self.predict_state(theta, dtheta, dx, 0.0)
        
        # How far will the tip of the pole move horizontally?
        # This is our main target for the cart to move under
        pole_tip_velocity = pred_dtheta * self.l
        predicted_tip_position = x + dx * DT + pole_tip_velocity * DT
        
        # Move cart to predicted tip position (with some lookahead)
        k_position = 2.0  # Position gain
        k_velocity = 1.5  # Velocity gain
        force = k_position * (predicted_tip_position - x) - k_velocity * dx
        
        # --- REFLEXIVE CORRECTIONS ---
        # If pole is falling fast, apply immediate correction
        if abs(pred_theta) > 0.1 and pred_theta * pred_dtheta > 0:  # Same sign = falling away
            reflex_gain = 15.0 * np.tanh(3.0 * abs(pred_dtheta))
            force += reflex_gain * np.sign(pred_theta)
            
        # If cart is moving too fast, damp it
        if abs(dx) > 0.5:
            force -= 2.0 * dx * np.tanh(abs(dx))
            
        # --- ENERGY MANAGEMENT ---
        # Calculate current energy
        ke = self.calculate_kinetic_energy(dx, dtheta)
        pe = self.calculate_potential_energy(theta)
        total_energy = ke + pe
        
        # Keep a short history of energy
        self.energy_history.append(total_energy)
        if len(self.energy_history) > 10:
            self.energy_history.pop(0)
            
        # If energy is increasing when it should be decreasing, apply damping
        if len(self.energy_history) > 5:
            energy_trend = (self.energy_history[-1] - self.energy_history[-5]) / 5
            if energy_trend > 0.01 and abs(theta) < 0.5:  # Only when somewhat stable
                force -= 3.0 * dx * np.tanh(abs(dx))
                
        # If energy is very low and pole is falling, add energy
        if total_energy < 0.5 and abs(theta) > 0.3 and theta * dtheta > 0:
            energy_boost = 8.0 * np.sign(theta) * np.tanh(abs(dtheta))
            force += energy_boost
            
        # --- FINE TUNING FOR STABILITY ---
        # When very close to upright, fine-tune position
        if abs(theta) < 0.1:
            # Strong position correction
            force -= 3.0 * x
            # Damping to prevent oscillations
            force -= 1.0 * dx
            
        # Limit force
        force = np.clip(force, -100.0, 100.0)
        
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