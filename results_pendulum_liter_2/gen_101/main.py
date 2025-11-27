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
    Reflex-Driven Adaptive Controller with Phase-Specific Behaviors
    
    Three distinct operational modes:
    1. Emergency Reflex (rapid fall): Maximum corrective force
    2. Swing-Up Pumping (mid-angle): Energy injection when beneficial
    3. Balance Hold (near-vertical): Minimalist integral positioning
    """

    def __init__(self):
        # Control thresholds
        self.THETA_EMERGENCY = 0.6     # rad (~34 deg)
        self.DTHETA_FAST = 1.8         # rad/s
        self.THETA_BALANCE = 0.15      # rad (~8.6 deg)
        
        # Force magnitudes
        self.FORCE_MAX = 95.0
        self.FORCE_PUMP = 25.0
        self.FORCE_DAMP = 0.3
        
        # Integral control
        self.integral_gain = 1.2
        self.integral_windup = 2.0
        self.integral_decay = 0.98
        self.integral = 0.0
        
        # Mode tracking
        self.current_mode = "emergency"
        self.pumping_active = False

    def get_action(self, state):
        """Reflex-driven control with discrete behavioral modes"""
        x, theta, dx, dtheta = state
        
        # Normalize angle
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        abs_theta = abs(theta)
        abs_dtheta = abs(dtheta)
        
        # Determine control mode based on physical urgency
        if abs_theta > self.THETA_EMERGENCY or abs_dtheta > self.DTHETA_FAST:
            # EMERGENCY REFLEX: Stop the fall immediately
            self.current_mode = "emergency"
            self.pumping_active = False
            
            # Push toward falling side with maximum force
            emergency_force = -np.sign(theta) * self.FORCE_MAX
            
            # Add cart damping to prevent excessive movement
            damping_force = -self.FORCE_DAMP * dx
            total_force = emergency_force + damping_force
            
        elif abs_theta > self.THETA_BALANCE:
            # SWING-UP PUMPING: Inject energy strategically
            self.current_mode = "pumping"
            
            # Only pump when motion is helping us go upright
            energy_sign = np.sign(theta * dtheta)
            
            if energy_sign < 0:  # Moving toward upright
                if not self.pumping_active:
                    # Brief pulse when crossing threshold
                    self.pumping_active = True
                    pump_force = -np.sign(theta) * self.FORCE_PUMP
                else:
                    pump_force = 0.0
            else:
                self.pumping_active = False
                pump_force = 0.0
                
            # Damping to prevent overshoot
            damping_force = -0.15 * dx
            total_force = pump_force + damping_force
            
        else:
            # BALANCE HOLD: Focus on precision positioning
            self.current_mode = "balance"
            self.pumping_active = False
            
            # Update integral of position error
            self.integral += x * DT
            self.integral *= self.integral_decay  # Natural decay
            
            # Anti-windup protection
            self.integral = np.clip(self.integral, -self.integral_windup, self.integral_windup)
            
            # Position correction only (let natural dynamics handle balance)
            pos_correction = -self.integral_gain * x
            integral_correction = -self.integral_gain * self.integral
            
            total_force = pos_correction + integral_correction
            
            # Very light damping for energy efficiency
            total_force -= 0.05 * dx

        return float(np.clip(total_force, -100.0, 100.0))

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