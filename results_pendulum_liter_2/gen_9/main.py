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
import numpy as np

# --- Physics Constants ---
M_CART = 1.0
M_POLE = 0.35
L_POLE = 2.5
L_COM = L_POLE / 2
G = 9.81
FRICTION_CART = 0.35
FRICTION_JOINT = 0.25
DT = 0.02

class Controller:
    """
    Phase-Based Orchestrator with Discrete Behavioral Modes
    
    Four distinct phases with sharp transitions:
    - EMERGENCY: Falling fast, use predictive momentum compensation
    - SWING-UP: Large angles, energy pumping with friction compensation  
    - CONVERGENCE: Moderate angles, velocity matching and positioning
    - PRECISION: Small angles, gentle stabilization with drift control
    """
    
    def __init__(self):
        # Phase thresholds
        self.emergency_threshold = 0.7  # rad - falling fast
        self.swing_threshold = 0.4      # rad - need energy pumping
        self.convergence_threshold = 0.15  # rad - moderate control
        self.precision_threshold = 0.05   # rad - fine control
        
        # Emergency phase parameters
        self.emergency_gain = 45.0
        self.momentum_prediction_horizon = 0.08  # seconds
        
        # Swing-up phase parameters  
        self.swing_energy_gain = 28.0
        self.swing_friction_compensation = 12.0
        
        # Convergence phase parameters
        self.convergence_angle_gain = 35.0
        self.convergence_velocity_gain = 8.0
        self.position_correction_gain = 2.5
        
        # Precision phase parameters
        self.precision_angle_gain = 42.0
        self.precision_velocity_gain = 10.0
        self.integral_gain = 0.6
        self.position_integral = 0.0
        
        # Memory for phase transitions
        self.last_phase = "EMERGENCY"
        self.phase_transition_smoothing = 0.0

    def get_action(self, state):
        """Orchestrate control based on current phase detection"""
        x, theta, dx, dtheta = state
        
        # Normalize angle
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Detect current phase
        current_phase = self._detect_phase(theta, dtheta, x, dx)
        
        # Apply phase-specific control law
        if current_phase == "EMERGENCY":
            force = self._emergency_control(theta, dtheta, x, dx)
        elif current_phase == "SWING-UP":
            force = self._swing_up_control(theta, dtheta, x, dx)
        elif current_phase == "CONVERGENCE":
            force = self._convergence_control(theta, dtheta, x, dx)
        else:  # PRECISION
            force = self._precision_control(theta, dtheta, x, dx)
            
        # Smooth phase transitions to avoid jerky behavior
        if current_phase != self.last_phase:
            self.phase_transition_smoothing = 0.3
        else:
            self.phase_transition_smoothing *= 0.8
            
        if self.phase_transition_smoothing > 0.01:
            # Blend with previous phase's behavior during transition
            if self.last_phase == "EMERGENCY":
                prev_force = self._emergency_control(theta, dtheta, x, dx)
            elif self.last_phase == "SWING-UP":
                prev_force = self._swing_up_control(theta, dtheta, x, dx)
            elif self.last_phase == "CONVERGENCE":
                prev_force = self._convergence_control(theta, dtheta, x, dx)
            else:
                prev_force = self._precision_control(theta, dtheta, x, dx)
                
            force = (force * (1 - self.phase_transition_smoothing) + 
                    prev_force * self.phase_transition_smoothing)
        
        self.last_phase = current_phase
        return float(np.clip(force, -100.0, 100.0))

    def _detect_phase(self, theta, dtheta, x, dx):
        """Determine which control phase we're in"""
        abs_theta = abs(theta)
        abs_dtheta = abs(dtheta)
        
        # Emergency: falling fast or extreme angle
        if (abs_theta > self.emergency_threshold or 
            (abs_theta > 0.5 and abs_dtheta > 2.0) or
            (abs_theta > 0.3 and abs_dtheta > 3.0)):
            return "EMERGENCY"
        
        # Swing-up: large angle but not emergency
        elif abs_theta > self.swing_threshold:
            return "SWING-UP"
        
        # Convergence: moderate angle, getting close
        elif abs_theta > self.convergence_threshold:
            return "CONVERGENCE"
        
        # Precision: small angle, fine control needed
        else:
            return "PRECISION"

    def _emergency_control(self, theta, dtheta, x, dx):
        """Emergency response to prevent falling"""
        # Predict where we'll be in the near future
        theta_pred = theta + dtheta * self.momentum_prediction_horizon
        
        # Calculate emergency force based on predicted state
        emergency_force = -self.emergency_gain * theta_pred
        
        # Add momentum compensation
        momentum_compensation = -8.0 * dtheta * np.tanh(2.0 * abs(theta))
        
        # Cart position correction (secondary priority)
        position_correction = -1.5 * x - 0.8 * dx
        
        total_force = emergency_force + momentum_compensation + position_correction
        
        # Ensure decisive action in emergency
        if abs(total_force) < 25.0 and abs(theta) > 0.5:
            total_force = 25.0 * np.sign(total_force) if total_force != 0 else 25.0 * np.sign(theta)
            
        return total_force

    def _swing_up_control(self, theta, dtheta, x, dx):
        """Energy pumping strategy for swing-up phase"""
        # Energy-based control: pump when moving away from upright
        energy_pumping = self.swing_energy_gain * np.sign(theta) * np.tanh(2.0 * abs(dtheta))
        
        # Friction compensation (critical for heavy pole)
        friction_comp = self.swing_friction_compensation * np.sign(dtheta) * np.tanh(abs(dtheta))
        
        # Direction-aware damping: stronger when moving away from upright
        direction_factor = np.tanh(3.0 * theta * dtheta)  # Positive when moving away
        velocity_damping = -4.0 * dtheta * (1.0 + 0.5 * max(0, direction_factor))
        
        # Position management
        position_control = -1.2 * x - 0.6 * dx
        
        total_force = energy_pumping + friction_comp + velocity_damping + position_control
        return total_force

    def _convergence_control(self, theta, dtheta, x, dx):
        """Bring system to near-vertical with good positioning"""
        # Angle control with velocity matching
        angle_control = -self.convergence_angle_gain * theta
        velocity_control = -self.convergence_velocity_gain * dtheta
        
        # Smart position correction - scale with angle improvement
        position_urgency = 1.0 - np.tanh(6.0 * abs(theta))
        position_control = -self.position_correction_gain * x * position_urgency
        velocity_correction = -1.2 * dx * position_urgency
        
        # Gentle velocity damping
        velocity_damping = -2.5 * dtheta * np.exp(-3.0 * abs(theta))
        
        total_force = (angle_control + velocity_control + 
                      position_control + velocity_correction + velocity_damping)
        return total_force

    def _precision_control(self, theta, dtheta, x, dx):
        """Fine stabilization near perfect balance"""
        # High-gain angle control
        angle_control = -self.precision_angle_gain * theta
        velocity_control = -self.precision_velocity_gain * dtheta
        
        # Integral control for position drift (only in precision mode)
        self.position_integral += x * DT
        self.position_integral = np.clip(self.position_integral, -0.8, 0.8)
        integral_control = -self.integral_gain * self.position_integral
        
        # Gentle position correction
        position_control = -1.8 * x - 1.0 * dx
        
        total_force = angle_control + velocity_control + integral_control + position_control
        
        # Very gentle actions near perfect balance
        if abs(theta) < 0.02 and abs(dtheta) < 0.1:
            total_force *= 0.7
            
        return total_force

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
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