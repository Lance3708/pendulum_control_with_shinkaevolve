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
    Phase-Based Adaptive Controller with Predictive Positioning
    
    Philosophy: "Move to where the pole will be, not where it is"
    
    Four distinct phases with crisp transitions:
    1. Emergency rescue (|theta| > 0.5): Aggressive catch-up
    2. Recovery swing (0.2 < |theta| < 0.5): Controlled energy management
    3. Stabilization (0.05 < |theta| < 0.2): PD control with damping
    4. Precision hold (|theta| < 0.05): Fine-tuned integral control
    """

    def __init__(self):
        # Physical constants for calculations
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        
        # Natural frequency - key timescale
        self.omega_n = np.sqrt(G / L_COM)
        
        # Integral state for precision phase
        self.integral_x = 0.0
        self.integral_theta = 0.0
        
        # Previous state for derivative calculations
        self.prev_theta = None
        self.prev_x = None
        
        # Energy tracking for adaptive control
        self.prev_energy = None

    def compute_energy(self, theta, dtheta, dx):
        """Compute total mechanical energy of the system"""
        # Kinetic energy (cart + pole)
        KE_cart = 0.5 * self.M * dx**2
        KE_pole = 0.5 * self.m * (dx**2 + (self.l * dtheta)**2 + 2 * dx * self.l * dtheta * np.cos(theta))
        
        # Potential energy (pole, relative to upright)
        PE = self.m * self.g * self.l * (1 - np.cos(theta))
        
        return KE_cart + KE_pole + PE

    def predict_theta(self, theta, dtheta, steps=3):
        """Predict future theta using simplified dynamics"""
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Approximate angular acceleration
        denom = self.l * (4.0/3.0 - self.m * cos_theta**2 / self.Mtot)
        theta_acc = (self.g * sin_theta - FRICTION_JOINT * dtheta / (self.m * self.l)) / denom
        
        # Simple Euler extrapolation
        theta_pred = theta + dtheta * DT * steps + 0.5 * theta_acc * (DT * steps)**2
        dtheta_pred = dtheta + theta_acc * DT * steps
        
        return theta_pred, dtheta_pred

    def get_action(self, state):
        """Phase-based control with distinct strategies"""
        x, theta, dx, dtheta = state
        
        # Normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Compute current energy
        energy = self.compute_energy(theta, dtheta, dx)
        
        # Predict future state
        theta_pred, dtheta_pred = self.predict_theta(theta, dtheta, steps=2)
        
        # Determine phase based on angle magnitude
        abs_theta = abs(theta)
        
        force = 0.0
        
        # ============ PHASE 1: EMERGENCY RESCUE (|theta| > 0.5 rad) ============
        if abs_theta > 0.5:
            # Core idea: Push hard toward the falling direction
            # The cart must "catch up" to the pole's base
            
            # Direction of fall
            fall_dir = np.sign(theta)
            
            # Urgency factor: more urgent as angle increases
            urgency = 1.0 + 2.0 * (abs_theta - 0.5)
            
            # Base rescue force - proportional to how far we've fallen
            rescue_force = 12.0 * theta * urgency
            
            # Add momentum compensation - if pole is accelerating its fall, push harder
            if theta * dtheta > 0:  # Falling away from vertical
                rescue_force += 6.0 * dtheta * fall_dir
            
            # Predictive component - where will the pole be?
            rescue_force += 3.0 * (theta_pred - theta) * urgency
            
            # Position correction - don't let cart drift too far
            pos_correction = -0.8 * x - 0.3 * dx
            
            force = rescue_force + pos_correction
            
        # ============ PHASE 2: RECOVERY SWING (0.2 < |theta| < 0.5) ============
        elif abs_theta > 0.2:
            # Core idea: Controlled energy management during transition
            
            # Proportional control on angle
            K_p = 18.0
            force = K_p * theta
            
            # Derivative damping - stronger when moving fast
            K_d = 4.5
            force += K_d * dtheta
            
            # Direction-aware damping
            if theta * dtheta > 0:  # Moving away from vertical
                force += 2.5 * np.sign(theta) * abs(dtheta)
            else:  # Returning to vertical - lighter touch
                force += 0.8 * np.sign(theta) * abs(dtheta)
            
            # Cart positioning - start bringing cart back
            force -= 1.2 * x + 0.6 * dx
            
            # Predictive nudge
            force += 1.5 * (theta_pred - theta)
            
        # ============ PHASE 3: STABILIZATION (0.05 < |theta| < 0.2) ============
        elif abs_theta > 0.05:
            # Core idea: Smooth PD control with careful damping
            
            # Strong proportional gain
            K_p = 28.0
            force = K_p * theta
            
            # Velocity damping
            K_d = 6.0
            force += K_d * dtheta
            
            # Cart centering becomes more important
            force -= 2.5 * x + 1.2 * dx
            
            # Small predictive term
            force += 0.8 * (theta_pred - theta)
            
            # Start building integral for position
            if abs(dtheta) < 0.5:
                self.integral_x += x * DT
                self.integral_x = np.clip(self.integral_x, -0.8, 0.8)
            
        # ============ PHASE 4: PRECISION HOLD (|theta| < 0.05) ============
        else:
            # Core idea: Fine-tuned control with integral action
            
            # Gentle proportional control
            K_p = 35.0
            force = K_p * theta
            
            # Light damping
            K_d = 7.0
            force += K_d * dtheta
            
            # Strong cart centering with integral
            force -= 3.5 * x + 1.5 * dx
            
            # Integral control for zero steady-state error
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, -0.5, 0.5)
            self.integral_theta += theta * DT
            self.integral_theta = np.clip(self.integral_theta, -0.1, 0.1)
            
            force -= 1.2 * self.integral_x
            force += 8.0 * self.integral_theta
        
        # ============ UNIVERSAL CORRECTIONS ============
        
        # Friction compensation for cart
        if abs(dx) > 0.01:
            force += FRICTION_CART * dx * 0.5
        
        # Anti-windup: reset integrals if we leave precision phase
        if abs_theta > 0.15:
            self.integral_x *= 0.9
            self.integral_theta *= 0.9
        
        # Safety limits
        force = np.clip(force, -100.0, 100.0)
        
        # Store state for next iteration
        self.prev_theta = theta
        self.prev_x = x
        self.prev_energy = energy
        
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