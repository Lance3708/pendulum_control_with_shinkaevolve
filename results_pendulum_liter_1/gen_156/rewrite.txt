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
    Phase-Aware Adaptive Controller with Energy-Shaped Swing-Up
    Dynamically adapts control strategy based on operational phase for optimal performance.
    """

    def __init__(self):
        # System parameters
        m = M_POLE
        M = M_CART
        l = L_COM
        g = G
        Mtot = M + m
        denom0 = l * (4.0 / 3.0 - m / Mtot)
        b_c = FRICTION_CART
        b_j = FRICTION_JOINT

        # Physically accurate linearized A matrix
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[3, 1] = g / denom0
        A[3, 2] = b_c / (Mtot * denom0)
        A[3, 3] = -b_j / (m * l * denom0)
        A[2, 1] = -(m * l / Mtot) * A[3, 1]
        A[2, 2] = -b_c / Mtot - (m * l / Mtot) * A[3, 2]
        A[2, 3] = b_j / (Mtot * denom0)

        # B matrix
        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        B[3, 0] = -1.0 / (Mtot * denom0)

        # Multiple LQR designs for different operational phases
        # Aggressive swing-up LQR
        Q_swing = np.diag([2.0, 30.0, 0.3, 2.5])
        R_swing = np.array([[1.0]])
        self.K_swing = self.solve_lqr(A, B, Q_swing, R_swing)
        
        # Precision stabilization LQR
        Q_stable = np.diag([5.0, 50.0, 0.8, 4.0])
        R_stable = np.array([[1.0]])
        self.K_stable = self.solve_lqr(A, B, Q_stable, R_stable)
        
        # Transition LQR
        Q_trans = np.diag([3.5, 40.0, 0.5, 3.0])
        R_trans = np.array([[1.0]])
        self.K_trans = self.solve_lqr(A, B, Q_trans, R_trans)

        # Integral control parameters
        self.integral_x = 0.0
        self.K_i = 0.85
        
        # Energy parameters
        self.E_target = M_POLE * G * L_COM  # Potential energy at top
        
        # Phase tracking
        self.prev_phase = "STABILIZE"
        self.phase_timer = 0

    def solve_lqr(self, A, B, Q, R):
        """Solve continuous-time LQR"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def classify_phase(self, theta, dtheta):
        """Classify operational phase based on state"""
        angle_abs = abs(theta)
        velocity_abs = abs(dtheta)
        
        # Momentum-based classification
        momentum = theta * dtheta
        energy_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
        energy_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
        energy_current = energy_kinetic + energy_potential
        energy_deficit = max(0.0, 1.0 - energy_current / self.E_target)
        
        # Phase classification logic
        if angle_abs > 1.0 or (angle_abs > 0.8 and velocity_abs > 2.0):
            return "SWING_UP"
        elif angle_abs > 0.3 or (angle_abs > 0.15 and velocity_abs > 0.8):
            return "TRANSITION"
        else:
            return "STABILIZE"

    def get_action(self, state):
        """Phase-aware adaptive control with energy-shaped swing-up"""
        x, theta, dx, dtheta = state

        # Robust angle normalization
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        # Classify current operational phase
        phase = self.classify_phase(theta, dtheta)
        
        # Update phase timer for hysteresis
        if phase != self.prev_phase:
            self.phase_timer = 0
        else:
            self.phase_timer += 1
        self.prev_phase = phase

        # Select control law based on phase
        if phase == "SWING_UP":
            force = self.swing_up_control(x, theta, dx, dtheta)
        elif phase == "TRANSITION":
            force = self.transition_control(x, theta, dx, dtheta)
        else:  # STABILIZE
            force = self.stabilize_control(x, theta, dx, dtheta)

        # Phase-specific integral action
        force += self.integral_action(x, theta, dtheta, phase)
        
        return float(force)

    def swing_up_control(self, x, theta, dx, dtheta):
        """Aggressive energy-shaping control for swing-up phase"""
        # Energy-based swing-up with directional awareness
        energy_kinetic = 0.5 * M_POLE * (L_COM * dtheta)**2
        energy_potential = M_POLE * G * L_COM * (1 - np.cos(theta))
        energy_current = energy_kinetic + energy_potential
        energy_deficit = max(0.0, 1.0 - energy_current / self.E_target)
        
        # Activation function based on angle and energy deficit
        swing_activation = np.tanh(5.0 * (abs(theta) - 0.7)) * energy_deficit
        
        # Directional momentum consideration
        momentum = theta * dtheta
        if momentum > 0:  # Falling away
            u_swing = 10.0 * swing_activation * np.sign(theta)
        else:  # Recovering
            u_swing = 7.0 * swing_activation * np.sign(theta)
            
        # Base LQR for stability during swing
        state_vec = np.array([x, theta, dx, dtheta])
        u_lqr = -self.K_swing @ state_vec
        
        # Blend strategies
        return u_swing + 0.7 * u_lqr

    def transition_control(self, x, theta, dx, dtheta):
        """Mixed strategy for transition phase"""
        state_vec = np.array([x, theta, dx, dtheta])
        u_lqr = -self.K_trans @ state_vec
        
        # Moderate swing-up assist if needed
        if abs(theta) > 0.5:
            swing_activation = np.tanh(4.0 * (abs(theta) - 0.4))
            u_swing = 4.0 * swing_activation * np.sign(theta)
            return u_lqr + u_swing
        
        return u_lqr

    def stabilize_control(self, x, theta, dx, dtheta):
        """Precision control for stabilization phase"""
        state_vec = np.array([x, theta, dx, dtheta])
        u_lqr = -self.K_stable @ state_vec
        
        # Additional damping when very close to equilibrium
        if abs(theta) < 0.1 and abs(dtheta) < 0.5:
            u_damp = -0.15 * dtheta * (0.1 - abs(theta)) / 0.1
            return u_lqr + u_damp
            
        return u_lqr

    def integral_action(self, x, theta, dtheta, phase):
        """Phase-specific integral control with adaptive anti-windup"""
        # Integral gate based on operational phase
        if phase == "STABILIZE":
            # Tight gate for precision
            integral_gate = np.tanh(15.0 * (0.08 - abs(theta))) * np.tanh(10.0 * (0.5 - abs(dtheta)))
        elif phase == "TRANSITION":
            # Moderate gate
            integral_gate = np.tanh(10.0 * (0.15 - abs(theta))) * np.tanh(8.0 * (0.8 - abs(dtheta)))
        else:  # SWING_UP
            # Very loose gate to prevent windup
            integral_gate = np.tanh(5.0 * (0.3 - abs(theta))) * np.tanh(4.0 * (1.5 - abs(dtheta)))
        
        # Update integral term
        if integral_gate > 0.1 and abs(x) > 0.01:
            self.integral_x += x * DT
            # Phase-specific anti-windup limits
            if phase == "STABILIZE":
                self.integral_x = np.clip(self.integral_x, -1.0, 1.0)
            elif phase == "TRANSITION":
                self.integral_x = np.clip(self.integral_x, -2.0, 2.0)
            else:  # SWING_UP
                self.integral_x = np.clip(self.integral_x, -3.0, 3.0)
        else:
            # Adaptive decay based on phase
            if phase == "STABILIZE":
                decay = 0.98
            elif phase == "TRANSITION":
                decay = 0.95
            else:  # SWING_UP
                decay = 0.90
            self.integral_x *= decay
        
        return self.K_i * integral_gate * self.integral_x

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
    # Apply force limits
    return float(np.clip(force, -100.0, 100.0))
# EVOLVE-BLOCK-END

def run_simulation(seed=None):
    """
    Runs the simulation loop.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initial state: 0.4 rad (~23 degrees)
    # 更大初始角度配合更重更长的杆子，极具挑战性
    state = np.array([0.0, 0.9, 0.0, 0.0])

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