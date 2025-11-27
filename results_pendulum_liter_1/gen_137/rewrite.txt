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
    Energy-Shaping Hybrid Controller with Phase-Based Switching
    Uses energy-based control for swing-up and LQR for stabilization,
    with smooth transitions between phases for optimal performance.
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        
        # Compute natural frequency
        self.omega_n = np.sqrt(self.g / self.l)
        
        # Design LQR controller for stabilization phase
        self.setup_lqr()
        
        # Integral control state
        self.integral_x = 0.0
        self.integral_theta = 0.0
        
        # Energy calculations
        self.E_upright = self.m * self.g * self.l  # Potential energy at upright position
        
        # Phase tracking
        self.phase = 'swing_up'  # 'swing_up', 'transition', 'stabilize'

    def setup_lqr(self):
        """Setup LQR controller for stabilization phase"""
        # Physically accurate linearized A matrix
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        
        denom0 = self.l * (4.0 / 3.0 - self.m / self.Mtot)
        A[3, 1] = self.g / denom0
        A[3, 2] = FRICTION_CART / (self.Mtot * denom0)
        A[3, 3] = -FRICTION_JOINT / (self.m * self.l * denom0)
        A[2, 1] = -(self.m * self.l / self.Mtot) * A[3, 1]
        A[2, 2] = -FRICTION_CART / self.Mtot - (self.m * self.l / self.Mtot) * A[3, 2]
        A[2, 3] = FRICTION_JOINT / (self.Mtot * denom0)

        # B matrix
        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / self.Mtot + (self.m * self.l) / (self.Mtot**2 * denom0)
        B[3, 0] = -1.0 / (self.Mtot * denom0)

        # LQR weights - optimized for fast stabilization with minimal overshoot
        Q = np.diag([10.0, 100.0, 2.0, 8.0])  # Higher weight on angle for precision
        R = np.array([[1.0]])

        # Solve LQR gains
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        self.K_lqr = np.linalg.inv(R) @ B.T @ P

    def calculate_pendulum_energy(self, theta, dtheta):
        """Calculate total mechanical energy of the pendulum"""
        T = 0.5 * self.m * (self.l * dtheta)**2  # Kinetic energy
        V = self.m * self.g * self.l * (1 - np.cos(theta))  # Potential energy
        return T + V

    def energy_shaping_control(self, state):
        """Energy-based control for swing-up phase"""
        x, theta, dx, dtheta = state
        
        # Calculate current energy and energy error
        E_current = self.calculate_pendulum_energy(theta, dtheta)
        E_error = self.E_upright - E_current
        
        # Energy shaping control law with centrifugal compensation
        # u = k * E_error * sign(theta * dtheta) - b * dx
        k_energy = 0.8  # Energy shaping gain
        b_damping = 0.5  # Damping gain
        
        # Centrifugal compensation term
        centrifugal_comp = 0.1 * self.m * self.l * dtheta**2 * np.sin(theta)
        
        # Control law
        u_energy = k_energy * E_error * np.sign(theta * dtheta) - b_damping * dx - centrifugal_comp
        
        return u_energy

    def lqr_control(self, state):
        """Standard LQR control with integral action"""
        x, theta, dx, dtheta = state
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        state_vec = np.array([x, theta, dx, dtheta])
        u_lqr = -self.K_lqr @ state_vec
        
        # Integral action with anti-windup
        self.integral_x += x * DT
        self.integral_theta += theta * DT
        
        # Anti-windup clamping
        self.integral_x = np.clip(self.integral_x, -2.0, 2.0)
        self.integral_theta = np.clip(self.integral_theta, -1.0, 1.0)
        
        # Integral gains
        k_ix = 0.5
        k_itheta = 2.0
        
        u_integral = -k_ix * self.integral_x - k_itheta * self.integral_theta
        
        return u_lqr[0] + u_integral

    def transition_control(self, state, alpha):
        """Smooth transition between energy shaping and LQR"""
        u_energy = self.energy_shaping_control(state)
        u_lqr = self.lqr_control(state)
        
        # Linear interpolation
        return alpha * u_lqr + (1 - alpha) * u_energy

    def determine_phase(self, state):
        """Determine current control phase based on state"""
        x, theta, dx, dtheta = state
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        # Calculate energy
        E_current = self.calculate_pendulum_energy(theta, dtheta)
        E_normalized = E_current / self.E_upright
        
        # Phase determination criteria
        angle_threshold = 0.3  # radians
        velocity_threshold = 1.0  # rad/s
        energy_threshold = 0.9  # 90% of upright energy
        
        # If close to upright and slow, stabilize
        if abs(theta) < angle_threshold and abs(dtheta) < velocity_threshold:
            return 'stabilize'
        # If almost enough energy and approaching upright, transition
        elif E_normalized > energy_threshold and abs(theta) < 0.6:
            return 'transition'
        # Otherwise, keep swinging up
        else:
            return 'swing_up'

    def get_action(self, state):
        """Main control function implementing hybrid strategy"""
        x, theta, dx, dtheta = state
        
        # Determine current phase
        current_phase = self.determine_phase(state)
        
        if current_phase == 'swing_up':
            # Pure energy shaping control
            force = self.energy_shaping_control(state)
            
            # Reset integrals during swing-up to prevent windup
            self.integral_x = 0.0
            self.integral_theta = 0.0
            
        elif current_phase == 'transition':
            # Calculate transition parameter based on angle
            # Alpha goes from 0 (pure energy shaping) to 1 (pure LQR)
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))
            alpha = np.clip(abs(theta_norm) / 0.6, 0.0, 1.0)
            force = self.transition_control(state, alpha)
            
        else:  # stabilizing phase
            force = self.lqr_control(state)
        
        return float(force)

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