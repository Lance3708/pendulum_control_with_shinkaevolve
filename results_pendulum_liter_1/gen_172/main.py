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
    Phase-Aware Momentum-Cancellation LQR Controller
    Uses sliding-mode inspired phase blending with momentum-aware integral action.
    """

    def __init__(self):
        m = M_POLE
        M = M_CART
        l = L_COM
        g = G
        Mtot = M + m
        denom0 = l * (4.0 / 3.0 - m / Mtot)
        b_c = FRICTION_CART
        b_j = FRICTION_JOINT

        # Linearized dynamics matrices
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[3, 1] = g / denom0
        A[3, 2] = b_c / (Mtot * denom0)
        A[3, 3] = -b_j / (m * l * denom0)
        A[2, 1] = -(m * l / Mtot) * A[3, 1]
        A[2, 2] = -b_c / Mtot - (m * l / Mtot) * A[3, 2]
        A[2, 3] = b_j / (Mtot * denom0)

        B = np.zeros((4, 1))
        B[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        B[3, 0] = -1.0 / (Mtot * denom0)

        # Proven optimal LQR weights
        Q = np.diag([4.5, 44.0, 0.6, 3.2])
        R = np.array([[1.0]])
        self.K = self.solve_lqr(A, B, Q, R)

        # Physical constants for control
        self.omega_n = np.sqrt(G / L_COM)
        self.E_ref = M_POLE * G * L_COM
        self.I_pole = M_POLE * L_COM**2 / 3.0  # Moment of inertia

        # Integral states
        self.integral_x = 0.0
        self.integral_momentum = 0.0  # Novel: track angular momentum deficit
        
        # Tuned parameters
        self.K_i = 0.8
        self.K_mom = 0.015  # Momentum integral gain

    def solve_lqr(self, A, B, Q, R):
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def compute_stability_index(self, theta, dtheta):
        """Compute unified stability metric for phase determination"""
        E_kinetic = 0.5 * M_POLE * (dtheta * L_COM)**2
        E_potential = M_POLE * G * L_COM * (1.0 - np.cos(theta))
        E_current = E_kinetic + E_potential
        energy_deficit = max(0.0, 1.0 - E_current / self.E_ref)
        
        # Stability index: higher = more unstable
        sigma = abs(theta) + 0.3 * energy_deficit
        return sigma, energy_deficit

    def get_action(self, state):
        x, theta, dx, dtheta = state
        
        # Robust angle normalization
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        # Compute stability index for phase determination
        sigma, energy_deficit = self.compute_stability_index(theta, dtheta)
        
        # Phase weights via smooth sigmoid transitions
        recovery_weight = 1.0 / (1.0 + np.exp(-12.0 * (sigma - 0.7)))
        stabilize_weight = 1.0 - recovery_weight
        
        # === BASE LQR CONTROL ===
        state_vec = np.array([x, theta, dx, dtheta])
        base_force = -self.K @ state_vec
        
        # Gentle gain scheduling (proven optimal slopes)
        pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.6))
        vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 1.0))
        adaptive_gain = pos_gain * vel_gain
        
        force = base_force * adaptive_gain
        
        # === RECOVERY PHASE: Energy injection with momentum feedforward ===
        if recovery_weight > 0.05:
            swing_activation = np.tanh(6.0 * max(0.0, abs(theta) - 0.8))
            
            # Normalized momentum-based falling severity
            norm_momentum = (theta * dtheta) / (self.omega_n * L_COM)
            falling_factor = 1.0 + np.tanh(2.5 * norm_momentum)
            
            # Energy-scaled assist
            energy_scale = 1.0 + 0.4 * energy_deficit
            
            # Predictive feedforward: estimate where pole will be
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            Mtot = M_CART + M_POLE
            denom0 = L_COM * (4.0/3.0 - M_POLE * cos_t**2 / Mtot)
            temp_pred = (M_POLE * L_COM * dtheta**2 * sin_t) / Mtot
            theta_acc_pred = (G * sin_t - cos_t * temp_pred) / denom0
            
            # One-step prediction
            theta_pred = theta + dtheta * DT + 0.5 * theta_acc_pred * DT**2
            predictive_boost = 1.0 + 0.15 * np.tanh(3.0 * (abs(theta_pred) - abs(theta)))
            
            u_recovery = 8.0 * swing_activation * np.sign(theta) * falling_factor * energy_scale * predictive_boost
            force = force + recovery_weight * u_recovery
        
        # === STABILIZATION PHASE: Precision control with dual-gated integral ===
        # Dual gating: angle AND velocity (proven superior)
        angle_gate = np.tanh(12.0 * (0.1 - abs(theta)))
        velocity_gate = np.tanh(8.0 * (1.0 - abs(dtheta)))
        integral_gate = angle_gate * velocity_gate
        
        if integral_gate > 0.1:
            # Standard position integral
            self.integral_x += x * DT
            
            # Novel: Angular momentum deficit integral
            # Target momentum is zero; track cumulative deficit
            momentum_error = -dtheta  # Want dtheta -> 0
            self.integral_momentum += momentum_error * DT * stabilize_weight
            self.integral_momentum = np.clip(self.integral_momentum, -0.5, 0.5)
        else:
            # Leaky decay when gates closed
            self.integral_x *= 0.95
            self.integral_momentum *= 0.9
        
        # Combined integral force
        integral_force = self.K_i * integral_gate * self.integral_x
        momentum_force = self.K_mom * integral_gate * self.integral_momentum * np.sign(theta + 1e-8)
        
        force = force + integral_force + momentum_force
        
        # === TRANSITION SMOOTHING: Sliding surface inspired damping ===
        # In transition zone, add extra damping proportional to sliding variable
        if 0.15 < abs(theta) < 0.6:
            # Sliding surface: s = dtheta + lambda * theta
            lambda_slide = 2.0 * self.omega_n
            s = dtheta + lambda_slide * theta
            
            # Smooth reaching law
            transition_damp = -0.5 * np.tanh(2.0 * s)
            force = force + transition_damp
        
        # Near-equilibrium cart damping
        if abs(theta) < 0.12:
            cart_damp = -0.1 * dx * (0.12 - abs(theta)) / 0.12
            force = force + cart_damp

        return float(force[0])

# Initialize controller
controller = Controller()

def get_control_action(state):
    force = controller.get_action(state)
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