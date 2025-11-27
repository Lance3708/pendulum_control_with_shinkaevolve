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
class DiscreteLQRController:
    """
    Discrete-time LQR Controller with Phase-Based Adaptive Control
    
    Key innovations:
    1. Discrete-time LQR using zero-order hold discretization
    2. Phase-based control architecture with smooth transitions
    3. Enhanced state estimation with filtered derivatives
    4. Energy-based swing-up assistance
    5. Adaptive integral control with state-dependent activation
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

        # Continuous-time state-space matrices
        A_cont = np.zeros((4, 4))
        A_cont[0, 2] = 1.0
        A_cont[1, 3] = 1.0
        A_cont[3, 1] = g / denom0
        A_cont[3, 2] = b_c / (Mtot * denom0)
        A_cont[3, 3] = -b_j / (m * l * denom0)
        A_cont[2, 1] = -(m * l / Mtot) * A_cont[3, 1]
        A_cont[2, 2] = -b_c / Mtot - (m * l / Mtot) * A_cont[3, 2]
        A_cont[2, 3] = b_j / (Mtot * denom0)

        B_cont = np.zeros((4, 1))
        B_cont[2, 0] = 1.0 / Mtot + (m * l) / (Mtot**2 * denom0)
        B_cont[3, 0] = -1.0 / (Mtot * denom0)

        # Discretize using zero-order hold
        self.A_disc, self.B_disc = self._zero_order_hold(A_cont, B_cont, DT)

        # Optimized LQR weights (proven configuration)
        Q = np.diag([4.5, 44.0, 0.6, 3.2])
        R = np.array([[1.0]])

        # Solve discrete-time LQR
        self.K = self.solve_dlqr(self.A_disc, self.B_disc, Q, R)
        
        # Control phases
        self.phase = 'swing_up'
        self.phase_transition_smooth = 0.0
        
        # State estimation
        self.prev_state = None
        self.filtered_dtheta = 0.0
        self.filtered_dx = 0.0
        
        # Integral control
        self.integral_x = 0.0
        self.integral_theta = 0.0
        
        # Energy calculation
        self.omega_n = np.sqrt(G / L_COM)
        
        # Performance metrics
        self.step_count = 0

    def _zero_order_hold(self, A, B, dt):
        """Convert continuous system to discrete using zero-order hold"""
        n = A.shape[0]
        M = np.zeros((2*n, 2*n))
        M[:n, :n] = A
        M[:n, n:] = B @ B.T
        M[n:, :n] = np.zeros((n, n))
        M[n:, n:] = -A.T
        
        phi = np.linalg.matrix_power(np.eye(2*n) + 0.5 * dt * M, 2)
        A_disc = phi[:n, :n]
        B_disc = phi[:n, n:] @ np.linalg.pinv(B) if np.linalg.matrix_rank(B) == B.shape[1] else phi[:n, n:]
        
        return A_disc, B_disc

    def solve_dlqr(self, A, B, Q, R):
        """Solve discrete-time LQR using DARE"""
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def _update_state_estimation(self, state):
        """Update filtered state derivatives and detect phase transitions"""
        if self.prev_state is not None:
            # Simple low-pass filtering for derivatives
            alpha = 0.7
            dx_meas = (state[0] - self.prev_state[0]) / DT
            dtheta_meas = (state[1] - self.prev_state[1]) / DT
            
            self.filtered_dx = alpha * self.filtered_dx + (1 - alpha) * dx_meas
            self.filtered_dtheta = alpha * self.filtered_dtheta + (1 - alpha) * dtheta_meas
            
            # Phase transition logic
            theta_mag = abs(state[1])
            energy_level = self._calculate_energy(state)
            
            if theta_mag > 0.8:
                self.phase = 'swing_up'
            elif theta_mag < 0.3 and energy_level < 0.1:
                self.phase = 'fine_tuning'
            else:
                self.phase = 'stabilization'
                
            # Smooth phase transition
            self.phase_transition_smooth = 0.9 * self.phase_transition_smooth + 0.1 * (
                1.0 if self.phase == 'fine_tuning' else 0.0
            )
        
        self.prev_state = state.copy()
        self.step_count += 1

    def _calculate_energy(self, state):
        """Calculate normalized system energy"""
        x, theta, dx, dtheta = state
        # Kinetic + potential energy normalized by upright energy
        kinetic = 0.5 * M_CART * dx**2 + 0.5 * M_POLE * (dx**2 + (L_COM * dtheta)**2 + 2 * dx * L_COM * dtheta * np.cos(theta))
        potential = M_POLE * G * L_COM * (1 - np.cos(theta))
        total_energy = kinetic + potential
        upright_energy = M_POLE * G * L_COM * 2  # Energy at horizontal position
        return min(abs(total_energy / upright_energy), 2.0)

    def _swing_up_assist(self, state):
        """Energy-based swing-up assistance"""
        x, theta, dx, dtheta = state
        
        if abs(theta) > 0.8:
            # Physics-informed assist with energy consideration
            swing_activation = np.tanh(6.0 * (abs(theta) - 0.8))
            
            # Direction-aware falling severity
            falling_direction = np.sign(theta * dtheta)
            normalized_severity = (theta * dtheta) / (L_COM * self.omega_n)
            falling_severity = 1.0 + np.tanh(3.0 * normalized_severity)
            
            # Energy-based scaling - reduce assist when system has sufficient energy
            energy_level = self._calculate_energy(state)
            energy_scale = max(0.5, 1.0 - 0.5 * np.tanh(5.0 * (energy_level - 0.5)))
            
            u_swing = 8.0 * swing_activation * np.sign(theta) * falling_severity * energy_scale
            return u_swing
        
        return 0.0

    def _adaptive_gain_scheduling(self, state):
        """Phase-aware gain scheduling"""
        x, theta, dx, dtheta = state
        
        # Base gains depend on phase
        if self.phase == 'swing_up':
            pos_gain = 1.0 + 0.8 * np.tanh(6.0 * max(0.0, abs(theta) - 0.6))
            vel_gain = 1.0 + 0.4 * np.tanh(5.0 * max(0.0, abs(dtheta) - 1.2))
        elif self.phase == 'stabilization':
            pos_gain = 1.0 + 0.5 * np.tanh(5.0 * max(0.0, abs(theta) - 0.4))
            vel_gain = 1.0 + 0.3 * np.tanh(4.0 * max(0.0, abs(dtheta) - 0.8))
        else:  # fine_tuning
            pos_gain = 1.0 + 0.2 * np.tanh(8.0 * max(0.0, abs(theta) - 0.1))
            vel_gain = 1.0 + 0.1 * np.tanh(6.0 * max(0.0, abs(dtheta) - 0.3))
        
        return pos_gain * vel_gain

    def _integral_control(self, state):
        """Adaptive integral control with state-dependent activation"""
        x, theta, dx, dtheta = state
        
        # Only activate integral control in fine-tuning phase
        integral_gate = self.phase_transition_smooth
        
        if integral_gate > 0.5:
            # Update integrals with state-dependent leakage
            leak_factor = 1.0 - 0.1 * np.tanh(10.0 * abs(theta))
            self.integral_x = self.integral_x * leak_factor + x * DT
            self.integral_theta = self.integral_theta * leak_factor + theta * DT
            
            # Integral gains scale with phase
            K_i_x = 0.8 * integral_gate
            K_i_theta = 0.3 * integral_gate
            
            return K_i_x * self.integral_x + K_i_theta * self.integral_theta
        else:
            # Reset integrals when not in fine-tuning
            self.integral_x *= 0.95
            self.integral_theta *= 0.95
            return 0.0

    def get_action(self, state):
        """Main control action with phase-based adaptation"""
        # Update state estimation and phase detection
        self._update_state_estimation(state)
        
        x, theta, dx, dtheta = state
        
        # Normalize angle
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        
        # Use filtered derivatives for more robust control
        state_vec = np.array([x, theta, self.filtered_dx, self.filtered_dtheta])
        
        # Discrete-time LQR base control
        base_force = -self.K @ state_vec
        
        # Adaptive gain scheduling
        adaptive_gain = self._adaptive_gain_scheduling(state)
        force = base_force * adaptive_gain
        
        # Swing-up assistance
        swing_force = self._swing_up_assist(state)
        force = force + swing_force
        
        # Integral control for fine-tuning
        integral_force = self._integral_control(state)
        force = force + integral_force
        
        return float(force[0])

# Initialize controller
controller = DiscreteLQRController()

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