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
    Adaptive LQR Controller with Integral Action for Single Inverted Pendulum.
    
    Key improvements over baseline:
    1. Aggressive Q weights for fast theta stabilization
    2. Lower R for more control authority
    3. Adaptive gain scheduling based on state magnitude
    4. Integral action to counteract friction-induced steady-state error
    5. Smooth transitions between control regimes
    """

    def __init__(self):
        # 系统参数
        m = M_POLE  # 摆杆质量
        M = M_CART  # 小车质量  
        l = L_COM   # 质心距离
        g = G       # 重力加速度
        
        # 线性化状态空间模型 (with friction approximation)
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, -FRICTION_CART/M, 0],
            [0, (m+M)*g/(M*l), FRICTION_CART/(M*l), -FRICTION_JOINT/(m*l*l)]
        ])
        
        B = np.array([
            [0],
            [0],
            [1/M],
            [-1/(M*l)]
        ])
        
        # Aggressive LQR weights for fast stabilization
        # High theta weight for quick angle correction
        Q_aggressive = np.diag([25.0, 200.0, 2.0, 15.0])  # [x, theta, dx, dtheta]
        R_aggressive = np.array([[0.15]])  # Allow more control force
        
        # Precision LQR weights for fine-tuning near equilibrium
        Q_precision = np.diag([40.0, 300.0, 5.0, 20.0])
        R_precision = np.array([[0.25]])
        
        # Solve both LQR gains
        self.K_aggressive = self.solve_lqr(A, B, Q_aggressive, R_aggressive)
        self.K_precision = self.solve_lqr(A, B, Q_precision, R_precision)
        
        # Integral state for steady-state error correction
        self.theta_integral = 0.0
        self.x_integral = 0.0
        self.Ki_theta = 3.5  # Integral gain for theta
        self.Ki_x = 0.8      # Integral gain for x
        
        # Anti-windup limits
        self.integral_limit_theta = 0.5
        self.integral_limit_x = 2.0
        
        # Previous state for derivative estimation
        self.prev_theta = None
        
    def solve_lqr(self, A, B, Q, R):
        """求解连续时间LQR问题"""
        from scipy.linalg import solve_continuous_are
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except:
            # Fallback gains if LQR fails
            return np.array([[15.0, 120.0, 12.0, 25.0]])
    
    def get_action(self, state):
        """Adaptive LQR with integral action"""
        x, theta, dx, dtheta = state
        
        # Proper angle wrapping
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        abs_theta = abs(theta)
        
        # Adaptive gain blending based on angle magnitude
        if abs_theta > 0.5:
            # Large angle: use aggressive gains
            alpha = 1.0
        elif abs_theta > 0.15:
            # Transition region: blend gains
            alpha = (abs_theta - 0.15) / 0.35
        else:
            # Near upright: use precision gains
            alpha = 0.0
        
        K = alpha * self.K_aggressive + (1 - alpha) * self.K_precision
        
        state_vec = np.array([x, theta, dx, dtheta])
        
        # Base LQR control
        force = -K @ state_vec
        force = float(force[0])
        
        # Integral action (only when near equilibrium to prevent windup during swing)
        if abs_theta < 0.4:
            # Update integrals
            self.theta_integral += theta * DT
            self.x_integral += x * DT
            
            # Anti-windup
            self.theta_integral = np.clip(self.theta_integral, 
                                          -self.integral_limit_theta, 
                                          self.integral_limit_theta)
            self.x_integral = np.clip(self.x_integral,
                                      -self.integral_limit_x,
                                      self.integral_limit_x)
            
            # Add integral correction
            force += -self.Ki_theta * self.theta_integral
            force += -self.Ki_x * self.x_integral
        else:
            # Reset integrals when far from equilibrium
            self.theta_integral *= 0.95
            self.x_integral *= 0.95
        
        # Friction compensation - add force to overcome static friction
        if abs(dx) < 0.1 and abs(force) > 0.5:
            force += np.sign(force) * FRICTION_CART * 0.3
        
        # Additional damping for high velocities to prevent overshoot
        if abs(dtheta) > 1.0:
            force += -2.0 * dtheta
        if abs(dx) > 1.5:
            force += -1.5 * dx
            
        # Soft saturation to avoid abrupt changes
        force = 100.0 * np.tanh(force / 80.0)
        
        self.prev_theta = theta
        
        return force

# Initialize controller  
controller = Controller()

def get_control_action(state):
    return float(controller.get_action(state))
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