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
    Adaptive LQR with Integral Action for Single Inverted Pendulum Stabilization.
    """

    def __init__(self):
        # 系统参数
        m = M_POLE  # 摆杆质量
        M = M_CART  # 小车质量  
        l = L_COM   # 质心距离
        g = G       # 重力加速度
        
        # 扩展状态空间模型以包含积分项 (LQI)
        # 新增状态: int_theta (theta的积分)
        A_ext = np.array([
            [0, 0, 0, 1, 0],      # x_dot = dx
            [0, 0, 0, 0, 1],      # theta_dot = dtheta
            [0, 0, 0, 0, 0],      # int_theta_dot = theta
            [0, -m*g/M, 0, 0, 0], # dx_dot = ...
            [0, (m+M)*g/(M*l), 0, 0, 0] # dtheta_dot = ...
        ])
        
        B_ext = np.array([
            [0],
            [0],
            [0],
            [1/M],
            [-1/(M*l)]
        ])
        
        # LQR权重矩阵 (为扩展状态设计)
        # 对积分项给予适中权重以消除稳态误差
        Q_ext = np.diag([10.0, 50.0, 30.0, 1.0, 2.0])  # [x, theta, int_theta, dx, dtheta]
        R_ext = np.array([[1.0]])
        
        # 求解LQR增益
        self.K_ext = self.solve_lqr(A_ext, B_ext, Q_ext, R_ext)
        
        # 初始化积分项
        self.int_theta = 0.0
        
    def solve_lqr(self, A, B, Q, R):
        """求解连续时间LQR问题"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    
    def get_action(self, state):
        """Adaptive LQR control law with integral action"""
        x, theta, dx, dtheta = state
        
        # 角度归一化
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # 非线性状态变换以实现自适应增益调度
        # 当角度增大时，控制作用自然饱和，避免过冲
        scale_factor = 1.0  # 基础增益
        adaptive_gain = 1.0 + 4.0 * np.tanh(np.abs(theta) * 2)  # 根据角度调整增益
        
        # 更新积分项 (带抗饱和)
        # 只在系统接近平衡时积分，防止启动时积分饱和
        if np.abs(theta) < 0.5:  # 只在小角度范围内积分
            self.int_theta += theta * DT
        else:
            # 在大角度时重置或限制积分项
            self.int_theta = 0.99 * self.int_theta  # 缓慢衰减
            
        # 构造扩展状态向量
        state_ext = np.array([x, theta, self.int_theta, dx, dtheta])
        
        # 计算控制力
        force = -self.K_ext @ state_ext
        
        # 应用自适应增益
        force = force * adaptive_gain
        
        return float(force[0])

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