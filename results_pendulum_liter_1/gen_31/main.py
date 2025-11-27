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
    Advanced Friction-Compensated LQR Controller for Single Inverted Pendulum Stabilization.
    """

    def __init__(self):
        # 系统参数
        m = M_POLE  # 摆杆质量
        M = M_CART  # 小车质量
        l = L_COM   # 质心距离
        g = G       # 重力加速度

        # 精确线性化状态空间模型，针对重长杆和高摩擦环境优化
        Mtot = M + m
        denom = l * (4.0/3.0 - m * cos_theta**2 / Mtot)  # 匹配非线性动力学的分母项

        # 在平衡点(theta=0)精确线性化
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/Mtot, -FRICTION_CART/Mtot, 0],
            [0, (Mtot*g)/(Mtot*l), 0, -FRICTION_JOINT/(m*l**2)]
        ])

        # 精确输入矩阵，考虑杆质量对系统惯性的影响
        B = np.array([
            [0],
            [0],
            [1.0/Mtot + m/(Mtot**2 * (4.0/3.0 - m/Mtot))],
            [-1.0/(Mtot * l * (4.0/3.0 - m/Mtot))]
        ])

        # 优化的LQR权重矩阵
        # 针对重长杆挑战调整：增加角速度权重以加速振荡衰减
        Q = np.diag([5.0, 32.0, 0.5, 2.8])    # [x, theta, dx, dtheta] - 增加theta和dtheta权重
        R = np.array([[0.95]])                 # 略微降低控制惩罚以允许稍强响应

        # 求解LQR增益
        self.K_base = self.solve_lqr(A, B, Q, R)

        # 大偏差时的增益提升因子
        self.high_gain_factor = 2.0

    def solve_lqr(self, A, B, Q, R):
        """求解连续时间LQR问题"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """改进的LQR控制法则: u = -K * x，带平滑自适应增益调节和精确摩擦补偿"""
        x, theta, dx, dtheta = state

        # 角度归一化到[-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        state_vec = np.array([x, theta, dx, dtheta])

        # 基础LQR控制律
        force = -self.K_base @ state_vec

        # 精确非线性摩擦补偿项
        # 针对高摩擦环境优化补偿强度
        friction_compensation = 0.12 * np.tanh(8 * dx) + 0.06 * np.tanh(4 * dtheta)
        force += friction_compensation

        # 平滑增益调度：使用tanh函数实现连续过渡
        # 在角度较大时更激进，接近平衡点时更温和
        angle_magnitude = abs(theta)
        gain_modulation = 1.0 + 0.8 * np.tanh(3.0 * (angle_magnitude - 0.5))
        force *= gain_modulation

        # 限制控制力在物理可行范围内
        force = np.clip(force, -100.0, 100.0)

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