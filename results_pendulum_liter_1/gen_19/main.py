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
    Suboptimal LQR Controller for Single Inverted Pendulum Stabilization.
    次优LQR控制器 - 能稳住，但参数故意调得"懒惰"。

    特点：
    1. 物理环境更难（杆更长更重，摩擦更大）。
    2. Q矩阵参数较小：对误差容忍度高 -> 精度分低。
    3. R矩阵参数较大：不愿用大力 -> 响应慢，时间分低。

    目标：初始分数 ~3000 分，进化后可达 9000+ 分。
    """

    def __init__(self):
        # 系统参数
        m = M_POLE  # 摆杆质量
        M = M_CART  # 小车质量
        l = L_COM   # 质心距离
        g = G       # 重力加速度

        # 线性化状态空间模型
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -m*g/M, 0, 0],
            [0, (m+M)*g/(M*l), 0, 0]
        ])

        B = np.array([
            [0],
            [0],
            [1/M],
            [-1/(M*l)]
        ])

        # LQR权重矩阵（更保守的次优设置，留出充分进化空间）
        # Q: 显著降低权重，高度容忍误差 - 导致基础分数低
        # R: 大幅增加权重，极度限制出力 - 导致响应慢，能耗高
        Q = np.diag([2.0, 8.0, 0.05, 0.4])    # [x, theta, dx, dtheta] (更保守)
        R = np.array([[3.0]])                  # 控制力惩罚极大 (非常不愿意用大力)

        # 求解LQR增益
        self.K = self.solve_lqr(A, B, Q, R)

    def solve_lqr(self, A, B, Q, R):
        """求解连续时间LQR问题"""
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        """LQR控制法则: u = -K * x"""
        x, theta, dx, dtheta = state
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        # Friction compensation to counteract known nonlinear effects
        # Cart friction compensation (Coulomb friction model)
        cart_friction_compensation = FRICTION_CART * np.sign(dx) if abs(dx) > 1e-6 else 0.0

        # Joint friction compensation (viscous friction model)
        joint_friction_compensation = FRICTION_JOINT * dtheta

        # Total feedforward compensation
        friction_compensation = cart_friction_compensation + joint_friction_compensation

        state_vec = np.array([x, theta, dx, dtheta])
        lqr_force = -self.K @ state_vec

        # Apply friction compensation before limiting
        force = lqr_force[0] + friction_compensation

        return float(force)

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