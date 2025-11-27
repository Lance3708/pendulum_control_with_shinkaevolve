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
    Phase-Reflex Controller with Energy Shaping

    Philosophy: "Think in distinct physical regimes, not continuous adaptation.
                 React like a human - panic when falling, guide when recovering, hold when balanced."

    Three distinct operational phases:
    1. Emergency Recovery (|θ| > 0.8): Aggressive energy-aware swing-up with predictive cart damping
    2. Recovery Transition (0.15 < |θ| ≤ 0.8): Physics-based predictive catching with smart damping
    3. Precision Hold (|θ| ≤ 0.15): Fine-grained integral control with adaptive anti-windup
    """

    def __init__(self):
        # System parameters
        self.m = M_POLE
        self.M = M_CART
        self.l = L_COM
        self.g = G
        self.Mtot = self.M + self.m
        self.b_c = FRICTION_CART
        self.b_j = FRICTION_JOINT

        # Integral state with asymmetric bounds for faster response
        self.integral_x = 0.0
        self.omega_n = np.sqrt(G / L_COM)  # Natural frequency for normalization

    def _predict_theta_acc(self, theta, dtheta, dx, force):
        """Physics-based angular acceleration prediction for predictive control"""
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        f_cart = -self.b_c * dx
        f_joint = -self.b_j * dtheta
        temp = (force + f_cart + self.m * self.l * dtheta**2 * sin_t) / self.Mtot
        denom = self.l * (4.0/3.0 - self.m * cos_t**2 / self.Mtot)
        return (self.g * sin_t - cos_t * temp + f_joint / (self.m * self.l)) / denom

    def get_action(self, state):
        x, theta, dx, dtheta = state
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi  # Robust normalization
        abs_theta = abs(theta)

        # === PHASE 1: EMERGENCY RECOVERY (|θ| > 0.8) ===
        if abs_theta > 0.8:
            # Energy-risk aware swing-up: prevent over-pumping when already energetic
            E_risk = 0.5*self.M*dx**2 + self.m*self.g*self.l*(1 - np.cos(theta))
            swing_scale = min(1.0, 1.3 / (1.0 + max(0.0, E_risk - 1.0)))

            # Aggressive swing-up with direction awareness
            swing_activation = np.tanh(8.0 * (abs_theta - 0.8))
            # Push harder when falling away, ease off when returning
            energy_pump = 1.0 + np.tanh(3.0 * theta * dtheta / (self.l * self.omega_n))
            u_swing = 12.0 * swing_activation * np.sign(theta) * energy_pump * swing_scale
            force = u_swing

            # Predictive cart damping: counteract momentum buildup
            force -= 0.25 * dx * np.tanh(2.5 * abs_theta)

            # Reset integral during emergency to prevent windup
            self.integral_x *= 0.95
            return float(force)

        # === PHASE 2: RECOVERY TRANSITION (0.15 < |θ| ≤ 0.8) ===
        elif abs_theta > 0.15:
            # Base corrective force: simple proportional feedback
            force = -4.0 * theta - 1.2 * dtheta

            # Direction-aware mid-swing damping (only when returning to center)
            # theta*dtheta < 0 means pole is moving toward vertical
            is_returning = theta * dtheta < 0
            if is_returning and 0.25 < abs_theta < 0.75:
                mid_activation = np.exp(-5.0 * (abs_theta - 0.5)**2)
                # Stronger damping when returning fast
                K_damp = 2.6 * (1.0 + np.tanh(3.0 * abs(theta * dtheta)))
                force += K_damp * dtheta * mid_activation

            # Predictive momentum compensation: catch the pole at the right moment
            theta_acc = self._predict_theta_acc(theta, dtheta, dx, force)
            dtheta_pred = dtheta + theta_acc * DT

            # Correct based on predicted divergence (only when actually diverging)
            pred_div = theta * dtheta_pred / (self.omega_n * self.l)
            if pred_div > 0:  # Only correct when moving away from vertical
                force -= 0.14 * pred_div * np.tanh(3.0 * abs_theta)

            # Gentle integral decay during transition
            self.integral_x *= 0.97
            return float(force)

        # === PHASE 3: PRECISION HOLD (|θ| ≤ 0.15) ===
        else:
            # Fine-grained integral control with position-aware bounds
            # Expand windup limits in direction of needed correction
            lower_bound = -1.5 + 0.4 * x
            upper_bound = 1.5 + 0.4 * x
            self.integral_x += x * DT
            self.integral_x = np.clip(self.integral_x, lower_bound, upper_bound)

            # Adaptive integral gain: stronger near equilibrium, includes position error
            K_i = 0.95 * np.exp(-3.0 * (abs_theta + 0.6*abs(dtheta) + 0.3*abs(x)))
            K_i = np.clip(K_i, 0.30, 0.95)

            # Dual gating: only integrate when stable and slow
            gate = np.tanh(15.0 * (0.15 - abs_theta)) * np.tanh(10.0 * (1.0 - abs(dtheta)))
            gate = max(0, gate)  # One-sided gate

            integral_force = K_i * gate * self.integral_x
            force = -4.5 * theta - 1.5 * dtheta - integral_force

            # Direct position centering when very stable
            stability = np.exp(-8.0 * (theta**2 + 0.3*dtheta**2))
            if stability > 0.4:
                force -= 1.8 * x * stability + 0.9 * dx * stability

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