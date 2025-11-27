# Inverted Pendulum Control Optimization with ShinkaEvolve

<p align="center">
<i>Automated discovery of high-performance control strategies using LLM-driven evolutionary programming</i>
</p>

---

## 📖 Project Overview

This project uses the [ShinkaEvolve](https://github.com/SakanaAI/shinka) framework (an LLM-driven evolutionary programming library developed by Sakana AI) to automatically optimize single inverted pendulum control algorithms through multi-generational evolution. The experiments demonstrate **LLM agents' automated innovation capabilities in complex control problems**, with the evolved controllers significantly outperforming the initial design across multiple metrics.

### Key Achievements
- ✅ **180+ generations of evolutionary optimization**, achieving over 20% performance improvement
- ✅ **Multi-model collaboration**: 9 state-of-the-art LLMs including DeepSeek, Claude, Gemini, and Qwen participating in evolution
- ✅ **Fully automated**: Complete pipeline from code generation, evaluation, selection to optimization executed by LLM agents
- ✅ **Two-round experimental validation**: Complete exploration path from conservative optimization to aggressive innovation

---

## 🎯 Control Problem Definition

### Physical System
**Single Inverted Pendulum**: A long rod balanced on a horizontally movable cart.

**System Parameters** (high-difficulty configuration):
- 🔹 Cart mass: 1.0 kg
- 🔹 Pole mass: 0.35 kg (heavy with large inertia)
- 🔹 Pole length: 2.5 m (extremely long, highly unstable)
- 🔹 Friction coefficients:
  - Cart friction: 0.35 (high energy dissipation)
  - Joint friction: 0.25 (significant damping)
- 🔹 Initial angle:
  - Experiment 1: 0.4 rad (~23°)
  - Experiment 2: 1.02 rad (~58°) - **Extreme challenge**

### Control Objectives
**Multi-Objective Optimization (MOP)**:
1. ⚡ **Fast stabilization**: Minimize stabilization time
2. 🎯 **High-precision balancing**: Pole vertical (θ → 0), cart centered (x → 0)
3. 💚 **Energy-efficient control**: Minimize energy consumption
4. 🚫 **Avoid failure**: |θ| < 1.5 rad, |x| < 10 m

### Scoring System
**Maximum ~7300 points**, including:
- **Base stability**: ~500-800 points (angle, position, velocity errors)
- **Time efficiency bonus**: up to +3000 points (exponential reward, faster = higher)
- **Energy efficiency bonus**: up to +2500 points (lower energy consumption)
- **Success bonus**: up to +800 points (high-precision stability)

---

## 🧬 Experimental Design

### Experiment 1: `results_pendulum_liter_1`
**Incremental evolution around LQR algorithm**

**Strategy**:
- Initial program: Adaptive controller based on LQR (Linear Quadratic Regulator)
- Evolution direction: Parameter tuning, feature engineering, predictive compensation
- Generations: 180
- Initial angle: 0.4 rad (~23°)

**Key innovations** (automatically discovered during evolution):
1. **Adaptive gain scheduling**: Dynamically adjusts control gains based on angle and angular velocity
2. **Mid-swing damping**: Suppresses momentum overshoot in the 0.3-0.7 rad range
3. **Predictive momentum compensation**: Predicts future angular acceleration based on physics model and counteracts in advance
4. **Dual-gated integral control**: Dual gating on angle and velocity to prevent integral windup

**Final performance**:
- Initial score: ~3900 points
- Best score: ~4900 points
- Improvement: **+25%**

---

### Experiment 2: `results_pendulum_liter_2`
**Aggressive innovation breaking traditional frameworks**

**Strategy adjustments**:
- 🔥 **Banned traditional terminology**: Explicitly prohibited terms like "LQR", "Riccati", "Q matrix" in prompts
- 🚀 **Encouraged paradigm shifts**: Required LLMs to rethink from perspectives like physical intuition, biological inspiration, and phase-based control
- ⚡ **Increased difficulty**: Initial angle increased from 0.4 rad to 1.02 rad (~58°)
- 📈 **Larger innovation space**: max_tokens increased from 4096 to 16384

**Key innovations** (autonomously proposed by LLMs):
1. **Three-phase adaptive control** (Phase-Adaptive Control):
   - Emergency Phase (|θ| > 0.8): Aggressive swing-up + energy pumping
   - Recovery Phase (0.3 < |θ| ≤ 0.8): Predictive damping + adaptive gains
   - Balancing Phase (|θ| ≤ 0.3): Precision LQR + enhanced integral control
2. **Energy-efficient damping**: Intelligently identifies motion direction, applies damping only when moving away from vertical
3. **Enhanced position correction**: Adaptive position gain + velocity-dependent damping
4. **Direction-sensitive integral bounds**: Dynamically adjusts integral limits based on displacement direction

**Final performance**:
- Initial score: ~3900 points
- Best score: **~4870 points**
- Improvement: **+24.9%**
- Successfully handles 58° initial angle (near physical limit)

---

## 📊 "Path to Best" Evolution Analysis

### Experiment 2 Key Evolution Nodes

**Generational evolution visualization**:
```
Gen 0 → Gen 16 → Gen 22 → Gen 36 → Gen 47 → Gen 64 → Gen 80 → Gen 90
3931    3970     4636     4213     4233     4697     4467     4871 points
```

#### 🔹 Gen 0: `initial_program`
- **Score**: 3931.61
- **Description**: Initial LQR-based controller
- **Features**: Adaptive cross-coupling, mid-swing damping, predictive compensation

#### 🔹 Gen 16: `physics_enhanced_prediction_and_damping`
- **Score**: 3970.15 (+38.5)
- **Innovations**: 
  1. Direction-aware mid-swing damping
  2. High-fidelity predictive compensation (considering control force output, friction, centripetal terms)

#### 🔹 Gen 22: `fix_integral_sign_and_damp_logic`
- **Score**: 4636.78 (+666.6) 🚀 **Critical breakthrough**
- **Fix**: Integral term sign error (negative feedback → positive feedback)
- **Optimization**: Refined mid-swing damping logic, activates only when returning to vertical

#### 🔹 Gen 36-47: Exploratory adjustments
- **Score**: 4213 → 4233
- **Features**: Short-term performance decline followed by recovery (exploration vs exploitation)

#### 🔹 Gen 64: `fix_integral_sign_and_position_centering`
- **Score**: 4697.90 (+464.1)
- **Optimizations**: 
  1. Re-corrected integral force sign
  2. Enhanced position correction (lowered activation threshold, increased gain)

#### 🔹 Gen 80: `emergency_phase_cart_damping`
- **Score**: 4467.48 (-230.4)
- **Attempt**: Introduced cart damping in emergency phase (short-term regression)

#### 🔹 Gen 90: `enhanced_integral_windup_and_position_control`
- **Score**: 4871.36 (+403.9) 🏆 **Best performance**
- **Innovations**: 
  1. Direction-sensitive integral saturation limits
  2. Adaptive position correction gain

---

## 🛠️ Project Structure

```
.
├── initial.py                   # Initial controller implementation (LQR baseline)
├── evaluate.py                  # Performance evaluation and scoring system
├── run_evo.py                   # Evolution run configuration
├── test_performance.py          # Performance testing script
├── visualization_utils.py       # Visualization tools
├── viz_pendulum.ipynb          # Interactive visualization notebook
│
├── results_pendulum_liter_1/   # Experiment 1 results (180 gen, 23° initial)
│   ├── best/                   # Best program
│   ├── gen_*/                  # Candidate programs per generation
│   ├── evolution_db.sqlite     # Evolution database
│   ├── experiment_config.yaml  # Experiment configuration
│   └── evolution_run.log       # Run log
│
└── results_pendulum_liter_2/   # Experiment 2 results (102 gen, 58° initial)
    ├── best/                   # Best program (4871 points)
    ├── gen_*/                  # Candidate programs per generation
    ├── evolution_db.sqlite     # Evolution database
    └── experiment_config.yaml  # Experiment config (aggressive prompts)
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install ShinkaEvolve
pip install shinka-evolve

# Install dependencies
pip install numpy scipy matplotlib
```

### 2. Test Initial Controller
```bash
python test_performance.py
```

### 3. Run Evolution Experiment
```bash
# Start evolution (requires LLM API keys configuration)
python run_evo.py
```

### 4. Visualize Best Results
```bash
# Open Jupyter Notebook
jupyter notebook viz_pendulum.ipynb
```

---

## 📝 Code Example

### Best Controller Core Logic (Experiment 2)
```python
def get_action(self, state):
    x, theta, dx, dtheta = state
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    # Three-phase determination
    if abs(theta) > 0.8:
        current_phase = "emergency"     # Emergency swing-up
    elif abs(theta) > 0.3:
        current_phase = "recovery"      # Recovery phase
    else:
        current_phase = "balancing"     # Precision balancing
    
    # Base LQR force
    base_force = -self.K @ state_vec
    
    # Phase-adaptive gain scheduling
    if current_phase == "emergency":
        # High gain + energy pumping
        pos_gain = 1.0 + 0.8 * tanh(6.0 * (abs(theta) - 0.8))
        u_swing = 12.0 * swing_activation * sign(theta)
        
    elif current_phase == "recovery":
        # Predictive damping + moderate gain
        theta_acc_est = (g * sin(theta) - ...) / denominator
        predictive_correction = -0.08 * pred_divergence
        
    else:  # balancing
        # Enhanced integral control + position correction
        integral_force = K_i_adaptive * integral_gate * integral_x
        position_correction = -1.6 * x * stability_factor
    
    return clip(force, -100, 100)
```

---

## 🎓 Technical Details

### Evolution Configuration
```yaml
# Experiment 2 Core Parameters
llm_models:
  - deepseek-reasoner
  - claude-opus-4-5
  - gemini-2.5-pro
  - gemini-3-pro-preview
  - xai/grok-4-1-fast-reasoning
  - glm-4.5
  - glm-4.6
  - dashscope/qwen3-coder-plus
  - dashscope/qwen-plus-2025-07-28

llm_kwargs:
  max_tokens: 16384
  temperatures: [0.0, 0.5, 1.0]
  reasoning_efforts: [auto, high]

database_config:
  num_islands: 2
  archive_size: 40
  migration_interval: 10
  parent_selection_strategy: weighted
  parent_selection_lambda: 10.0
```

### Scoring Function Core
```python
# Time efficiency bonus (exponential function)
time_bonus = 3000.0 * exp(-8.0 * (stabilization_time / total_steps)^2)

# Energy efficiency bonus
energy_bonus = 2500.0 * exp(-25.0 * avg_energy_per_step^1.8)

# Success bonus
if θ_final < 0.03 and x_final < 0.8:
    success_bonus = 800.0
```

---

## 🔬 Experimental Results Comparison

| Metric | Initial Controller | Exp 1 Best | Exp 2 Best |
|--------|-------------------|------------|------------|
| **Total Score** | 3931 | 4900 | 4871 |
| **Stabilization Time** | ~11s | 5s | 4s |
| **Final Precision** | OK | Good | **Excellent** |
| **Initial Angle** | 23° | 23° | **58°** ⚡ |

> **Note**: Although Experiment 2's score (4871) is slightly lower than Experiment 1 (4900), it tackles a **much harder problem** with a 58° initial angle (vs 23°). The scoring system penalizes larger initial angles, making high scores exponentially harder to achieve. In terms of **control effectiveness relative to task difficulty**, Experiment 2 demonstrates superior performance.

---

## 🤝 Contributions & Acknowledgments

- **ShinkaEvolve**: [SakanaAI/shinka](https://github.com/SakanaAI/shinka)
- **LLM Models**: DeepSeek, Anthropic, Google, xAI, Zhipu AI, Alibaba Cloud

---

## 📄 License

This project follows the open-source license of ShinkaEvolve.

---

## 📮 Contact

For questions or suggestions, feel free to submit an Issue or Pull Request.

---

<p align="center">
<b>🌟 Let LLM Agents automatically optimize your control algorithms! 🌟</b>
</p>
