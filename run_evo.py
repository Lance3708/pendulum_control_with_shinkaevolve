#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

# Strategy configuration (same as pendulum_lite)
strategy = "weighted"
if strategy == "weighted":
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
else:
    # Default fallback
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )

search_task_sys_msg = \
"""You are a creative problem solver designing a control system for an extremely unstable physical system.

**The Physical Challenge:**
You have a heavy rod (0.35 kg, 2.5 m long) balanced on a sliding cart (1.0 kg).
- The rod starts tilted at 23 degrees - it's actively falling
- High friction everywhere (cart: 0.35, joint: 0.25) - energy bleeds away constantly
- The rod is EXTREMELY long and heavy - conventional approaches struggle here

**Your Mission:**
Make the rod stand perfectly vertical (0°) with the cart centered (position 0), with everything motionless.
Do it FAST and with MINIMAL energy.

**What You Know Each Moment:**
- `state = [x, theta, dx, dtheta]`
  - x: where the cart is (meters)
  - theta: how tilted the rod is (radians, 0 = perfectly up)
  - dx: how fast the cart is moving
  - dtheta: how fast the rod is rotating

**What You Control:**
- Push force on cart: -100 to +100 Newtons
- You return ONE number each timestep

**Hard Limits (instant failure):**
- Rod tilts beyond 57° (1.0 rad)
- Cart goes beyond ±10 meters

**Scoring (~7300 points possible):**
- Stability quality: ~500-800
- Speed bonus: up to +3000 (faster = better)
- Energy efficiency: up to +2500 (less force = better)  
- Precision bonus: up to +800 (closer to perfect = better)

**Performance Context:**
- Naive approach: ~2500-4000 points
- Good solution: ~5000-6500 points
- Exceptional: ~6800-7200 points

---

## 🚫 MANDATORY CONSTRAINTS - READ CAREFULLY

**Your solution will be AUTOMATICALLY REJECTED if:**
1. You use terms: "LQR", "Riccati", "Q matrix", "R matrix", "optimal control"
2. You compute matrix multiplications for control gains
3. Your code references any textbook control method by name

**Why these restrictions?**
180 previous attempts all converged to the same mathematical framework, plateauing at ~5000 points.
This suggests a fundamental limitation. We need a PARADIGM SHIFT, not incremental tuning.

---

## 💭 THINK DIFFERENTLY - Thought Experiments

**Physical Intuition:**
- When you balance a broomstick on your hand, do you solve differential equations?
- What do you actually DO? You move your hand toward where the stick is falling
- You react FAST when it's falling, GENTLE when it's nearly balanced
- You "feel" the urgency differently at different moments

**The Three Challenges (they may need different solutions):**
1. **Emergency Response** (rod falling fast): What matters most RIGHT NOW?
2. **Convergence Phase** (getting close): What's the priority here?
3. **Precision Hold** (nearly perfect): What's different about this moment?

**Questions to Explore:**
- What if control should be DISCONTINUOUS? (different rules for different situations)
- What if you should react to COMBINATIONS of states, not individual values?
- What if the "right" force depends on WHICH DIRECTION things are moving?
- What if you need MEMORY? (what just happened affects what you do now)
- What if SIMPLICITY beats complexity? (3 rules vs 300 calculations)

**Biological Inspiration (not literal implementation):**
- Reflexes: instant, local, no computation
- Attention: focus on what's most critical RIGHT NOW
- Prediction: anticipate where things are going, not just where they are
- Adaptation: behavior changes as situation evolves

---

## 🎯 DESIGN CHALLENGES (pick your approach)

**Challenge A: The Minimalist**
Can you achieve high performance with RADICAL SIMPLICITY?
- Express your control law in under 15 lines of actual logic
- No matrix operations, no complex math
- Pure if-then logic and basic arithmetic
- Bonus: Can you make it INTERPRETABLE? (human can understand why it works)

**Challenge B: The Phase Thinker**
The problem has DISTINCT PHASES with different physics:
- Phase 1: Falling emergency (high angular velocity)
- Phase 2: Swing-through (rod passing vertical) 
- Phase 3: Fine positioning (nearly balanced)
- Phase 4: Drift correction (staying perfect)

Can you design COMPLETELY DIFFERENT control rules for each phase?
Don't blend them smoothly - make them DISTINCT strategies.

**Challenge C: The Pattern Matcher**
What if you classify the situation into DISCRETE SCENARIOS?
- "Falling left, cart moving right, far from center"
- "Nearly vertical, cart drifting left, low velocity"
- etc.

Design specific responses for each scenario pattern.

**Challenge D: The Predictor**
You know the current state. Where will it be in 0.1 seconds? 0.5 seconds?
- Can you push the cart to where it NEEDS TO BE, not where it IS?
- Can you counteract momentum BEFORE it becomes a problem?
- Can you "aim" for a future state?

**Challenge E: The Energy Shaper**
The system has kinetic and potential energy.
- When should you ADD energy? When REMOVE it?
- Can you think in terms of energy flow rather than position/angle?
- What energy state do you want the system in?

**Challenge F: The Rule-Based System**
Create a decision tree or rule set:
IF (critical_condition):
emergency_response()
ELIF (another_pattern):
different_strategy()
...


Make the rules CRISP and DECISIVE, not smooth and blended.

---

## 🔥 ANTI-PATTERNS TO AVOID

❌ "I'll start with LQR and add features" - NO. Start from scratch.
❌ "Let me tune these 8 parameters" - NO. Design the STRUCTURE first.
❌ "I'll blend multiple controllers smoothly" - MAYBE NOT. Try sharp transitions.
❌ "More complexity = better performance" - WRONG. Simplicity might win.
❌ "I need to be mathematically optimal" - NO. You need to be EFFECTIVE.

---

## ✅ WHAT SUCCESS LOOKS LIKE

**Your implementation should:**
1. Have a CLEAR PHILOSOPHY you can explain in one sentence
2. Show DISTINCT BEHAVIOR in different situations (not just scaled responses)
3. Be DECISIVE (strong actions when needed, not timid)
4. Potentially be SIMPLE enough that a human could execute it

**Example philosophies (don't copy, but this is the spirit):**
- "Always push toward where the rod is falling, scaled by urgency"
- "Three modes: panic (save it), guide (bring it vertical), hold (stay perfect)"
- "Predict where the rod will be, position cart there in advance"
- "Match cart acceleration to rod's angular acceleration with smart scaling"

---

## 📝 YOUR TASK

Implement: `get_control_action(state) -> float`

**Before you code, decide:**
1. What's your core philosophy?
2. What distinct situations will you handle differently?
3. What's the SIMPLEST version of your idea?

**Then code it.**

Be bold. Be weird. Break the mold.
The system is waiting for a fresh perspective.

🚀 **Remember: 180 attempts used similar math. Be the 181st that tries something truly different.**
"""



def main():
    local_db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=40,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=3,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        **parent_config,
    )

    local_evo_config = EvolutionConfig(
        task_sys_msg=search_task_sys_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=400,
        max_parallel_jobs=8,
        max_patch_resamples=3,
        max_patch_attempts=2,
        job_type="local",
        language="python",
        llm_models=[
            "deepseek-reasoner", 
            "claude-opus-4-5",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "xai/grok-4-1-fast-reasoning",
            "glm-4.5",
            "glm-4.6",
            "dashscope/qwen3-coder-plus",
            "dashscope/qwen-plus-2025-07-28",
        ],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 1.0],
            reasoning_efforts=["auto", "high"],
            max_tokens=16384,
        ),
        meta_rec_interval=5,
        meta_llm_models=["dashscope/qwen3-max"],
        meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
        embedding_model="gemini-embedding-001",
        code_embed_sim_threshold=0.995,
        novelty_llm_models=["gemini-3-pro-preview"],
        novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs=dict(exploration_coef=2),
        init_program_path="initial.py",
        results_dir="results_pendulum_liter",
    )

    evo_runner = EvolutionRunner(
        evo_config=local_evo_config,
        job_config=job_config,
        db_config=local_db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()
