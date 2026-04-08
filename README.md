# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法

Here's the English version of the Gorge Chase PPO 2026 Competition Update documentation:

---

# Gorge Chase PPO - 2026 Competition Version

## I. Modified Files and Content

| File | Modifications |
|------|---------------|
| `agent_ppo/conf/conf.py` | 1. Action space: 8 → 16<br>2. Entropy coefficient: 0.001 → 0.01<br>3. Feature dimension: 40 → 1863<br>4. New environment params: MAX_STEPS=1000, TREASURE_NUM=10, MONSTER2_SPAWN_STEP=300, MONSTER_SPEED_UP_STEP=500<br>5. New reward params: REW_STEP=1.5, REW_TREASURE=100, REW_SURVIVE=0.05, REW_MONSTER_DISTANCE=0.3, PENALTY_HIT_WALL=0.1 |
| `agent_ppo/feature/preprocessor.py` | 1. New OrganManager class: manages treasure chest and buff position estimation (tracking objects outside FOV)<br>2. New MapManager class: manages 21×21 local map (obstacles, memory, treasure, buff)<br>3. Feature extraction: expanded from 40 to 1863 dimensions (hero 5 + monster 10 + map 1764 + treasure 60 + buff 6 + action mask 16 + progress 2)<br>4. Action mask: expanded to 16 dimensions, supports blink cooldown and wall collision detection<br>5. Reward function: completely rewritten |
| `agent_ppo/feature/definition.py` | 1. legal_action dimension: 8 → 16<br>2. prob dimension: 8 → 16 |
| `agent_ppo/model/model.py` | 1. Network structure: single MLP → CNN + MLP<br>2. New CNN map encoder: processes 4×21×21 map → 512 dimensions<br>3. New MLP other feature encoder: processes 99 dimensions → 128 dimensions |
| Other files | algorithm.py, train_workflow.py, agent.py: unchanged |

## II. Reward and Penalty Mechanism

| Type | Mechanism | Value |
|------|-----------|-------|
| Positive | Step reward | +1.5 per step |
| Positive | Survival reward | +0.05 per step |
| Positive | Treasure pickup | +100 each |
| Positive | Moving away from monster | +0.3 × (distance/50) |
| Positive | Blink usage | +0.2 per use |
| Positive | Buff pickup | +1.0 each |
| Positive | Approaching treasure | Higher when closer |
| Negative | Too close to monster | -0.3 × (5-distance) |
| Negative | Wall collision | -0.1 per hit |

## III. How to Change Map Size from 21×21 to 41×41

| Step | File | Line Location | Original Value | Change To |
|------|------|---------------|----------------|-----------|
| 1 | `conf.py` | Line 24 | `MAP_SIZE = 21` | `MAP_SIZE = 41` |
| 2 | `conf.py` | Line 25 | `4 * 21 * 21` | `4 * 41 * 41` |
| 3 | `preprocessor.py` | Lines 103-104 | `range(-10, 11)` | `range(-20, 21)` |
| 4 | `preprocessor.py` | Line 155 | `size = 21` | `size = 41` |
| 5 | `model.py` | Line 56 | `64 * 6 * 6` | `64 * 11 * 11` |

---

## Additional Notes

**CNN Output Size Calculation for Step 5:**

| Map Size | Conv Layers Output | Flattened Size |
|----------|-------------------|----------------|
| 21×21 → 11×11 → 6×6 | 64 × 6 × 6 = 2304 | `64 * 6 * 6` |
| 41×41 → 21×21 → 11×11 | 64 × 11 × 11 = 7744 | `64 * 11 * 11` |

**Conv layer calculations:**
- Conv1 (kernel=7, stride=2): ⌊(41 - 7)/2⌋ + 1 = 18 → 18? Actually 41→21 (padding=3)
- Conv2 (kernel=5, stride=2): 21→11
- Conv3 (kernel=3, stride=1): 11→11 (with padding=1)
| 5 | `model.py` | 第56行 | `64 * 6 * 6` | `64 * 11 * 11` |
