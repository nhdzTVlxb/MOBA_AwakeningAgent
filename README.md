# MOBA_AwakeningAgent

# Gorge Chase PPO - DMSD

## Lightweight Design

Due to device constraints, the model adopts a lightweight design:
- **Total observation dimension**: 1053
- **Model size**: ~0.6 MB
- **Design philosophy**: Small CNN encoder, low-dimensional MLPs, no RNN (temporal memory delegated to preprocessor)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Episode Loop                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Environment → Raw Observation
         │
         ▼
Step 2: Preprocessor
         │
         ├── 2.1 State extraction (hero position, flash cooldown, buff remaining)
         ├── 2.2 Movement state update (hit wall, stagnation, oscillation, backtrack)
         ├── 2.3 Legal action mask construction
         ├── 2.4 Monster feature extraction (distance, direction, threat score for 2 monsters)
         ├── 2.5 Dual-domain memory sync (treasure/buff position + refresh estimation)
         ├── 2.6 Situational signal computation (danger, pinch, pressure, greed window)
         ├── 2.7 Semantic map construction (8×11×11)
         ├── 2.8 Feature concatenation (652 dims)
         └── 2.9 Reward computation (50+ shaping terms)
         │
         ▼
Step 3: Model Forward
         │
         ├── Split features by semantic groups
         ├── Encode each group independently (MLP/CNN)
         ├── Fuse → Actor logits + Critic value
         │
         ▼
Step 4: Action Sampling
         │
         ├── Apply legal action mask
         ├── Sample action from policy distribution
         └── Return action to environment
         │
         ▼
Step 5: Update Episode Buffer
         │
         └── Store (obs, action, reward, value, log_prob, legal_mask, done)
         │
         ▼
Step 6: Terminal Check
         │
         ├── Not done → goto Step 1
         └── Done → PPO training
```

---

## Observation Features (652 dims)

```
Observation = [
    hero (15)                    # Hero state
    + monster1 (10)              # Monster 1 features
    + monster2 (10)              # Monster 2 features
    + treasure (10)              # Current target treasure guidance
    + semantic_map (968)         # 8×11×11 = 968
    + legal_action (16)          # 16-action legality mask
    + progress (24)              # Progress and situational signals
]
```

---

## Semantic Map (8 channels × 11×11)

Local perception window centered on hero, size **11×11** (radius 5).

| Channel | Content | Description |
|:-------:|---------|-------------|
| 0 | Walkable | Passable area (0/1) |
| 1 | Visit Heat | min(visit_count/5, 1.0) |
| 2 | Treasure | 1.0=confirmed, 0.5=direction estimate |
| 3 | Buff | 1.0=confirmed, 0.5=direction estimate |
| 4 | Monster Risk | Monster threat intensity with decay spread |
| 5 | Openness | Local open area ratio |
| 6 | Corridor | Corridor strength (avg of two deepest directions) |
| 7 | Dead-end Risk | Dead end risk score |

### Topology Computation Details

```
Openness = walkable_cells / total_cells within radius 2

Corridor = (depth1 + depth2) / (2 × LOCAL_HALF)

Dead-end Risk = 0.45×(1-Openness) + 0.35×branch_risk + 0.2×escape_risk
    branch_risk = max(0, (4 - branch_count) / 3)
    escape_risk = 1 - min(escape_depth / LOCAL_HALF, 1.0)
```

---

## Situational Signals

### Danger Level
```
danger = 0.75×distance_pressure + 0.25×pinch_risk + 0.15×speedup_reached
```

### Pinch Risk (double monster pincer)
```
pinch = angle_risk × proximity_risk
```

### Survival Pressure
```
pressure = 0.60×danger + 0.20×pinch + 0.15×speedup + 0.05×second_monster_pressure
```

### Greed Window (safe opportunity for treasure)
```
greed = treasure_opportunity + phase_bonus - 0.75×pressure - speed_penalty
```

---

## Model Architecture

```
                                ┌─────────────────────────────────────┐
                                │         Observation (652 dims)       │
                                └─────────────────────────────────────┘
                                                  │
                                    torch.split(dim=1)
                                                  │
        ┌──────────┬──────────┬──────────┬───────┴───────┬──────────┬──────────┐
        │          │          │          │               │          │          │
        ▼          ▼          ▼          ▼               ▼          ▼          ▼
    hero(15)  monster1(10) monster2(10) treasure(10)  map(968)  legal(16)  progress(24)
        │          │          │          │               │          │          │
        │          └────┬─────┘          │               │          └────┬─────┘
        │               ▼                │               │               ▼
        │      concat(20)                │               │        concat(40)
        │               │                │               │               │
        ▼               ▼                ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ HeroEncoder   │ │MonsterEncoder │ │TreasureEncoder│ │  MapEncoder   │ │ControlEncoder │
│               │ │               │ │               │ │               │ │               │
│ Linear(15,32) │ │Linear(20,64)  │ │Linear(10,16)  │ │ Conv2d(8,16)  │ │Linear(40,32)  │
│     ReLU      │ │    ReLU       │ │    ReLU       │ │     ReLU      │ │    ReLU       │
└───────────────┘ └───────────────┘ └───────────────┘ │ Conv2d(16,32) │ └───────────────┘
        │                 │                 │        │     ReLU      │          │
        │                 │                 │        │ Conv2d(32,32) │          │
        │                 │                 │        │     ReLU      │          │
        │                 │                 │        │ AdaptiveAvgPool2d(3,3) │
        │                 │                 │        │    Flatten    │          │
        │                 │                 │        │Linear(288,128)│          │
        │                 │                 │        │     ReLU      │          │
        │                 │                 │        └───────────────┘          │
        │                 │                 │               │                   │
        └─────────────────┴─────────────────┴───────────────┴───────────────────┘
                                              │
                              concat(32 + 64 + 16 + 128 + 32 = 272)
                                              │
                                              ▼
                              ┌─────────────────────────────┐
                              │         Backbone            │
                              │                             │
                              │    Linear(272, 128)         │
                              │         ReLU                │
                              │    Linear(128, 128)         │
                              │         ReLU                │
                              └─────────────────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                              ▼                               ▼
              ┌───────────────────────────┐   ┌───────────────────────────┐
              │       Actor Head           │   │       Critic Head         │
              │                           │   │                           │
              │    Linear(128, 16)         │   │    Linear(128, 1)         │
              └───────────────────────────┘   └───────────────────────────┘
                              │                               │
                              ▼                               ▼
                         logits(16)                        value(1)
                              │
                              ▼
              ┌───────────────────────────┐
              │    Apply Legal Mask        │
              │    Categorical Sample      │
              └───────────────────────────┘
                              │
                              ▼
                           action
```

### MapEncoder Details

```
map_feat (batch × 8 × 11 × 11)
         │
         ▼
    Conv2d(8 → 16, kernel=3, padding=1)
         │
         ▼ ReLU
    Conv2d(16 → 32, kernel=3, stride=2, padding=1)
         │  (11×11 → 6×6)
         ▼ ReLU
    Conv2d(32 → 32, kernel=3, stride=2, padding=1)
         │  (6×6 → 3×3)
         ▼ ReLU
    AdaptiveAvgPool2d(3×3)
         │
         ▼ Flatten (32 × 3 × 3 = 288)
         │
    Linear(288 → 128)
         │
         ▼ ReLU
    map_encoded (128)
```

---

## Dual-Domain Memory System

### 1 Treasure Memory
- **Position memory**: Position remembered once seen
- **Availability tracking**: Based on remaining treasures from environment
- **First-seen flag**: Controls first discovery reward

### 2 Buff Memory
- **Position memory**: Position remembered once seen
- **Refresh estimation**: Buff cooldown ~200 steps
  ```
  estimated_ready_step = last_unavailable_step + BUFF_REFRESH_STEPS (200)
  ```
- **Availability logic**:
  - `status == 0` → unavailable, start refresh estimation
  - `step >= estimated_ready_step` → available again + `just_refreshed=True`
- **Wait strategy**: Only allow waiting ≤ 18 steps, and monster must not catch up first

### 3 Visit Heat
- Global 128×128 grid tracking visit counts
- Used for `revisit_intensity` and semantic map heat channel

### 4 10-step Position History
- Records last `STALL_WINDOW+1` positions (default 11)
- Used for `stall_window_penalty`: penalty if movement < 5 over 10 steps

### 5 Post-Flash Window Tracking
- Records after flash: `post_flash_origin_pos`, `post_flash_origin_danger`, `post_flash_origin_min_dist`, `post_flash_origin_openness`
- 8-step window to detect behavior quality, with additional rewards/penalties

---

## Curriculum Learning Stages

| Stage | Episodes | Monster Interval | Speedup Step | Difficulty |
|-------|----------|------------------|--------------|------------|
| warmup_stable | 0-499 | step 800 | 1000 | Easy: slow |
| mid_pressure | 500-1299 | step 500 | 700 | Medium: double monster pressure |
| late_speedup_survival | 1300-1499 | step 300 | 500 | Hard:  early speedup |
| hard_generalization | 1500+ | step 300 | 500 | Extreme: sustained high pressure |

### Stage Transition Logic
```
current_stage = None
for stage in CURRICULUM_STAGES:
    if current_train_episode <= stage["max_train_episode"]:
        current_stage = stage
        break

env_conf = {
    "monster_interval": stage["monster_interval"],
    "monster_speedup": stage["monster_speedup"],
    "max_step": stage["max_step"],
}
```

---

## PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| GAMMA | 0.995 | Discount factor |
| LAMDA | 0.95 | GAE λ |
| INIT_LEARNING_RATE | 0.0001 | Initial learning rate |
| BETA_START | 0.003 | Initial entropy coefficient |
| BETA_END | 0.0005 | Final entropy coefficient |
| BETA_DECAY_STEPS | 4000 | Entropy decay steps |
| CLIP_PARAM | 0.15 | PPO clip range |
| VF_COEF | 1.0 | Value loss coefficient |
| GRAD_CLIP_RANGE | 0.5 | Gradient clipping |
| USE_ADVANTAGE_NORM | True | Advantage normalization |
| TARGET_KL | 0.015 | KL early stopping threshold |
| TRAIN_BATCH_EPISODES | 4 | Episodes per training batch |

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PPO Training Loop                                 │
└─────────────────────────────────────────────────────────────────────────────┘

For each batch of 4 episodes:
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Collect Trajectories                                                     │
│    for step in episode:                                                     │
│        obs → model → action → env.step → reward → next_obs                  │
│        store (obs, action, reward, value, log_prob, legal_mask, done)      │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. GAE Advantage Estimation                                                 │
│    returns = []                                                             │
│    advantages = []                                                          │
│    for t from T-1 to 0:                                                     │
│        delta = reward[t] + γ × value[t+1] × (1-done) - value[t]             │
│        advantage[t] = delta + γ × λ × advantage[t+1]                        │
│        return[t] = advantage[t] + value[t]                                  │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. PPO Multiple Epochs Update                                               │
│    for epoch in range(PPO_EPOCHS):                                          │
│        logits, values = model(obs)                                          │
│        logits_masked = logits + (1-legal_mask)×(-1e9)                       │
│        dist = Categorical(logits=logits_masked)                             │
│        new_log_prob = dist.log_prob(action)                                 │
│        entropy = dist.entropy().mean()                                      │
│                                                                              │
│        ratio = exp(new_log_prob - old_log_prob)                             │
│        surr1 = ratio × advantages                                           │
│        surr2 = clip(ratio, 1-ε, 1+ε) × advantages                          │
│        policy_loss = -min(surr1, surr2).mean()                              │
│                                                                              │
│        value_loss = MSE(values.squeeze(), returns)                          │
│                                                                              │
│        if KL > TARGET_KL: break  # early stopping                           │
│                                                                              │
│        loss = policy_loss + 0.5×value_loss - β×entropy                      │
│        loss.backward()                                                      │
│        clip_grad_norm_(params, GRAD_CLIP_RANGE)                             │
│        optimizer.step()                                                     │
│        lr_scheduler.step()                                                  │
│        β = linear_decay(BETA_START, BETA_END, step, BETA_DECAY_STEPS)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions Summary

1. **Structured Observation**: Semantic grouping enables the network to learn different modality abstractions more easily

2. **Dual-Domain Memory**: Combines local immediate perception with semi-global memory to mitigate partial observability

3. **Situation-Aware Rewards**: Same action (e.g., backtracking) receives different signals under high vs low pressure

4. **Fine-Grained Flash Shaping**: Distinguishes good flashes (wall-through escape, resource pickup) from bad ones (waste, wall-hit, suicide)

5. **Intelligent Buff Management**: 200-step refresh, allows short waits but never full-cycle idling

6. **Anti-Grinding Mechanisms**: Stagnation, oscillation, revisit, 10-step window, low-pressure loop detection

7. **Curriculum Learning**: Progresses from single monster slow to double monster high pressure
