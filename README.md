# MOBA_AwakeningAgent

# Gorge Chase PPO - DMSD (Dual-domain Memory and Situational Decision)

## Core Idea

A reinforcement learning agent for treasure collection with monster evasion, featuring **structured observation encoding** and **behavioral reward shaping**.

---

## Key Innovations

### 1. Dual-Domain Memory System
| Domain | Content |
|--------|---------|
| **Local Immediate** | Walkability, visit heat, treasure/buff positions, monster risk, openness, corridor, dead-end |
| **Semi-global Memory** | Visit heatmap, treasure/buff positions with refresh estimation, position history |

### 2. Situational Awareness Signals
- `danger_level` - distance + pinch risk + speedup state
- `pinch_risk` - angle between two approaching monsters
- `survival_pressure` - combined pressure from threats
- `greed_window` - safe opportunity to hunt treasure

### 3. Reward Shaping Components

#### Survival & Movement
| Reward | Purpose |
|--------|---------|
| `SURVIVE_REWARD` | Small per-step survival bonus |
| `DIST_SHAPING` | Moving away from monsters |
| `STALL_PENALTY` | 10-step ≥5 distance check (anti-grinding) |

#### Treasure Collection
| Reward | Purpose |
|--------|---------|
| `TREASURE_REWARD = 2.5` | Per treasure collected |
| `FIRST_SEEN_TREASURE_REWARD` | First discovery bonus |
| `TREASURE_DIST_COEF` | Moving toward nearest treasure |
| `TREASURE_MISS_PENALTY` | Abandoning close treasure |

#### Flash (Teleport) Usage
| Reward | Purpose |
|--------|---------|
| `FLASH_ESCAPE_REWARD` | Escaping danger via flash |
| `FLASH_THROUGH_WALL_REWARD` | Legitimate wall-through flash |
| `FLASH_WASTE_PENALTY` | Useless flash with no gain |
| `FLASH_HIT_WALL_PENALTY` | Attempting flash into wall |
| `FLASH_SUICIDE_PENALTY` | Flash making situation worse |

#### Buff System (200-step refresh)
| Reward | Purpose |
|--------|---------|
| `BUFF_REWARD = 2.0` | Per buff collected |
| `BUFF_APPROACH_REWARD` | Moving toward available buff |
| `BUFF_FLASH_PICKUP_BONUS` | Using flash to grab buff |
| `BUFF_WAIT_PENALTY` | Waiting too long for refresh |
| `BUFF_REFRESH_PICKUP_BONUS` | Grabbing freshly spawned buff |

#### Behavior Constraints
| Penalty | Trigger |
|---------|---------|
| `HIT_WALL_PENALTY` | Attempting move into wall |
| `STAGNATION_PENALTY` | Repeated small/no movement |
| `OSCILLATION_PENALTY` | Back-and-forth movement |
| `REVISIT_PENALTY` | Revisiting same area |

### 4. Structured Observation (1053 dims)

```
Observation = [
    hero (15)                    # position, flash/buff status, progress
    + monster1 (10)              # position, distance, direction, threat
    + monster2 (10)              # same as monster1
    + treasure (10)              # target treasure guidance
    + semantic_map (968)         # 8×11×11: walkable, heat, treasure, buff, risk, topology
    + legal_action (16)          # valid action mask
    + progress (24)              # survival pressure, greed window, etc.
]
```

### 5. Semantic Map (8 channels, 11×11)
| Channel | Content |
|---------|---------|
| 0 | Walkable area |
| 1 | Visit heat |
| 2 | Treasure positions (positive) |
| 3 | Buff positions (positive) |
| 4 | Monster risk |
| 5 | Openness |
| 6 | Corridor |
| 7 | Dead-end risk |

### 6. Curriculum Training Stages

| Stage | Episodes | Monster Interval | Speedup Step | Difficulty |
|-------|----------|------------------|--------------|------------|
| warmup_stable | 0-499 | 800 | 1000 | Easy |
| mid_pressure | 500-1299 | 500 | 700 | Medium |
| late_speedup_survival | 1300-1499 | 300 | 500 | Hard |
| hard_generalization | 1500+ | 300 | 500 | Extreme |

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.995 | Discount factor |
| `LAMDA` | 0.95 | GAE lambda |
| `LR` | 0.0001 | Learning rate |
| `CLIP_PARAM` | 0.15 | PPO clip range |
| `ACTION_NUM` | 16 | 8 move + 8 flash |
| `LOCAL_MAP_SIZE` | 17 | Local perception range |

---

## Architecture

```
Input (1053)
    ↓
[Split by semantic]
    ↓
Hero(15) → MLP(32)
Monster(20) → MLP(64)  
Treasure(10) → MLP(16)
Map(2312) → CNN(128)
Control(40) → MLP(32)
    ↓
Concat (272) → MLP(128) → MLP(128)
    ↓
┌─────────────┴─────────────┐
↓                           ↓
Actor (16 actions)        Critic (1 value)
```

---

## Design Philosophy

1. **Don't punish what you want to learn** - Positive shaping for desired behaviors
2. **Situational context matters** - Same action yields different rewards based on danger/pressure
3. **Memory reduces partial observability** - Remember treasure/buff locations after they leave view
4. **Conservative flash usage** - Flash only when it provides meaningful escape/resource gain
5. **Anti-grinding mechanisms** - Stagnation/oscillation/revisit penalties prevent exploitative loops


