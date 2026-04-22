# MOBA_AwakeningAgent

## Changelog

### 
### 2026-04-22 PPO4.2 Update（running）

## I. Observation Dimension Changes

| Module     | Dimension |
|------------|-----------|
| hero       | 15        |
| monster1   | 10        |
| monster2   | 10        |
| treasure   | 10        |
| map        | 847       |
| legal      | 16        |
| progress   | 24        |
| **Total**  | 932       |

### New Hero Features (3-dim)
- `history_dx_norm`: x-offset from 10 steps ago
- `history_dz_norm`: z-offset from 10 steps ago
- `history_dist_norm`: distance from 10 steps ago

---

## II. Core Parameter Changes

### Anti-Stalling (10-step lookback)
- `STALL_WINDOW = 10`
- `STALL_DISTANCE_THRESHOLD = 5.0`
- `STALL_PENALTY = 0.10`
- `HISTORY_POSITION_NORM = 32.0`

### First-Time Treasure Sight Reward
- `FIRST_SEEN_TREASURE_REWARD = 0.20`

### Learning Rate
- `INIT_LEARNING_RATE_START = 0.0001` (1e-4)

---

## III. Flash Strategy Changes

### Weakened Normal Flash Rewards (limit abuse)
- `FLASH_ESCAPE_REWARD_COEF = 0.005`
- `FLASH_DIRECTION_REWARD_COEF = 0.004`

### Post-Flash Exploration Rewards (replace direct flash gains)
- `POST_FLASH_EXPLORE_BONUS = 0.01`: rewarded when flashing into unexplored area
- `POST_FLASH_FRONTIER_BONUS = 0.01`: rewarded when flashing closer to frontier

> 🎯 Purpose: Shift flash reward from "using flash" to "actually exploring new areas after flash", discouraging meaningless flashes.

### Flash Danger Threshold
- `FLASH_DANGER_DISTANCE = 10.0`

### Non-Wall Flash Penalty
- `NON_WALL_FLASH_BASE_PENALTY = 0.12`
- `POST_SPEEDUP_NON_WALL_FLASH_MULTIPLIER = 1.4`

### Flash Hold Bonus (save flash)
- `FLASH_HOLD_BONUS = 0.04`
- `FLASH_HOLD_SAFE_DANGER_THRESHOLD = 0.40`
- `FLASH_HOLD_POST_SPEEDUP_SCALE = 0.5`

### Wall Hit / Wall Pass
- `FLASH_HIT_WALL_PENALTY = 0.35`
- `FLASH_THROUGH_WALL_BONUS_MULTIPLIER = 2.20`
- `FLASH_THROUGH_WALL_REWARD_COEF = 0.01`

### Safe Flash / Suicide Flash Penalty
- `SAFE_FLASH_PENALTY = 0.35`
- `SAFE_FLASH_DANGER_THRESHOLD = 0.18`
- `FLASH_SUICIDE_DISTANCE_MARGIN = 4.0`
- `FLASH_SUICIDE_PENALTY = 0.60`

### Early / Blind / Toward-Monster Flash Penalty
- `EARLY_FLASH_STEP_LIMIT = 20`
- `EARLY_FLASH_PENALTY = 0.35`
- `FLASH_BLIND_PENALTY = 0.40`
- `OPEN_AREA_FLASH_TOWARD_MONSTER_OPENNESS = 0.60`
- `FLASH_TOWARD_MONSTER_PENALTY = 0.35`

### Trapped Dead-End Flash Bonus
- `TRAPPED_DEAD_END_THRESHOLD = 0.65`
- `TRAPPED_FLASH_ESCAPE_BONUS = 0.45`
- `TRAPPED_FLASH_MONSTER_CROSS_BONUS = 0.55`
- `TRAPPED_WAIT_FLASH_PENALTY = 0.18`

### High-Pressure Backtrack Exemption / Bonus
- `HIGH_PRESSURE_BACKTRACK_THRESHOLD = 0.70`
- `HIGH_PRESSURE_BACKTRACK_PENALTY_SCALE = 0.25`
- `HIGH_PRESSURE_BACKTRACK_BONUS = 0.12`

---

## IV. Buff Strategy Enhancements

- `BUFF_REWARD = 1.30`
- `BUFF_APPROACH_REWARD = 0.10`
- `BUFF_HIGH_PRESSURE_PICKUP_BONUS = 0.50`
- `BUFF_ESCAPE_COMBO_REWARD = 0.30`
- `BUFF_FLASH_CD_PICKUP_BONUS = 0.25`
- `BUFF_HIGH_PRESSURE_THRESHOLD = 0.45`
- `BUFF_POST_SPEEDUP_PRIORITY_MULTIPLIER = 1.4`

---

## V. New Preprocessor Features

- Position history buffer (supports 10-step lookback)
- First-time treasure sight memory (prevents repeated rewards)
- Treasure / Buff memory cache (supports returning to known targets)

---

## VI. Reward Terms Added to Total Reward

- `stall_window_penalty`
- `first_seen_treasure_reward`
- `early_flash_penalty`
- `blind_flash_penalty`
- `flash_toward_monster_penalty`
- `trapped_flash_escape_bonus`
- `trapped_wait_flash_penalty`
- `high_pressure_backtrack_bonus`
- `buff_pickup_priority_bonus`
- `flash_hold_bonus`
- `non_wall_flash_base_penalty`
- `post_flash_explore_bonus`
- `post_flash_frontier_bonus`

---

## VII. Behavioral Goals Summary

| ✅ Rewarded Behaviors | ❌ Penalized Behaviors |
|----------------------|------------------------|
| No stalling / corner grinding | Almost no movement over 10 steps |
| First-time treasure discovery | Flash hitting a wall |
| Picking up / approaching treasure | Flashing when safe |
| Approaching or picking up buff | Flashing without seeing any monster |
| Picking up buff under high pressure | Flashing toward monster in open area |
| Successful wall-pass flash | Becoming more dangerous after flash |
| Escaping dead-end with flash | Wasting time in dead-end while waiting for CD |
| Successful backtracking under high pressure | Using non-wall flash carelessly |
| Holding flash when safe (not using it) | Backtracking with no benefit under low pressure |
| **Actually exploring new area / frontier after flash** | Wall-hugging / oscillating / repeated revisiting |

---

## VIII. Flash Abuse Limitation Mechanisms

| Mechanism | Effect |
|-----------|--------|
| Lower `FLASH_ESCAPE_REWARD_COEF` | Flash no longer gives high escape reward directly |
| Lower `FLASH_DIRECTION_REWARD_COEF` | Flash no longer rewarded just for correct direction |
| `NON_WALL_FLASH_BASE_PENALTY` | Base penalty for non-wall flash |
| `SAFE_FLASH_PENALTY` | Direct penalty for flashing when safe |
| `FLASH_BLIND_PENALTY` | Penalty for flashing without seeing any monster |
| `POST_FLASH_EXPLORE_BONUS` | Reward only if flash leads to new area exploration |
| `POST_FLASH_FRONTIER_BONUS` | Reward only if flash gets closer to frontier |


