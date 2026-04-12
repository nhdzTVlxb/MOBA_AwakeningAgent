# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法

## Changelog

### 
### 2026-04-13 PPO3.0 Update（PPO 3.0 is running.）

#### Overview
This update shifts the training focus from "survival by walking" to "active treasure collection and strategic flash usage". The hero feature dimension is expanded from 6 to 10, adding 4 flash-prior features. A staged flash reward system and segmented treasure rewards are introduced.

---

### Configuration Changes (`conf.py`)

| Parameter | Previous | Current | Purpose |
|-----------|----------|---------|---------|
| `HERO_FEATURE_DIM` | 6 | **10** | Add 4 flash-prior features |
| `FEATURE_LEN` | 1878 | **1882** | Auto-adjusted |
| `REW_SURVIVE` | 0.01 | **0.005** | Reduce passive survival weight |
| `REW_DISTANCE_EXPLORE` | 0.01 | **0.005** | Reduce exploration weight |
| `TREASURE_REWARD_SCHEDULE` | - | **[3.5, 2.5, 1.5]** | Segmented treasure rewards |
| `TREASURE_REWARD_FALLBACK` | - | **1.0** | Reward for 4th+ treasure |
| `FLASH_HIT_STREAK_THRESHOLD` | - | **2** | Wall hits to trigger high-value window |
| `FLASH_PRE_DANGER_DIST` | - | **12.0** | Danger threshold before speedup |
| `FLASH_POST_DANGER_DIST` | - | **8.0** | Danger threshold after speedup |
| `REW_FLASH_PRE_WALL` | - | **0.4** | Flash over wall (pre-speedup) |
| `REW_FLASH_PRE_ESCAPE` | - | **0.7** | Flash escape (pre-speedup) |
| `REW_FLASH_PRE_OPEN` | - | **0.25** | Flash to open area (pre-speedup) |
| `REW_FLASH_POST_WALL` | - | **0.8** | Flash over wall (post-speedup) |
| `REW_FLASH_POST_ESCAPE` | - | **1.1** | Flash escape (post-speedup) |
| `REW_FLASH_POST_COMBO` | - | **0.4** | Flash combo bonus (post-speedup) |
| `PENALTY_FLASH_HOLD_PRE` | - | **0.15** | Not flashing when should (pre-speedup) |
| `PENALTY_FLASH_HOLD_POST` | - | **0.35** | Not flashing when should (post-speedup) |

---

### Reward Comparison (1000 Steps + 10 Treasures)

| Source | Previous | Current |
|--------|----------|---------|
| Step Reward (1000 steps) | 11.0 | **6.0** |
| Treasure Reward (10 treasures) | 10.0 | **14.5** |
| **Total Theoretical Max** | 21.0 | **20.5** |
| **Treasure:Step Ratio** | ~0.9:1 | **~2.4:1** |

---

### Preprocessor Changes (`preprocessor.py`)

#### New State Variables
| Variable | Purpose |
|----------|---------|
| `wall_hit_streak` | Count consecutive wall hits |
| `_last_step_hit_wall` | Whether last step hit a wall |
| `prev_dead_end_risk` | Dead-end risk from previous frame |
| `prev_monster_closing` | Monster closing trend from previous frame |
| `prev_flash_high_value` | Whether previous frame was high-value flash window |
| `prev_is_speedup` | Whether previous frame was in speedup phase |

#### New Methods
| Method | Purpose |
|--------|---------|
| `_update_hit_state()` | Update wall hit streak based on action result |
| `_calc_flash_high_value_window()` | Determine if current state is a high-value flash opportunity |
| `_calc_treasure_pickup_reward()` | Calculate segmented treasure reward |

#### Hero Feature Expansion (6 → 10 dimensions)
| Dim | Feature | Description |
|-----|---------|-------------|
| 1-6 | Original | Position, flash ready, cooldown, buff, distance from start |
| 7 | `wall_hit_streak_norm` | Normalized consecutive wall hits (max 3) |
| 8 | `dead_end_risk` | 1.0 - openness_score |
| 9 | `monster_closing` | Whether monster distance is decreasing |
| 10 | `flash_high_value` | Binary: is this a high-value flash window |

#### Flash Reward Logic (Refactored)
- **Pre-speedup (low pressure)**: Gentle rewards for correct flash usage
- **Post-speedup (high pressure)**: Stronger rewards + combo bonus
- **Penalty**: Punish not flashing when in high-value window

---

### Files Modified
| File | Change Type |
|------|-------------|
| `agent_ppo/conf/conf.py` | Configuration update |
| `agent_ppo/feature/preprocessor.py` | Feature expansion + reward refactor |

### Files Unchanged (Auto-adapt)
| File | Reason |
|------|--------|
| `agent_ppo/model/model.py` | Reads `HERO_FEATURE_DIM` dynamically |
| `agent_ppo/algorithm/algorithm.py` | Not affected |
| `agent_ppo/feature/definition.py` | Reads `DIM_OF_OBSERVATION` dynamically |
| `agent_ppo/workflow/train_workflow.py` | Not affected |
| `agent_ppo/workflow/val_workflow.py` | Not affected |

---

### Expected Training Improvements
1. **Higher treasure collection rate** - Segmented rewards incentivize early treasures
2. **Flash usage > 0** - Prior features + staged rewards teach proper flash timing
3. **Better high-pressure survival** - Post-speedup rewards guide emergency flash usage
4. **Reduced passive behavior** - Lower `REW_SURVIVE` discourages pure survival strategy

---

### Monitoring Focus
After this update, watch these metrics:
- `treasures` - Should increase and stabilize
- `flash_count` - Should become > 0
- `terminated_rate` - Should not spike
- `completed_rate` - Should maintain or improve
```
