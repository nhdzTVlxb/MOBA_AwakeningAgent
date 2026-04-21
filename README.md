# MOBA_AwakeningAgent

## Changelog

### 
### 2026-04-21 PPO4.1 Update（PPO 3.0 is running.）

---
FLASH_DANGER_DISTANCE=15 did not constrain the use of flash.
## Key Changes in `conf.py` & `preprocessor.py`

- **Increased observation dimension** from 929 to 932 by expanding `hero` features from 12 to 15 (added 10-step historical position).
- **Added anti-stalling mechanism**:
  - `STALL_WINDOW = 10`, `STALL_DISTANCE_THRESHOLD = 5.0`, `STALL_PENALTY = 0.10`
  - Penalizes the agent if it hasn't moved meaningfully from its position 10 steps ago.
- **Added first-seen treasure reward**:
  - `FIRST_SEEN_TREASURE_REWARD = 0.20`
  - One-time positive reward when a treasure first enters the agent's view.
- **Disabled `TRUNCATED_BONUS`** (changed from `4.0` to `0.0`).
- **Preserved all existing shaping rewards** (flash escape, cooldown escape, safe flash penalty, buff rewards, etc.).
- **Curriculum learning is not important; just train with the weekly competition configuration.
  
- **In `preprocessor.py`**:
  - Added `position_history` buffer (stores last `STALL_WINDOW + 1` positions).
  - Added `first_seen_rewarded` and `last_seen_step` to `TargetMemory` for per-episode treasure tracking.
  - `_sync_collectible_memory()` now returns `newly_seen_treasure_count`.
  - `hero_feat` now includes 3 history-based features: `history_dx_norm`, `history_dz_norm`, `history_dist_norm`.
  - Reward now includes:
    - `first_seen_treasure_reward`
    - `stall_window_penalty`

---

**Summary:**  
- **932-dim observation** (added 10-step history)  
- **Stall penalty** for staying near a 10-step-old position  
- **First-seen treasure reward** to encourage exploration
