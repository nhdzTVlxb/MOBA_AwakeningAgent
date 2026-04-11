# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法

## Changelog

### 2026-04-12 Major Update

#### New Features
- **Monster Memory Features**: Added 5-dimensional memory features per monster (visibility, memory direction, memory distance, confidence), maintaining danger awareness even when monsters leave line of sight.
- **Primary Treasure Target Features**: Added 4-dimensional features (direction_x, direction_z, distance, validity) directly connected to output head, giving model explicit knowledge of the nearest treasure location.
- **Flash-Over-Wall Reward**: Detects whether flash skill crosses obstacles and grants a 0.15 mechanism reward, teaching the model that "flash can go through walls".
- **Flash Diagnostic Logging**: Tracks legal flash steps, legal flash steps in dangerous situations, and average flash probability to diagnose why flash is not being used.
- **Map Organ Channels**: Treasure and buff positions now written to CNN input channels, making map features more complete.

#### Configuration Adjustments
- `MAX_STEPS`: 2000 → 1000 (aligned with environment TOML)
- `MONSTER_SPEED_UP_STEP`: 1200 → 500
- `MONSTER2_SPAWN_STEP`: 600 → 300
- Curriculum learning stage timing parameters adapted for 1000-step limit

#### Training Stability Improvements
- PPO advantage normalization
- Fixed `explained_var` calculation (now uses current `value_pred`)
- Added `approx_kl` monitoring

#### Bug Fixes
- Fixed monster feature concatenation order (aligned with model reshape semantics)
- Fixed first-frame configuration order (now calls `set_episode_config` before `observation_process`)
- Aligned curriculum learning field names with environment TOML (`treasure_count`, `buff_count`, `monster_interval`, `monster_speedup`)
- Unified checkpoint id to `"latest"`
- Dynamic speedup step now correctly propagated to preprocessor

#### File Change Summary
| File | Change Type |
|------|-------------|
| `conf.py` | Configuration update |
| `preprocessor.py` | Refactored |
| `agent.py` | Adapted |
| `model.py` | Slice order adjusted |
| `algorithm.py` | Stability improvements |
| `train_workflow.py` | Field alignment + diagnostics |
| `val_workflow.py` | Field alignment |
