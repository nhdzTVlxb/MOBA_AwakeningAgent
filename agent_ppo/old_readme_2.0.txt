# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法

## Code Modification Summary

### 1. Modifications Compared to Previous Version

| File | Modifications |
|------|---------------|
| `conf/conf.py` | GAMMA: 0.99→0.995, MAX_STEPS: 1000→2000; added curriculum learning config and network config |
| `feature/preprocessor.py` | Return values changed from 3 to 5 (added shaped_reward and visible_treasure_ratio); added detailed reward functions (corridor, pincer, blink, dead-end, etc.) |
| `model/model.py` | Upgraded from simple CNN+MLP to entity-wise encoding + multi-head attention network |
| `algorithm/algorithm.py` | Added ClipFrac, ExplainedVar, AdvMean, RetMean metric outputs |
| `workflow/train_workflow.py` | Integrated curriculum learning, metric statistics, and validation workflow |
| `agent.py` | Adapted to preprocessor's 5 return values |
| `curriculum.py` | **New** - 4-stage curriculum learning scheduler |
| `metrics.py` | **New** - Pre/post speedup metric statistics tool |
| `workflow/val_workflow.py` | **New** - Validation workflow |

---

### 2. Commonalities with Reference Paper

| Reference Requirements | Implementation Status |
|------------------------|----------------------|
| **Feature Engineering** | |
| Distance to nearest monster | ✅ |
| Distance to second nearest monster | ✅ |
| Direction and distance to nearest treasure | ✅ |
| Blink availability | ✅ |
| Speedup phase indicator | ✅ |
| Surrounding terrain dead-end detection | ✅ |
| Recent stuck/backtracking detection | ✅ |
| Danger quantification | ✅ |
| **Reward Function** | |
| Survival reward, step penalty | ✅ |
| Treasure reward, buff reward | ✅ |
| Treasure approach reward | ✅ |
| Monster distance shaping | ✅ |
| Pre-speedup buffer reward | ✅ |
| Late-stage survival reward | ✅ |
| Corridor reward | ✅ |
| Pincer penalty | ✅ |
| Dead-end penalty | ✅ |
| Danger penalty | ✅ |
| Wall collision penalty | ✅ |
| Repeated exploration penalty | ✅ |
| Blink escape reward | ✅ |
| Blink overuse penalty | ✅ |
| Second monster pressure penalty | ✅ |
| Endgame reward | ✅ |
| **Curriculum Learning** | |
| warmup_stable (epochs 0-150) | ✅ |
| mid_pressure (epochs 150-500) | ✅ |
| late_speedup (epochs 500-900) | ✅ |
| hard_generalization (900+ epochs) | ✅ |
| Dynamic treasure count, buff count | ✅ |
| Dynamic monster parameters | ✅ |
| **Train/Val Split** | |
| Training maps 1-8, validation maps 9-10 | ✅ |
| Validation every 10 episodes | ✅ |
| **Monitoring Metrics** | |
| Pre/post speedup steps, rewards, treasure scores | ✅ |
| Death rate, completion rate | ✅ |
| Blink statistics | ✅ |
| Algorithm metrics (ClipFrac, ExplainedVar, etc.) | ✅ |

---

### 3. Differences from Reference Paper

| Aspect | Reference Paper | Current Implementation |
|--------|----------------|------------------------|
| **Network Architecture** | Entity-wise encoding + attention | ✅ Implemented |
| **Reward Function Details** | Full version | Full version (detailed function) |
| **Feature Engineering** | Full | ✅ Full |
| **Curriculum Learning** | 4 stages | ✅ 4 stages |
| **Train/Val Split** | Yes | ✅ Yes |
| **Monitoring Metrics** | 20+ items | ✅ 20+ items |
| **Redundant Features** | None | Minor (repeated exploration penalty, continuous action penalty, etc., do not affect training) |

---

### 4. Core Improvements

1. **Network Upgrade**: From simple concatenation → entity-wise encoding + multi-head attention
2. **Reward Enhancement**: Detailed reward functions including corridor, pincer, dead-end, blink, etc.
3. **Curriculum Learning**: 4 difficulty stages with dynamically adjusted environment parameters
4. **Validation Mechanism**: Separate training/validation maps, evaluation every 10 episodes
5. **Monitoring Enhancement**: Pre/post speedup split statistics, 20+ metrics reporting
