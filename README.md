# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法

# 峡谷追猎 PPO - 2026比赛版本

## 一、修改的文件及内容

| 文件 | 修改内容 |
|------|----------|
| `agent_ppo/conf/conf.py` | 1. 动作空间：8 → 16<br>2. 熵系数：0.001 → 0.01<br>3. 特征维度：40 → 1863<br>4. 新增环境参数：MAX_STEPS=1000，TREASURE_NUM=10，MONSTER2_SPAWN_STEP=300，MONSTER_SPEED_UP_STEP=500<br>5. 新增奖励参数：REW_STEP=1.5，REW_TREASURE=100，REW_SURVIVE=0.05，REW_MONSTER_DISTANCE=0.3，PENALTY_HIT_WALL=0.1 |
| `agent_ppo/feature/preprocessor.py` | 1. 新增OrganManager类：管理宝箱和buff的位置估计（视野外物体追踪）<br>2. 新增MapManager类：管理21×21局部地图（障碍物、记忆、宝箱、buff）<br>3. 特征提取：从40维扩展到1863维（英雄5+怪物10+地图1764+宝箱60+buff6+动作掩码16+进度2）<br>4. 动作掩码：扩展到16维，支持闪现冷却和撞墙检测<br>5. 奖励函数：完全重写 |
| `agent_ppo/feature/definition.py` | 1. legal_action维度：8 → 16<br>2. prob维度：8 → 16 |
| `agent_ppo/model/model.py` | 1. 网络结构：单MLP → CNN+MLP<br>2. 新增CNN地图编码器：处理4×21×21地图 → 512维<br>3. 新增MLP其他特征编码器：处理99维 → 128维 |
| 其他文件 | algorithm.py、train_workflow.py、agent.py 未修改 |

## 二、奖励惩罚机制

| 类型 | 机制 | 数值 |
|------|------|------|
| 正向 | 步数奖励 | +1.5/步 |
| 正向 | 存活奖励 | +0.05/步 |
| 正向 | 宝箱拾取 | +100/个 |
| 正向 | 远离怪物 | +0.3 × (距离/50) |
| 正向 | 使用闪现 | +0.2/次 |
| 正向 | 捡到buff | +1.0/次 |
| 正向 | 靠近宝箱 | 越近越高 |
| 负向 | 太靠近怪物 | -0.3 × (5-距离) |
| 负向 | 撞墙 | -0.1/次 |

## 三、如何将地图从21×21改为41×41

| 顺序 | 文件 | 位置 | 原值 | 改为 |
|------|------|------|------|------|
| 1 | `conf.py` | 第24行 | `MAP_SIZE = 21` | `MAP_SIZE = 41` |
| 2 | `conf.py` | 第25行 | `4 * 21 * 21` | `4 * 41 * 41` |
| 3 | `preprocessor.py` | 第103-104行 | `range(-10, 11)` | `range(-20, 21)` |
| 4 | `preprocessor.py` | 第155行 | `size = 21` | `size = 41` |
| 5 | `model.py` | 第56行 | `64 * 6 * 6` | `64 * 11 * 11` |
