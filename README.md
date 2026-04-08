# MOBA_AwakeningAgent
2026年腾讯开悟_智能体决策算法
# README.md

```markdown
# Gorge Chase PPO - 2026比赛版本

基于开悟平台峡谷追猎（Gorge Chase）2026比赛任务的PPO算法实现。

## 文件结构

```
agent_ppo/
├── conf/
│   └── conf.py                 # 配置文件（特征维度、奖励参数、超参数）
├── feature/
│   ├── definition.py           # 数据定义（SampleData, GAE计算）
│   └── preprocessor.py         # 特征预处理（核心：宝箱/buff/怪物/地图管理）
├── model/
│   └── model.py                # 神经网络（CNN+MLP结构）
├── algorithm/
│   └── algorithm.py            # PPO算法实现
├── workflow/
│   └── train_workflow.py       # 训练工作流
└── agent.py                    # Agent主类
```

## 相比2026 Baseline的修改

### 1. 动作空间扩展（8 → 16）

| 文件 | 修改内容 |
|------|----------|
| `conf.py` | `ACTION_NUM = 16` |
| `definition.py` | `legal_action` 和 `prob` 维度改为16 |
| `preprocessor.py` | 动作掩码扩展到16维，闪现冷却时屏蔽后8维 |

### 2. 特征维度大幅增加（40 → 1863维）

| 特征组 | Baseline | 本实现 | 说明 |
|--------|----------|--------|------|
| 英雄特征 | 4维 | 5维 | 增加闪现cd |
| 怪物特征 | 10维 | 10维 | 相同 |
| 地图特征 | 16维(5×5) | **1764维(4×21×21)** | CNN输入 |
| 宝箱特征 | 无 | **60维(10×6)** | 10个宝箱位置估计 |
| buff特征 | 无 | **6维** | buff位置估计 |
| 动作掩码 | 8维 | 16维 | 对应16动作 |
| 进度特征 | 2维 | 2维 | 相同 |

### 3. 网络结构升级（MLP → CNN+MLP）

| Baseline | 本实现 |
|----------|--------|
| 单MLP：40→128→64→8/1 | CNN编码地图(4×21×21→512) + MLP编码其他特征 → 合并 → 输出 |

### 4. 新增OrganManager（宝箱/buff管理）

- 视野外物体位置估计（基于方向+距离）
- 10个宝箱独立追踪
- 每个宝箱输出6维特征（是否发现、相对位置、距离、绝对位置）

### 5. 新增MapManager（地图管理）

- 4通道21×21局部地图
- 障碍物、记忆、宝箱、buff
- 支持撞墙检测和动作掩码

### 6. 奖励函数重设计

| 奖励项 | Baseline | 本实现 |
|--------|----------|--------|
| 步数奖励 | 0.01 | 1.5（匹配计分规则） |
| 宝箱奖励 | 无 | 100/个 |
| 远离怪物 | 距离塑形(0.1) | 距离公式（近惩罚/远奖励） |
| 存活奖励 | 无 | 0.05/步 |
| 闪现鼓励 | 无 | 0.2/次 |
| 撞墙惩罚 | 无 | -0.1/次 |

### 7. 熵系数调整

| Baseline | 本实现 |
|----------|--------|
| 0.001 | 0.01（增加探索） |

## 核心模块说明

### preprocessor.py

```
Preprocessor
├── OrganManager    # 宝箱/buff位置估计（视野内外）
├── MapManager      # 21×21局部地图（障碍物、记忆、宝箱、buff）
├── _get_reward()   # 奖励计算（步数+宝箱+远离怪物+技能使用）
└── _get_action_mask() # 16维动作掩码（撞墙+闪现冷却）
```

### model.py

```
Model
├── map_encoder     # CNN: 4×21×21 → 512
├── other_encoder   # MLP: 99维 → 128
└── actor/critic    # 合并后输出16动作 + 1价值
```

## 如何修改地图尺寸（21×21 → 41×41）

### 需要修改3个文件：

#### 文件1: `agent_ppo/conf/conf.py`

```python
# 原代码
MAP_SIZE = 21
MAP_FEATURE_DIM = 4 * 21 * 21  # 1764

# 修改为
MAP_SIZE = 41
MAP_FEATURE_DIM = 4 * 41 * 41  # 6724
```

#### 文件2: `agent_ppo/feature/preprocessor.py`

在 `MapManager.get_around_feature()` 方法中：

```python
# 原代码
size = 21
half = size // 2

# 修改为
size = 41
half = size // 2
```

在 `MapManager.update_obstacles()` 方法中：

```python
# 原代码
for i in range(-10, 11):  # 半径10，总21
    for j in range(-10, 11):

# 修改为
for i in range(-20, 21):  # 半径20，总41
    for j in range(-20, 21):
```

在 `Preprocessor.feature_process()` 中，地图特征提取保持不变（自动适配）。

#### 文件3: `agent_ppo/model/model.py`

在 `Model.__init__()` 中，CNN的输出维度需要重新计算：

```python
# 41×41输入经过3层卷积后的尺寸：
# 输入: 4×41×41
# Conv(7,2,padding=3): 41→21
# Conv(5,2,padding=2): 21→11  
# Conv(3,1,padding=1): 11→11
# Flatten: 64×11×11 = 7744

# 原代码
make_fc_layer(64 * 6 * 6, 512)

# 修改为
make_fc_layer(64 * 11 * 11, 512)  # 7744 → 512
```

### 修改汇总表

| 文件 | 位置 | 原值 | 新值 |
|------|------|------|------|
| `conf.py` | `MAP_SIZE` | 21 | 41 |
| `conf.py` | `MAP_FEATURE_DIM` | 1764 | 6724 |
| `preprocessor.py` | `get_around_feature()` size | 21 | 41 |
| `preprocessor.py` | `update_obstacles()` 循环范围 | -10~10 | -20~20 |
| `model.py` | `make_fc_layer` 输入 | 64*6*6=2304 | 64*11*11=7744 |

### 注意事项

1. **特征维度变化**：总特征维度会从1863变为约6823，需要同步更新其他相关配置
2. **内存/显存增加**：CNN参数量和输入尺寸都会增加
3. **训练时间变长**：更大的网络需要更多训练步数
4. **视野半径同步**：`update_obstacles` 中的循环范围必须与 `MAP_SIZE` 匹配

## 训练监控指标

| 指标 | 含义 | 目标值 |
|------|------|--------|
| 终止步数 | 每局存活步数 | 800-1000 |
| 宝箱收集数 | 每局收集宝箱 | 8-10 |
| total_loss | 总损失 | 逐渐下降 |
| policy_loss | 策略损失 | -1 ~ -5 |
| entropy | 熵 | 逐渐下降 |

## 环境要求

- Python 3.11
- PyTorch
- NumPy
- 开悟平台框架

## 快速开始

```bash
# 测试运行
python train_test.py

# 正式训练（根据平台命令）
python train.py
```

## 参考

- 开悟平台文档
- PPO: Proximal Policy Optimization (Schulman et al., 2017)
```

---

这个README包含：
1. 文件结构说明
2. 相比Baseline的7大修改点（表格清晰）
3. 核心模块说明
4. **41×41地图修改指南**（具体到哪个文件的哪一行）
5. 训练监控指标
6. 快速开始命令
