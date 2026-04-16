#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共929维）
    FEATURES = [
        12,
        10,
        10,
        10,
        847,
        16,
        24,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：16 个离散动作（8 移动 + 8 闪现）
    ACTION_NUM = 16

    # Value head / 价值头：单头价值估计
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0002
    BETA_START = 0.003
    BETA_END = 0.0005
    BETA_DECAY_STEPS = 4000
    CLIP_PARAM = 0.15
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    USE_ADVANTAGE_NORM = True
    ADVANTAGE_NORM_EPS = 1e-8
    TARGET_KL = 0.015

    # Reward shaping / 奖励设计
    # --- 生存类（降权，避免压过宝箱信号） ---
    SURVIVE_REWARD = 0.001                          # 0.005→0.001：2000步累积 10→2
    DIST_SHAPING_COEF = 0.03                        # 0.05→0.03：距离 shaping 降权
    POST_SPEEDUP_SURVIVE_MULTIPLIER = 1.15          # 1.5→1.15：加速后生存乘数大幅降低
    POST_SPEEDUP_DIST_MULTIPLIER = 1.1              # 1.4→1.1：加速后距离 shaping 降低
    TRUNCATED_BONUS = 4.0                           # 8.0→4.0：存活终局奖励减半
    TERMINATED_PENALTY = -8.0                       # -12→-8：死亡惩罚降低（避免过度恐惧）

    # --- 宝箱类（大幅强化） ---
    TREASURE_REWARD = 2.5                           # 1.2→2.5：单个宝箱奖励翻倍+
    BUFF_REWARD = 0.5                               # 0.3→0.5：buff 奖励提升
    TREASURE_DIST_COEF = 0.18                       # 0.1→0.18：接近宝箱 shaping 大幅增强
    CLOSE_TREASURE_APPROACH_COEF = 0.15             # 0.08→0.15：近距离冲刺奖励翻倍
    TREASURE_MISS_PENALTY = 0.3                     # 0.15→0.3：走近又走开的惩罚翻倍
    TREASURE_MISS_DISTANCE = 22.0                   # 18→22：miss 判定距离放宽
    TREASURE_MISS_MARGIN = 2.5                      # 3→2.5：更敏感的 miss 判定
    TREASURE_URGENCY_DISTANCE = 28.0                # 20→28：紧急接近距离放宽
    EXIT_DIST_COEF = 0.06                           # 0.04→0.06：无宝箱时趋近 buff 增强

    # --- 宝箱优先级调整 ---
    TREASURE_PRIORITY_DISTANCE = 36.0               # 32→36：宝箱优先判定距离略放宽
    SINGLE_MONSTER_TREASURE_PRESSURE_DISTANCE = 65.0  # 60→65：单怪压力距离略放宽
    SINGLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 2.2   # 1.8→2.2：单怪阶段宝箱优先级大幅提升
    DOUBLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 1.6   # 1.35→1.6：双怪阶段宝箱优先级提升
    POST_SPEEDUP_TREASURE_PRIORITY_MULTIPLIER = 0.8     # 0.65→0.8：加速后宝箱不要降太多

    # --- 前期捡箱窗口（整体强化） ---
    EARLY_LOOT_SAFE_DISTANCE = 85.0
    EARLY_LOOT_TREASURE_PRIORITY_MULTIPLIER = 2.2  # 1.75→2.2：前期捡箱宝箱优先级提升
    EARLY_LOOT_DIST_SHAPING_MULTIPLIER = 0.55      # 0.45→0.55：前期距离 shaping 增强
    EARLY_LOOT_REVISIT_PENALTY_MULTIPLIER = 0.4    # 0.6→0.4：前期绕圈惩罚增强（系数越小=惩罚越重）
    EARLY_LOOT_EXPLORE_BONUS_MULTIPLIER = 0.5      # 0.0→0.5：前期探索奖励重新开放
    EARLY_LOOT_COLLECTION_BONUS = 0.6              # 0.25→0.6：前期捡箱额外奖励翻倍+
    EARLY_LOOT_FIRST_TREASURE_BONUS = 1.0          # 0.4→1.0：第一个宝箱额外奖励大幅提升
    EARLY_LOOT_STALL_STEP_THRESHOLD = 15           # 20→15：更早触发停滞惩罚
    EARLY_LOOT_STALL_PROGRESS_THRESHOLD = 1.5      # 1.0→1.5：停滞判定更宽松（进步少就算停滞）
    EARLY_LOOT_STALL_PENALTY = 0.03                # 0.012→0.03：停滞惩罚加强

    # --- 双怪与压力 ---
    DOUBLE_MONSTER_PINCH_DISTANCE = 90.0
    DOUBLE_MONSTER_PINCH_COS_THRESHOLD = -0.25
    PRE_SPEEDUP_BUFFER_WINDOW = 120
    PRE_SPEEDUP_BUFFER_SAFE_DISTANCE = 55.0        # 60→55：缓冲安全距离收紧
    PRE_SPEEDUP_BUFFER_COEF = 0.03                 # 0.04→0.03：缓冲期奖励降权
    SECOND_MONSTER_PRESSURE_THRESHOLD = 70.0
    SECOND_MONSTER_PRESSURE_COEF = 0.025           # 0.03→0.025：第二只怪压力略降

    # --- 闪现奖励 ---
    FLASH_ESCAPE_REWARD_COEF = 0.05
    FLASH_DANGER_DISTANCE = 55.0
    FLASH_DIRECTION_REWARD_COEF = 0.04
    FLASH_DIRECTION_MAX_DISTANCE_DROP = 12.0
    FLASH_THROUGH_WALL_REWARD_COEF = 0.06
    FLASH_THROUGH_WALL_MIN_MOVE_DISTANCE = 4.0
    FLASH_THROUGH_WALL_SCAN_STEPS = 4
    FLASH_THROUGH_WALL_MAX_DISTANCE_DROP = 6.0
    FLASH_WASTE_PENALTY = 0.08
    FLASH_WASTE_MIN_ESCAPE_GAIN = 8.0
    FLASH_FAR_WASTE_MULTIPLIER = 1.5

    # --- 行为约束（不变） ---
    HIT_WALL_PENALTY = 0.05
    HIT_WALL_DISTANCE_THRESHOLD = 0.5
    STAGNATION_MOVE_THRESHOLD = 0.75
    STAGNATION_MAX_STEPS = 6
    STAGNATION_PENALTY_COEF = 0.05
    OSCILLATION_RETURN_DISTANCE = 1.25
    OSCILLATION_MAX_STEPS = 4
    OSCILLATION_PENALTY_COEF = 0.06
    NO_VISION_STAGNATION_MULTIPLIER = 1.5
    NO_VISION_PATROL_MOVE_DISTANCE = 1.0
    NO_VISION_PATROL_BONUS_COEF = 0.02
    REVISIT_PENALTY_COEF = 0.02
    REVISIT_WINDOW_SIZE = 3

    # Monitor reporting / 监控上报
    EPISODE_PROGRESS_REPORT_INTERVAL = 50
    EPISODE_PROGRESS_REPORT_EPISODE_INTERVAL = 10

    # Lightweight exploration bonus / 轻量探索奖励
    ENABLE_EXPLORE_BONUS = True
    EXPLORE_BONUS_SCALE = 0.01
    EXPLORE_BONUS_GRID_SIZE = 16
    EXPLORE_BONUS_MIN_RATIO = 0.25

    # Semantic map / 局部语义地图
    LOCAL_MAP_SIZE = 11
    LOCAL_MAP_CHANNEL = 7

    # Structured observation encoder / 结构化观测编码
    HERO_ENCODER_DIM = 32
    MONSTER_ENCODER_DIM = 64
    TREASURE_ENCODER_DIM = 16
    MAP_ENCODER_DIM = 128
    CONTROL_ENCODER_DIM = 32
    FUSION_HIDDEN_DIM = 128

    # Episode curriculum / 课程式训练分布
    RESUME_CURRICULUM_STAGE_NAME = "hard_generalization"
    RESUME_CURRICULUM_STAGE_NAME = "hard_generalization"
    CURRICULUM_STAGES = (
        {
            "name": "warmup_stable",
            "max_train_episode": 149,
            "treasure_count": (9, 10),
            "buff_count": (2, 2),
            "monster_interval": (220, 300),
            "monster_speedup": (360, 460),
            "max_step": 2000,
        },
        {
            "name": "mid_pressure",
            "max_train_episode": 499,
            "treasure_count": (8, 10),
            "buff_count": (1, 2),
            "monster_interval": (160, 280),
            "monster_speedup": (240, 420),
            "max_step": 2000,
        },
        {
            "name": "late_speedup_survival",
            "max_train_episode": 899,
            "treasure_count": (7, 10),
            "buff_count": (1, 2),
            "monster_interval": (120, 220),
            "monster_speedup": (180, 320),
            "max_step": 2000,
        },
        {
            "name": "hard_generalization",
            "max_train_episode": 10**9,
            "treasure_count": (6, 10),
            "buff_count": (0, 2),
            "monster_interval": (120, 320),
            "monster_speedup": (140, 420),
            "max_step": 2000,
        },
    )
