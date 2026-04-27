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

    # Feature dimensions / 特征维度（共1053维）
    # hero(15) + monster1(10) + monster2(10) + treasure(10)
    # + map(8*11*11=968) + legal(16) + progress(24)
    FEATURES = [
        15,
        10,
        10,
        10,
        968,  # map: 8 * 11 * 11
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
    INIT_LEARNING_RATE_START = 0.0001
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
    # --- 生存类 ---
    SURVIVE_REWARD = 0.001
    DIST_SHAPING_COEF = 0.03
    POST_SPEEDUP_SURVIVE_MULTIPLIER = 1.15
    POST_SPEEDUP_DIST_MULTIPLIER = 1.1
    TRUNCATED_BONUS = 0.0
    TERMINATED_PENALTY = -8.0

    # --- 宝箱类 ---
    TREASURE_REWARD = 2.5
    TREASURE_DIST_COEF = 0.18
    CLOSE_TREASURE_APPROACH_COEF = 0.15
    TREASURE_MISS_PENALTY = 0.3
    TREASURE_MISS_DISTANCE = 22.0
    TREASURE_MISS_MARGIN = 2.5
    TREASURE_URGENCY_DISTANCE = 28.0
    EXIT_DIST_COEF = 0.06

    # --- 首次看见宝箱奖励 ---
    FIRST_SEEN_TREASURE_REWARD = 0.20

    # --- 宝箱优先级调整 ---
    TREASURE_PRIORITY_DISTANCE = 36.0
    SINGLE_MONSTER_TREASURE_PRESSURE_DISTANCE = 65.0
    SINGLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 2.2
    DOUBLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 1.6
    POST_SPEEDUP_TREASURE_PRIORITY_MULTIPLIER = 0.8

    # --- 前期捡箱窗口 ---
    EARLY_LOOT_SAFE_DISTANCE = 80.0
    EARLY_LOOT_TREASURE_PRIORITY_MULTIPLIER = 2.2
    EARLY_LOOT_DIST_SHAPING_MULTIPLIER = 0.55
    EARLY_LOOT_REVISIT_PENALTY_MULTIPLIER = 0.4
    EARLY_LOOT_EXPLORE_BONUS_MULTIPLIER = 0.5
    EARLY_LOOT_COLLECTION_BONUS = 0.6
    EARLY_LOOT_FIRST_TREASURE_BONUS = 1.0
    EARLY_LOOT_STALL_STEP_THRESHOLD = 15
    EARLY_LOOT_STALL_PROGRESS_THRESHOLD = 1.5
    EARLY_LOOT_STALL_PENALTY = 0.03

    # --- 双怪与压力 ---
    DOUBLE_MONSTER_PINCH_DISTANCE = 95.0
    DOUBLE_MONSTER_PINCH_COS_THRESHOLD = 0.00
    PRE_SPEEDUP_BUFFER_WINDOW = 120
    PRE_SPEEDUP_BUFFER_SAFE_DISTANCE = 55.0
    PRE_SPEEDUP_BUFFER_COEF = 0.03
    SECOND_MONSTER_PRESSURE_THRESHOLD = 75.0
    SECOND_MONSTER_PRESSURE_COEF = 0.025

    # --- 闪现奖励 ---
    FLASH_ESCAPE_REWARD_COEF = 0.005
    FLASH_DANGER_DISTANCE = 10.0
    FLASH_DIRECTION_REWARD_COEF = 0.004
    FLASH_DIRECTION_MAX_DISTANCE_DROP = 12.0
    FLASH_THROUGH_WALL_REWARD_COEF = 0.01
    FLASH_THROUGH_WALL_MIN_MOVE_DISTANCE = 4.0
    FLASH_THROUGH_WALL_SCAN_STEPS = 4
    FLASH_THROUGH_WALL_MAX_DISTANCE_DROP = 6.0
    FLASH_WASTE_PENALTY = 0.18
    FLASH_WASTE_MIN_ESCAPE_GAIN = 8.0
    FLASH_FAR_WASTE_MULTIPLIER = 2.5

    # ========== 等闪失败惩罚 ==========
    WAIT_FLASH_PENALTY = 0.16
    WAIT_FLASH_OSCILLATION_MULTIPLIER = 1.8
    WAIT_FLASH_HIT_WALL_MULTIPLIER = 2.5

    NON_WALL_FLASH_BASE_PENALTY = 0.12
    POST_SPEEDUP_NON_WALL_FLASH_MULTIPLIER = 1.4

    # ========== 闪现质量相关 ==========
    SAFE_FLASH_DANGER_THRESHOLD = 0.18
    SAFE_FLASH_PENALTY = 0.35
    FLASH_HOLD_BONUS = 0.04
    FLASH_HOLD_SAFE_DANGER_THRESHOLD = 0.40
    FLASH_HOLD_POST_SPEEDUP_SCALE = 0.5

    FLASH_SUICIDE_DISTANCE_MARGIN = 4.0
    FLASH_SUICIDE_PENALTY = 0.60

    POST_FLASH_CONFUSION_WINDOW = 3
    POST_FLASH_STALL_PENALTY = 0.12
    POST_FLASH_OSCILLATION_PENALTY = 0.12
    POST_FLASH_HIT_WALL_PENALTY = 0.18
    POST_FLASH_BACKTRACK_PENALTY = 0.12

    POST_FLASH_EXPLORE_BONUS = 0.10
    POST_FLASH_FRONTIER_BONUS = 0.01

    FLASH_HIT_WALL_PENALTY = 0.35
    FLASH_THROUGH_WALL_BONUS_MULTIPLIER = 2.20

    # ========== 早闪 / 盲闪 / 朝怪闪 ==========
    EARLY_FLASH_STEP_LIMIT = 20
    EARLY_FLASH_PENALTY = 0.35
    FLASH_BLIND_PENALTY = 0.40
    OPEN_AREA_FLASH_TOWARD_MONSTER_OPENNESS = 0.60
    FLASH_TOWARD_MONSTER_PENALTY = 0.35

    # ========== 死胡同贴墙穿墙闪 ==========
    TRAPPED_DEAD_END_THRESHOLD = 0.65
    TRAPPED_FLASH_ESCAPE_BONUS = 0.45
    TRAPPED_FLASH_MONSTER_CROSS_BONUS = 0.55
    TRAPPED_WAIT_FLASH_PENALTY = 0.18

    # ========== 高压回头豁免 ==========
    HIGH_PRESSURE_BACKTRACK_THRESHOLD = 0.70
    HIGH_PRESSURE_BACKTRACK_PENALTY_SCALE = 0.25
    HIGH_PRESSURE_BACKTRACK_BONUS = 0.12

    # ========== buff 战略奖励 / DMSD resource memory ==========
    # 目标：
    # 1. 鼓励主动拿 buff；
    # 2. 允许安全短等刷新；
    # 3. 禁止怪物快追上时原地等 buff；
    # 4. 鼓励闪现直接拿已刷新的近距离 buff；
    # 5. buff 总收益接近宝箱，但不远远超过宝箱。
    #
    # 宝箱基础 TREASURE_REWARD = 2.5；
    # buff 基础设为 2.0，略低于宝箱，但靠近、闪现、刷新窗口可以补足。
    BUFF_REWARD = 2.00

    # 走向 buff 的过程奖励，不能太大，否则会为了 buff 乱绕路。
    BUFF_APPROACH_REWARD = 0.12
    BUFF_MAX_CHASE_DISTANCE = 38.0

    # buff 刷新记忆：环境刷新约 200 步。
    BUFF_REFRESH_STEPS = 200

    # 只允许短等，不允许原地等完整 200 步。
    BUFF_WAIT_MAX_STEPS = 18
    BUFF_WAIT_SAFE_MARGIN_STEPS = 8.0
    BUFF_WAIT_PENALTY = 0.16
    BUFF_DANGEROUS_WAIT_PENALTY = 0.26

    # 估算走到 buff 的步数，只用于 reward 判断。
    BUFF_ESTIMATED_HERO_SPEED = 4.0

    # 闪现拿 buff：你说闪现大概 10 步，这里给一点余量。
    BUFF_FLASH_PICKUP_DISTANCE = 8.5
    BUFF_FLASH_PICKUP_BONUS = 0.45

    # buff 刚刷新时拿到，给额外鼓励。
    BUFF_REFRESH_PICKUP_BONUS = 0.40

    # 高压/后期拿 buff 的额外价值，但封顶，避免超过宝箱太多。
    BUFF_HIGH_PRESSURE_PICKUP_BONUS = 0.45
    BUFF_PICKUP_BONUS_CAP = 1.0

    # 拿到 buff 后确实拉开怪物距离，给小额连招奖励。
    BUFF_ESCAPE_COMBO_REWARD = 0.35

    # 如果这次闪现确实拿到了 buff，就不要再按普通非穿墙闪现惩罚。
    BUFF_RESOURCE_FLASH_PENALTY_SCALE = 0.0

    # 兼容旧逻辑。你现在的新逻辑主要用 BUFF_FLASH_PICKUP_BONUS。
    BUFF_FLASH_CD_PICKUP_BONUS = 0.20

    BUFF_HIGH_PRESSURE_THRESHOLD = 0.45
    BUFF_POST_SPEEDUP_PRIORITY_MULTIPLIER = 1.25

    # ========== 10步防磨蹭 ==========
    STALL_WINDOW = 10
    STALL_DISTANCE_THRESHOLD = 5.0
    STALL_PENALTY = 0.10
    HISTORY_POSITION_NORM = 32.0

    # ========== Cooldown-aware 等待闪判断 ==========
    WAIT_FLASH_SAFE_MARGIN_STEPS = 6.0
    WAIT_FLASH_DANGER_THRESHOLD = 0.45
    WAIT_FLASH_MAX_STAGNATION_TOLERANCE = 2

    # ========== CD中强制走脱奖励 ==========
    COOLDOWN_ESCAPE_REWARD_COEF = 0.15
    COOLDOWN_ESCAPE_MIN_DIST_GAIN = 6.0
    COOLDOWN_ESCAPE_OPENNESS_COEF = 0.05

    # ========== 连续撞墙回头惩罚 ==========
    WALL_BACKTRACK_PENALTY = 0.15
    CONSECUTIVE_HIT_WALL_PENALTY = 0.08

    # ========== 低压探索惩罚 ==========
    LOW_PRESSURE_THRESHOLD = 0.25
    LOW_PRESSURE_STALL_PENALTY = 0.10
    LOW_PRESSURE_SMALL_LOOP_PENALTY = 0.12
    LOW_PRESSURE_BACKTRACK_PENALTY = 0.10
    LOW_PRESSURE_EXPLORE_BONUS = 0.06
    LOW_PRESSURE_FRONTIER_BONUS = 0.08
    LOW_PRESSURE_MIN_MOVE_DISTANCE = 1.0
    LOW_PRESSURE_LOCAL_LOOP_RADIUS = 3.0

    # ========== 普通走路脱困奖励 ==========
    NORMAL_ESCAPE_REWARD_COEF = 0.08
    NORMAL_ESCAPE_OPENNESS_COEF = 0.05
    NORMAL_ESCAPE_DEAD_END_RELIEF_COEF = 0.05
    CORNER_ESCAPE_BONUS = 0.08

    # ========== 现有行为约束 ==========
    HIT_WALL_PENALTY = 0.12
    STAGNATION_PENALTY_COEF = 0.12
    OSCILLATION_PENALTY_COEF = 0.12
    REVISIT_PENALTY_COEF = 0.05
    NO_VISION_PATROL_BONUS_COEF = 0.03

    # --- 行为约束阈值 ---
    HIT_WALL_DISTANCE_THRESHOLD = 0.5
    STAGNATION_MOVE_THRESHOLD = 0.75
    STAGNATION_MAX_STEPS = 6
    OSCILLATION_RETURN_DISTANCE = 1.25
    OSCILLATION_MAX_STEPS = 4
    NO_VISION_STAGNATION_MULTIPLIER = 1.5
    NO_VISION_PATROL_MOVE_DISTANCE = 1.0
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
    LOCAL_MAP_CHANNEL = 8

    # Structured observation encoder / 结构化观测编码
    HERO_ENCODER_DIM = 32
    MONSTER_ENCODER_DIM = 64
    TREASURE_ENCODER_DIM = 16
    MAP_ENCODER_DIM = 128
    CONTROL_ENCODER_DIM = 32
    FUSION_HIDDEN_DIM = 128

    TRAIN_BATCH_EPISODES = 4

    # Episode curriculum / 课程式训练分布
    RESUME_CURRICULUM_STAGE_NAME = "hard_generalization"
    CURRICULUM_STAGES = (
        {
            "name": "warmup_stable",
            "max_train_episode": 399,
            "treasure_count": (10, 10),
            "buff_count": (2, 2),
            "monster_interval": (800, 800),
            "monster_speedup": (1000, 1000),
            "max_step": 1000,
        },
        {
            "name": "mid_pressure",
            "max_train_episode": 799,
            "treasure_count": (10, 10),
            "buff_count": (2, 2),
            "monster_interval": (500, 500),
            "monster_speedup": (700, 700),
            "max_step": 1000,
        },
        {
            "name": "late_speedup_survival",
            "max_train_episode": 1099,
            "treasure_count": (10, 10),
            "buff_count": (2, 2),
            "monster_interval": (300, 300),
            "monster_speedup": (500, 500),
            "max_step": 1000,
        },
        {
            "name": "hard_generalization",
            "max_train_episode": 10**9,
            "treasure_count": (10, 10),
            "buff_count": (2, 2),
            "monster_interval": (300, 300),
            "monster_speedup": (500, 500),
            "max_step": 1000,
        },
    )
