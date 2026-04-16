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
    # --- 生存类 ---
    SURVIVE_REWARD = 0.001
    DIST_SHAPING_COEF = 0.03
    POST_SPEEDUP_SURVIVE_MULTIPLIER = 1.15
    POST_SPEEDUP_DIST_MULTIPLIER = 1.1
    TRUNCATED_BONUS = 4.0
    TERMINATED_PENALTY = -8.0

    # --- 宝箱类 ---
    TREASURE_REWARD = 2.5
    BUFF_REWARD = 0.5
    TREASURE_DIST_COEF = 0.18
    CLOSE_TREASURE_APPROACH_COEF = 0.15
    TREASURE_MISS_PENALTY = 0.3
    TREASURE_MISS_DISTANCE = 22.0
    TREASURE_MISS_MARGIN = 2.5
    TREASURE_URGENCY_DISTANCE = 28.0
    EXIT_DIST_COEF = 0.06

    # --- 宝箱优先级调整 ---
    TREASURE_PRIORITY_DISTANCE = 36.0
    SINGLE_MONSTER_TREASURE_PRESSURE_DISTANCE = 65.0
    SINGLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 2.2
    DOUBLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER = 1.6
    POST_SPEEDUP_TREASURE_PRIORITY_MULTIPLIER = 0.8

    # --- 前期捡箱窗口 ---
    EARLY_LOOT_SAFE_DISTANCE = 85.0
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
    DOUBLE_MONSTER_PINCH_DISTANCE = 90.0
    DOUBLE_MONSTER_PINCH_COS_THRESHOLD = -0.25
    PRE_SPEEDUP_BUFFER_WINDOW = 120
    PRE_SPEEDUP_BUFFER_SAFE_DISTANCE = 55.0
    PRE_SPEEDUP_BUFFER_COEF = 0.03
    SECOND_MONSTER_PRESSURE_THRESHOLD = 70.0
    SECOND_MONSTER_PRESSURE_COEF = 0.025

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

    # ========== 新增：Cooldown-aware 等待闪判断 ==========
    WAIT_FLASH_SAFE_MARGIN_STEPS = 6.0      # 怪物追上时间需比CD多多少步才允许等
    WAIT_FLASH_DANGER_THRESHOLD = 0.45      # 危险度超过此值不允许等
    WAIT_FLASH_MAX_STAGNATION_TOLERANCE = 2  # 最多允许原地磨几步

    # ========== 新增：CD中强制走脱奖励 ==========
    COOLDOWN_ESCAPE_REWARD_COEF = 0.08      # 走脱奖励系数
    COOLDOWN_ESCAPE_MIN_DIST_GAIN = 6.0     # 最小有效距离增益
    COOLDOWN_ESCAPE_OPENNESS_COEF = 0.05    # 开阔度增益系数

    # ========== 新增：等闪失败惩罚 ==========
    WAIT_FLASH_PENALTY = 0.12               # 不该等却等的惩罚
    WAIT_FLASH_OSCILLATION_MULTIPLIER = 1.8  # 等闪时振荡惩罚倍数
    WAIT_FLASH_HIT_WALL_MULTIPLIER = 2.0    # 等闪时撞墙惩罚倍数

    # ========== 新增：连续撞墙回头惩罚 ==========
    WALL_BACKTRACK_PENALTY = 0.15           # 撞墙后回头惩罚
    CONSECUTIVE_HIT_WALL_PENALTY = 0.08     # 连续撞墙累积惩罚

    # ========== 现有行为约束加强（系数调整） ==========
    HIT_WALL_PENALTY = 0.10                 # 0.05 → 0.10
    STAGNATION_PENALTY_COEF = 0.10          # 0.05 → 0.10
    OSCILLATION_PENALTY_COEF = 0.10         # 0.06 → 0.10
    REVISIT_PENALTY_COEF = 0.04             # 0.02 → 0.04
    NO_VISION_PATROL_BONUS_COEF = 0.03      # 0.02 → 0.03

    # --- 行为约束 ---
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
    LOCAL_MAP_CHANNEL = 7

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
            "max_train_episode": 299, 
            "treasure_count": (9, 10), 
            "buff_count": (2, 2), 
            "monster_interval": (500, 700), 
            "monster_speedup": (700, 900), 
            "max_step": 2000, 
        }, 
        { 
            "name": "mid_pressure", 
            "max_train_episode": 899, 
            "treasure_count": (8, 10), 
            "buff_count": (1, 2), 
            "monster_interval": (500, 500),
            "monster_speedup": (700, 700), 
            "max_step": 2000, 
        }, 
        { 
            "name": "late_speedup_survival", 
            "max_train_episode": 1599, 
             "treasure_count": (8, 10), 
            "buff_count": (1, 2), 
            "monster_interval": (300, 500), 
            "monster_speedup": (500, 700), 
            "max_step": 2000, 
        }, 
        { 
            "name": "hard_generalization", 
            "max_train_episode": 10**9, 
            "treasure_count": (6, 10), 
            "buff_count": (1, 2), 
            "monster_interval": (300, 300), 
            "monster_speedup": (500, 500), 
            "max_step": 1000, 
        }, 
    )
