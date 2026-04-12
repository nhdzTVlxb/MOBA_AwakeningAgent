#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置文件。
"""


class Config:
    # ========== 训练配置 ==========
    GAMMA = 0.99
    LAMDA = 0.95
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    BETA_START = 0.01
    GRAD_CLIP_RANGE = 0.5
    INIT_LEARNING_RATE_START = 1e-4

    # ========== 环境配置 ==========
    MAX_STEPS = 1000
    MONSTER_SPEED_UP_STEP = 500
    MONSTER2_SPAWN_STEP = 300

    # ========== 特征维度配置 ==========
    # 地图特征
    MAP_SIZE = 21
    MAP_CHANNELS = 4
    MAP_RAW_CHANNELS = 4
    MAP_FEATURE_DIM = MAP_CHANNELS * MAP_SIZE * MAP_SIZE  # 1764

    # 英雄特征（从6维扩展到10维）
    HERO_FEATURE_DIM = 10
    HERO_RAW_DIM = HERO_FEATURE_DIM

    # 怪物特征（原始5维 + 记忆5维 = 10维）
    MONSTER_FEATURE_DIM = 10
    MONSTER_NUM = 2
    MONSTER_TOTAL_DIM = MONSTER_NUM * MONSTER_FEATURE_DIM  # 20
    MONSTER_PER_RAW_DIM = MONSTER_FEATURE_DIM

    # 宝箱特征
    TREASURE_FEATURE_DIM = 6
    TREASURE_NUM = 10
    TREASURE_TOTAL_DIM = TREASURE_NUM * TREASURE_FEATURE_DIM  # 60
    TREASURE_PER_RAW_DIM = TREASURE_FEATURE_DIM

    # 主目标宝箱特征
    PRIMARY_TREASURE_FEATURE_DIM = 4

    # Buff特征
    BUFF_FEATURE_DIM = 6
    BUFF_RAW_DIM = BUFF_FEATURE_DIM

    # 动作掩码
    ACTION_NUM = 16

    # 进度特征
    PROGRESS_FEATURE_DIM = 2

    # 总特征维度
    FEATURE_LEN = (
        HERO_FEATURE_DIM +                    # 10
        MONSTER_TOTAL_DIM +                   # 20
        MAP_FEATURE_DIM +                     # 1764
        TREASURE_TOTAL_DIM +                  # 60
        PRIMARY_TREASURE_FEATURE_DIM +        # 4
        BUFF_FEATURE_DIM +                    # 6
        ACTION_NUM +                          # 16
        PROGRESS_FEATURE_DIM                  # 2
    )  # = 1882

    DIM_OF_OBSERVATION = FEATURE_LEN

    # ========== 网络配置 ==========
    EMBEDDING_DIM = 128
    MAP_EMBEDDING_DIM = 128
    NUM_HEADS = 4
    VALUE_NUM = 1

    # ========== 奖励系数 ==========
    REW_STEP = 0.001
    REW_SURVIVE = 0.005
    REW_TREASURE = 1.0
    REW_MONSTER_DISTANCE = 0.05
    REW_CORRIDOR = 0.1
    REW_FLASH_ESCAPE = 0.5
    REW_PRE_SPEEDUP = 0.02
    REW_DISTANCE_EXPLORE = 0.005
    REW_FLASH_OVER_WALL = 0.15

    # ========== 宝箱分段奖励 ==========
    TREASURE_REWARD_SCHEDULE = [3.5, 2.5, 1.5]
    TREASURE_REWARD_FALLBACK = 1.0

    # ========== 闪现高价值窗口参数 ==========
    FLASH_HIT_STREAK_THRESHOLD = 2
    FLASH_PRE_DANGER_DIST = 12.0
    FLASH_POST_DANGER_DIST = 8.0

    # 加速前：温和引导
    REW_FLASH_PRE_WALL = 0.4
    REW_FLASH_PRE_ESCAPE = 0.7
    REW_FLASH_PRE_OPEN = 0.25

    # 加速后：高压强引导
    REW_FLASH_POST_WALL = 0.8
    REW_FLASH_POST_ESCAPE = 1.1
    REW_FLASH_POST_COMBO = 0.4

    # 高危却不闪的惩罚
    PENALTY_FLASH_HOLD_PRE = 0.15
    PENALTY_FLASH_HOLD_POST = 0.35

    # ========== 惩罚系数 ==========
    PENALTY_DEAD_END = 0.1
    PENALTY_PINCH = 0.15
    PENALTY_SECOND_MONSTER = 0.05
    PENALTY_FLASH_ABUSE = 0.3
    PENALTY_HIT_WALL = 0.05
    PENALTY_ZIGZAG = 0.1

    # ========== 其他配置 ==========
    DANGER_THRESHOLD = 20.0
    DANGER_THRESHOLD_POST = 25.0
    DISTANCE_FEATURE_NORM = 200.0
    MAX_DISTANCE_REWARD = 0.5


class CurriculumConfig:
    """课程学习配置"""

    # 训练地图ID列表
    TRAIN_MAP_IDS = [1, 2, 3, 10, 5, 6, 7, 8]

    # 验证地图ID列表
    VALID_MAP_IDS = [9, 4]

    # 验证间隔（局数）
    VALIDATION_INTERVAL = 100

    # ========== 阶段划分（按episode数） ==========
    STAGE0_END = 30000
    STAGE1_END = 40000
    STAGE2_END = 45000
    # > 45000: Stage 3

    # ========== Stage 0: 热身稳定 ==========
    STAGE0_TREASURE_NUM = (8, 10)
    STAGE0_BUFF_NUM = (0, 1)
    STAGE0_MONSTER2_SPAWN = (500, 700)
    STAGE0_SPEED_UP = (700, 900)

    # ========== Stage 1: 中期压力 ==========
    STAGE1_TREASURE_NUM = (8, 10)
    STAGE1_BUFF_NUM = (0, 1)
    STAGE1_MONSTER2_SPAWN = (400, 600)
    STAGE1_SPEED_UP = (600, 800)

    # ========== Stage 2: 后期加速生存 ==========
    STAGE2_TREASURE_NUM = (8, 10)
    STAGE2_BUFF_NUM = (0, 1)
    STAGE2_MONSTER2_SPAWN = (300, 500)
    STAGE2_SPEED_UP = (500, 700)

    # ========== Stage 3: 困难泛化 ==========
    STAGE3_TREASURE_NUM = (8, 10)
    STAGE3_BUFF_NUM = (0, 1)
    STAGE3_MONSTER2_SPAWN = (200, 400)
    STAGE3_SPEED_UP = (400, 600)
