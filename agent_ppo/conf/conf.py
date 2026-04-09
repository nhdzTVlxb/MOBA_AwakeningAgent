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

import numpy as np


class Config:
    # ==================== PPO算法参数 ====================
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.1
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # ==================== 环境参数 ====================
    MAX_STEPS = 2000
    MONSTER2_SPAWN_STEP = 300
    MONSTER_SPEED_UP_STEP = 500
    FLASH_COOLDOWN = 100
    BUFF_DURATION = 50
    BUFF_RESPAWN = 200
    TREASURE_NUM = 10

    # ==================== 奖励参数 ====================
    REW_STEP = 1.5
    REW_TREASURE = 100
    REW_FLASH = 0.1
    REW_SURVIVE = 0.05
    REW_MONSTER_DISTANCE = 0.3
    PENALTY_HIT_WALL = 0.5
    
    # 新增奖励参数
    REW_CORRIDOR = 0.2
    REW_PRE_SPEEDUP = 0.5
    REW_FLASH_ESCAPE = 2.0
    PENALTY_PINCH = 0.3
    PENALTY_DEAD_END = 0.4
    PENALTY_SECOND_MONSTER = 0.2
    PENALTY_FLASH_ABUSE = 0.5
    DANGER_THRESHOLD = 10.0
    DANGER_THRESHOLD_POST = 15.0

    # ==================== 特征维度 ====================
    HERO_FEATURE_DIM = 5
    MONSTER_FEATURE_DIM = 5
    MONSTER_NUM = 2
    MONSTER_TOTAL_DIM = 10
    MAP_CHANNELS = 4
    MAP_SIZE = 21
    MAP_FEATURE_DIM = 4 * 21 * 21
    TREASURE_FEATURE_DIM = 6
    TREASURE_TOTAL_DIM = TREASURE_FEATURE_DIM * TREASURE_NUM
    BUFF_FEATURE_DIM = 6
    ACTION_NUM = 16
    PROGRESS_FEATURE_DIM = 2

    FEATURE_LEN = (
        HERO_FEATURE_DIM + MONSTER_TOTAL_DIM + MAP_FEATURE_DIM +
        TREASURE_TOTAL_DIM + BUFF_FEATURE_DIM + ACTION_NUM + PROGRESS_FEATURE_DIM
    )
    DIM_OF_OBSERVATION = FEATURE_LEN
    VALUE_NUM = 1

    DATA_SPLIT_SHAPE = [
        FEATURE_LEN,      # obs
        ACTION_NUM,       # legal_action
        1,                # act
        1,                # reward
        1,                # done
        1,                # value
        1,                # next_value
        1,                # advantage
        1,                # reward_sum
        ACTION_NUM,       # prob
    ]
    
    # ==================== 网络配置（新增） ====================
    # 实体embedding维度
    EMBEDDING_DIM = 64
    MAP_EMBEDDING_DIM = 128
    NUM_HEADS = 4
    
    # 各实体特征维度
    HERO_RAW_DIM = HERO_FEATURE_DIM
    TREASURE_PER_RAW_DIM = TREASURE_FEATURE_DIM
    MONSTER_PER_RAW_DIM = MONSTER_FEATURE_DIM
    BUFF_RAW_DIM = BUFF_FEATURE_DIM
    MAP_RAW_CHANNELS = MAP_CHANNELS
    MAP_RAW_SIZE = MAP_SIZE
    
    # 注意力的键/值维度
    KEY_DIM = 64
    VALUE_DIM = 64


class CurriculumConfig:
    """课程学习配置"""
    
    STAGE0_END = 150
    STAGE0_TREASURE_NUM = (9, 10)
    STAGE0_BUFF_NUM = (1, 2)
    STAGE0_MONSTER2_SPAWN = (220, 300)
    STAGE0_SPEED_UP = (360, 460)
    
    STAGE1_END = 500
    STAGE1_TREASURE_NUM = (8, 10)
    STAGE1_BUFF_NUM = (1, 2)
    STAGE1_MONSTER2_SPAWN = (160, 280)
    STAGE1_SPEED_UP = (240, 420)
    
    STAGE2_END = 900
    STAGE2_TREASURE_NUM = (7, 10)
    STAGE2_BUFF_NUM = (1, 2)
    STAGE2_MONSTER2_SPAWN = (120, 220)
    STAGE2_SPEED_UP = (180, 320)
    
    STAGE3_TREASURE_NUM = (6, 10)
    STAGE3_BUFF_NUM = (0, 2)
    STAGE3_MONSTER2_SPAWN = (120, 320)
    STAGE3_SPEED_UP = (140, 420)
    
    VALIDATION_INTERVAL = 10
    TRAIN_MAP_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
    VALID_MAP_IDS = [9, 10]


if __name__ == '__main__':
    print(f"FEATURE_LEN = {Config.FEATURE_LEN}")
    print(f"ACTION_NUM = {Config.ACTION_NUM}")
