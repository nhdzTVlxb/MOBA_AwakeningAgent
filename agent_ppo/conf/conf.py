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
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.1  # 从 0.001 改为 0.01-0.1，增加探索
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # ==================== 环境参数 ====================
    MAX_STEPS = 1000
    MONSTER2_SPAWN_STEP = 300
    MONSTER_SPEED_UP_STEP = 500
    FLASH_COOLDOWN = 100
    BUFF_DURATION = 50
    BUFF_RESPAWN = 200
    TREASURE_NUM = 10

    # ==================== 奖励参数 ====================
    REW_STEP = 1.5                    # 每步奖励
    REW_TREASURE = 100                # 宝箱奖励
    REW_FLASH = 0.1                   # 闪现奖励系数
    REW_SURVIVE = 0.05                # 新增：存活奖励（每步额外）
    REW_MONSTER_DISTANCE = 0.3        # 新增：远离怪物奖励系数
    PENALTY_HIT_WALL = 0.5      #from 0.1 to 0.5      # 撞墙惩罚

    # ==================== 特征维度 ====================
    # 英雄特征：位置(2) + 闪现可用(1) + 闪现cd(1) + buff状态(1) = 5
    HERO_FEATURE_DIM = 5

    # 怪物特征：2只 × 5维 = 10
    MONSTER_FEATURE_DIM = 5
    MONSTER_NUM = 2
    MONSTER_TOTAL_DIM = 10

    # 地图特征：4通道 × 21×21 = 1764
    MAP_CHANNELS = 4
    MAP_SIZE = 21
    MAP_FEATURE_DIM = 4 * 21 * 21

    # 宝箱特征：10个 × 6维 = 60
    TREASURE_FEATURE_DIM = 6
    TREASURE_TOTAL_DIM = TREASURE_FEATURE_DIM * TREASURE_NUM

    # buff特征：6维
    BUFF_FEATURE_DIM = 6

    # 动作掩码：16
    ACTION_NUM = 16

    # 进度特征：2维
    PROGRESS_FEATURE_DIM = 2

    # 总特征维度
    FEATURE_LEN = (
        HERO_FEATURE_DIM + MONSTER_TOTAL_DIM + MAP_FEATURE_DIM +
        TREASURE_TOTAL_DIM + BUFF_FEATURE_DIM + ACTION_NUM + PROGRESS_FEATURE_DIM
    )
    DIM_OF_OBSERVATION = FEATURE_LEN

    VALUE_NUM = 1

    # ==================== 数据维度 ====================
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


if __name__ == '__main__':
    print(f"FEATURE_LEN = {Config.FEATURE_LEN}")
    print(f"ACTION_NUM = {Config.ACTION_NUM}")
