#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# ObsData: feature and legal_action mask / 特征向量与合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action, d_action(greedy), prob, value / 动作、贪心动作、概率、价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: single-frame sample / 单帧样本
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=1,
    done=1,
    value=1,
    next_value=1,
    advantage=1,
    reward_sum=1,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = delta + gamma * lamda * gae
        sample.advantage = gae
        sample.reward_sum = gae + sample.value