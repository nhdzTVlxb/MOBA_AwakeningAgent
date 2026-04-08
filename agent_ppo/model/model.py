#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_cnn_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """Create a convolutional layer with orthogonal initialization.

    创建正交初始化的卷积层。
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """CNN + MLP backbone for map + other features.

    CNN + MLP 骨干网络，处理地图特征和其他特征。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn"
        self.device = device

        # 地图编码器: 4×21×21 -> 512
        # 卷积计算:
        # 输入: 4×21×21
        # Conv2d(4,32,7,2,padding=3): 输出 32×11×11
        # Conv2d(32,64,5,2,padding=2): 输出 64×6×6
        # Conv2d(64,64,3,1,padding=1): 输出 64×6×6
        # Flatten: 64×6×6 = 2304
        # FC: 2304 -> 512
        self.map_encoder = nn.Sequential(
            make_cnn_layer(Config.MAP_CHANNELS, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            make_cnn_layer(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            make_cnn_layer(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(64 * 6 * 6, 512),
            nn.ReLU(),
        )

        # 非地图特征维度
        other_dim = (
            Config.HERO_FEATURE_DIM +
            Config.MONSTER_TOTAL_DIM +
            Config.TREASURE_TOTAL_DIM +
            Config.BUFF_FEATURE_DIM +
            Config.ACTION_NUM +
            Config.PROGRESS_FEATURE_DIM
        )
        self.other_encoder = nn.Sequential(
            make_fc_layer(other_dim, 128),
            nn.ReLU(),
        )

        # 合并后: 512 + 128 = 640
        # Actor head / 策略头
        self.actor_head = make_fc_layer(512 + 128, Config.ACTION_NUM)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(512 + 128, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        """Forward pass.

        前向传播。
        """
        # 分割特征：地图特征和其他特征
        map_feat = obs[:, :Config.MAP_FEATURE_DIM].reshape(-1, Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
        other_feat = obs[:, Config.MAP_FEATURE_DIM:]

        # 编码
        map_encoded = self.map_encoder(map_feat)
        other_encoded = self.other_encoder(other_feat)

        # 合并
        combined = torch.cat([map_encoded, other_encoded], dim=1)

        # 输出
        logits = self.actor_head(combined)
        value = self.critic_head(combined)

        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
