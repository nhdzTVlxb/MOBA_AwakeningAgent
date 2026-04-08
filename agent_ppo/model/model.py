#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_cnn_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

        # 地图编码器: 4×21×21 -> 512
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

        # 合并后
        self.action_head = make_fc_layer(512 + 128, Config.ACTION_NUM)
        self.value_head = make_fc_layer(512 + 128, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        # 分割特征
        map_feat = obs[:, :Config.MAP_FEATURE_DIM].reshape(-1, Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
        other_feat = obs[:, Config.MAP_FEATURE_DIM:]

        # 编码
        map_encoded = self.map_encoder(map_feat)
        other_encoded = self.other_encoder(other_feat)

        # 合并
        combined = torch.cat([map_encoded, other_encoded], dim=1)

        # 输出
        logits = self.action_head(combined)
        value = self.value_head(combined)

        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
