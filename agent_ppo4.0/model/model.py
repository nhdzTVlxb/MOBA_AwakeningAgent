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

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Structured feature encoder + Actor/Critic dual heads.

    按语义拆分观测后的结构化编码器 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_memory_map_ppo"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.feature_splits = Config.FEATURE_SPLIT_SHAPE
        hero_dim, monster_dim, _, treasure_dim, map_dim, legal_dim, progress_dim = self.feature_splits
        control_dim = legal_dim + progress_dim
        monster_pair_dim = monster_dim * 2

        self.hero_encoder = nn.Sequential(
            make_fc_layer(hero_dim, Config.HERO_ENCODER_DIM),
            nn.ReLU(),
        )
        self.monster_encoder = nn.Sequential(
            make_fc_layer(monster_pair_dim, Config.MONSTER_ENCODER_DIM),
            nn.ReLU(),
        )
        self.treasure_encoder = nn.Sequential(
            make_fc_layer(treasure_dim, Config.TREASURE_ENCODER_DIM),
            nn.ReLU(),
        )

        self.map_shape = (
            Config.LOCAL_MAP_CHANNEL,
            Config.LOCAL_MAP_SIZE,
            Config.LOCAL_MAP_SIZE,
        )
        expected_map_dim = self.map_shape[0] * self.map_shape[1] * self.map_shape[2]
        if map_dim != expected_map_dim:
            raise ValueError(f"map feature dim mismatch: {map_dim} != {expected_map_dim}")

        self.map_encoder = nn.Sequential(
            nn.Conv2d(Config.LOCAL_MAP_CHANNEL, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(32 * 3 * 3, Config.MAP_ENCODER_DIM),
            nn.ReLU(),
        )
        self.control_encoder = nn.Sequential(
            make_fc_layer(control_dim, Config.CONTROL_ENCODER_DIM),
            nn.ReLU(),
        )

        fusion_dim = (
            Config.HERO_ENCODER_DIM
            + Config.MONSTER_ENCODER_DIM
            + Config.TREASURE_ENCODER_DIM
            + Config.MAP_ENCODER_DIM
            + Config.CONTROL_ENCODER_DIM
        )
        self.backbone = nn.Sequential(
            make_fc_layer(fusion_dim, Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            make_fc_layer(Config.FUSION_HIDDEN_DIM, Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(Config.FUSION_HIDDEN_DIM, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(Config.FUSION_HIDDEN_DIM, value_num)

    def forward(self, obs, inference=False):
        hero_feat, monster_1, monster_2, treasure_feat, map_feat, legal_action, progress_feat = torch.split(
            obs, self.feature_splits, dim=1
        )
        monster_feat = torch.cat([monster_1, monster_2], dim=1)
        control_feat = torch.cat([legal_action, progress_feat], dim=1)
        map_feat = map_feat.view(-1, *self.map_shape)

        hidden = torch.cat(
            [
                self.hero_encoder(hero_feat),
                self.monster_encoder(monster_feat),
                self.treasure_encoder(treasure_feat),
                self.map_encoder(map_feat),
                self.control_encoder(control_feat),
            ],
            dim=1,
        )
        hidden = self.backbone(hidden)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
