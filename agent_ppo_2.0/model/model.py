#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiHeadAttention(nn.Module):
    """多头注意力机制，用于实体间交互"""
    def __init__(self, embed_dim, num_heads, key_dim=None, value_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim if key_dim is not None else embed_dim
        self.value_dim = value_dim if value_dim is not None else embed_dim
        
        assert self.key_dim % num_heads == 0, f"key_dim {self.key_dim} must be divisible by num_heads {num_heads}"
        assert self.value_dim % num_heads == 0, f"value_dim {self.value_dim} must be divisible by num_heads {num_heads}"
        
        self.head_dim_key = self.key_dim // num_heads
        self.head_dim_value = self.value_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, self.key_dim)
        self.k_proj = nn.Linear(embed_dim, self.key_dim)
        self.v_proj = nn.Linear(embed_dim, self.value_dim)
        self.out_proj = nn.Linear(self.value_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.reshape(batch_size, q_len, self.num_heads, self.head_dim_key).transpose(1, 2)
        K = K.reshape(batch_size, k_len, self.num_heads, self.head_dim_key).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim_value).transpose(1, 2)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim_key ** 0.5)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, self.value_dim)
        
        return self.out_proj(attn_output)


class EntityEncoder(nn.Module):
    """实体编码器：将原始特征编码为embedding"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            make_fc_layer(input_dim, output_dim * 2),
            nn.ReLU(),
            make_fc_layer(output_dim * 2, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.encoder(x)


class MapEncoder(nn.Module):
    """地图编码器：将4×21×21地图编码为embedding"""
    def __init__(self, in_channels, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            make_cnn_layer(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            make_cnn_layer(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            make_cnn_layer(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(64 * 6 * 6, output_dim * 2),
            nn.ReLU(),
            make_fc_layer(output_dim * 2, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.cnn(x)


class Model(nn.Module):
    """
    实体分别编码 + 注意力机制的PPO网络
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.embed_dim = Config.EMBEDDING_DIM
        self.map_embed_dim = Config.MAP_EMBEDDING_DIM
        self.num_heads = Config.NUM_HEADS
        
        # 1. 各实体编码器
        self.hero_encoder = EntityEncoder(Config.HERO_RAW_DIM, self.embed_dim)
        self.buff_encoder = EntityEncoder(Config.BUFF_RAW_DIM, self.embed_dim)
        self.treasure_encoder = EntityEncoder(Config.TREASURE_PER_RAW_DIM, self.embed_dim)
        self.monster_encoder = EntityEncoder(Config.MONSTER_PER_RAW_DIM, self.embed_dim)
        self.map_encoder = MapEncoder(Config.MAP_RAW_CHANNELS, self.map_embed_dim)
        
        # 2. 将地图embedding投影到与实体相同的维度
        self.map_proj = nn.Linear(self.map_embed_dim, self.embed_dim)
        
        # 3. 注意力机制
        self.attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            key_dim=64,
            value_dim=64
        )
        
        # 4. 可学习的query
        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # 5. 聚合层
        num_entities = 1 + 1 + 1 + Config.TREASURE_NUM + Config.MONSTER_NUM  # 15
        aggregator_input_dim = self.embed_dim * (num_entities + 1)
        
        self.aggregator = nn.Sequential(
            make_fc_layer(aggregator_input_dim, self.embed_dim * 2),
            nn.ReLU(),
            make_fc_layer(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
        )
        
        # 6. 额外特征：动作掩码 + 进度特征 + 主目标宝箱特征
        extra_dim = Config.ACTION_NUM + Config.PROGRESS_FEATURE_DIM + Config.PRIMARY_TREASURE_FEATURE_DIM
        
        # 7. 输出头
        self.action_head = make_fc_layer(self.embed_dim + extra_dim, Config.ACTION_NUM)
        self.value_head = make_fc_layer(self.embed_dim + extra_dim, Config.VALUE_NUM)
        
    def forward(self, obs, inference=False):
        """
        Args:
            obs: [B, FEATURE_LEN] 原始特征向量
        Returns:
            logits: [B, ACTION_NUM] 动作概率logits
            value: [B, VALUE_NUM] 状态价值
        """
        batch_size = obs.shape[0]
        
        # ========== 1. 解析原始特征（顺序与 preprocessor 一致） ==========
        idx = 0
        
        # 1.1 英雄特征: [B, HERO_FEATURE_DIM]
        hero_raw = obs[:, idx:idx + Config.HERO_FEATURE_DIM]
        idx += Config.HERO_FEATURE_DIM
        
        # 1.2 怪物特征: [B, MONSTER_TOTAL_DIM] -> [B, 2, MONSTER_FEATURE_DIM]
        monster_raw = obs[:, idx:idx + Config.MONSTER_TOTAL_DIM].reshape(
            batch_size, Config.MONSTER_NUM, Config.MONSTER_FEATURE_DIM
        )
        idx += Config.MONSTER_TOTAL_DIM
        
        # 1.3 地图特征: [B, MAP_FEATURE_DIM] -> [B, C, H, W]
        map_feat = obs[:, idx:idx + Config.MAP_FEATURE_DIM].reshape(
            batch_size, Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE
        )
        idx += Config.MAP_FEATURE_DIM
        
        # 1.4 宝箱特征: [B, TREASURE_TOTAL_DIM] -> [B, 10, TREASURE_FEATURE_DIM]
        treasure_raw = obs[:, idx:idx + Config.TREASURE_TOTAL_DIM].reshape(
            batch_size, Config.TREASURE_NUM, Config.TREASURE_FEATURE_DIM
        )
        idx += Config.TREASURE_TOTAL_DIM
        
        # 1.5 主目标宝箱特征: [B, PRIMARY_TREASURE_FEATURE_DIM]（新增）
        primary_treasure_raw = obs[:, idx:idx + Config.PRIMARY_TREASURE_FEATURE_DIM]
        idx += Config.PRIMARY_TREASURE_FEATURE_DIM
        
        # 1.6 buff特征: [B, BUFF_FEATURE_DIM]
        buff_raw = obs[:, idx:idx + Config.BUFF_FEATURE_DIM]
        idx += Config.BUFF_FEATURE_DIM
        
        # 1.7 动作掩码: [B, ACTION_NUM]
        legal_action = obs[:, idx:idx + Config.ACTION_NUM]
        idx += Config.ACTION_NUM
        
        # 1.8 进度特征: [B, PROGRESS_FEATURE_DIM]
        progress = obs[:, idx:idx + Config.PROGRESS_FEATURE_DIM]
        
        # ========== 2. 各实体分别编码 ==========
        # 地图编码
        map_embedding = self.map_encoder(map_feat)
        map_embedding = self.map_proj(map_embedding)
        
        # 英雄编码
        hero_embedding = self.hero_encoder(hero_raw)
        
        # buff编码
        buff_embedding = self.buff_encoder(buff_raw)
        
        # 宝箱编码
        treasure_flat = treasure_raw.reshape(batch_size * Config.TREASURE_NUM, -1)
        treasure_embeddings = self.treasure_encoder(treasure_flat).reshape(
            batch_size, Config.TREASURE_NUM, self.embed_dim
        )
        
        # 怪物编码
        monster_flat = monster_raw.reshape(batch_size * Config.MONSTER_NUM, -1)
        monster_embeddings = self.monster_encoder(monster_flat).reshape(
            batch_size, Config.MONSTER_NUM, self.embed_dim
        )
        
        # ========== 3. 构建实体序列 ==========
        all_entities = torch.cat([
            map_embedding.unsqueeze(1),      # [B, 1, embed_dim]
            hero_embedding.unsqueeze(1),     # [B, 1, embed_dim]
            buff_embedding.unsqueeze(1),     # [B, 1, embed_dim]
            treasure_embeddings,              # [B, 10, embed_dim]
            monster_embeddings,               # [B, 2, embed_dim]
        ], dim=1)  # [B, 15, embed_dim]
        
        # ========== 4. 多头注意力交互 ==========
        query = self.query.expand(batch_size, -1, -1)
        attended = self.attention(query, all_entities, all_entities)
        
        # ========== 5. 聚合所有实体 ==========
        all_flat = torch.cat([
            attended.squeeze(1),               # [B, embed_dim]
            map_embedding,                     # [B, embed_dim]
            hero_embedding,                    # [B, embed_dim]
            buff_embedding,                    # [B, embed_dim]
            treasure_embeddings.reshape(batch_size, -1),  # [B, 10*embed_dim]
            monster_embeddings.reshape(batch_size, -1),   # [B, 2*embed_dim]
        ], dim=1)
        
        aggregated = self.aggregator(all_flat)  # [B, embed_dim]
        
        # ========== 6. 合并额外特征 ==========
        # 新增：把主目标宝箱特征也加进来
        extra = torch.cat([primary_treasure_raw, legal_action, progress], dim=1)
        
        combined = torch.cat([aggregated, extra], dim=1)
        
        # ========== 7. 输出 ==========
        logits = self.action_head(combined)
        value = self.value_head(combined)
        
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()