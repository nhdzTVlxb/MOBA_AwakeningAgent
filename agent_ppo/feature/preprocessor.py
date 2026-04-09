#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import math
import numpy as np
from agent_ppo.conf.conf import Config


def norm(x, min_x, max_x, eps=1e-8):
    if not isinstance(x, np.ndarray):
        x = np.array(x, np.float32)
    x = np.maximum(np.minimum(max_x, x), min_x)
    return (x - min_x) / (max_x - min_x + eps)


def get_hero_info_and_pos(frame_state):
    """获取英雄信息和位置 - 适配 dict 格式"""
    heroes = frame_state.get('heroes', {})
    if isinstance(heroes, dict):
        hero = heroes
    elif isinstance(heroes, list) and len(heroes) > 0:
        hero = heroes[0]
    else:
        hero = {}
    hero_pos = [hero.get('pos', {}).get('x', 0), hero.get('pos', {}).get('z', 0)]
    return hero, hero_pos


class OrganManager:
    """宝箱/buff位置估计"""
    def __init__(self, name, config_id=None):
        self.name = name
        self.config_id = config_id
        self.found = False
        self.pos = np.array([-1, -1], np.float32)
        self.last_real_distance = -1
        self.real_distance = -1
        self.available = True

    def update(self, organ, hero_pos):
        self.last_real_distance = self.real_distance

        if organ.get('status') == 0:
            self.available = False

        if self.found:
            self.real_distance = np.linalg.norm(self.pos - np.array(hero_pos, np.float32))
            return

        if organ.get('status') != -1:
            pos_data = organ.get('pos', {})
            self.pos = np.array([pos_data.get('x', 0), pos_data.get('z', 0)], np.float32)
            self.found = True
            self.real_distance = np.linalg.norm(self.pos - np.array(hero_pos, np.float32))
        elif self.last_real_distance == -1:
            self.real_distance = 1.0

        if self.last_real_distance == -1:
            self.last_real_distance = self.real_distance

    def get_feature(self, hero_pos):
        if self.pos[0] == -1 or not self.available:
            x = np.full((6,), 0, np.float32)
            x[3] = 1.0
            return x

        relative_pos = [self.pos[0] - hero_pos[0], self.pos[1] - hero_pos[1]]
        delta_distance = max(np.linalg.norm(relative_pos), 1e-4)

        feature = np.array([
            float(self.found),
            norm(relative_pos[0] / delta_distance, -1, 1),
            norm(relative_pos[1] / delta_distance, -1, 1),
            norm(delta_distance, 0, math.sqrt(2) * 128),
            norm(self.pos[0], -128, 128),
            norm(self.pos[1], -128, 128),
        ], np.float32)
        return feature


class MapManager:
    """地图管理（21×21局部地图）"""
    def __init__(self):
        self.obstacles = np.full((128, 128), -1.0, np.float32)
        self.memory = np.zeros((128, 128), np.float32)
        self.treasures = np.zeros((128, 128), np.float32)
        self.buffs = np.zeros((128, 128), np.float32)
        self.hero_pos = None
        self.step = 0

    def update_hero(self, hero_pos):
        self.hero_pos = [int(round(hero_pos[0])), int(round(hero_pos[1]))]
        self.memory[self.hero_pos[0], self.hero_pos[1]] += 1.0
        self.step += 1

    def update_obstacles(self, hero_pos, map_info):
        """更新障碍物地图 - map_info 是二维列表"""
        hero_pos = [int(round(hero_pos[0])), int(round(hero_pos[1]))]
        if not map_info:
            return
        
        if isinstance(map_info[0], list):
            map_grid = np.array(map_info, np.float32)
        else:
            map_grid = np.array([line if isinstance(line, list) else line.get('values', []) for line in map_info], np.float32)
        
        if map_grid.size == 0:
            return
        
        map_size = map_grid.shape[0]
        if map_size == 0:
            return

        for i in range(-10, 11):
            for j in range(-10, 11):
                u, v = hero_pos[0] + i, hero_pos[1] + j
                if u < 0 or u >= 128 or v < 0 or v >= 128:
                    continue

                grid_i = int((i + 10) * map_size / 21)
                grid_j = int((j + 10) * map_size / 21)
                grid_i = max(0, min(map_size - 1, grid_i))
                grid_j = max(0, min(map_size - 1, grid_j))

                if self.obstacles[u, v] == 0:
                    continue
                self.obstacles[u, v] = map_grid[grid_i, grid_j]

    def get_around_feature(self):
        """获取21×21局部地图特征"""
        size = 21
        half = size // 2
        feature = np.zeros((4, size, size), np.float32)

        if self.hero_pos is None:
            return feature

        for i in range(size):
            for j in range(size):
                u = self.hero_pos[0] + i - half
                v = self.hero_pos[1] + j - half
                if u < 0 or u >= 128 or v < 0 or v >= 128:
                    continue

                feature[0, i, j] = self.obstacles[u, v]
                feature[1, i, j] = np.clip(self.memory[u, v] / 10, 0, 1)
                feature[2, i, j] = self.treasures[u, v]
                feature[3, i, j] = self.buffs[u, v]

        return feature
    
    def get_local_memory_avg(self):
        """获取周围21×21区域的平均记忆值（用于探索奖励）"""
        if self.hero_pos is None:
            return 1.0
        
        size = 21
        half = size // 2
        total = 0.0
        count = 0
        
        for i in range(size):
            for j in range(size):
                u = self.hero_pos[0] + i - half
                v = self.hero_pos[1] + j - half
                if 0 <= u < 128 and 0 <= v < 128:
                    total += self.memory[u, v]
                    count += 1
        
        if count == 0:
            return 1.0
        return total / count / 10


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = Config.MAX_STEPS
        self.last_action = -1
        self.last_min_monster_dist_norm = 0.5
        self.talent_max_cd = 0
        self.hit_actions = set()
        self.last_hero_pos = None
        self.last_treasure_count = 0
        self._last_buff_count = 0
        self._last_min_dist = 1000
        self.same_action_count = 0
        self.last_move_action = -1

        self.map_manager = MapManager()
        self.buff = OrganManager('buff')
        self.treasures = [OrganManager('treasure', i + 1) for i in range(Config.TREASURE_NUM)]

    def feature_process(self, env_obs, last_action):
        self.last_action = last_action
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])

        self.step_no = observation.get("step_no", 0)
        self.max_step = env_info.get("max_step", Config.MAX_STEPS)

        hero_info, hero_pos = get_hero_info_and_pos(frame_state)

        talent = hero_info.get('talent', {})
        self.talent_max_cd = max(self.talent_max_cd, talent.get('cooldown', 0))

        # 更新宝箱和buff
        for organ in frame_state.get('organs', []):
            sub_type = organ.get('sub_type', 0)
            if sub_type == 1:
                idx = organ.get('config_id', 1) - 1
                if 0 <= idx < Config.TREASURE_NUM:
                    self.treasures[idx].update(organ, hero_pos)
            elif sub_type == 2:
                self.buff.update(organ, hero_pos)

        self.map_manager.update_hero(hero_pos)
        self.map_manager.update_obstacles(hero_pos, map_info)

        # 1. 英雄特征 (5维)
        hero_feat = np.array([
            norm(hero_pos[0], -128, 128),
            norm(hero_pos[1], -128, 128),
            float(talent.get('status', 0) == 1),
            norm(talent.get('cooldown', 0), 0, max(self.talent_max_cd, 1)),
            float(hero_info.get('buff_remain_time', 0) > 0)
        ], dtype=np.float32)

        # 2. 怪物特征 (10维)
        monsters = frame_state.get('monsters', [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m.get('pos', {})
                m_x = m_pos.get('x', 0)
                m_z = m_pos.get('z', 0)
                dx = m_x - hero_pos[0]
                dz = m_z - hero_pos[1]
                distance = math.sqrt(dx * dx + dz * dz)
                in_view = distance <= 10.0

                monster_feats.extend([
                    float(in_view),
                    norm(dx / max(distance, 1e-4), -1, 1),
                    norm(dz / max(distance, 1e-4), -1, 1),
                    norm(distance, 0, 180),
                    float(self.step_no >= Config.MONSTER_SPEED_UP_STEP)
                ])
            else:
                monster_feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])

        # 3. 地图特征 (1764维)
        map_feat = self.map_manager.get_around_feature().reshape(-1)

        # 4. 宝箱特征 (60维)
        treasure_feats = []
        for i in range(Config.TREASURE_NUM):
            treasure_feats.append(self.treasures[i].get_feature(hero_pos))
        treasure_feats = sorted(treasure_feats, key=lambda x: x[3])
        treasure_feat = np.concatenate(treasure_feats)

        # 5. buff特征 (6维)
        buff_feat = self.buff.get_feature(hero_pos)

        # 6. 动作掩码 (16维)
        legal_action = self._get_action_mask(hero_info, hero_pos)

        # 7. 进度特征 (2维)
        step_norm = norm(self.step_no, 0, self.max_step)
        progress_feat = np.array([step_norm, step_norm], dtype=np.float32)

        # 拼接
        feature = np.concatenate([
            hero_feat,
            np.array(monster_feats, dtype=np.float32),
            map_feat,
            treasure_feat,
            buff_feat,
            np.array(legal_action, dtype=np.float32),
            progress_feat
        ])

        reward = self._get_reward(hero_pos, frame_state)

        return feature, legal_action, [reward]

    def _get_action_mask(self, hero_info, hero_pos):
        mask = [1] * Config.ACTION_NUM

        # 撞墙检测
        if self.last_hero_pos is not None and self.last_action != -1:
            dx = self.last_hero_pos[0] - hero_pos[0]
            dz = self.last_hero_pos[1] - hero_pos[1]
            if abs(dx) < 0.1 and abs(dz) < 0.1:
                self.hit_actions.add(self.last_action % 8)
            else:
                self.hit_actions = set()

        for action in self.hit_actions:
            if action < 8:
                mask[action] = 0

        if sum(mask[:8]) == 0:
            self.hit_actions = set()
            for i in range(8):
                mask[i] = 1

        # 闪现冷却mask
        talent = hero_info.get('talent', {})
        if talent.get('status', 0) == 0:
            for i in range(8):
                mask[i + 8] = 0

        self.last_hero_pos = hero_pos.copy()
        return mask

    def _get_reward(self, hero_pos, frame_state):
        """计算奖励 - 逃命优先，安全时探索"""
        r = Config.REW_STEP + Config.REW_SURVIVE
        
        # ========== 1. 怪物距离计算 ==========
        monsters = frame_state.get('monsters', [])
        min_dist = 1000
        for m in monsters:
            m_pos = m.get('pos', {})
            dx = m_pos.get('x', 0) - hero_pos[0]
            dz = m_pos.get('z', 0) - hero_pos[1]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < min_dist:
                min_dist = dist
        
        # ========== 2. 逃命优先（怪物近时）==========
        if min_dist < 10:
            # 危险！逃命第一
            r += Config.REW_MONSTER_DISTANCE * 2 * (1 - min_dist / 50)
            
            # 危险时用闪现给超高奖励
            if self.last_action >= 8:
                r += 2.0
            
            # 检查闪现后是否远离了怪物
            if hasattr(self, '_last_min_dist') and self.last_action >= 8:
                if min_dist > self._last_min_dist + 5:
                    r += 0.5
            
        # ========== 3. 安全时探索（怪物远时）==========
        elif min_dist > 30:
            # 安全！主动探索
            r += Config.REW_MONSTER_DISTANCE * 0.3 * min(min_dist / 50, 1.0)
            
            # 探索新区域奖励
            local_memory_avg = self.map_manager.get_local_memory_avg()
            if local_memory_avg < 0.1:
                r += 0.4
            elif local_memory_avg > 0.5:
                r -= 0.1
            
            # 宝箱导向奖励
            nearest_treasure_dist = 1000
            for treasure in self.treasures:
                if treasure.available and treasure.pos[0] != -1:
                    dist = treasure.real_distance
                    if dist < nearest_treasure_dist:
                        nearest_treasure_dist = dist
            
            if nearest_treasure_dist < 1000 and nearest_treasure_dist < 100:
                r += 0.05 * (1 - nearest_treasure_dist / 100)
            
            # 闪现奖励
            if self.last_action >= 8:
                r += 0.2
            
        # ========== 4. 中等距离（边逃边探索）==========
        else:
            r += Config.REW_MONSTER_DISTANCE * 0.8 * min(min_dist / 50, 1.0)
            
            # 轻微探索奖励
            local_memory_avg = self.map_manager.get_local_memory_avg()
            if local_memory_avg < 0.1:
                r += 0.15
            
            # 轻微宝箱导向（宝箱比怪物近时才考虑）
            nearest_treasure_dist = 1000
            for treasure in self.treasures:
                if treasure.available and treasure.pos[0] != -1:
                    dist = treasure.real_distance
                    if dist < nearest_treasure_dist:
                        nearest_treasure_dist = dist
            
            if nearest_treasure_dist < 1000 and nearest_treasure_dist < min_dist:
                r += 0.03 * (1 - nearest_treasure_dist / 100)
            
            # 闪现奖励
            if self.last_action >= 8:
                r += 0.5
        
        # 记录上一帧的怪物距离
        self._last_min_dist = min_dist
        
        # ========== 5. 宝箱拾取奖励 ==========
        score_info = frame_state.get('score_info', {})
        curr_count = score_info.get('treasure_collected_count', 0) if score_info else 0
        treasure_get = curr_count - self.last_treasure_count
        if treasure_get > 0:
            r += Config.REW_TREASURE * treasure_get
            r += Config.REW_SURVIVE * 10
        self.last_treasure_count = curr_count
        
        # ========== 6. 增益拾取奖励 ==========
        buff_count = score_info.get('buff_count', 0) if score_info else 0
        if hasattr(self, '_last_buff_count'):
            if buff_count > self._last_buff_count:
                r += Config.REW_SURVIVE * 20
        self._last_buff_count = buff_count
        
        # ========== 7. 连续相同动作惩罚（减少原地抽搐）==========
        current_move_action = self.last_action % 8 if self.last_action != -1 else -1
        if current_move_action != -1:
            if current_move_action == self.last_move_action:
                self.same_action_count += 1
            else:
                self.same_action_count = 0
            
            if self.same_action_count >= 3:
                r -= 0.05 * (self.same_action_count - 2)
        self.last_move_action = current_move_action
        
        # ========== 8. 撞墙惩罚 ==========
        if self.last_action != -1 and self.last_action < 8:
            if self.last_hero_pos is not None:
                dx = self.last_hero_pos[0] - hero_pos[0]
                dz = self.last_hero_pos[1] - hero_pos[1]
                if abs(dx) < 0.1 and abs(dz) < 0.1:
                    r -= Config.PENALTY_HIT_WALL
        
        return r
