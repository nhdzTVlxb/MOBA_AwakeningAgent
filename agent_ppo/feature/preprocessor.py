#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

import math
import numpy as np
from collections import deque
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
        
        # 拆分为两个历史位置变量
        self.last_hero_pos_for_mask = None   # 专门给 action mask 用
        self.last_hero_pos_for_reward = None # 专门给 reward 位移判断用
        
        # 宝箱和buff计数（用于增量奖励）
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_min_dist = None              # 改为None，首帧不计算delta
        self.same_action_count = 0
        self.last_move_action = -1

        self.map_manager = MapManager()
        self.buff = OrganManager('buff')
        self.treasures = [OrganManager('treasure', i + 1) for i in range(Config.TREASURE_NUM)]
        
        # 历史怪物位置（真正存储上一帧）
        self.last_monsters_pos = []
        self.current_monsters_pos = []
        
        self._visit_map = np.zeros((128, 128), np.float32)
        
        # 距离探索奖励相关
        self.start_pos = None
        self.last_distance_from_start = 0.0
        
        # Z型惩罚相关
        self._action_sequence = []

    def set_start_pos(self, hero_pos):
        """设置起始位置（对局开始时调用）"""
        self.start_pos = hero_pos.copy()
        self.last_distance_from_start = 0.0

    def feature_process(self, env_obs, last_action):
        self.last_action = last_action
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])

        self.step_no = observation.get("step_no", 0)
        self.max_step = env_info.get("max_step", Config.MAX_STEPS)

        hero_info, hero_pos = get_hero_info_and_pos(frame_state)

        # 自动初始化 start_pos
        if self.start_pos is None:
            self.start_pos = hero_pos.copy()
            self.last_distance_from_start = 0.0

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

        # 先保存当前怪物位置到临时变量
        monsters = frame_state.get('monsters', [])
        self.current_monsters_pos = []
        for m in monsters:
            m_pos = m.get('pos', {})
            self.current_monsters_pos.append([m_pos.get('x', 0), m_pos.get('z', 0)])

        # ========== 获取 score_info 用于宝箱/buff增量奖励 ==========
        score_info = frame_state.get('score_info', {})
        current_treasure_count = score_info.get('treasure_collected_count', 0) if score_info else 0
        current_buff_count = score_info.get('buff_count', 0) if score_info else 0
        
        # 计算增量
        treasure_get = current_treasure_count - self.last_treasure_count
        buff_get = current_buff_count - self.last_buff_count

        # 计算距离起始位置的欧几里得距离
        distance_from_start = 0.0
        if self.start_pos is not None:
            dx = hero_pos[0] - self.start_pos[0]
            dz = hero_pos[1] - self.start_pos[1]
            distance_from_start = math.sqrt(dx*dx + dz*dz)

        # 1. 英雄特征 (6维)
        hero_feat = np.array([
            norm(hero_pos[0], -128, 128),
            norm(hero_pos[1], -128, 128),
            float(talent.get('status', 0) == 1),
            norm(talent.get('cooldown', 0), 0, max(self.talent_max_cd, 1)),
            float(hero_info.get('buff_remain_time', 0) > 0),
            norm(distance_from_start, 0, Config.DISTANCE_FEATURE_NORM),
        ], dtype=np.float32)

        # 2. 怪物特征 (10维)
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

        # ========== 计算奖励 ==========
        total_reward, shaped_reward, distance_reward = self._get_reward(
            hero_pos, monsters, treasure_get, buff_get
        )

        # 更新历史计数
        self.last_treasure_count = current_treasure_count
        self.last_buff_count = current_buff_count

        # 计算视野内可见宝箱比例
        visible_treasure_ratio = self._get_visible_treasure_ratio(hero_pos)

        # 更新 reward 专用的历史位置
        self.last_hero_pos_for_reward = hero_pos.copy()
        
        # 奖励计算完成后，更新历史怪物位置
        self.last_monsters_pos = self.current_monsters_pos.copy()

        return feature, legal_action, [total_reward], shaped_reward, visible_treasure_ratio, distance_reward

    def _get_action_mask(self, hero_info, hero_pos):
        mask = [1] * Config.ACTION_NUM

        # 撞墙检测 - 使用 mask 专用的历史位置
        if self.last_hero_pos_for_mask is not None and self.last_action != -1:
            dx = self.last_hero_pos_for_mask[0] - hero_pos[0]
            dz = self.last_hero_pos_for_mask[1] - hero_pos[1]
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

        # 更新 mask 专用的历史位置
        self.last_hero_pos_for_mask = hero_pos.copy()
        return mask

    # ==================== 奖励函数增强方法 ====================
    
    def _calc_openness_score(self, hero_pos):
        """计算某个位置的开阔度评分（0-1）"""
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        walkable = 0
        for dx, dz in directions:
            nx, nz = x + dx, z + dz
            if 0 <= nx < 128 and 0 <= nz < 128:
                if self.map_manager.obstacles[nx, nz] <= 0:
                    walkable += 1
        
        return walkable / 8.0

    def _calc_corridor_reward_detailed(self, hero_pos):
        """详细版开阔度奖励"""
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        walkable_count = 0
        total_depth = 0
        depth_list = []
        
        for dx, dz in directions:
            depth = 0
            for step in range(1, 11):
                nx, nz = x + dx * step, z + dz * step
                if 0 <= nx < 128 and 0 <= nz < 128:
                    if self.map_manager.obstacles[nx, nz] <= 0:
                        depth += 1
                    else:
                        break
                else:
                    break
            
            if depth > 0:
                walkable_count += 1
                total_depth += depth
                depth_list.append(depth)
        
        if walkable_count == 0:
            return -Config.PENALTY_DEAD_END
        
        direction_ratio = walkable_count / 8.0
        avg_depth = total_depth / walkable_count
        depth_score = min(avg_depth / 10.0, 1.0)
        long_path_count = sum(1 for d in depth_list if d > 5)
        openness_score = min(long_path_count / 4.0, 1.0)
        
        openness = direction_ratio * 0.3 + depth_score * 0.4 + openness_score * 0.3
        
        if openness > 0.7:
            return Config.REW_CORRIDOR * openness
        elif openness < 0.3:
            return -Config.PENALTY_DEAD_END * (1 - openness)
        return openness * 0.05

    def _calc_pinch_penalty_detailed(self, hero_pos, monsters):
        """详细版包夹惩罚 - 使用 self.last_monsters_pos 作为历史位置"""
        if len(monsters) < 2:
            return 0.0
        
        m1_pos = [monsters[0].get('pos', {}).get('x', 0), monsters[0].get('pos', {}).get('z', 0)]
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        
        v1 = [m1_pos[0] - hero_pos[0], m1_pos[1] - hero_pos[1]]
        v2 = [m2_pos[0] - hero_pos[0], m2_pos[1] - hero_pos[1]]
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = max(np.linalg.norm(v1), 1e-6)
        norm2 = max(np.linalg.norm(v2), 1e-6)
        cos_angle = dot / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle) * 180 / math.pi
        
        distance_factor = 1.0
        if norm1 < 20 and norm2 < 20:
            distance_factor = 2.0
        elif norm1 < 30 and norm2 < 30:
            distance_factor = 1.5
        
        # 使用真正的历史怪物位置判断逼近
        speed_factor = 1.0
        if len(self.last_monsters_pos) >= 2:
            last_m1 = self.last_monsters_pos[0] if len(self.last_monsters_pos) > 0 else m1_pos
            last_m2 = self.last_monsters_pos[1] if len(self.last_monsters_pos) > 1 else m2_pos
            
            dist1_prev = np.linalg.norm([last_m1[0] - hero_pos[0], last_m1[1] - hero_pos[1]])
            dist2_prev = np.linalg.norm([last_m2[0] - hero_pos[0], last_m2[1] - hero_pos[1]])
            
            if norm1 < dist1_prev and norm2 < dist2_prev:
                speed_factor = 1.5
        
        if angle > 120 and norm1 < 35 and norm2 < 35:
            pinch_severity = (angle / 180.0) * distance_factor * speed_factor
            return -Config.PENALTY_PINCH * pinch_severity * (2 - min(norm1, norm2) / 35)
        
        return 0.0

    def _calc_second_monster_penalty(self, hero_pos, monsters):
        """计算第二只怪压力惩罚"""
        if len(monsters) < 2:
            return 0.0
        
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        dx = m2_pos[0] - hero_pos[0]
        dz = m2_pos[1] - hero_pos[1]
        dist = math.sqrt(dx*dx + dz*dz)
        
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = 25 if is_speedup else 20
        
        if dist < threshold:
            return -Config.PENALTY_SECOND_MONSTER * (1 - dist / threshold)
        return 0.0

    def _calc_pre_speedup_bonus(self):
        """计算临近加速缓冲奖励"""
        steps_to_speedup = Config.MONSTER_SPEED_UP_STEP - self.step_no
        if 0 < steps_to_speedup <= 50:
            bonus = Config.REW_PRE_SPEEDUP * (1 - steps_to_speedup / 50)
            if self.last_min_dist is not None and self.last_min_dist < 20:
                bonus *= 2
            return bonus
        return 0.0

    def _calc_flash_reward_detailed(self, hero_pos, min_dist, treasure_get):
        """
        详细版闪现奖励/惩罚
        
        Args:
            hero_pos: 当前位置
            min_dist: 当前最近怪物距离
            treasure_get: 这一步是否拿到了宝箱（增量）
        """
        if self.last_action < 8:
            return 0.0
        
        reward = 0.0
        
        # 检查是否远离了怪物
        if self.last_min_dist is not None:
            if min_dist > self.last_min_dist + 5:
                reward += Config.REW_FLASH_ESCAPE
            elif min_dist <= self.last_min_dist:
                reward -= Config.PENALTY_FLASH_ABUSE * 0.5
        
        # 检查是否到了更开阔的位置
        current_openness = self._calc_openness_score(hero_pos)
        if self.last_hero_pos_for_reward is not None:
            last_openness = self._calc_openness_score(self.last_hero_pos_for_reward)
            if current_openness > last_openness + 0.3:
                reward += Config.REW_FLASH_ESCAPE * 0.5
            elif current_openness < last_openness - 0.3:
                reward -= Config.PENALTY_FLASH_ABUSE * 0.3
        
        # 修复2: 检查这一步是否拿到了宝箱（使用增量，不是累计值）
        if treasure_get > 0:
            reward += Config.REW_TREASURE * 0.05 #old 0.2
        
        # 闪现后进入危险区
        if min_dist < 10:
            reward -= Config.PENALTY_FLASH_ABUSE
        
        return reward

    def _calc_danger_penalty(self, min_dist):
        """计算危险惩罚（非线性）"""
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = Config.DANGER_THRESHOLD_POST if is_speedup else Config.DANGER_THRESHOLD
        
        if min_dist < threshold:
            penalty = Config.REW_MONSTER_DISTANCE * 2 * (1 - min_dist / threshold)
            return -penalty
        return 0.0

    def _calc_dead_end_penalty(self, hero_pos):
        """详细版死角惩罚"""
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        main_directions = [(1,0), (-1,0), (0,1), (0,-1)]
        depths = []
        
        for dx, dz in main_directions:
            depth = 0
            for step in range(1, 6):
                nx, nz = x + dx * step, z + dz * step
                if 0 <= nx < 128 and 0 <= nz < 128:
                    if self.map_manager.obstacles[nx, nz] <= 0:
                        depth += 1
                    else:
                        break
                else:
                    break
            depths.append(depth)
        
        depths_sorted = sorted(depths, reverse=True)
        
        if depths_sorted[0] > 3 and depths_sorted[1] < 2:
            return -Config.PENALTY_DEAD_END * 1.5
        elif depths_sorted[0] > 2 and depths_sorted[1] < 2:
            return -Config.PENALTY_DEAD_END
        
        return 0.0

    def _calc_repeat_explore_penalty(self, hero_pos):
        """计算重复探索惩罚"""
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        x = max(0, min(127, x))
        z = max(0, min(127, z))
        
        self._visit_map[x, z] += 1
        
        total_visits = 0
        count = 0
        for i in range(-4, 5):
            for j in range(-4, 5):
                nx, nz = x + i, z + j
                if 0 <= nx < 128 and 0 <= nz < 128:
                    total_visits += self._visit_map[nx, nz]
                    count += 1
        
        avg_visits = total_visits / max(count, 1)
        
        if avg_visits > 5:
            return -0.05 * (avg_visits - 4)
        return 0.0

    def _calc_distance_explore_reward(self, hero_pos):
        """欧几里得距离探索奖励"""
        if self.start_pos is None:
            return 0.0
        
        dx = hero_pos[0] - self.start_pos[0]
        dz = hero_pos[1] - self.start_pos[1]
        current_distance = math.sqrt(dx*dx + dz*dz)
        
        distance_increase = current_distance - self.last_distance_from_start
        reward = 0.0
        
        if distance_increase > 0:
            reward = Config.REW_DISTANCE_EXPLORE * min(distance_increase, 10.0)
            reward = min(reward, Config.MAX_DISTANCE_REWARD)
        
        self.last_distance_from_start = current_distance
        return reward

    def _calc_zigzag_penalty(self):
        """Z型走/抽搐惩罚 - 检测相反方向往返"""
        if self.last_action < 0 or self.last_action >= 8:
            return 0.0
        
        self._action_sequence.append(self.last_action)
        if len(self._action_sequence) > 10:
            self._action_sequence.pop(0)
        
        if len(self._action_sequence) >= 4:
            # 相反方向映射
            opposite_map = {
                0: 4, 4: 0,  # 右左
                1: 5, 5: 1,  # 下上
                2: 6, 6: 2,  # 左右
                3: 7, 7: 3,  # 上下
            }
            
            a = self._action_sequence[-4]
            b = self._action_sequence[-3]
            c = self._action_sequence[-2]
            d = self._action_sequence[-1]
            
            # 检测 A-B-A-B 模式且 B 是 A 的相反方向
            if (a == c and b == d and opposite_map.get(a) == b):
                return -Config.PENALTY_ZIGZAG
        
        return 0.0

    def _calc_monster_distance_shaping(self, min_dist):
        """
        修复3: 怪物距离 shaping - 使用距离增量，远离怪物给正奖励
        首帧不计算（last_min_dist为None时直接返回0）
        """
        # 首帧不计算delta奖励
        if self.last_min_dist is None:
            return 0.0
        
        # 计算距离变化
        delta = min_dist - self.last_min_dist
        
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        shaping_weight = 1.5 if is_speedup else 1.0
        
        if delta > 0:
            # 远离怪物，给正奖励
            reward = Config.REW_MONSTER_DISTANCE * shaping_weight * min(delta / 10.0, 1.0)
            return reward
        elif delta < 0:
            # 靠近怪物，给负奖励（惩罚）
            penalty = Config.REW_MONSTER_DISTANCE * shaping_weight * min(-delta / 10.0, 1.0)
            return -penalty
        
        return 0.0

    def _get_visible_treasure_ratio(self, hero_pos):
        """计算当前视野内可见宝箱的比例"""
        visible_count = 0
        total_count = len(self.treasures)
        
        if total_count == 0:
            return 0.0
        
        for treasure in self.treasures:
            if treasure.found and treasure.available:
                dist = np.linalg.norm(treasure.pos - np.array(hero_pos, np.float32))
                if dist <= 10.0:
                    visible_count += 1
        
        return visible_count / total_count

    def _get_reward(self, hero_pos, monsters, treasure_get, buff_get):
        """计算奖励 - 完整修复版"""
        base_reward = Config.REW_STEP + Config.REW_SURVIVE
        shaped_reward = 0.0
        
        # ========== 怪物距离计算 ==========
        min_dist = 1000
        for m in monsters:
            m_pos = m.get('pos', {})
            dx = m_pos.get('x', 0) - hero_pos[0]
            dz = m_pos.get('z', 0) - hero_pos[1]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < min_dist:
                min_dist = dist
        
        # ========== 1. 危险惩罚 ==========
        shaped_reward += self._calc_danger_penalty(min_dist)
        
        # ========== 2. 怪物距离 shaping（首帧不计算） ==========
        shaped_reward += self._calc_monster_distance_shaping(min_dist)
        
        # ========== 3. 第二只怪压力惩罚 ==========
        shaped_reward += self._calc_second_monster_penalty(hero_pos, monsters)
        
        # ========== 4. 包夹惩罚 ==========
        shaped_reward += self._calc_pinch_penalty_detailed(hero_pos, monsters)
        
        # ========== 5. 开阔度奖励 ==========
        shaped_reward += self._calc_corridor_reward_detailed(hero_pos)
        
        # ========== 6. 死角惩罚 ==========
        shaped_reward += self._calc_dead_end_penalty(hero_pos)
        
        # ========== 7. 闪现奖励/惩罚 ==========
        shaped_reward += self._calc_flash_reward_detailed(hero_pos, min_dist, treasure_get)
        
        # ========== 8. 临近加速缓冲奖励 ==========
        shaped_reward += self._calc_pre_speedup_bonus()
        
        # ========== 9. 重复探索惩罚 ==========
        shaped_reward += self._calc_repeat_explore_penalty(hero_pos)
        
        # ========== 10. Z型走/抽搐惩罚 ==========
        shaped_reward += self._calc_zigzag_penalty()
        
        # ========== 11. 宝箱接近奖励 ==========
        nearest_treasure_dist = 1000
        for treasure in self.treasures:
            if treasure.available and treasure.pos[0] != -1:
                dist = treasure.real_distance
                if dist < nearest_treasure_dist:
                    nearest_treasure_dist = dist
        
        if nearest_treasure_dist < 1000 and nearest_treasure_dist < min_dist:
            treasure_approach = 0.05 * (1 - nearest_treasure_dist / 100)
            shaped_reward += treasure_approach
        
        # ========== 12. 宝箱拾取奖励（修复1：恢复增量奖励） ==========
        if treasure_get > 0:
            treasure_pickup = Config.REW_TREASURE * treasure_get + Config.REW_SURVIVE * 10
            shaped_reward += treasure_pickup
        
        # ========== 13. 增益拾取奖励（修复1：恢复增量奖励） ==========
        if buff_get > 0:
            shaped_reward += Config.REW_SURVIVE * 20
        
        # ========== 14. 连续相同动作惩罚 ==========
        current_move_action = self.last_action % 8 if self.last_action != -1 else -1
        if current_move_action != -1:
            if current_move_action == self.last_move_action:
                self.same_action_count += 1
            else:
                self.same_action_count = 0
            
            if self.same_action_count >= 3:
                shaped_reward -= 0.05 * (self.same_action_count - 2)
        self.last_move_action = current_move_action
        
        # ========== 15. 撞墙惩罚 ==========
        if self.last_action != -1 and self.last_action < 8:
            if self.last_hero_pos_for_reward is not None:
                dx = self.last_hero_pos_for_reward[0] - hero_pos[0]
                dz = self.last_hero_pos_for_reward[1] - hero_pos[1]
                if abs(dx) < 0.1 and abs(dz) < 0.1:
                    shaped_reward -= Config.PENALTY_HIT_WALL
        
        # ========== 16. 欧几里得距离探索奖励 ==========
        distance_reward = self._calc_distance_explore_reward(hero_pos)
        shaped_reward += distance_reward
        
        # 更新历史距离
        self.last_min_dist = min_dist
        
        total_reward = base_reward + shaped_reward
        
        return total_reward, shaped_reward, distance_reward
