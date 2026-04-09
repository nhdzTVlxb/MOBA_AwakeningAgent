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
        self.last_hero_pos = None
        self.last_treasure_count = 0
        self._last_buff_count = 0
        self._last_min_dist = 1000
        self.same_action_count = 0
        self.last_move_action = -1

        self.map_manager = MapManager()
        self.buff = OrganManager('buff')
        self.treasures = [OrganManager('treasure', i + 1) for i in range(Config.TREASURE_NUM)]
        
        # 奖励函数增强所需的状态变量
        self._last_monsters_pos = []
        self._visit_map = np.zeros((128, 128), np.float32)

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

        # 保存怪物位置供奖励函数使用
        self._last_monsters_pos = []
        for m in monsters:
            m_pos = m.get('pos', {})
            self._last_monsters_pos.append([m_pos.get('x', 0), m_pos.get('z', 0)])

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

        # 计算奖励（返回 total_reward 和 shaped_reward）
        total_reward, shaped_reward = self._get_reward(hero_pos, frame_state)

        # 计算视野内可见宝箱比例（用于监控）
        visible_treasure_ratio = self._get_visible_treasure_ratio(hero_pos)

        return feature, legal_action, [total_reward], shaped_reward, visible_treasure_ratio

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
        """
        详细版开阔度奖励（corridor reward）
        根据周围8个方向的通路长度、平均可走深度以及附近区域是否开阔来计算
        
        参考文献描述：
        - 根据当前位置周围几个方向的通路长度
        - 平均可走深度
        - 附近区域是否开阔
        """
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        # 检查8个方向
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        walkable_count = 0
        total_depth = 0
        depth_list = []
        
        for dx, dz in directions:
            # 计算该方向的可走深度（最大10步）
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
        
        # 1. 可走方向比例
        direction_ratio = walkable_count / 8.0
        
        # 2. 平均深度（归一化到0-1，最大深度10）
        avg_depth = total_depth / walkable_count
        depth_score = min(avg_depth / 10.0, 1.0)
        
        # 3. 开阔度：是否有长通道（深度>5的方向数）
        long_path_count = sum(1 for d in depth_list if d > 5)
        openness_score = min(long_path_count / 4.0, 1.0)
        
        # 综合开阔度
        openness = direction_ratio * 0.3 + depth_score * 0.4 + openness_score * 0.3
        
        if openness > 0.7:
            return Config.REW_CORRIDOR * openness
        elif openness < 0.3:
            return -Config.PENALTY_DEAD_END * (1 - openness)
        return openness * 0.05  # 中等开阔度给少量奖励

    def _calc_pinch_penalty_detailed(self, hero_pos, monsters):
        """
        详细版包夹惩罚
        根据两只怪物与英雄的相对位置关系、相对速度、合围趋势来判断
        
        参考文献描述：
        - 根据两只怪物和自己的相对位置关系来判断是否有合围趋势
        - 活动空间被两只怪一起压缩的程度
        """
        if len(monsters) < 2:
            return 0.0
        
        # 获取两只怪物的位置
        m1_pos = [monsters[0].get('pos', {}).get('x', 0), monsters[0].get('pos', {}).get('z', 0)]
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        
        # 计算向量
        v1 = [m1_pos[0] - hero_pos[0], m1_pos[1] - hero_pos[1]]
        v2 = [m2_pos[0] - hero_pos[0], m2_pos[1] - hero_pos[1]]
        
        # 1. 夹角（包夹的核心指标）
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = max(np.linalg.norm(v1), 1e-6)
        norm2 = max(np.linalg.norm(v2), 1e-6)
        cos_angle = dot / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle) * 180 / math.pi
        
        # 2. 距离因子（两只怪越近越危险）
        distance_factor = 1.0
        if norm1 < 20 and norm2 < 20:
            distance_factor = 2.0
        elif norm1 < 30 and norm2 < 30:
            distance_factor = 1.5
        
        # 3. 相对速度（如果有历史数据）
        speed_factor = 1.0
        if hasattr(self, '_last_monsters_pos') and len(self._last_monsters_pos) >= 2:
            # 计算两只怪是否在向英雄靠近
            last_m1 = self._last_monsters_pos[0] if len(self._last_monsters_pos) > 0 else m1_pos
            last_m2 = self._last_monsters_pos[1] if len(self._last_monsters_pos) > 1 else m2_pos
            
            dist1_prev = np.linalg.norm([last_m1[0] - hero_pos[0], last_m1[1] - hero_pos[1]])
            dist2_prev = np.linalg.norm([last_m2[0] - hero_pos[0], last_m2[1] - hero_pos[1]])
            
            if norm1 < dist1_prev and norm2 < dist2_prev:
                speed_factor = 1.5  # 两只都在靠近，更危险
        
        # 综合包夹程度
        if angle > 120 and norm1 < 35 and norm2 < 35:
            pinch_severity = (angle / 180.0) * distance_factor * speed_factor
            return -Config.PENALTY_PINCH * pinch_severity * (2 - min(norm1, norm2) / 35)
        
        return 0.0

    def _calc_second_monster_penalty(self, hero_pos, monsters):
        """
        计算第二只怪压力惩罚
        """
        if len(monsters) < 2:
            return 0.0
        
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        dx = m2_pos[0] - hero_pos[0]
        dz = m2_pos[1] - hero_pos[1]
        dist = math.sqrt(dx*dx + dz*dz)
        
        # 加速后第二只怪的压力更大
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = 25 if is_speedup else 20
        
        if dist < threshold:
            return -Config.PENALTY_SECOND_MONSTER * (1 - dist / threshold)
        return 0.0

    def _calc_pre_speedup_bonus(self):
        """
        计算临近加速缓冲奖励
        在加速前N步提前鼓励拉距离
        """
        steps_to_speedup = Config.MONSTER_SPEED_UP_STEP - self.step_no
        if 0 < steps_to_speedup <= 50:
            # 离加速越近，奖励系数越高
            bonus = Config.REW_PRE_SPEEDUP * (1 - steps_to_speedup / 50)
            # 如果离怪物近，额外奖励拉距离
            if hasattr(self, '_last_min_dist'):
                if self._last_min_dist < 20:
                    bonus *= 2
            return bonus
        return 0.0

    def _calc_flash_reward_detailed(self, hero_pos, monsters, min_dist):
        """
        详细版闪现奖励/惩罚
        检查闪现后是否：
        - 明显拉开了与怪物的距离
        - 改善了站位（到了更开阔的位置）
        - 拿到了资源
        
        参考文献描述：
        - 如果使用闪现后，和怪物的距离明显拉开，或者自己到了更安全、更开阔的位置，就给奖励
        - 如果使用闪现后既没有明显脱险，也没有改善站位、拿到资源，就给惩罚
        """
        if self.last_action < 8:
            return 0.0
        
        reward = 0.0
        
        # 1. 检查是否远离了怪物
        if hasattr(self, '_last_min_dist'):
            if min_dist > self._last_min_dist + 5:
                # 成功脱险
                reward += Config.REW_FLASH_ESCAPE
            elif min_dist <= self._last_min_dist:
                # 闪现无效或更糟
                reward -= Config.PENALTY_FLASH_ABUSE * 0.5
        
        # 2. 检查是否到了更开阔的位置
        if hasattr(self, '_last_hero_pos'):
            # 计算当前位置的开阔度
            current_openness = self._calc_openness_score(hero_pos)
            last_openness = self._calc_openness_score(self._last_hero_pos)
            
            if current_openness > last_openness + 0.3:
                reward += Config.REW_FLASH_ESCAPE * 0.5
            elif current_openness < last_openness - 0.3:
                reward -= Config.PENALTY_FLASH_ABUSE * 0.3
        
        # 3. 检查是否拿到了宝箱或buff（通过前后帧比较）
        if hasattr(self, '_last_treasure_count'):
            if self.last_treasure_count > self._last_treasure_count:
                reward += Config.REW_TREASURE * 0.2  # 闪现拿宝箱给额外奖励
        
        # 4. 检查是否进入了更危险的区域
        if min_dist < 10 and self.last_action >= 8:
            # 闪现后反而更危险（距离<10），给惩罚
            reward -= Config.PENALTY_FLASH_ABUSE
        
        return reward

    def _calc_danger_penalty(self, min_dist):
        """
        计算危险惩罚（非线性，加速后阈值更高）
        """
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = Config.DANGER_THRESHOLD_POST if is_speedup else Config.DANGER_THRESHOLD
        
        if min_dist < threshold:
            # 非线性惩罚：越近惩罚越重
            penalty = Config.REW_MONSTER_DISTANCE * 2 * (1 - min_dist / threshold)
            return -penalty
        return 0.0

    def _calc_dead_end_penalty(self, hero_pos):
        """
        详细版死角惩罚
        根据附近是否开阔、内圈是否容易转身来判断
        
        参考文献描述：
        - 根据当前位置附近是否开阔、内圈是否容易转身来判断
        - 越像死角或短胡同，惩罚越重
        """
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        # 检查4个主要方向的可走深度
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
        
        # 判断是否为死角：只有一个方向有较长通路，其他方向都很短
        depths_sorted = sorted(depths, reverse=True)
        
        if depths_sorted[0] > 3 and depths_sorted[1] < 2:
            # 只有一个方向可走，典型死角
            return -Config.PENALTY_DEAD_END * 1.5
        elif depths_sorted[0] > 2 and depths_sorted[1] < 2:
            # 接近死角
            return -Config.PENALTY_DEAD_END
        
        return 0.0

    def _calc_repeat_explore_penalty(self, hero_pos):
        """
        计算重复探索惩罚
        基于访问历史，如果在同一个区域停留太久就惩罚
        """
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        # 确保坐标在范围内
        x = max(0, min(127, x))
        z = max(0, min(127, z))
        
        # 更新访问记录
        self._visit_map[x, z] += 1
        
        # 计算周围9x9区域的访问次数
        total_visits = 0
        count = 0
        for i in range(-4, 5):
            for j in range(-4, 5):
                nx, nz = x + i, z + j
                if 0 <= nx < 128 and 0 <= nz < 128:
                    total_visits += self._visit_map[nx, nz]
                    count += 1
        
        avg_visits = total_visits / max(count, 1)
        
        # 平均访问次数过高说明在重复绕路
        if avg_visits > 5:
            return -0.05 * (avg_visits - 4)
        return 0.0

    def _get_visible_treasure_ratio(self, hero_pos):
        """
        计算当前视野内可见宝箱的比例
        """
        visible_count = 0
        total_count = len(self.treasures)
        
        if total_count == 0:
            return 0.0
        
        for treasure in self.treasures:
            if treasure.found and treasure.available:
                # 计算距离
                dist = np.linalg.norm(treasure.pos - np.array(hero_pos, np.float32))
                if dist <= 10.0:  # 视野范围
                    visible_count += 1
        
        return visible_count / total_count

    def _get_reward(self, hero_pos, frame_state):
        """计算奖励 - 详细版，包含所有参考文献要求的奖励项"""
        # 基础奖励
        base_reward = Config.REW_STEP + Config.REW_SURVIVE
        shaped_reward = 0.0
        
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
        
        # ========== 2. 危险惩罚（非线性） ==========
        shaped_reward += self._calc_danger_penalty(min_dist)
        
        # ========== 3. 怪物距离shaping ==========
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        shaping_weight = 1.5 if is_speedup else 1.0
        shaped_reward += Config.REW_MONSTER_DISTANCE * shaping_weight * (1 - min(min_dist / 50, 1.0))
        
        # ========== 4. 第二只怪压力惩罚 ==========
        shaped_reward += self._calc_second_monster_penalty(hero_pos, monsters)
        
        # ========== 5. 包夹惩罚（详细版） ==========
        shaped_reward += self._calc_pinch_penalty_detailed(hero_pos, monsters)
        
        # ========== 6. 开阔度奖励（详细版） ==========
        shaped_reward += self._calc_corridor_reward_detailed(hero_pos)
        
        # ========== 7. 死角惩罚 ==========
        shaped_reward += self._calc_dead_end_penalty(hero_pos)
        
        # ========== 8. 闪现奖励/惩罚（详细版） ==========
        shaped_reward += self._calc_flash_reward_detailed(hero_pos, monsters, min_dist)
        
        # ========== 9. 临近加速缓冲奖励 ==========
        shaped_reward += self._calc_pre_speedup_bonus()
        
        # ========== 10. 重复探索惩罚 ==========
        shaped_reward += self._calc_repeat_explore_penalty(hero_pos)
        
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
        
        # ========== 12. 宝箱拾取奖励 ==========
        score_info = frame_state.get('score_info', {})
        curr_count = score_info.get('treasure_collected_count', 0) if score_info else 0
        treasure_get = curr_count - self.last_treasure_count
        if treasure_get > 0:
            treasure_pickup = Config.REW_TREASURE * treasure_get + Config.REW_SURVIVE * 10
            shaped_reward += treasure_pickup
        self.last_treasure_count = curr_count
        
        # ========== 13. 增益拾取奖励 ==========
        buff_count = score_info.get('buff_count', 0) if score_info else 0
        if hasattr(self, '_last_buff_count'):
            if buff_count > self._last_buff_count:
                shaped_reward += Config.REW_SURVIVE * 20
        self._last_buff_count = buff_count
        
        # ========== 14. 连续相同动作惩罚（减少原地抽搐） ==========
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
            if self.last_hero_pos is not None:
                dx = self.last_hero_pos[0] - hero_pos[0]
                dz = self.last_hero_pos[1] - hero_pos[1]
                if abs(dx) < 0.1 and abs(dz) < 0.1:
                    shaped_reward -= Config.PENALTY_HIT_WALL
        
        # 记录上一帧的怪物距离
        self._last_min_dist = min_dist
        
        total_reward = base_reward + shaped_reward
        
        return total_reward, shaped_reward
