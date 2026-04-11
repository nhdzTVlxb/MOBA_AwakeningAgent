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
    """Get hero info and position (supports both dict and list formats)."""
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
    """Treasure/buff position estimator."""
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
    """Map manager for 21x21 local-map features."""
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
        """Update obstacle map from local map_info (2D grid/list)."""
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
        """Get 21x21 local map features around the hero."""
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
        """Return average visit density in the nearby 21x21 area for exploration shaping.

        This computes the mean value of `self.memory` in a 21x21 window centered on
        the hero. Higher values mean the area has been visited more frequently;
        lower values mean the area is relatively unexplored.
        """
        if self.hero_pos is None:
            # No valid hero location: return explored-like value to avoid false exploration bonus.
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
                    if self.obstacles[u, v] <= 0:
                        total += self.memory[u, v]
                        count += 1
        
        if count == 0:
            return 1.0
        # avg visits per walkable cell in the local area
        return total / count


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
        self.monster_last_seen = [0, 0]
        self.monster_last_pos = [[0.0, 0.0], [0.0, 0.0]]

        self.map_manager = MapManager()
        self.buff = OrganManager('buff')
        self.treasures = [OrganManager('treasure', i + 1) for i in range(Config.TREASURE_NUM)]
        
        # State variables used by enhanced reward shaping.
        self._last_monsters_pos = []
        self._visit_map = np.zeros((128, 128), np.float32)
        self._tracked_visible_treasures = {}

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

        # Update tracked treasures and buffs.
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

        # 1. Hero features (5 dims)
        hero_feat = np.array([
            norm(hero_pos[0], -128, 128),
            norm(hero_pos[1], -128, 128),
            float(talent.get('status', 0) == 1),
            norm(talent.get('cooldown', 0), 0, max(self.talent_max_cd, 1)),
            float(hero_info.get('buff_remain_time', 0) > 0)
        ], dtype=np.float32)

        # 2. Monster features (16 dims)
        monsters = frame_state.get('monsters', [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m.get('pos', {})
                m_x = m_pos.get('x', 0)
                m_z = m_pos.get('z', 0)
                self.monster_last_pos[i] = [m_x, m_z]
                self.monster_last_seen[i] = 20
                distance = math.sqrt((m_x - hero_pos[0])**2 + (m_z - hero_pos[1])**2)
                in_view = distance <= 10.0
            else:
                m_x, m_z = self.monster_last_pos[i]
                self.monster_last_seen[i] = max(0, self.monster_last_seen[i] - 1)
                in_view = False

            dx = m_x - hero_pos[0]
            dz = m_z - hero_pos[1]
            distance = math.sqrt(dx * dx + dz * dz)
            
            if self.monster_last_seen[i] > 0:
                monster_feats.extend([
                    float(in_view),
                    norm(dx / max(distance, 1e-4), -1, 1),
                    norm(dz / max(distance, 1e-4), -1, 1),
                    norm(distance, 0, 180),
                    float(self.step_no >= Config.MONSTER_SPEED_UP_STEP),
                    norm(dx, -256, 256),
                    norm(dz, -256, 256),
                    self.monster_last_seen[i] / 20.0
                ])
            else:
                monster_feats.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # Store monster positions for reward calculations.
        self._last_monsters_pos = []
        for m in monsters:
            m_pos = m.get('pos', {})
            self._last_monsters_pos.append([m_pos.get('x', 0), m_pos.get('z', 0)])

        # 3. Map features (1764 dims)
        map_feat = self.map_manager.get_around_feature().reshape(-1)

        # 4. Treasure features (60 dims)
        treasure_feats = []
        for i in range(Config.TREASURE_NUM):
            treasure_feats.append(self.treasures[i].get_feature(hero_pos))
        treasure_feats = sorted(treasure_feats, key=lambda x: x[3])
        treasure_feat = np.concatenate(treasure_feats)

        # 5. Buff features (6 dims)
        buff_feat = self.buff.get_feature(hero_pos)

        # 6. Action mask (16 dims)
        legal_action = self._get_action_mask(hero_info, hero_pos)

        # 7. Progress features (2 dims)
        step_norm = norm(self.step_no, 0, self.max_step)
        progress_feat = np.array([step_norm, step_norm], dtype=np.float32)

        # Concatenate all feature groups.
        feature = np.concatenate([
            hero_feat,
            np.array(monster_feats, dtype=np.float32),
            map_feat,
            treasure_feat,
            buff_feat,
            np.array(legal_action, dtype=np.float32),
            progress_feat
        ])

        # Compute reward (returns total_reward, shaped_reward, and reward_components).
        total_reward, shaped_reward, reward_components = self._get_reward(hero_pos, frame_state)

        # Compute visible treasure ratio in current view (for monitoring).
        visible_treasure_ratio = self._get_visible_treasure_ratio(hero_pos)

        return feature, legal_action, total_reward, shaped_reward, visible_treasure_ratio, reward_components

    def _get_action_mask(self, hero_info, hero_pos):
        mask = [1] * Config.ACTION_NUM

        # Collision check: detect moves that produce almost no displacement.
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

        # Flash cooldown mask.
        talent = hero_info.get('talent', {})
        if talent.get('status', 0) == 0:
            for i in range(8):
                mask[i + 8] = 0

        self.last_hero_pos = hero_pos.copy()
        return mask

    # ==================== Enhanced Reward Helpers ====================
    
    def _calc_openness_score(self, hero_pos):
        """
        Compute openness score (0-1) for a given position.
        
        Math details:
        - Checks the 8 immediately adjacent neighboring cells (distance 1).
        - Sums up how many of these adjacent cells are walkable (obstacle <= 0).
        - Returns the ratio: (walkable cells) / 8.0.
        - A score of 1.0 means perfectly open space, lower scores indicate wall adjacency.
        """
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
        Detailed corridor/openness reward.

        Uses path lengths in 8 directions, average walkable depth, and local
        openness around the hero.

        Math details:
        - Rays are projected in 8 directions up to a max depth of 10 cells until hitting an obstacle or map boundary.
        - direction_ratio = (number of directions with depth > 0) / 8.0.
        - depth_score = min((average depth of all walkable directions) / 10.0, 1.0).
        - openness_score = min((number of directions with depth > 5) / 4.0, 1.0).
        - open_val = direction_ratio * 0.3 + depth_score * 0.4 + openness_score * 0.3.
        - Returns a positive structured reward if combined openness > 0.7, or scales a dead-end penalty if < 0.3.
        """
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        # Check 8 directions.
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        walkable_count = 0
        total_depth = 0
        depth_list = []
        
        for dx, dz in directions:
            # Walkable depth in this direction (up to 10 cells).
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
            return 0.0
        
        # 1) Ratio of walkable directions.
        direction_ratio = walkable_count / 8.0
        
        # 2) Average depth (normalized to 0-1 with max depth 10).
        avg_depth = total_depth / walkable_count
        depth_score = min(avg_depth / 10.0, 1.0)
        
        # 3) Long-path openness: number of directions with depth > 5.
        long_path_count = sum(1 for d in depth_list if d > 5)
        openness_score = min(long_path_count / 4.0, 1.0)
        
        # Combined openness score.
        openness = direction_ratio * 0.3 + depth_score * 0.4 + openness_score * 0.3
        
        if openness > 0.7:
            return Config.REW_CORRIDOR * openness
        return 0  # no reward unless in open space

    def _calc_pinch_penalty_detailed(self, hero_pos, monsters):
        """
        Detailed pinch/encirclement penalty.

        Estimates encirclement risk from relative geometry between two monsters
        and the hero, plus approach trend when historical data is available.

        Literature-aligned criteria:
        - Relative position relationship indicating converging pressure
        - Degree of movement-space compression by two monsters
        """
        if len(monsters) < 2:
            return 0.0
        
        # Positions of two monsters.
        m1_pos = [monsters[0].get('pos', {}).get('x', 0), monsters[0].get('pos', {}).get('z', 0)]
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        
        # Relative vectors from hero to each monster.
        v1 = [m1_pos[0] - hero_pos[0], m1_pos[1] - hero_pos[1]]
        v2 = [m2_pos[0] - hero_pos[0], m2_pos[1] - hero_pos[1]]
        
        # 1) Angle between vectors (core pinch indicator).
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = max(np.linalg.norm(v1), 1e-6)
        norm2 = max(np.linalg.norm(v2), 1e-6)
        cos_angle = dot / (norm1 * norm2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle) * 180 / math.pi
        
        # 2) Distance factor: closer monsters imply higher danger.
        distance_factor = 1.0
        if norm1 < 20 and norm2 < 20:
            distance_factor = 2.0
        elif norm1 < 30 and norm2 < 30:
            distance_factor = 1.5
        
        # 3) Relative approach trend (if history exists).
        speed_factor = 1.0
        if hasattr(self, '_last_monsters_pos') and len(self._last_monsters_pos) >= 2:
            # Check whether both monsters are approaching the hero.
            last_m1 = self._last_monsters_pos[0] if len(self._last_monsters_pos) > 0 else m1_pos
            last_m2 = self._last_monsters_pos[1] if len(self._last_monsters_pos) > 1 else m2_pos
            
            dist1_prev = np.linalg.norm([last_m1[0] - hero_pos[0], last_m1[1] - hero_pos[1]])
            dist2_prev = np.linalg.norm([last_m2[0] - hero_pos[0], last_m2[1] - hero_pos[1]])
            
            if norm1 < dist1_prev and norm2 < dist2_prev:
                speed_factor = 1.5  # Both are approaching -> more dangerous.
        
        # Combined pinch severity.
        if angle > 120 and norm1 < 35 and norm2 < 35:
            pinch_severity = (angle / 180.0) * distance_factor * speed_factor
            return -Config.PENALTY_PINCH * pinch_severity * (2 - min(norm1, norm2) / 35)
        
        return 0.0

    def _calc_second_monster_penalty(self, hero_pos, monsters):
        """
        Compute pressure penalty from the second monster.
        """
        if len(monsters) < 2:
            return 0.0
        
        m2_pos = [monsters[1].get('pos', {}).get('x', 0), monsters[1].get('pos', {}).get('z', 0)]
        dx = m2_pos[0] - hero_pos[0]
        dz = m2_pos[1] - hero_pos[1]
        dist = math.sqrt(dx*dx + dz*dz)
        
        # The second monster is more threatening after speedup.
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = 25 if is_speedup else 20
        
        if dist < threshold:
            return -Config.PENALTY_SECOND_MONSTER * (1 - dist / threshold)
        return 0.0

    def _calc_pre_speedup_bonus(self):
        """
        Compute pre-speedup buffer bonus.

        Encourages early spacing in the N steps before monster speedup.
        """
        steps_to_speedup = Config.MONSTER_SPEED_UP_STEP - self.step_no
        if 0 < steps_to_speedup <= 50:
            # Bonus increases as speedup gets closer.
            bonus = Config.REW_PRE_SPEEDUP * (1 - steps_to_speedup / 50)
            # Extra incentive when staying far away from monsters.
            if hasattr(self, '_last_min_dist'):
                if self._last_min_dist > 40:
                    bonus *= 2
            return bonus
        return 0.0

    def _calc_flash_reward_detailed(self, hero_pos, monsters, min_dist):
        """
        Detailed flash reward/penalty.

        Evaluates whether flash:
        - meaningfully increases distance from monsters,
        - improves positioning (to a more open/safe area),
        - helps secure resources.

        Literature-aligned intent:
        - reward effective defensive/positioning flash usage,
        - penalize flash usage that does not improve safety or outcomes.
        """
        if self.last_action < 8:
            return 0.0
        else:
            return -Config.PENALTY_FLASH_ABUSE

        reward = 0.0
        
        # 1) Check if hero moved farther from monsters.
        if hasattr(self, '_last_min_dist'):
            if min_dist > self._last_min_dist + 5:
                # Successful disengage.
                reward += Config.REW_FLASH_ESCAPE
            elif min_dist <= self._last_min_dist:
                # Flash was ineffective or worse.
                reward -= Config.PENALTY_FLASH_ABUSE * 0.5
        
        # 2) Check if flash improved local openness.
        if hasattr(self, '_last_hero_pos'):
            # Compare openness at current vs previous hero position.
            current_openness = self._calc_openness_score(hero_pos)
            last_openness = self._calc_openness_score(self._last_hero_pos)
            
            if current_openness > last_openness + 0.3:
                reward += Config.REW_FLASH_ESCAPE * 0.5
            elif current_openness < last_openness - 0.3:
                reward -= Config.PENALTY_FLASH_ABUSE * 0.3
        
        # 3) Check if flash likely helped secure resources.
        if hasattr(self, '_last_treasure_count'):
            if self.last_treasure_count > self._last_treasure_count:
                reward += Config.REW_TREASURE * 0.1  # Extra reward when flash helps collect treasure.
        
        # 4) Penalize flash into high danger.
        if min_dist < 10 and self.last_action >= 8:
            # Flash resulted in higher danger (distance < 10).
            reward -= Config.PENALTY_FLASH_ABUSE
        
        return reward

    def _calc_danger_penalty(self, min_dist):
        """
        Compute nonlinear danger penalty (higher threshold after speedup).
        """
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        threshold = Config.DANGER_THRESHOLD_POST if is_speedup else Config.DANGER_THRESHOLD
        
        if min_dist < threshold:
            # Nonlinear: penalty grows faster as monster gets closer.
            penalty = Config.REW_MONSTER_DISTANCE * 2 * (1 - min_dist / threshold)
            return -penalty
        return 0.0

    def _calc_dead_end_penalty(self, hero_pos):
        """
        Detailed dead-end penalty.

        Uses local openness and turning space to detect dead-end-like geometry.

        Literature-aligned intent:
        - evaluate openness and maneuverability around current position,
        - penalize stronger when layout resembles a dead end or short corridor.
        """
        x, z = int(round(hero_pos[0])), int(round(hero_pos[1]))
        
        # Check walkable depth along 4 main directions.
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
        
        # Dead-end heuristic: one long exit and other directions short.
        depths_sorted = sorted(depths, reverse=True)
        
        if depths_sorted[0] > 3 and depths_sorted[1] < 2:
            # Only one meaningful direction is open: strong dead-end.
            return -Config.PENALTY_DEAD_END * 1.5
        elif depths_sorted[0] > 2 and depths_sorted[1] < 2:
            # Near dead-end.
            return -Config.PENALTY_DEAD_END
        
        return 0.0

    def _calc_repeat_explore_penalty(self, hero_pos):
        """
        Compute repeated-exploration penalty using local memory average.
        """
        # map_manager.get_local_memory_avg() returns local average divided by 10
        avg_visits = self.map_manager.get_local_memory_avg()
        
        return -Config.PENALTY_REPEAT_EXPLORE * avg_visits

    def _get_visible_treasure_ratio(self, hero_pos):
        """
        Compute ratio of treasures currently visible to the hero.
        """
        visible_count = 0
        total_count = len(self.treasures)
        
        if total_count == 0:
            return 0.0
        
        for treasure in self.treasures:
            if treasure.found and treasure.available:
                # Distance check to determine visibility.
                dist = np.linalg.norm(treasure.pos - np.array(hero_pos, np.float32))
                if dist <= 10.0:  # View radius.
                    visible_count += 1
        
        return visible_count / total_count

    def _get_reward(self, hero_pos, frame_state):
        """Compute final reward with detailed shaping terms.

        Returns:
        - total_reward: base reward + shaped reward
        - shaped_reward: sum of all shaping/penalty components only

        The function combines risk-aware survival signals, positioning quality,
        exploration control, resource incentives, and anti-stall penalties.
        """
        # Base reward.
        base_reward = Config.REW_STEP
        shaped_reward = 0.0
        reward_components = {
            'base_reward': base_reward,
            'danger_penalty': 0.0,
            'monster_distance': 0.0,
            'second_monster_penalty': 0.0,
            'pinch_penalty': 0.0,
            'corridor_reward': 0.0,
            'dead_end_penalty': 0.0,
            'flash_reward': 0.0,
            'pre_speedup_bonus': 0.0,
            'repeat_explore_penalty': 0.0,
            'treasure_approach': 0.0,
            'treasure_pickup': 0.0,
            'buff_pickup': 0.0,
            'wall_hit_penalty': 0.0
        }
        
        # ========== 1. Nearest-monster distance ==========
        monsters = frame_state.get('monsters', [])
        min_dist = 1000
        for m in monsters:
            m_pos = m.get('pos', {})
            dx = m_pos.get('x', 0) - hero_pos[0]
            dz = m_pos.get('z', 0) - hero_pos[1]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < min_dist:
                min_dist = dist
        
        # ========== 2. Nonlinear danger penalty ==========
        danger_penalty = self._calc_danger_penalty(min_dist)
        shaped_reward += danger_penalty
        reward_components['danger_penalty'] += danger_penalty
        
        # ========== 3. Monster-distance shaping ==========
        is_speedup = self.step_no >= Config.MONSTER_SPEED_UP_STEP
        shaping_weight = 1.5 if is_speedup else 1.0
        monster_distance_reward = Config.REW_MONSTER_DISTANCE * shaping_weight * min(min_dist / 50.0, 1.0)
        shaped_reward += monster_distance_reward
        reward_components['monster_distance'] += monster_distance_reward
        
        # ========== 4. Second-monster pressure penalty ==========
        second_monster_penalty = self._calc_second_monster_penalty(hero_pos, monsters)
        shaped_reward += second_monster_penalty
        reward_components['second_monster_penalty'] += second_monster_penalty
        
        # ========== 5. Detailed pinch/encirclement penalty ==========
        pinch_penalty = self._calc_pinch_penalty_detailed(hero_pos, monsters)
        shaped_reward += pinch_penalty
        reward_components['pinch_penalty'] += pinch_penalty
        
        # ========== 6. Detailed corridor/openness reward ==========
        corridor_reward = self._calc_corridor_reward_detailed(hero_pos)
        shaped_reward += corridor_reward
        reward_components['corridor_reward'] += corridor_reward
        
        # ========== 7. Dead-end penalty ==========
        dead_end_penalty = self._calc_dead_end_penalty(hero_pos)
        shaped_reward += dead_end_penalty
        reward_components['dead_end_penalty'] += dead_end_penalty
        
        # ========== 8. Detailed flash reward/penalty ==========
        flash_reward = self._calc_flash_reward_detailed(hero_pos, monsters, min_dist)
        shaped_reward += flash_reward
        reward_components['flash_reward'] += flash_reward
        
        # ========== 9. Pre-speedup buffer bonus ==========
        pre_speedup_bonus = self._calc_pre_speedup_bonus()
        shaped_reward += pre_speedup_bonus
        reward_components['pre_speedup_bonus'] += pre_speedup_bonus
        
        # ========== 10. Repeated-exploration penalty ==========
        repeat_explore_penalty = self._calc_repeat_explore_penalty(hero_pos)
        shaped_reward += repeat_explore_penalty
        reward_components['repeat_explore_penalty'] += repeat_explore_penalty
        
        # ========== 11. Treasure-approach reward ==========
        if not hasattr(self, '_tracked_visible_treasures'):
            self._tracked_visible_treasures = {}
            
        current_tracked = {}
        treasure_approach = 0.0
        
        for i, treasure in enumerate(self.treasures):
            # Check if treasure is still active on the map
            if treasure.available and treasure.pos[0] != -1:
                dist = treasure.real_distance
                is_visible = (dist <= 10.0)  # Treasure is within view radius
                
                if i in self._tracked_visible_treasures:
                    last_dist = self._tracked_visible_treasures[i]
                    if is_visible:
                        current_tracked[i] = dist
                        if dist < last_dist:
                            # Agent moved closer during its action
                            treasure_approach += Config.REW_TREASURE_APPROACH
                    else:
                        # Treasure disappeared from view but is still available on the map
                        treasure_approach -= Config.REW_TREASURE_APPROACH
                else:
                    if is_visible:
                        # Agent found a new treasure tracking
                        current_tracked[i] = dist
                        
        self._tracked_visible_treasures = current_tracked
        
        shaped_reward += treasure_approach
        reward_components['treasure_approach'] += treasure_approach
        
        # Extract hero info to get reliable pickup metrics
        hero_info, _ = get_hero_info_and_pos(frame_state)
        
        # ========== 12. Treasure pickup reward ==========
        current_treasure_collected = hero_info.get('treasure_collected_count', 0)
        
        if not hasattr(self, '_last_collected_treasure_count'):
            self._last_collected_treasure_count = current_treasure_collected
            
        if current_treasure_collected > self._last_collected_treasure_count:
            treasure_pickup = Config.REW_TREASURE * (current_treasure_collected - self._last_collected_treasure_count)
            shaped_reward += treasure_pickup
            reward_components['treasure_pickup'] += treasure_pickup
            
        self._last_collected_treasure_count = current_treasure_collected
        
        # ========== 13. Buff pickup reward ==========
        current_buff_time = hero_info.get('buff_remain_time', 0)
        
        if not hasattr(self, '_last_buff_time'):
            self._last_buff_time = current_buff_time
            
        if current_buff_time > self._last_buff_time:
            buff_pickup = Config.REW_BUFF
            shaped_reward += buff_pickup
            reward_components['buff_pickup'] += buff_pickup
            
        self._last_buff_time = current_buff_time
        
        # ========== 14. Wall-hit penalty ==========
        if self.last_action != -1 and self.last_action < 16:
            if self.last_hero_pos is not None:
                dx = self.last_hero_pos[0] - hero_pos[0]
                dz = self.last_hero_pos[1] - hero_pos[1]
                if abs(dx) == 0.0 and abs(dz) == 0.0:
                    wall_hit_penalty = -Config.PENALTY_HIT_WALL
                    shaped_reward += wall_hit_penalty
                    reward_components['wall_hit_penalty'] += wall_hit_penalty
        
        # Save nearest-monster distance for next-step comparisons.
        self._last_min_dist = min_dist
        
        total_reward = base_reward + shaped_reward
        
        return total_reward, shaped_reward, reward_components
