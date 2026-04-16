#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import math
from dataclasses import dataclass, field

import numpy as np

from agent_ppo.conf.conf import Config

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_DIAGONAL = math.sqrt(2.0) * MAP_SIZE
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
LOCAL_HALF = Config.LOCAL_MAP_SIZE // 2
TARGET_DIST_SCALE = 16.0
TOPOLOGY_RADIUS = 2

DIST_BUCKET_TO_DISTANCE = {
    0: 15.0,
    1: 45.0,
    2: 75.0,
    3: 105.0,
    4: 135.0,
    5: 165.0,
}

DIRECTION_TO_VECTOR = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (math.sqrt(0.5), math.sqrt(0.5)),
    3: (0.0, 1.0),
    4: (-math.sqrt(0.5), math.sqrt(0.5)),
    5: (-1.0, 0.0),
    6: (-math.sqrt(0.5), -math.sqrt(0.5)),
    7: (0.0, -1.0),
    8: (math.sqrt(0.5), -math.sqrt(0.5)),
}

ACTION_TO_ROW_COL_DELTA = {
    0: (0, 1),
    1: (-1, 1),
    2: (-1, 0),
    3: (-1, -1),
    4: (0, -1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1),
}

OPPOSITE_MOVE_ACTION = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    4: 0,
    5: 1,
    6: 2,
    7: 3,
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""

    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _clip_position(pos):
    clipped = np.clip(np.asarray(pos, dtype=np.float32), 0.0, MAP_SIZE - 1.0)
    return clipped.astype(np.float32)


def _is_valid_position(pos):
    return pos is not None and float(pos[0]) >= 0.0 and float(pos[1]) >= 0.0


def _get_nested(data, keys, default=None):
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _extract_env_conf(usr_conf):
    if not isinstance(usr_conf, dict):
        return {}
    env_conf = usr_conf.get("env_conf", usr_conf)
    return env_conf if isinstance(env_conf, dict) else {}


def _distance_from_bucket(dist_bucket):
    return DIST_BUCKET_TO_DISTANCE.get(int(dist_bucket), DIST_BUCKET_TO_DISTANCE[int(MAX_DIST_BUCKET)])


def _direction_to_sin_cos(direction):
    dx, dz = DIRECTION_TO_VECTOR.get(int(direction), (0.0, 0.0))
    return float(dz), float(dx)


def _estimate_position_from_relative(hero_pos, direction, dist_bucket):
    dx, dz = DIRECTION_TO_VECTOR.get(int(direction), (0.0, 0.0))
    distance = _distance_from_bucket(dist_bucket)
    return _clip_position(hero_pos + np.array([dx * distance, dz * distance], dtype=np.float32))


def _relative_direction_to_move_action(direction):
    direction = int(direction)
    if 1 <= direction <= 8:
        return direction - 1
    return None


def _circular_action_gap(action_a, action_b):
    diff = abs(int(action_a) - int(action_b)) % 8
    return min(diff, 8 - diff)


@dataclass
class TargetMemory:
    config_id: int
    sub_type: int
    pos: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0], dtype=np.float32))
    found: bool = False
    visible_this_step: bool = False
    available: bool = True
    direction: int = 0
    distance_bucket: int = int(MAX_DIST_BUCKET)
    last_distance: float = MAP_DIAGONAL
    distance: float = MAP_DIAGONAL


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self, usr_conf=None):
        self.step_no = 0
        self.max_step = 1000
        self.monster_speedup_step = 500
        self.monster_interval_step = 300
        self.initial_monster_speed = 0.0
        self.speedup_reached = False
        self.last_min_monster_distance = MAP_DIAGONAL * 0.5
        self.visit_counts = {}
        self.last_grid = None
        self.last_hero_pos = None
        self.prev_hero_pos = None
        self.last_move_action = -1
        self.blocked_move_actions = set()
        self.last_map_array = None
        self.stagnation_steps = 0
        self.oscillation_steps = 0
        self.visit_heat = np.zeros((int(MAP_SIZE), int(MAP_SIZE)), dtype=np.float32)
        self.treasure_memory = {}
        self.buff_memory = {}
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_flash_count = 0
        self.last_target_treasure_id = None
        self.last_target_treasure_distance = None
        self.last_nearest_visible_treasure_distance = None
        self.last_target_buff_id = None
        self.last_target_buff_distance = None
        self.early_loot_stall_steps = 0
        self.configure_episode(usr_conf)

    def configure_episode(self, usr_conf=None):
        env_conf = _extract_env_conf(usr_conf)
        if not env_conf:
            return

        self.max_step = max(1, int(env_conf.get("max_step", self.max_step)))

        monster_speedup = int(env_conf.get("monster_speedup", self.monster_speedup_step))
        if monster_speedup > 0:
            self.monster_speedup_step = min(max(1, monster_speedup), self.max_step)

        monster_interval = int(env_conf.get("monster_interval", self.monster_interval_step))
        if monster_interval > 0:
            self.monster_interval_step = monster_interval

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, reward and stats."""

        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        hero = self._extract_hero(frame_state)
        hero_pos = self._extract_position(hero)
        if hero_pos is None:
            hero_pos = np.array([0.0, 0.0], dtype=np.float32)

        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
        self.max_step = max(1, int(env_info.get("max_step", self.max_step)))
        self.monster_speedup_step = min(max(1, self.monster_speedup_step), self.max_step)
        map_array = self._to_map_array(observation.get("map_info"))

        movement_state = self._update_movement_state(hero_pos, last_action)
        self._register_visit(hero_pos)

        legal_action = self._build_legal_action_mask(
            observation.get("legal_act", observation.get("legal_action")),
            map_info=map_array,
        )
        monsters, min_monster_distance = self._build_monster_features(frame_state.get("monsters", []), hero_pos)
        second_monster_distance = monsters[1]["distance"] if len(monsters) > 1 else MAP_DIAGONAL
        treasure_target, buff_target = self._sync_collectible_memory(frame_state.get("organs", []), env_info, hero_pos)
        speedup_state = self._compute_speedup_state(env_info)
        pinch_risk = self._compute_pinch_risk(hero_pos, monsters)
        danger_level = self._compute_danger_level(
            min_monster_distance=min_monster_distance,
            pinch_risk=pinch_risk,
            speedup_state=speedup_state,
        )
        semantic_map, topology_summary = self._build_semantic_map(
            map_info=map_array,
            hero_pos=hero_pos,
            monsters=monsters,
        )

        treasure_total = max(1, int(env_info.get("total_treasure", 10)))
        treasure_count = int(
            env_info.get(
                "treasures_collected",
                hero.get("treasure_collected_count", 0),
            )
        )
        buff_count = int(env_info.get("collected_buff", 0))
        flash_count = int(env_info.get("flash_count", 0))
        flash_config_cd = max(1, int(env_info.get("flash_cooldown", 100)))
        flash_cooldown = self._extract_flash_cooldown(hero)
        flash_ready = 1.0 if flash_cooldown <= 1e-6 else 0.0
        buff_remaining = self._extract_buff_remaining(hero)
        buff_active = 1.0 if buff_remaining > 1e-6 else 0.0

        step_norm = _norm(self.step_no, self.max_step)
        treasure_collected_ratio = treasure_count / float(treasure_total)
        remaining_treasure_ratio = max(0.0, 1.0 - treasure_collected_ratio)
        flash_opportunity = max(1.0, self.step_no / float(flash_config_cd) + 1.0)
        flash_used_ratio = float(np.clip(flash_count / flash_opportunity, 0.0, 1.0))
        treasure_dir_sin, treasure_dir_cos, treasure_distance_norm = self._target_guidance(
            hero_pos,
            treasure_target,
        )
        buff_dir_sin, buff_dir_cos, buff_distance_norm = self._target_guidance(
            hero_pos,
            buff_target,
        )
        revisit_intensity = self._revisit_intensity(hero_pos)
        second_monster_active = 1.0 if len(monsters) > 1 and monsters[1]["active"] else 0.0
        single_monster_phase = 1.0 if self._is_single_monster_phase() else 0.0
        treasure_available = 1.0 if treasure_target is not None else 0.0
        treasure_visible = (
            1.0 if treasure_target is not None and treasure_target.visible_this_step else 0.0
        )
        buff_available = 1.0 if buff_target is not None else 0.0
        treasure_opportunity = self._compute_resource_opportunity(
            target=treasure_target,
            danger_level=danger_level,
            pinch_risk=pinch_risk,
            speedup_state=speedup_state,
            prefer_treasure=True,
        )
        survival_pressure = self._compute_survival_pressure(
            danger_level=danger_level,
            pinch_risk=pinch_risk,
            speedup_state=speedup_state,
            second_monster_distance=second_monster_distance,
        )
        greed_window = self._compute_greed_window(
            treasure_target=treasure_target,
            treasure_opportunity=treasure_opportunity,
            survival_pressure=survival_pressure,
            speedup_state=speedup_state,
        )

        hero_feat = np.array(
            [
                _norm(hero_pos[0], MAP_SIZE - 1.0),
                _norm(hero_pos[1], MAP_SIZE - 1.0),
                _norm(flash_cooldown, MAX_FLASH_CD),
                flash_ready,
                _norm(buff_remaining, MAX_BUFF_DURATION),
                buff_active,
                treasure_collected_ratio,
                step_norm,
                danger_level,
                movement_state["stagnation_ratio"],
                movement_state["oscillation_ratio"],
                revisit_intensity,
            ],
            dtype=np.float32,
        )

        walkable_channel = semantic_map[0]
        if treasure_target is not None and _is_valid_position(treasure_target.pos):
            treasure_passability = self._compute_treasure_passability(
                hero_pos, treasure_target, walkable_channel,
            )
            treasure_feat = np.array(
                [
                    treasure_visible,
                    _norm(treasure_target.pos[0], MAP_SIZE - 1.0),
                    _norm(treasure_target.pos[1], MAP_SIZE - 1.0),
                    treasure_passability,
                    treasure_distance_norm,
                    _norm(treasure_target.distance_bucket, MAX_DIST_BUCKET),
                    treasure_dir_sin,
                    treasure_dir_cos,
                    treasure_available,
                    treasure_opportunity,
                ],
                dtype=np.float32,
            )
        else:
            treasure_feat = np.zeros(Config.FEATURES[3], dtype=np.float32)

        progress_feat = np.array(
            [
                remaining_treasure_ratio,
                flash_used_ratio,
                speedup_state["speedup_reached"],
                speedup_state["time_to_speedup_norm"],
                single_monster_phase,
                second_monster_active,
                _norm(min_monster_distance, MAP_DIAGONAL),
                _norm(second_monster_distance, MAP_DIAGONAL),
                pinch_risk,
                topology_summary["local_openness"],
                topology_summary["local_corridor"],
                topology_summary["local_dead_end_risk"],
                treasure_available,
                treasure_visible,
                treasure_distance_norm,
                treasure_dir_sin,
                treasure_dir_cos,
                treasure_opportunity,
                buff_available,
                buff_distance_norm,
                buff_dir_sin,
                buff_dir_cos,
                survival_pressure,
                greed_window,
            ],
            dtype=np.float32,
        )

        feature = np.concatenate(
            [
                hero_feat,
                monsters[0]["feature"],
                monsters[1]["feature"],
                treasure_feat,
                semantic_map.reshape(-1),
                np.asarray(legal_action, dtype=np.float32),
                progress_feat,
            ]
        ).astype(np.float32)
        if feature.shape[0] != Config.DIM_OF_OBSERVATION:
            raise ValueError(f"feature dim mismatch: {feature.shape[0]} != {Config.DIM_OF_OBSERVATION}")

        reward, reward_terms = self._build_reward(
            hero_pos=hero_pos,
            min_monster_distance=min_monster_distance,
            treasure_target=treasure_target,
            buff_target=buff_target,
            treasure_count=treasure_count,
            buff_count=buff_count,
            flash_count=flash_count,
            last_action=last_action,
            movement_state=movement_state,
            monsters=monsters,
            second_monster_distance=second_monster_distance,
            speedup_state=speedup_state,
        )

        self.last_min_monster_distance = min_monster_distance
        self.last_target_treasure_id = treasure_target.config_id if treasure_target is not None else None
        self.last_target_treasure_distance = treasure_target.distance if treasure_target is not None else None
        nearest_visible = self._pick_nearest_visible(self.treasure_memory)
        self.last_nearest_visible_treasure_distance = nearest_visible.distance if nearest_visible is not None else None
        self.last_target_buff_id = buff_target.config_id if buff_target is not None else None
        self.last_target_buff_distance = buff_target.distance if buff_target is not None else None
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_count
        self.last_flash_count = flash_count
        self.prev_hero_pos = None if self.last_hero_pos is None else np.asarray(self.last_hero_pos, dtype=np.float32)
        self.last_hero_pos = hero_pos
        self.last_map_array = None if map_array is None else np.asarray(map_array, dtype=np.float32)
        self.last_move_action = int(last_action) if 0 <= int(last_action) < 8 else -1

        return feature, legal_action, [reward], {
            "reward_terms": reward_terms,
            "speedup_state": speedup_state,
        }

    def _extract_hero(self, frame_state):
        hero = frame_state.get("heroes", {})
        if isinstance(hero, list):
            return hero[0] if hero else {}
        return hero

    def _extract_position(self, obj):
        pos = None
        if isinstance(obj, dict):
            if "pos" in obj and isinstance(obj["pos"], dict):
                pos = obj["pos"]
            elif "x" in obj and "z" in obj:
                pos = obj
        if not isinstance(pos, dict) or "x" not in pos or "z" not in pos:
            return None
        return _clip_position([pos["x"], pos["z"]])

    def _extract_flash_cooldown(self, hero):
        return float(
            hero.get(
                "flash_cooldown",
                _get_nested(hero, ["talent", "cooldown"], 0.0),
            )
        )

    def _extract_buff_remaining(self, hero):
        return float(
            hero.get(
                "buff_remaining_time",
                hero.get("buff_remain_time", hero.get("buff_remain", 0.0)),
            )
        )

    def _extract_relative_direction(self, obj):
        return int(
            obj.get(
                "hero_relative_direction",
                _get_nested(obj, ["relative_pos", "direction"], 0),
            )
        )

    def _extract_relative_distance_bucket(self, obj):
        return int(
            obj.get(
                "hero_l2_distance",
                _get_nested(obj, ["relative_pos", "l2_distance"], int(MAX_DIST_BUCKET)),
            )
        )

    def _compute_speedup_state(self, env_info):
        monster_speed = float(env_info.get("monster_speed", 0.0))
        if self.initial_monster_speed <= 0.0 and monster_speed > 0.0:
            self.initial_monster_speed = monster_speed
        if self.initial_monster_speed <= 0.0:
            self.initial_monster_speed = 1.0

        speedup_step = min(max(1, int(self.monster_speedup_step)), self.max_step)
        speedup_by_step = self.step_no >= speedup_step
        speedup_by_speed = monster_speed > self.initial_monster_speed + 1e-6
        self.speedup_reached = bool(self.speedup_reached or speedup_by_step or speedup_by_speed)

        time_to_speedup = max(0, speedup_step - self.step_no)
        time_to_speedup_norm = 0.0
        if not self.speedup_reached:
            time_to_speedup_norm = float(np.clip(time_to_speedup / float(max(1, speedup_step)), 0.0, 1.0))

        pre_speedup_buffer_ratio = 0.0
        if not self.speedup_reached and time_to_speedup <= Config.PRE_SPEEDUP_BUFFER_WINDOW:
            pre_speedup_buffer_ratio = 1.0 - float(
                np.clip(
                    time_to_speedup / float(max(1, Config.PRE_SPEEDUP_BUFFER_WINDOW)),
                    0.0,
                    1.0,
                )
            )

        return {
            "speedup_reached": float(self.speedup_reached),
            "time_to_speedup": float(time_to_speedup),
            "time_to_speedup_norm": time_to_speedup_norm,
            "pre_speedup_buffer_ratio": pre_speedup_buffer_ratio,
            "monster_speed": monster_speed,
            "speedup_step": float(speedup_step),
        }

    def _update_movement_state(self, hero_pos, last_action):
        movement_state = {
            "hit_wall": False,
            "moved_distance": 0.0,
            "stagnation_ratio": 0.0,
            "oscillation_ratio": 0.0,
            "flash_origin_blocked": False,
            "flash_path_blocked_cells": 0.0,
            "flash_through_wall": False,
        }

        if self.last_hero_pos is None or last_action == -1:
            self.blocked_move_actions.clear()
            self.stagnation_steps = 0
            self.oscillation_steps = 0
            return movement_state

        moved_distance = float(np.linalg.norm(hero_pos - self.last_hero_pos))
        movement_state["moved_distance"] = moved_distance

        if 8 <= int(last_action) < 16:
            flash_path = self._scan_flash_path(int(last_action) - 8)
            movement_state["flash_origin_blocked"] = flash_path["origin_blocked"]
            movement_state["flash_path_blocked_cells"] = float(flash_path["blocked_cells"])
            movement_state["flash_through_wall"] = bool(
                flash_path["blocked_cells"] > 0
                and moved_distance >= Config.FLASH_THROUGH_WALL_MIN_MOVE_DISTANCE
            )

        if moved_distance < Config.STAGNATION_MOVE_THRESHOLD:
            self.stagnation_steps = min(self.stagnation_steps + 1, Config.STAGNATION_MAX_STEPS)
        else:
            self.stagnation_steps = max(self.stagnation_steps - 1, 0)

        if 0 <= int(last_action) < 8 and moved_distance < Config.HIT_WALL_DISTANCE_THRESHOLD:
            movement_state["hit_wall"] = True
            self.blocked_move_actions.add(int(last_action))
        else:
            self.blocked_move_actions.clear()

        bounced_back = False
        if 0 <= int(last_action) < 8 and 0 <= self.last_move_action < 8 and self.prev_hero_pos is not None:
            returned_to_prev = (
                float(np.linalg.norm(hero_pos - self.prev_hero_pos))
                <= Config.OSCILLATION_RETURN_DISTANCE
            )
            bounced_back = (
                int(last_action) == OPPOSITE_MOVE_ACTION.get(self.last_move_action, -1)
                and returned_to_prev
            )

        if bounced_back:
            self.oscillation_steps = min(self.oscillation_steps + 1, Config.OSCILLATION_MAX_STEPS)
        else:
            self.oscillation_steps = max(self.oscillation_steps - 1, 0)

        movement_state["stagnation_ratio"] = float(
            np.clip(self.stagnation_steps / float(max(1, Config.STAGNATION_MAX_STEPS)), 0.0, 1.0)
        )
        movement_state["oscillation_ratio"] = float(
            np.clip(self.oscillation_steps / float(max(1, Config.OSCILLATION_MAX_STEPS)), 0.0, 1.0)
        )
        return movement_state

    def _register_visit(self, hero_pos):
        x = int(np.clip(round(hero_pos[0]), 0, MAP_SIZE - 1))
        z = int(np.clip(round(hero_pos[1]), 0, MAP_SIZE - 1))
        self.visit_heat[x, z] += 1.0

    def _build_legal_action_mask(self, legal_act_raw, map_info=None):
        legal_action = [1] * Config.ACTION_NUM

        if isinstance(legal_act_raw, (list, tuple, np.ndarray)) and len(legal_act_raw) > 0:
            first = legal_act_raw[0]
            if isinstance(first, (bool, np.bool_)):
                legal_action = [int(bool(v)) for v in legal_act_raw[: Config.ACTION_NUM]]
                if len(legal_action) < Config.ACTION_NUM:
                    legal_action.extend([1] * (Config.ACTION_NUM - len(legal_action)))
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
                legal_action = [1 if idx in valid_set else 0 for idx in range(Config.ACTION_NUM)]

        for blocked_action in self._obstacle_blocked_actions(map_info):
            legal_action[blocked_action] = 0

        for blocked_action in self.blocked_move_actions:
            if 0 <= blocked_action < 8:
                legal_action[blocked_action] = 0

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM

        return legal_action

    def _obstacle_blocked_actions(self, map_info):
        map_array = self._to_map_array(map_info)
        if map_array is None or map_array.ndim != 2:
            return set()

        center_row = map_array.shape[0] // 2
        center_col = map_array.shape[1] // 2
        blocked_actions = set()
        for action, (d_row, d_col) in ACTION_TO_ROW_COL_DELTA.items():
            row = center_row + d_row
            col = center_col + d_col
            if not (0 <= row < map_array.shape[0] and 0 <= col < map_array.shape[1]):
                blocked_actions.add(action)
                continue
            if float(map_array[row, col]) <= 0.0:
                blocked_actions.add(action)
        return blocked_actions

    def _scan_flash_path(self, move_action):
        if self.last_map_array is None or self.last_map_array.ndim != 2:
            return {"origin_blocked": False, "blocked_cells": 0}

        if move_action not in ACTION_TO_ROW_COL_DELTA:
            return {"origin_blocked": False, "blocked_cells": 0}

        d_row, d_col = ACTION_TO_ROW_COL_DELTA[move_action]
        center_row = self.last_map_array.shape[0] // 2
        center_col = self.last_map_array.shape[1] // 2
        blocked_cells = 0
        origin_blocked = False
        max_steps = max(1, int(getattr(Config, "FLASH_THROUGH_WALL_SCAN_STEPS", 1)))

        for step_idx in range(1, max_steps + 1):
            row = center_row + d_row * step_idx
            col = center_col + d_col * step_idx
            if not (0 <= row < self.last_map_array.shape[0] and 0 <= col < self.last_map_array.shape[1]):
                break
            if float(self.last_map_array[row, col]) <= 0.0:
                blocked_cells += 1
                if step_idx == 1:
                    origin_blocked = True

        return {"origin_blocked": origin_blocked, "blocked_cells": blocked_cells}

    def _build_monster_features(self, monsters, hero_pos):
        monster_infos = []
        for idx in range(2):
            if idx >= len(monsters):
                monster_infos.append(
                    {
                        "feature": np.zeros(10, dtype=np.float32),
                        "distance": MAP_DIAGONAL,
                        "pos": None,
                        "active": False,
                        "visible": False,
                        "direction": 0,
                        "threat_score": 0.0,
                    }
                )
                continue

            monster = monsters[idx]
            pos = self._extract_position(monster)
            direction = self._extract_relative_direction(monster)
            dist_bucket = self._extract_relative_distance_bucket(monster)
            active_flag = 1.0

            if pos is not None:
                is_visible = float(monster.get("is_in_view", 1.0))
                est_pos = pos
                distance = float(np.linalg.norm(est_pos - hero_pos))
            else:
                is_visible = float(monster.get("is_in_view", 0.0))
                est_pos = _estimate_position_from_relative(hero_pos, direction, dist_bucket)
                distance = _distance_from_bucket(dist_bucket)

            dir_sin, dir_cos = _direction_to_sin_cos(direction)
            speed_norm = _norm(monster.get("speed", 1), MAX_MONSTER_SPEED)
            distance_norm = _norm(distance, MAP_DIAGONAL)
            threat_score = self._compute_monster_threat(
                distance=distance,
                is_visible=is_visible,
                speed_norm=speed_norm,
                active_flag=active_flag,
            )
            feature = np.array(
                [
                    is_visible,
                    _norm(est_pos[0], MAP_SIZE - 1.0),
                    _norm(est_pos[1], MAP_SIZE - 1.0),
                    speed_norm,
                    distance_norm,
                    _norm(dist_bucket, MAX_DIST_BUCKET),
                    dir_sin,
                    dir_cos,
                    active_flag,
                    threat_score,
                ],
                dtype=np.float32,
            )
            monster_infos.append(
                {
                    "feature": feature,
                    "distance": float(distance),
                    "pos": est_pos,
                    "active": True,
                    "visible": bool(is_visible >= 0.5),
                    "direction": int(direction),
                    "threat_score": threat_score,
                }
            )

        min_distance_candidates = [info["distance"] for info in monster_infos if info["active"]]
        min_distance = min(min_distance_candidates) if min_distance_candidates else MAP_DIAGONAL
        return monster_infos, float(min_distance)

    def _compute_monster_threat(self, distance, is_visible, speed_norm, active_flag):
        proximity = 1.0 - _norm(distance, MAP_DIAGONAL)
        threat_score = (0.65 * proximity) + (0.2 * float(is_visible)) + (0.15 * float(speed_norm))
        return float(np.clip(threat_score * float(active_flag), 0.0, 1.0))

    def _compute_pinch_risk(self, hero_pos, monsters):
        nearby_monsters = [
            monster
            for monster in monsters
            if monster["active"] and monster["pos"] is not None
        ]
        if len(nearby_monsters) < 2:
            return 0.0

        nearby_monsters = sorted(nearby_monsters, key=lambda monster: monster["distance"])[:2]
        vector_a = np.asarray(nearby_monsters[0]["pos"], dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        vector_b = np.asarray(nearby_monsters[1]["pos"], dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if norm_a <= 1e-6 or norm_b <= 1e-6:
            return 0.0

        cos_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        angle_risk = float(np.clip((1.0 - cos_similarity) * 0.5, 0.0, 1.0))
        proximity_risk = 1.0 - float(
            np.clip(
                (nearby_monsters[0]["distance"] + nearby_monsters[1]["distance"])
                / float(max(1.0, 2.0 * Config.DOUBLE_MONSTER_PINCH_DISTANCE)),
                0.0,
                1.0,
            )
        )
        return float(np.clip(angle_risk * proximity_risk, 0.0, 1.0))

    def _compute_danger_level(self, min_monster_distance, pinch_risk, speedup_state):
        distance_pressure = 1.0 - _norm(min_monster_distance, MAP_DIAGONAL)
        speedup_pressure = 0.15 * float(speedup_state.get("speedup_reached", 0.0))
        danger_level = (0.75 * distance_pressure) + (0.25 * float(pinch_risk)) + speedup_pressure
        return float(np.clip(danger_level, 0.0, 1.0))

    def _compute_resource_opportunity(self, target, danger_level, pinch_risk, speedup_state, prefer_treasure):
        if target is None:
            return 0.0

        distance_term = 1.0 - _norm(target.distance, MAP_DIAGONAL)
        visibility_term = 1.0 if target.visible_this_step else 0.65
        safety_term = max(0.0, 1.0 - (0.75 * float(danger_level) + 0.25 * float(pinch_risk)))

        phase_multiplier = 1.0
        if prefer_treasure and self._is_single_monster_phase() and speedup_state["speedup_reached"] < 1.0:
            phase_multiplier *= 1.15
        if prefer_treasure and speedup_state["speedup_reached"] >= 1.0:
            phase_multiplier *= 0.7

        opportunity = distance_term * visibility_term * safety_term * phase_multiplier
        return float(np.clip(opportunity, 0.0, 1.0))

    def _compute_survival_pressure(self, danger_level, pinch_risk, speedup_state, second_monster_distance):
        second_monster_pressure = 1.0 - _norm(second_monster_distance, MAP_DIAGONAL)
        pressure = (
            (0.6 * float(danger_level))
            + (0.2 * float(pinch_risk))
            + (0.15 * float(speedup_state.get("speedup_reached", 0.0)))
            + (0.05 * second_monster_pressure)
        )
        return float(np.clip(pressure, 0.0, 1.0))

    def _compute_greed_window(self, treasure_target, treasure_opportunity, survival_pressure, speedup_state):
        if treasure_target is None:
            return 0.0

        phase_bonus = 0.2 if self._is_single_monster_phase() and speedup_state["speedup_reached"] < 1.0 else 0.0
        speed_penalty = 0.2 if speedup_state["speedup_reached"] >= 1.0 else 0.0
        greed_window = treasure_opportunity + phase_bonus - (0.75 * float(survival_pressure)) - speed_penalty
        return float(np.clip(greed_window, 0.0, 1.0))

    def _compute_treasure_passability(self, hero_pos, treasure_target, walkable_channel):
        if treasure_target is None or not _is_valid_position(treasure_target.pos):
            return 0.0

        delta = np.asarray(treasure_target.pos, dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return 1.0

        dx, dz = float(delta[0]), float(delta[1])
        best_action = None
        best_alignment = -2.0
        for action, (d_row, d_col) in ACTION_TO_ROW_COL_DELTA.items():
            norm = math.sqrt(d_row * d_row + d_col * d_col)
            if norm < 1e-6:
                continue
            alignment = (d_col * dx + d_row * dz) / (norm * dist)
            if alignment > best_alignment:
                best_alignment = alignment
                best_action = action

        if best_action is None:
            return 0.0

        d_row, d_col = ACTION_TO_ROW_COL_DELTA[best_action]
        center = LOCAL_HALF
        walkable_count = 0
        check_count = 0
        for step in range(1, 4):
            r = center + d_row * step
            c = center + d_col * step
            if 0 <= r < Config.LOCAL_MAP_SIZE and 0 <= c < Config.LOCAL_MAP_SIZE:
                walkable_count += float(walkable_channel[r, c] > 0.0)
                check_count += 1

        return walkable_count / float(max(1, check_count))

    def _sync_collectible_memory(self, organs, env_info, hero_pos):
        for memory_bank in (self.treasure_memory, self.buff_memory):
            for target in memory_bank.values():
                target.visible_this_step = False

        remaining_treasures = {
            int(treasure_id)
            for treasure_id in (env_info.get("treasure_id", []) or [])
        }

        for organ in organs or []:
            sub_type = int(organ.get("sub_type", 0))
            if sub_type not in (1, 2):
                continue

            config_id = int(organ.get("config_id", 0))
            memory_bank = self.treasure_memory if sub_type == 1 else self.buff_memory
            target = memory_bank.get(config_id)
            if target is None:
                target = TargetMemory(config_id=config_id, sub_type=sub_type)
                memory_bank[config_id] = target

            status = int(organ.get("status", 1))
            target.direction = self._extract_relative_direction(organ)
            target.distance_bucket = self._extract_relative_distance_bucket(organ)

            available = status != 0
            if sub_type == 1 and remaining_treasures:
                available = config_id in remaining_treasures
            target.available = available

            pos = self._extract_position(organ)
            if pos is not None and status != -1:
                target.pos = pos
                target.found = True
                target.visible_this_step = True
            elif not target.found and target.direction > 0:
                target.pos = _estimate_position_from_relative(hero_pos, target.direction, target.distance_bucket)

            target.last_distance = target.distance
            if _is_valid_position(target.pos):
                target.distance = float(np.linalg.norm(target.pos - hero_pos))
            else:
                target.distance = _distance_from_bucket(target.distance_bucket)

        if remaining_treasures:
            for config_id, target in self.treasure_memory.items():
                target.available = config_id in remaining_treasures

        treasure_target = self._pick_nearest_available(self.treasure_memory, prefer_visible=True)
        buff_target = self._pick_nearest_available(self.buff_memory)
        return treasure_target, buff_target

    def _pick_nearest_available(self, memory_bank, prefer_visible=False):
        candidates = [
            target
            for target in memory_bank.values()
            if target.available and _is_valid_position(target.pos)
        ]
        if not candidates:
            return None
        if prefer_visible:
            return min(
                candidates,
                key=lambda target: (0 if target.visible_this_step else 1, target.distance),
            )
        return min(candidates, key=lambda target: target.distance)

    def _pick_nearest_visible(self, memory_bank):
        candidates = [
            target
            for target in memory_bank.values()
            if target.available and target.visible_this_step and _is_valid_position(target.pos)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda target: target.distance)

    def _target_guidance(self, hero_pos, target):
        if target is None:
            return 0.0, 0.0, 1.0

        if _is_valid_position(target.pos):
            delta = np.asarray(target.pos, dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                dir_cos = float(np.clip(delta[0] / distance, -1.0, 1.0))
                dir_sin = float(np.clip(delta[1] / distance, -1.0, 1.0))
            else:
                dir_sin, dir_cos = _direction_to_sin_cos(target.direction)
        else:
            distance = _distance_from_bucket(target.distance_bucket)
            dir_sin, dir_cos = _direction_to_sin_cos(target.direction)

        return dir_sin, dir_cos, _norm(distance, MAP_DIAGONAL)

    def _build_semantic_map(self, map_info, hero_pos, monsters):
        walkable_channel = self._build_walkable_channel(map_info)
        openness_channel, corridor_channel, dead_end_channel = self._build_topology_channels(walkable_channel)
        semantic_map = np.zeros(
            (Config.LOCAL_MAP_CHANNEL, Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE),
            dtype=np.float32,
        )
        semantic_map[0] = walkable_channel
        semantic_map[1] = self._build_visit_channel(hero_pos)
        semantic_map[2] = self._build_collectible_channel(hero_pos)
        semantic_map[3] = self._build_risk_channel(hero_pos, monsters)
        semantic_map[4] = openness_channel
        semantic_map[5] = corridor_channel
        semantic_map[6] = dead_end_channel
        topology_summary = self._summarize_topology(openness_channel, corridor_channel, dead_end_channel)
        return semantic_map, topology_summary

    def _to_map_array(self, map_info):
        if map_info is None:
            return None
        if isinstance(map_info, np.ndarray):
            return map_info.astype(np.float32)
        if isinstance(map_info, list) and len(map_info) > 0 and isinstance(map_info[0], dict):
            return np.asarray([line.get("values", []) for line in map_info], dtype=np.float32)
        return np.asarray(map_info, dtype=np.float32)

    def _build_walkable_channel(self, map_info):
        patch = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        map_array = self._to_map_array(map_info)
        if map_array is None or map_array.ndim != 2:
            return patch

        center = map_array.shape[0] // 2
        for row in range(Config.LOCAL_MAP_SIZE):
            for col in range(Config.LOCAL_MAP_SIZE):
                src_row = center - LOCAL_HALF + row
                src_col = center - LOCAL_HALF + col
                if 0 <= src_row < map_array.shape[0] and 0 <= src_col < map_array.shape[1]:
                    patch[row, col] = float(map_array[src_row, src_col] > 0)
        return patch

    def _build_topology_channels(self, walkable_channel):
        openness_channel = np.zeros_like(walkable_channel)
        corridor_channel = np.zeros_like(walkable_channel)
        dead_end_channel = np.zeros_like(walkable_channel)

        for row in range(Config.LOCAL_MAP_SIZE):
            for col in range(Config.LOCAL_MAP_SIZE):
                if walkable_channel[row, col] <= 0.0:
                    dead_end_channel[row, col] = 1.0
                    continue

                openness_score = self._local_open_area_ratio(walkable_channel, row, col)
                corridor_score = self._local_corridor_score(walkable_channel, row, col)
                dead_end_score = self._local_dead_end_risk(
                    walkable_channel,
                    row,
                    col,
                    openness_score,
                )
                openness_channel[row, col] = openness_score
                corridor_channel[row, col] = corridor_score
                dead_end_channel[row, col] = dead_end_score

        return openness_channel, corridor_channel, dead_end_channel

    def _summarize_topology(self, openness_channel, corridor_channel, dead_end_channel):
        center = LOCAL_HALF
        return {
            "local_openness": float(openness_channel[center, center]),
            "local_corridor": float(corridor_channel[center, center]),
            "local_dead_end_risk": float(dead_end_channel[center, center]),
        }

    def _local_open_area_ratio(self, walkable_channel, row, col):
        walkable_count = 0.0
        total_count = 0.0
        for d_row in range(-TOPOLOGY_RADIUS, TOPOLOGY_RADIUS + 1):
            for d_col in range(-TOPOLOGY_RADIUS, TOPOLOGY_RADIUS + 1):
                rr = row + d_row
                cc = col + d_col
                if 0 <= rr < Config.LOCAL_MAP_SIZE and 0 <= cc < Config.LOCAL_MAP_SIZE:
                    walkable_count += float(walkable_channel[rr, cc] > 0.0)
                    total_count += 1.0
        if total_count <= 0.0:
            return 0.0
        return float(np.clip(walkable_count / total_count, 0.0, 1.0))

    def _walkable_direction_depth(self, walkable_channel, row, col, d_row, d_col):
        depth = 0
        for step_idx in range(1, LOCAL_HALF + 1):
            rr = row + d_row * step_idx
            cc = col + d_col * step_idx
            if not (0 <= rr < Config.LOCAL_MAP_SIZE and 0 <= cc < Config.LOCAL_MAP_SIZE):
                break
            if walkable_channel[rr, cc] <= 0.0:
                break
            depth += 1
        return depth

    def _direction_depths(self, walkable_channel, row, col):
        return [
            self._walkable_direction_depth(walkable_channel, row, col, d_row, d_col)
            for d_row, d_col in ACTION_TO_ROW_COL_DELTA.values()
        ]

    def _local_corridor_score(self, walkable_channel, row, col):
        direction_depths = sorted(self._direction_depths(walkable_channel, row, col), reverse=True)
        if not direction_depths:
            return 0.0
        best_two = direction_depths[:2]
        corridor_depth = sum(best_two) / float(max(1, len(best_two)))
        return float(np.clip(corridor_depth / float(max(1, LOCAL_HALF)), 0.0, 1.0))

    def _walkable_neighbor_count(self, walkable_channel, row, col):
        return sum(1 for depth in self._direction_depths(walkable_channel, row, col) if depth > 0)

    def _local_dead_end_risk(self, walkable_channel, row, col, openness_score):
        direction_depths = sorted(self._direction_depths(walkable_channel, row, col), reverse=True)
        if not direction_depths:
            return 1.0

        branch_count = self._walkable_neighbor_count(walkable_channel, row, col)
        escape_margin = (
            sum(direction_depths[:2]) / float(max(1, min(2, len(direction_depths))))
        ) / float(max(1, LOCAL_HALF))
        openness_risk = 1.0 - float(openness_score)
        branch_risk = float(np.clip((4.0 - float(branch_count)) / 3.0, 0.0, 1.0))
        escape_risk = 1.0 - float(np.clip(escape_margin, 0.0, 1.0))
        dead_end_risk = (0.45 * openness_risk) + (0.35 * branch_risk) + (0.2 * escape_risk)
        return float(np.clip(dead_end_risk, 0.0, 1.0))

    def _build_visit_channel(self, hero_pos):
        channel = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        hero_x = int(round(hero_pos[0]))
        hero_z = int(round(hero_pos[1]))
        for row in range(Config.LOCAL_MAP_SIZE):
            for col in range(Config.LOCAL_MAP_SIZE):
                dx = col - LOCAL_HALF
                dz = LOCAL_HALF - row
                x = hero_x + dx
                z = hero_z + dz
                if 0 <= x < MAP_SIZE and 0 <= z < MAP_SIZE:
                    channel[row, col] = min(self.visit_heat[x, z] / 5.0, 1.0)
        return channel

    def _build_collectible_channel(self, hero_pos):
        channel = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)

        for target in self.treasure_memory.values():
            if target.available and _is_valid_position(target.pos):
                row, col = self._project_to_local(hero_pos, target.pos)
                self._mark_channel(channel, row, col, 1.0 if target.found else 0.5)

        for target in self.buff_memory.values():
            if target.available and _is_valid_position(target.pos):
                row, col = self._project_to_local(hero_pos, target.pos)
                self._mark_channel(channel, row, col, -1.0 if target.found else -0.5)

        return channel

    def _build_risk_channel(self, hero_pos, monsters):
        channel = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        for monster in monsters:
            if monster["pos"] is None:
                continue
            row, col = self._project_to_local(hero_pos, monster["pos"])
            strength = max(0.2, 1.0 - monster["distance"] / (MAP_DIAGONAL * 0.35))
            self._paint_blob(channel, row, col, strength)
        return np.clip(channel, 0.0, 1.0)

    def _project_to_local(self, hero_pos, target_pos):
        delta = np.asarray(target_pos, dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        dx, dz = float(delta[0]), float(delta[1])
        max_abs = max(abs(dx), abs(dz), 1.0)
        if max_abs > LOCAL_HALF:
            scale = LOCAL_HALF / max_abs
            dx *= scale
            dz *= scale

        row = int(np.clip(round(LOCAL_HALF - dz), 0, Config.LOCAL_MAP_SIZE - 1))
        col = int(np.clip(round(LOCAL_HALF + dx), 0, Config.LOCAL_MAP_SIZE - 1))
        return row, col

    def _mark_channel(self, channel, row, col, value):
        if abs(value) >= abs(channel[row, col]):
            channel[row, col] = value

    def _paint_blob(self, channel, row, col, value):
        for d_row in range(-1, 2):
            for d_col in range(-1, 2):
                rr = row + d_row
                cc = col + d_col
                if 0 <= rr < Config.LOCAL_MAP_SIZE and 0 <= cc < Config.LOCAL_MAP_SIZE:
                    distance = abs(d_row) + abs(d_col)
                    decay = 1.0 if distance == 0 else 0.5 if distance == 1 else 0.25
                    channel[rr, cc] = max(channel[rr, cc], value * decay)

    def _is_double_monster_pinched(self, hero_pos, monsters):
        nearby_monsters = [
            monster
            for monster in monsters
            if monster["active"]
            and monster["pos"] is not None
            and monster["distance"] <= Config.DOUBLE_MONSTER_PINCH_DISTANCE
        ]
        if len(nearby_monsters) < 2:
            return False

        nearby_monsters = sorted(nearby_monsters, key=lambda monster: monster["distance"])[:2]
        vector_a = np.asarray(nearby_monsters[0]["pos"], dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        vector_b = np.asarray(nearby_monsters[1]["pos"], dtype=np.float32) - np.asarray(hero_pos, dtype=np.float32)
        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if norm_a <= 1e-6 or norm_b <= 1e-6:
            return False

        cos_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        return cos_similarity <= Config.DOUBLE_MONSTER_PINCH_COS_THRESHOLD

    def _is_single_monster_phase(self):
        return self.step_no < max(1, int(self.monster_interval_step))

    def _is_early_loot_phase(self, min_monster_distance, speedup_state):
        return (
            speedup_state["speedup_reached"] < 1.0
            and self._is_single_monster_phase()
            and min_monster_distance >= Config.EARLY_LOOT_SAFE_DISTANCE
        )

    def _single_monster_time_decay(self):
        phase_window = float(max(1, min(self.monster_interval_step, self.max_step)))
        return float(np.clip(1.0 - self.step_no / phase_window, 0.0, 1.0))

    def _treasure_priority_multiplier(
        self,
        hero_pos,
        treasure_target,
        monsters,
        min_monster_distance,
        post_speedup,
    ):
        if treasure_target is None:
            return 1.0

        if post_speedup:
            return Config.POST_SPEEDUP_TREASURE_PRIORITY_MULTIPLIER

        if treasure_target.distance > Config.TREASURE_PRIORITY_DISTANCE:
            return 1.0

        active_monsters = [monster for monster in monsters if monster["active"]]
        if self._is_single_monster_phase():
            if min_monster_distance <= Config.SINGLE_MONSTER_TREASURE_PRESSURE_DISTANCE:
                return Config.SINGLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER
            return 1.0

        if not self._is_double_monster_pinched(hero_pos, active_monsters):
            return Config.DOUBLE_MONSTER_TREASURE_PRIORITY_MULTIPLIER

        return 1.0

    def _build_reward(
        self,
        hero_pos,
        min_monster_distance,
        treasure_target,
        buff_target,
        treasure_count,
        buff_count,
        flash_count,
        last_action,
        movement_state,
        monsters,
        second_monster_distance,
        speedup_state,
    ):
        post_speedup = speedup_state["speedup_reached"] >= 1.0
        early_loot_phase = self._is_early_loot_phase(min_monster_distance, speedup_state)
        early_loot_active = early_loot_phase and treasure_target is not None
        survive_multiplier = Config.POST_SPEEDUP_SURVIVE_MULTIPLIER if post_speedup else 1.0
        dist_multiplier = Config.POST_SPEEDUP_DIST_MULTIPLIER if post_speedup else 1.0
        treasure_priority_multiplier = self._treasure_priority_multiplier(
            hero_pos=hero_pos,
            treasure_target=treasure_target,
            monsters=monsters,
            min_monster_distance=min_monster_distance,
            post_speedup=post_speedup,
        )
        if early_loot_active:
            treasure_priority_multiplier *= Config.EARLY_LOOT_TREASURE_PRIORITY_MULTIPLIER

        survive_reward = Config.SURVIVE_REWARD * survive_multiplier
        dist_shaping = Config.DIST_SHAPING_COEF * dist_multiplier * self._away_reward(
            self.last_min_monster_distance,
            min_monster_distance,
        )
        if early_loot_active:
            dist_shaping *= Config.EARLY_LOOT_DIST_SHAPING_MULTIPLIER

        treasure_dist_reward = 0.0
        nearest_visible = self._pick_nearest_visible(self.treasure_memory)
        current_nearest_visible_dist = nearest_visible.distance if nearest_visible is not None else None
        if current_nearest_visible_dist is not None and self.last_nearest_visible_treasure_distance is not None:
            visible_progress = self._towards_reward(
                self.last_nearest_visible_treasure_distance,
                current_nearest_visible_dist,
            )
            treasure_dist_reward = (
                Config.TREASURE_DIST_COEF * treasure_priority_multiplier * visible_progress
            )

        if treasure_target is not None and self.last_target_treasure_distance is not None:
            treasure_progress = self._towards_reward(
                self.last_target_treasure_distance,
                treasure_target.distance,
            )
        else:
            treasure_progress = 0.0

        close_treasure_reward = 0.0
        if treasure_target is not None:
            close_ratio = float(
                np.clip(
                    (Config.TREASURE_URGENCY_DISTANCE - treasure_target.distance)
                    / float(max(1.0, Config.TREASURE_URGENCY_DISTANCE)),
                    0.0,
                    1.0,
                )
            )
            close_treasure_reward = (
                Config.CLOSE_TREASURE_APPROACH_COEF
                * treasure_priority_multiplier
                * close_ratio
                * max(treasure_progress, 0.0)
            )

        guide_dist_reward = 0.0
        if treasure_target is None and buff_target is not None and self.last_target_buff_distance is not None:
            guide_dist_reward = Config.EXIT_DIST_COEF * self._towards_reward(
                self.last_target_buff_distance,
                buff_target.distance,
            )

        treasure_reward = (
            max(0, treasure_count - self.last_treasure_count)
            * Config.TREASURE_REWARD
            * treasure_priority_multiplier
        )
        early_loot_collection_bonus = 0.0
        treasure_gain = max(0, treasure_count - self.last_treasure_count)
        if treasure_gain > 0 and not post_speedup and self._is_single_monster_phase():
            time_decay = self._single_monster_time_decay()
            early_loot_collection_bonus += treasure_gain * Config.EARLY_LOOT_COLLECTION_BONUS * time_decay
            if self.last_treasure_count <= 0:
                early_loot_collection_bonus += Config.EARLY_LOOT_FIRST_TREASURE_BONUS * time_decay
        buff_reward = max(0, buff_count - self.last_buff_count) * Config.BUFF_REWARD

        treasure_miss_penalty = 0.0
        if (
            treasure_target is not None
            and self.last_target_treasure_id == treasure_target.config_id
            and self.last_target_treasure_distance is not None
            and treasure_count <= self.last_treasure_count
        ):
            moved_away = treasure_target.distance - self.last_target_treasure_distance
            if (
                self.last_target_treasure_distance <= Config.TREASURE_MISS_DISTANCE
                and moved_away >= Config.TREASURE_MISS_MARGIN
            ):
                miss_ratio = float(np.clip(moved_away / TARGET_DIST_SCALE, 0.0, 1.0))
                treasure_miss_penalty = (
                    -Config.TREASURE_MISS_PENALTY
                    * treasure_priority_multiplier
                    * (0.5 + 0.5 * miss_ratio)
                )

        early_loot_stall_penalty = 0.0
        if (
            early_loot_active
            and self.last_target_treasure_id == treasure_target.config_id
            and self.last_target_treasure_distance is not None
        ):
            distance_gain = self.last_target_treasure_distance - treasure_target.distance
            if distance_gain <= Config.EARLY_LOOT_STALL_PROGRESS_THRESHOLD:
                max_stall = max(1, Config.EARLY_LOOT_STALL_STEP_THRESHOLD * 2)
                self.early_loot_stall_steps = min(self.early_loot_stall_steps + 1, max_stall)
            else:
                self.early_loot_stall_steps = 0

            if self.early_loot_stall_steps >= Config.EARLY_LOOT_STALL_STEP_THRESHOLD:
                stall_ratio = float(
                    np.clip(
                        (self.early_loot_stall_steps - Config.EARLY_LOOT_STALL_STEP_THRESHOLD + 1)
                        / float(max(1, Config.EARLY_LOOT_STALL_STEP_THRESHOLD)),
                        0.0,
                        1.0,
                    )
                )
                early_loot_stall_penalty = -Config.EARLY_LOOT_STALL_PENALTY * stall_ratio
        else:
            self.early_loot_stall_steps = 0

        pre_speedup_buffer_reward = 0.0
        if speedup_state["pre_speedup_buffer_ratio"] > 0.0:
            pressure_distance = min(min_monster_distance, second_monster_distance)
            safety_margin = float(
                np.clip(
                    (pressure_distance - Config.PRE_SPEEDUP_BUFFER_SAFE_DISTANCE) / TARGET_DIST_SCALE,
                    -1.0,
                    1.0,
                )
            )
            pre_speedup_buffer_reward = (
                Config.PRE_SPEEDUP_BUFFER_COEF
                * speedup_state["pre_speedup_buffer_ratio"]
                * safety_margin
            )

        second_monster_pressure_penalty = 0.0
        if second_monster_distance < Config.SECOND_MONSTER_PRESSURE_THRESHOLD:
            stage_multiplier = 1.25 if post_speedup else 1.0
            pressure_ratio = float(
                np.clip(
                    (Config.SECOND_MONSTER_PRESSURE_THRESHOLD - second_monster_distance)
                    / TARGET_DIST_SCALE,
                    0.0,
                    1.0,
                )
            )
            second_monster_pressure_penalty = (
                -Config.SECOND_MONSTER_PRESSURE_COEF * stage_multiplier * pressure_ratio
            )

        flash_escape_reward = 0.0
        flash_used = last_action >= 8
        if flash_used and self.last_min_monster_distance < Config.FLASH_DANGER_DISTANCE:
            flash_escape_reward = Config.FLASH_ESCAPE_REWARD_COEF * max(
                self._away_reward(self.last_min_monster_distance, min_monster_distance),
                0.0,
            )

        flash_waste_penalty = 0.0
        flash_direction_reward = 0.0
        flash_through_wall_reward = 0.0
        if flash_used:
            escape_gain = min_monster_distance - self.last_min_monster_distance
            gained_resource = (treasure_count > self.last_treasure_count) or (
                buff_count > self.last_buff_count
            )
            if self.last_min_monster_distance < Config.FLASH_DANGER_DISTANCE:
                nearest_monster = min(
                    (monster for monster in monsters if monster["active"]),
                    key=lambda monster: monster["distance"],
                    default=None,
                )
                flash_dir = int(last_action) - 8
                monster_dir = (
                    None
                    if nearest_monster is None
                    else _relative_direction_to_move_action(nearest_monster["direction"])
                )
                if monster_dir is not None:
                    gap = _circular_action_gap(flash_dir, monster_dir)
                    alignment = 1.0 if gap == 0 else 0.5 if gap == 1 else 0.0
                    danger_ratio = float(
                        np.clip(
                            (Config.FLASH_DANGER_DISTANCE - self.last_min_monster_distance)
                            / float(max(1.0, Config.FLASH_DANGER_DISTANCE)),
                            0.0,
                            1.0,
                        )
                    )
                    if (
                        alignment > 0.0
                        and escape_gain >= -Config.FLASH_DIRECTION_MAX_DISTANCE_DROP
                    ):
                        flash_direction_reward = (
                            Config.FLASH_DIRECTION_REWARD_COEF * danger_ratio * alignment
                        )
            if (
                movement_state["flash_through_wall"]
                and escape_gain >= -Config.FLASH_THROUGH_WALL_MAX_DISTANCE_DROP
            ):
                blocked_ratio = float(
                    np.clip(
                        movement_state["flash_path_blocked_cells"]
                        / float(max(1, Config.FLASH_THROUGH_WALL_SCAN_STEPS)),
                        0.0,
                        1.0,
                    )
                )
                move_ratio = float(
                    np.clip(
                        movement_state["moved_distance"]
                        / float(max(1.0, Config.FLASH_THROUGH_WALL_MIN_MOVE_DISTANCE * 2.0)),
                        0.0,
                        1.0,
                    )
                )
                utility_ratio = max(
                    0.0,
                    float(np.clip(escape_gain / float(max(1.0, Config.FLASH_WASTE_MIN_ESCAPE_GAIN)), 0.0, 1.0)),
                )
                if treasure_progress > 0.0:
                    utility_ratio = max(utility_ratio, float(np.clip(treasure_progress, 0.0, 1.0)))
                if gained_resource:
                    utility_ratio = 1.0
                flash_through_wall_reward = (
                    Config.FLASH_THROUGH_WALL_REWARD_COEF
                    * (0.5 + 0.5 * blocked_ratio)
                    * (0.5 + 0.5 * move_ratio)
                    * (0.6 + 0.4 * utility_ratio)
                )
            if escape_gain < Config.FLASH_WASTE_MIN_ESCAPE_GAIN and not gained_resource:
                waste_multiplier = (
                    Config.FLASH_FAR_WASTE_MULTIPLIER
                    if self.last_min_monster_distance >= Config.FLASH_DANGER_DISTANCE
                    else 1.0
                )
                flash_waste_penalty = -Config.FLASH_WASTE_PENALTY * waste_multiplier

        hit_wall_penalty = -Config.HIT_WALL_PENALTY if movement_state["hit_wall"] else 0.0
        revisit_intensity = self._revisit_intensity(hero_pos)
        revisit_penalty = -Config.REVISIT_PENALTY_COEF * revisit_intensity
        if early_loot_active:
            revisit_penalty *= Config.EARLY_LOOT_REVISIT_PENALTY_MULTIPLIER
        visible_monster_count = sum(1 for monster in monsters if monster["active"] and monster["visible"])
        stagnation_penalty = -Config.STAGNATION_PENALTY_COEF * movement_state["stagnation_ratio"]
        if visible_monster_count == 0:
            stagnation_penalty *= Config.NO_VISION_STAGNATION_MULTIPLIER
        oscillation_penalty = -Config.OSCILLATION_PENALTY_COEF * movement_state["oscillation_ratio"]
        no_vision_patrol_bonus = 0.0
        if (
            visible_monster_count == 0
            and treasure_target is None
            and buff_target is None
            and movement_state["moved_distance"] > 0.0
        ):
            moved_ratio = float(
                np.clip(
                    movement_state["moved_distance"] / float(max(1.0, Config.NO_VISION_PATROL_MOVE_DISTANCE)),
                    0.0,
                    1.0,
                )
            )
            no_vision_patrol_bonus = (
                Config.NO_VISION_PATROL_BONUS_COEF * moved_ratio * max(0.0, 1.0 - revisit_intensity)
            )
        explore_bonus = self._explore_bonus(hero_pos)
        if early_loot_active:
            explore_bonus *= Config.EARLY_LOOT_EXPLORE_BONUS_MULTIPLIER

        reward_terms = {
            "survive_reward": survive_reward,
            "dist_shaping": dist_shaping,
            "treasure_dist_reward": treasure_dist_reward,
            "close_treasure_reward": close_treasure_reward,
            "guide_dist_reward": guide_dist_reward,
            "treasure_reward": treasure_reward,
            "early_loot_collection_bonus": early_loot_collection_bonus,
            "buff_reward": buff_reward,
            "treasure_miss_penalty": treasure_miss_penalty,
            "early_loot_stall_penalty": early_loot_stall_penalty,
            "pre_speedup_buffer_reward": pre_speedup_buffer_reward,
            "second_monster_pressure_penalty": second_monster_pressure_penalty,
            "flash_escape_reward": flash_escape_reward,
            "flash_direction_reward": flash_direction_reward,
            "flash_through_wall_reward": flash_through_wall_reward,
            "flash_waste_penalty": flash_waste_penalty,
            "hit_wall_penalty": hit_wall_penalty,
            "stagnation_penalty": stagnation_penalty,
            "oscillation_penalty": oscillation_penalty,
            "revisit_penalty": revisit_penalty,
            "no_vision_patrol_bonus": no_vision_patrol_bonus,
            "explore_bonus": explore_bonus,
        }
        reward = float(sum(reward_terms.values()))
        return reward, reward_terms

    def _towards_reward(self, last_distance, current_distance):
        return float(np.clip((last_distance - current_distance) / TARGET_DIST_SCALE, -1.0, 1.0))

    def _away_reward(self, last_distance, current_distance):
        return float(np.clip((current_distance - last_distance) / TARGET_DIST_SCALE, -1.0, 1.0))

    def _revisit_intensity(self, hero_pos):
        hero_x = int(np.clip(round(hero_pos[0]), 0, MAP_SIZE - 1))
        hero_z = int(np.clip(round(hero_pos[1]), 0, MAP_SIZE - 1))
        half = Config.REVISIT_WINDOW_SIZE // 2

        values = []
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                x = hero_x + dx
                z = hero_z + dz
                if 0 <= x < MAP_SIZE and 0 <= z < MAP_SIZE:
                    values.append(self.visit_heat[x, z])

        if not values:
            return 0.0

        revisit_mean = float(np.mean(values))
        return float(np.clip((revisit_mean - 1.0) / 2.0, 0.0, 1.0))

    def _explore_bonus(self, hero_pos):
        if not Config.ENABLE_EXPLORE_BONUS:
            return 0.0

        grid = self._grid_cell(hero_pos)
        if grid == self.last_grid:
            return 0.0

        visit_count = self.visit_counts.get(grid, 0) + 1
        self.visit_counts[grid] = visit_count
        self.last_grid = grid

        novelty_ratio = 1.0 / math.sqrt(float(visit_count))
        if novelty_ratio < Config.EXPLORE_BONUS_MIN_RATIO:
            return 0.0

        return Config.EXPLORE_BONUS_SCALE * novelty_ratio

    def _grid_cell(self, hero_pos):
        grid_size = max(1, int(Config.EXPLORE_BONUS_GRID_SIZE))
        bucket_size = MAP_SIZE / grid_size
        x_idx = min(grid_size - 1, max(0, int(hero_pos[0] / bucket_size)))
        z_idx = min(grid_size - 1, max(0, int(hero_pos[1] / bucket_size)))
        return x_idx, z_idx
