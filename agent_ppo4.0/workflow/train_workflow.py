#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
"""

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.resume_utils import read_configured_resume_checkpoint, write_resume_progress_snapshot
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


DIST_BUCKET_TO_DISTANCE = {
    0: 15.0,
    1: 45.0,
    2: 75.0,
    3: 105.0,
    4: 135.0,
    5: 165.0,
}
MAX_DISTANCE = 180.0


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_position(obj):
    if isinstance(obj, dict):
        if "pos" in obj and isinstance(obj["pos"], dict):
            obj = obj["pos"]
        if "x" in obj and "z" in obj:
            return np.array(
                [_safe_float(obj.get("x", 0.0)), _safe_float(obj.get("z", 0.0))],
                dtype=np.float32,
            )
    return None


def _distance_from_bucket(dist_bucket):
    return DIST_BUCKET_TO_DISTANCE.get(_safe_int(dist_bucket, 5), DIST_BUCKET_TO_DISTANCE[5])


def _extract_observation_parts(env_obs):
    observation = env_obs.get("observation", {})
    return observation, observation.get("frame_state", {}), observation.get("env_info", {})


def _extract_hero(frame_state):
    heroes = frame_state.get("heroes", {})
    if isinstance(heroes, list):
        return heroes[0] if heroes else {}
    return heroes if isinstance(heroes, dict) else {}


def _score_snapshot(frame_state, env_info):
    hero = _extract_hero(frame_state)
    step_score = _safe_float(env_info.get("step_score", hero.get("step_score", 0.0)))
    treasure_score = _safe_float(env_info.get("treasure_score", hero.get("treasure_score", 0.0)))
    total_score = _safe_float(env_info.get("total_score", step_score + treasure_score))
    treasures_collected = _safe_int(
        env_info.get("treasures_collected", hero.get("treasure_collected_count", 0))
    )
    return total_score, step_score, treasure_score, treasures_collected


def _entity_distance(entity, hero_pos):
    pos = _extract_position(entity)
    if pos is not None and hero_pos is not None:
        return float(np.linalg.norm(pos - hero_pos))
    return float(_distance_from_bucket(entity.get("hero_l2_distance", entity.get("distance_bucket", 5))))


def _nearest_treasure_distance(frame_state, hero_pos):
    distances = []
    for organ in frame_state.get("organs", []) or []:
        if _safe_int(organ.get("sub_type", 0)) != 1:
            continue
        if _safe_int(organ.get("status", 1)) == 0:
            continue
        distances.append(_entity_distance(organ, hero_pos))
    if not distances:
        return -1.0
    return float(min(distances))


def _danger_level(frame_state, hero_pos):
    distances = [_entity_distance(monster, hero_pos) for monster in (frame_state.get("monsters", []) or [])]
    if not distances:
        return 0.0
    min_distance = min(distances)
    return float(np.clip(1.0 - min_distance / MAX_DISTANCE, 0.0, 1.0))


def _round_metric_dict(metric_dict):
    rounded = {}
    for key, value in metric_dict.items():
        if isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = round(float(value), 4)
    return rounded


def _mean_metric_dict(metric_dicts):
    if not metric_dicts:
        return {}
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    return {
        key: float(np.mean([_safe_float(metric_dict.get(key, 0.0)) for metric_dict in metric_dicts]))
        for key in keys
    }


def _extract_env_conf(conf):
    if not isinstance(conf, dict):
        return {}
    env_conf = conf.get("env_conf", conf)
    return env_conf if isinstance(env_conf, dict) else {}


def _describe_maps(conf):
    env_conf = _extract_env_conf(conf)
    return env_conf.get("map", "unknown")


def _curriculum_stage_start_episode(stage_name):
    previous_max_train_episode = -1
    for stage in Config.CURRICULUM_STAGES:
        if stage.get("name") == stage_name:
            return max(0, previous_max_train_episode + 1)
        previous_max_train_episode = int(stage.get("max_train_episode", previous_max_train_episode))
    return None


@dataclass
class EpisodeMetrics:
    speedup_reached: float = 0.0
    pre_speedup_steps: float = 0.0
    post_speedup_steps: float = 0.0
    phase_time_to_speedup: float = 0.0
    pre_speedup_shaped_reward: float = 0.0
    post_speedup_shaped_reward: float = 0.0
    early_loot_collection_bonus: float = 0.0
    early_loot_stall_penalty: float = 0.0
    pre_speedup_buffer_reward: float = 0.0
    second_monster_pressure_penalty: float = 0.0
    flash_direction_reward: float = 0.0
    flash_through_wall_reward: float = 0.0
    flash_waste_penalty: float = 0.0
    hit_wall_penalty: float = 0.0
    stagnation_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    treasure_miss_penalty: float = 0.0
    no_vision_patrol_bonus: float = 0.0
    pre_speedup_step_score_gain: float = 0.0
    post_speedup_step_score_gain: float = 0.0
    pre_speedup_treasure_gain: float = 0.0
    post_speedup_treasure_gain: float = 0.0
    pre_speedup_treasures_collected: float = 0.0
    post_speedup_treasures_collected: float = 0.0
    time_to_first_treasure: float = -1.0
    pre_speedup_total_score_gain: float = 0.0
    post_speedup_total_score_gain: float = 0.0
    pre_speedup_terminal_bonus: float = 0.0
    post_speedup_terminal_bonus: float = 0.0
    reward: float = 0.0
    total_score: float = 0.0
    step_score: float = 0.0
    treasure_score: float = 0.0
    treasures_collected: float = 0.0
    episode_steps: float = 0.0
    post_speedup_terminated: float = 0.0
    terminated_flag: float = 0.0
    completed_flag: float = 0.0
    abnormal_truncated_flag: float = 0.0
    danger_level: float = 0.0
    nearest_treasure_dist: float = -1.0
    _last_step_score: float = field(default=0.0, repr=False)
    _last_treasure_score: float = field(default=0.0, repr=False)
    _last_total_score: float = field(default=0.0, repr=False)
    _last_treasures_collected: float = field(default=0.0, repr=False)
    _phase_obs_count: float = field(default=0.0, repr=False)

    def observe_step(self, env_obs, shaped_reward, reward_terms=None, speedup_state=None, step_idx=0):
        observation, frame_state, env_info = _extract_observation_parts(env_obs)
        reward_terms = reward_terms or {}
        speedup_state = speedup_state or {}

        self.speedup_reached = max(
            self.speedup_reached,
            float(speedup_state.get("speedup_reached", 0.0)),
        )
        self.phase_time_to_speedup += float(speedup_state.get("time_to_speedup_norm", 0.0))
        self._phase_obs_count += 1.0

        phase = "post" if self.speedup_reached else "pre"
        setattr(self, f"{phase}_speedup_steps", getattr(self, f"{phase}_speedup_steps") + 1.0)
        setattr(
            self,
            f"{phase}_speedup_shaped_reward",
            getattr(self, f"{phase}_speedup_shaped_reward") + float(shaped_reward),
        )
        self.early_loot_collection_bonus += float(reward_terms.get("early_loot_collection_bonus", 0.0))
        self.early_loot_stall_penalty += float(reward_terms.get("early_loot_stall_penalty", 0.0))
        self.pre_speedup_buffer_reward += float(reward_terms.get("pre_speedup_buffer_reward", 0.0))
        self.second_monster_pressure_penalty += float(
            reward_terms.get("second_monster_pressure_penalty", 0.0)
        )
        self.flash_direction_reward += float(reward_terms.get("flash_direction_reward", 0.0))
        self.flash_through_wall_reward += float(reward_terms.get("flash_through_wall_reward", 0.0))
        self.flash_waste_penalty += float(reward_terms.get("flash_waste_penalty", 0.0))
        self.hit_wall_penalty += float(reward_terms.get("hit_wall_penalty", 0.0))
        self.stagnation_penalty += float(reward_terms.get("stagnation_penalty", 0.0))
        self.oscillation_penalty += float(reward_terms.get("oscillation_penalty", 0.0))
        self.treasure_miss_penalty += float(reward_terms.get("treasure_miss_penalty", 0.0))
        self.no_vision_patrol_bonus += float(reward_terms.get("no_vision_patrol_bonus", 0.0))

        total_score, step_score, treasure_score, treasures_collected = _score_snapshot(
            frame_state,
            env_info,
        )
        setattr(
            self,
            f"{phase}_speedup_step_score_gain",
            getattr(self, f"{phase}_speedup_step_score_gain") + (step_score - self._last_step_score),
        )
        setattr(
            self,
            f"{phase}_speedup_treasure_gain",
            getattr(self, f"{phase}_speedup_treasure_gain") + (treasure_score - self._last_treasure_score),
        )
        setattr(
            self,
            f"{phase}_speedup_total_score_gain",
            getattr(self, f"{phase}_speedup_total_score_gain") + (total_score - self._last_total_score),
        )
        treasure_delta = max(0.0, float(treasures_collected) - self._last_treasures_collected)
        setattr(
            self,
            f"{phase}_speedup_treasures_collected",
            getattr(self, f"{phase}_speedup_treasures_collected") + treasure_delta,
        )

        self._last_step_score = step_score
        self._last_treasure_score = treasure_score
        self._last_total_score = total_score
        self._last_treasures_collected = float(treasures_collected)

        self.total_score = total_score
        self.step_score = step_score
        self.treasure_score = treasure_score
        self.treasures_collected = float(treasures_collected)
        self.episode_steps = float(_safe_int(env_info.get("finished_steps", observation.get("step_no", step_idx))))
        if treasure_delta > 0.0 and self.time_to_first_treasure < 0.0:
            self.time_to_first_treasure = self.episode_steps

    def finalize(self, env_obs, final_bonus, terminated, truncated, step_idx):
        observation, frame_state, env_info = _extract_observation_parts(env_obs)
        phase = "post" if self.speedup_reached else "pre"
        setattr(
            self,
            f"{phase}_speedup_terminal_bonus",
            getattr(self, f"{phase}_speedup_terminal_bonus") + float(final_bonus),
        )

        total_score, step_score, treasure_score, treasures_collected = _score_snapshot(
            frame_state,
            env_info,
        )
        finished_steps = _safe_int(env_info.get("finished_steps", observation.get("step_no", step_idx)))
        finished_steps = finished_steps if finished_steps > 0 else step_idx
        max_step = max(1, _safe_int(env_info.get("max_step", finished_steps)))

        hero_pos = _extract_position(env_info.get("pos"))
        if hero_pos is None:
            hero_pos = _extract_position(_extract_hero(frame_state))

        self.total_score = total_score
        self.step_score = step_score
        self.treasure_score = treasure_score
        self.treasures_collected = float(treasures_collected)
        self.episode_steps = float(finished_steps)
        self.reward = (
            self.pre_speedup_shaped_reward
            + self.post_speedup_shaped_reward
            + self.pre_speedup_terminal_bonus
            + self.post_speedup_terminal_bonus
        )
        self.terminated_flag = 1.0 if terminated else 0.0
        self.completed_flag = 1.0 if (not terminated and finished_steps >= max_step) else 0.0
        self.abnormal_truncated_flag = 1.0 if (truncated and not terminated and finished_steps < max_step) else 0.0
        self.post_speedup_terminated = 1.0 if (terminated and self.speedup_reached) else 0.0
        self.danger_level = _danger_level(frame_state, hero_pos)
        self.nearest_treasure_dist = _nearest_treasure_distance(frame_state, hero_pos)
        if self.time_to_first_treasure < 0.0:
            self.time_to_first_treasure = float(max_step + 1)

    def as_train_monitor_dict(self):
        mean_phase_time_to_speedup = self.phase_time_to_speedup / max(1.0, self._phase_obs_count)
        total_treasures_collected = self.pre_speedup_treasures_collected + self.post_speedup_treasures_collected
        pre_speedup_treasure_rate = (
            self.pre_speedup_treasures_collected / total_treasures_collected
            if total_treasures_collected > 1e-6
            else 0.0
        )
        return {
            "train_reward": self.reward,
            "train_total_score": self.total_score,
            "train_step_score": self.step_score,
            "train_treasure_score": self.treasure_score,
            "train_treasures_collected": self.treasures_collected,
            "train_episode_steps": self.episode_steps,
            "train_speedup_reached": self.speedup_reached,
            "train_phase_time_to_speedup": mean_phase_time_to_speedup,
            "train_pre_speedup_steps": self.pre_speedup_steps,
            "train_post_speedup_steps": self.post_speedup_steps,
            "train_pre_speedup_reward": self.pre_speedup_shaped_reward + self.pre_speedup_terminal_bonus,
            "train_post_speedup_reward": self.post_speedup_shaped_reward + self.post_speedup_terminal_bonus,
            "train_pre_speedup_shaped_reward": self.pre_speedup_shaped_reward,
            "train_post_speedup_shaped_reward": self.post_speedup_shaped_reward,
            "train_early_loot_collection_bonus": self.early_loot_collection_bonus,
            "train_early_loot_stall_penalty": self.early_loot_stall_penalty,
            "train_pre_speedup_buffer_reward": self.pre_speedup_buffer_reward,
            "train_second_monster_pressure_penalty": self.second_monster_pressure_penalty,
            "train_flash_direction_reward": self.flash_direction_reward,
            "train_flash_through_wall_reward": self.flash_through_wall_reward,
            "train_flash_waste_penalty": self.flash_waste_penalty,
            "train_hit_wall_penalty": self.hit_wall_penalty,
            "train_stagnation_penalty": self.stagnation_penalty,
            "train_oscillation_penalty": self.oscillation_penalty,
            "train_treasure_miss_penalty": self.treasure_miss_penalty,
            "train_no_vision_patrol_bonus": self.no_vision_patrol_bonus,
            "train_time_to_first_treasure": self.time_to_first_treasure,
            "train_pre_speedup_step_score_gain": self.pre_speedup_step_score_gain,
            "train_post_speedup_step_score_gain": self.post_speedup_step_score_gain,
            "train_pre_speedup_treasure_gain": self.pre_speedup_treasure_gain,
            "train_post_speedup_treasure_gain": self.post_speedup_treasure_gain,
            "train_pre_speedup_treasures_collected": self.pre_speedup_treasures_collected,
            "train_post_speedup_treasures_collected": self.post_speedup_treasures_collected,
            "train_pre_speedup_treasure_rate": pre_speedup_treasure_rate,
            "train_pre_speedup_total_score_gain": self.pre_speedup_total_score_gain,
            "train_post_speedup_total_score_gain": self.post_speedup_total_score_gain,
        }

    def as_val_episode_dict(self):
        mean_phase_time_to_speedup = self.phase_time_to_speedup / max(1.0, self._phase_obs_count)
        total_treasures_collected = self.pre_speedup_treasures_collected + self.post_speedup_treasures_collected
        pre_speedup_treasure_rate = (
            self.pre_speedup_treasures_collected / total_treasures_collected
            if total_treasures_collected > 1e-6
            else 0.0
        )
        return {
            "reward": self.reward,
            "total_score": self.total_score,
            "step_score": self.step_score,
            "treasure_score": self.treasure_score,
            "treasures_collected": self.treasures_collected,
            "episode_steps": self.episode_steps,
            "speedup_reached": self.speedup_reached,
            "phase_time_to_speedup": mean_phase_time_to_speedup,
            "pre_speedup_steps": self.pre_speedup_steps,
            "post_speedup_steps": self.post_speedup_steps,
            "pre_speedup_reward": self.pre_speedup_shaped_reward + self.pre_speedup_terminal_bonus,
            "post_speedup_reward": self.post_speedup_shaped_reward + self.post_speedup_terminal_bonus,
            "pre_speedup_shaped_reward": self.pre_speedup_shaped_reward,
            "post_speedup_shaped_reward": self.post_speedup_shaped_reward,
            "early_loot_collection_bonus": self.early_loot_collection_bonus,
            "early_loot_stall_penalty": self.early_loot_stall_penalty,
            "pre_speedup_buffer_reward": self.pre_speedup_buffer_reward,
            "second_monster_pressure_penalty": self.second_monster_pressure_penalty,
            "flash_direction_reward": self.flash_direction_reward,
            "flash_through_wall_reward": self.flash_through_wall_reward,
            "flash_waste_penalty": self.flash_waste_penalty,
            "hit_wall_penalty": self.hit_wall_penalty,
            "stagnation_penalty": self.stagnation_penalty,
            "oscillation_penalty": self.oscillation_penalty,
            "treasure_miss_penalty": self.treasure_miss_penalty,
            "no_vision_patrol_bonus": self.no_vision_patrol_bonus,
            "time_to_first_treasure": self.time_to_first_treasure,
            "pre_speedup_step_score_gain": self.pre_speedup_step_score_gain,
            "post_speedup_step_score_gain": self.post_speedup_step_score_gain,
            "pre_speedup_treasure_gain": self.pre_speedup_treasure_gain,
            "post_speedup_treasure_gain": self.post_speedup_treasure_gain,
            "pre_speedup_treasures_collected": self.pre_speedup_treasures_collected,
            "post_speedup_treasures_collected": self.post_speedup_treasures_collected,
            "pre_speedup_treasure_rate": pre_speedup_treasure_rate,
            "pre_speedup_total_score_gain": self.pre_speedup_total_score_gain,
            "post_speedup_total_score_gain": self.post_speedup_total_score_gain,
            "pre_speedup_terminal_bonus": self.pre_speedup_terminal_bonus,
            "post_speedup_terminal_bonus": self.post_speedup_terminal_bonus,
            "post_speedup_terminated": self.post_speedup_terminated,
            "terminated_flag": self.terminated_flag,
            "completed_flag": self.completed_flag,
            "abnormal_truncated_flag": self.abnormal_truncated_flag,
            "danger_level": self.danger_level,
            "nearest_treasure_dist": self.nearest_treasure_dist,
        }


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    train_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    eval_conf = read_usr_conf("agent_ppo/conf/eval_env_conf.toml", logger)
    if train_conf is None:
        logger.error("train_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=train_conf,
        logger=logger,
        monitor=monitor,
        train_conf=train_conf,
        eval_conf=eval_conf,
        eval_every_n=50,
        eval_episodes=10,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(
        self,
        env,
        agent,
        usr_conf,
        logger,
        monitor,
        train_conf=None,
        eval_conf=None,
        eval_every_n=50,
        eval_episodes=10,
    ):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.completed_episode_count = 0
        self.last_get_training_metrics_time = 0

        self.train_conf = train_conf or usr_conf
        self.eval_conf = eval_conf
        self.eval_every_n = eval_every_n
        self.eval_episodes = eval_episodes
        self.is_eval_mode = False
        self.eval_episode_cnt = 0
        self.train_episode_since_last_eval = 0
        self.train_episode_total = 0
        self.eval_episode_metrics = []
        self._last_progress_report_step = None
        self.resume_checkpoint_state = read_configured_resume_checkpoint()
        self._apply_resume_checkpoint_state()
        self._sync_resume_progress_snapshot()

    def _apply_resume_checkpoint_state(self):
        if not self.resume_checkpoint_state.get("enabled", False):
            return

        metadata = self.resume_checkpoint_state.get("metadata") or {}
        if metadata:
            self.episode_cnt = _safe_int(metadata.get("episode_cnt"), 0)
            self.completed_episode_count = _safe_int(metadata.get("completed_episode_count"), 0)
            self.train_episode_total = _safe_int(metadata.get("train_episode_total"), 0)
            self.train_episode_since_last_eval = _safe_int(
                metadata.get("train_episode_since_last_eval"),
                0,
            )
            if self.logger:
                self.logger.info(
                    f"[RESUME] restore episode progress from {self.resume_checkpoint_state.get('model_file')}: "
                    f"episode_cnt={self.episode_cnt}, completed_episode_count={self.completed_episode_count}, "
                    f"train_episode_total={self.train_episode_total}, "
                    f"train_episode_since_last_eval={self.train_episode_since_last_eval}"
                )
            return

        manifest_metadata = self.resume_checkpoint_state.get("manifest_metadata") or {}
        manifest_train_step = _safe_int(manifest_metadata.get("train_step"), 0)
        if manifest_train_step > 0:
            self.episode_cnt = manifest_train_step
            self.completed_episode_count = manifest_train_step
            self.train_episode_total = manifest_train_step
            self.train_episode_since_last_eval = manifest_train_step % max(1, int(self.eval_every_n))
            if self.logger:
                self.logger.warning(
                    f"[RESUME] checkpoint {self.resume_checkpoint_state.get('configured_source')} "
                    f"does not contain exact episode metadata. Fallback to zip manifest train_step="
                    f"{manifest_train_step} as pseudo progress."
                )
            return

        resume_stage_name = getattr(Config, "RESUME_CURRICULUM_STAGE_NAME", None)
        if not resume_stage_name:
            return

        start_episode = _curriculum_stage_start_episode(resume_stage_name)
        if start_episode is None:
            if self.logger:
                self.logger.warning(
                    f"[CURRICULUM] resume stage {resume_stage_name} not found, keep train_episode_total=0"
                )
            return

        self.train_episode_total = start_episode
        if self.logger:
            self.logger.warning(
                f"[CURRICULUM] resume checkpoint detected at {self.resume_checkpoint_state.get('model_file')}, "
                f"but no episode metadata was found. Fallback train_episode_total={self.train_episode_total} "
                f"so training starts from stage={resume_stage_name}"
            )

    def _snapshot_resume_metadata(self):
        return {
            "episode_cnt": int(self.episode_cnt),
            "completed_episode_count": int(self.completed_episode_count),
            "train_episode_total": int(self.train_episode_total),
            "train_episode_since_last_eval": int(self.train_episode_since_last_eval),
        }

    def _sync_resume_progress_snapshot(self):
        metadata = write_resume_progress_snapshot(self._snapshot_resume_metadata())
        if metadata and hasattr(self.agent, "set_resume_metadata"):
            self.agent.set_resume_metadata(metadata)

    def _sample_from_range(self, bounds):
        low, high = int(bounds[0]), int(bounds[1])
        if low > high:
            low, high = high, low
        return int(np.random.randint(low, high + 1))

    def _select_curriculum_stage(self):
        for stage in Config.CURRICULUM_STAGES:
            if self.train_episode_total <= int(stage["max_train_episode"]):
                return stage
        return Config.CURRICULUM_STAGES[-1]

    def _build_train_episode_conf(self):
        episode_conf = copy.deepcopy(self.train_conf)
        env_conf = _extract_env_conf(episode_conf)
        stage = self._select_curriculum_stage()

        env_conf["map_random"] = True
        env_conf["treasure_count"] = self._sample_from_range(stage["treasure_count"])
        env_conf["buff_count"] = self._sample_from_range(stage["buff_count"])
        env_conf["monster_interval"] = self._sample_from_range(stage["monster_interval"])
        env_conf["monster_speedup"] = self._sample_from_range(stage["monster_speedup"])
        env_conf["max_step"] = int(stage.get("max_step", env_conf.get("max_step", 1000)))
        return episode_conf, stage["name"]

    def _report_episode_progress(self, step, force=False):
        if not self.monitor:
            return
        episode_interval = max(
            1,
            int(getattr(Config, "EPISODE_PROGRESS_REPORT_EPISODE_INTERVAL", 1)),
        )
        if self.episode_cnt > 0 and self.episode_cnt % episode_interval != 0:
            return
        interval = max(1, int(getattr(Config, "EPISODE_PROGRESS_REPORT_INTERVAL", 50)))
        step = int(step)
        if not force and step > 0 and step % interval != 0:
            return
        if not force and self._last_progress_report_step == step:
            return
        self._last_progress_report_step = step

        progress_metrics = {
            "current_episode_id": float(self.episode_cnt),
            "completed_episode_count": float(self.completed_episode_count),
            "current_episode_step": float(step),
            "current_episode_is_eval": 1.0 if self.is_eval_mode else 0.0,
            "train_episode_total": float(self.train_episode_total),
        }
        self.monitor.put_data({os.getpid(): _round_metric_dict(progress_metrics)})

    def _describe_episode_conf(self, conf):
        env_conf = _extract_env_conf(conf)
        return (
            f"maps={env_conf.get('map')} map_random={env_conf.get('map_random')} "
            f"treasure_count={env_conf.get('treasure_count')} buff_count={env_conf.get('buff_count')} "
            f"monster_interval={env_conf.get('monster_interval')} "
            f"monster_speedup={env_conf.get('monster_speedup')} max_step={env_conf.get('max_step')}"
        )

    def run_episodes(self):
        while True:
            if not self.is_eval_mode and self.eval_conf is not None:
                if self.train_episode_since_last_eval >= self.eval_every_n:
                    self.is_eval_mode = True
                    self.eval_episode_cnt = 0
                    self.eval_episode_metrics = []
                    self.logger.info(
                        f"[EVAL] Switching to eval mode on maps {_describe_maps(self.eval_conf)}"
                    )

            if self.is_eval_mode and self.eval_episode_cnt >= self.eval_episodes:
                self.is_eval_mode = False
                self.train_episode_since_last_eval = 0
                self._sync_resume_progress_snapshot()
                self.logger.info(
                    f"[EVAL] Eval completed. Back to training on maps {_describe_maps(self.train_conf)}"
                )

            stage_name = "eval_fixed"
            if self.is_eval_mode:
                current_conf = copy.deepcopy(self.eval_conf)
            else:
                current_conf, stage_name = self._build_train_episode_conf()

            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(current_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs, usr_conf=current_conf)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            episode_metrics = EpisodeMetrics()
            self.episode_cnt += 1
            if not self.is_eval_mode:
                self.train_episode_total += 1
            done = False
            step = 0
            total_reward = 0.0
            self._last_progress_report_step = None
            self._report_episode_progress(step=0, force=True)

            mode_str = "EVAL" if self.is_eval_mode else "TRAIN"
            if self.is_eval_mode:
                self.logger.info(
                    f"[{mode_str}] Episode {self.episode_cnt} start {self._describe_episode_conf(current_conf)}"
                )
            else:
                self.logger.info(
                    f"[{mode_str}] Episode {self.episode_cnt} start stage={stage_name} "
                    f"{self._describe_episode_conf(current_conf)}"
                )

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data, is_stochastic=not self.is_eval_mode)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated
                self._report_episode_progress(step=step)

                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                episode_metrics.observe_step(
                    env_obs,
                    float(reward[0]),
                    reward_terms=_remain_info.get("reward_terms", {}),
                    speedup_state=_remain_info.get("speedup_state", {}),
                    step_idx=step,
                )

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    _, _, env_info = _extract_observation_parts(env_obs)
                    total_score = _safe_float(env_info.get("total_score", 0.0))
                    treasures_collected = _safe_int(env_info.get("treasures_collected", 0))
                    flash_count = _safe_int(env_info.get("flash_count", 0))
                    finished_steps = _safe_int(env_info.get("finished_steps", step))
                    max_step = max(1, _safe_int(env_info.get("max_step", step)))

                    if terminated:
                        final_reward[0] = Config.TERMINATED_PENALTY
                        result_str = "FAIL"
                    elif finished_steps >= max_step:
                        final_reward[0] = Config.TRUNCATED_BONUS
                        result_str = "SURVIVE"
                    else:
                        result_str = "STOP"

                    self.logger.info(
                        f"[GAMEOVER][{mode_str}] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"treasures:{treasures_collected} flash:{flash_count} "
                        f"total_reward:{total_reward:.3f}"
                    )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    self.completed_episode_count += 1
                    self._report_episode_progress(step=step, force=True)

                    episode_metrics.finalize(
                        env_obs=env_obs,
                        final_bonus=float(final_reward[0]),
                        terminated=terminated,
                        truncated=truncated,
                        step_idx=step,
                    )

                    if self.is_eval_mode:
                        self.eval_episode_cnt += 1
                        self.eval_episode_metrics.append(episode_metrics.as_val_episode_dict())
                        if self.eval_episode_cnt >= self.eval_episodes and self.monitor:
                            self.monitor.put_data(
                                {
                                    os.getpid(): _round_metric_dict(
                                        self._build_val_monitor_data(self.eval_episode_metrics)
                                    )
                                }
                            )
                    else:
                        self.train_episode_since_last_eval += 1
                        self._sync_resume_progress_snapshot()
                        if self.monitor:
                            self.monitor.put_data(
                                {os.getpid(): _round_metric_dict(episode_metrics.as_train_monitor_dict())}
                            )

                    if not self.is_eval_mode and collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info

    def _build_val_monitor_data(self, val_episode_metrics):
        mean_metrics = _mean_metric_dict(val_episode_metrics)
        return {
            "val_reward": mean_metrics.get("reward", 0.0),
            "val_total_score": mean_metrics.get("total_score", 0.0),
            "val_step_score": mean_metrics.get("step_score", 0.0),
            "val_treasure_score": mean_metrics.get("treasure_score", 0.0),
            "val_treasures_collected": mean_metrics.get("treasures_collected", 0.0),
            "val_episode_steps": mean_metrics.get("episode_steps", 0.0),
            "val_speedup_reached": mean_metrics.get("speedup_reached", 0.0),
            "val_phase_time_to_speedup": mean_metrics.get("phase_time_to_speedup", 0.0),
            "val_pre_speedup_steps": mean_metrics.get("pre_speedup_steps", 0.0),
            "val_post_speedup_steps": mean_metrics.get("post_speedup_steps", 0.0),
            "val_pre_speedup_reward": mean_metrics.get("pre_speedup_reward", 0.0),
            "val_post_speedup_reward": mean_metrics.get("post_speedup_reward", 0.0),
            "val_pre_speedup_shaped_reward": mean_metrics.get("pre_speedup_shaped_reward", 0.0),
            "val_post_speedup_shaped_reward": mean_metrics.get("post_speedup_shaped_reward", 0.0),
            "val_early_loot_collection_bonus": mean_metrics.get("early_loot_collection_bonus", 0.0),
            "val_early_loot_stall_penalty": mean_metrics.get("early_loot_stall_penalty", 0.0),
            "val_pre_speedup_buffer_reward": mean_metrics.get("pre_speedup_buffer_reward", 0.0),
            "val_second_monster_pressure_penalty": mean_metrics.get(
                "second_monster_pressure_penalty", 0.0
            ),
            "val_flash_direction_reward": mean_metrics.get("flash_direction_reward", 0.0),
            "val_flash_through_wall_reward": mean_metrics.get("flash_through_wall_reward", 0.0),
            "val_flash_waste_penalty": mean_metrics.get("flash_waste_penalty", 0.0),
            "val_hit_wall_penalty": mean_metrics.get("hit_wall_penalty", 0.0),
            "val_stagnation_penalty": mean_metrics.get("stagnation_penalty", 0.0),
            "val_oscillation_penalty": mean_metrics.get("oscillation_penalty", 0.0),
            "val_treasure_miss_penalty": mean_metrics.get("treasure_miss_penalty", 0.0),
            "val_no_vision_patrol_bonus": mean_metrics.get("no_vision_patrol_bonus", 0.0),
            "val_time_to_first_treasure": mean_metrics.get("time_to_first_treasure", 0.0),
            "val_pre_speedup_step_score_gain": mean_metrics.get("pre_speedup_step_score_gain", 0.0),
            "val_post_speedup_step_score_gain": mean_metrics.get("post_speedup_step_score_gain", 0.0),
            "val_pre_speedup_treasure_gain": mean_metrics.get("pre_speedup_treasure_gain", 0.0),
            "val_post_speedup_treasure_gain": mean_metrics.get("post_speedup_treasure_gain", 0.0),
            "val_pre_speedup_treasures_collected": mean_metrics.get("pre_speedup_treasures_collected", 0.0),
            "val_post_speedup_treasures_collected": mean_metrics.get("post_speedup_treasures_collected", 0.0),
            "val_pre_speedup_treasure_rate": mean_metrics.get("pre_speedup_treasure_rate", 0.0),
            "val_pre_speedup_total_score_gain": mean_metrics.get("pre_speedup_total_score_gain", 0.0),
            "val_post_speedup_total_score_gain": mean_metrics.get("post_speedup_total_score_gain", 0.0),
            "val_pre_speedup_terminal_bonus": mean_metrics.get("pre_speedup_terminal_bonus", 0.0),
            "val_post_speedup_terminal_bonus": mean_metrics.get("post_speedup_terminal_bonus", 0.0),
            "val_post_speedup_terminated": mean_metrics.get("post_speedup_terminated", 0.0),
            "val_terminated_rate": mean_metrics.get("terminated_flag", 0.0),
            "val_completed_rate": mean_metrics.get("completed_flag", 0.0),
            "val_abnormal_truncated_rate": mean_metrics.get("abnormal_truncated_flag", 0.0),
            "val_danger_level": mean_metrics.get("danger_level", 0.0),
            "val_nearest_treasure_dist": mean_metrics.get("nearest_treasure_dist", -1.0),
        }
