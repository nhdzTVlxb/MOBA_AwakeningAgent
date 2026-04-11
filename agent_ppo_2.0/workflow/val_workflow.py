#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
验证流程

在验证地图上运行对局，评估模型泛化能力。
"""

import random
from agent_ppo.conf.conf import Config, CurriculumConfig
from agent_ppo.feature.preprocessor import get_hero_info_and_pos
from agent_ppo.metrics import EpisodeMetrics


def run_validation(agent, env, usr_conf, logger, episode_cnt=0):
    """
    运行一局验证对局
    """
    valid_map_ids = CurriculumConfig.VALID_MAP_IDS
    map_id = random.choice(valid_map_ids)

    # 保留 map_id 覆盖逻辑
    usr_conf['env_conf']['map_id'] = map_id

    logger.info(f"[VALIDATION] Episode {episode_cnt} on map {map_id}")

    env_obs = env.reset(usr_conf)
    agent.reset(env_obs)

    observation = env_obs.get("observation", {})
    frame_state = observation.get("frame_state", {})
    _, hero_pos = get_hero_info_and_pos(frame_state)

    # 关键修复：字段名与环境 TOML 对齐
    speed_up_step = usr_conf.get('env_conf', {}).get('monster_speedup', Config.MONSTER_SPEED_UP_STEP)
    monster2_spawn_step = usr_conf.get('env_conf', {}).get('monster_interval', Config.MONSTER2_SPAWN_STEP)

    agent.preprocessor.set_start_pos(hero_pos)
    agent.preprocessor.set_episode_config(
        speed_up_step=speed_up_step,
        monster2_spawn_step=monster2_spawn_step
    )

    obs_data, remain_info = agent.observation_process(env_obs)

    metrics = EpisodeMetrics(speed_up_step)

    done = False
    step = 0
    total_reward = 0.0
    total_distance_reward = 0.0

    while not done:
        act_data = agent.predict([obs_data])[0]
        act = agent.action_process(act_data, is_stochastic=False)

        _, env_obs = env.step(act)

        terminated = env_obs.get("terminated", False)
        truncated = env_obs.get("truncated", False)
        step += 1
        done = terminated or truncated

        _obs_data, _remain_info = agent.observation_process(env_obs)

        reward = _remain_info.get("reward", [0.0])[0] \
            if isinstance(_remain_info.get("reward"), list) \
            else _remain_info.get("reward", 0.0)

        shaped_reward = _remain_info.get("shaped_reward", 0.0)
        distance_reward = _remain_info.get("distance_reward", 0.0)

        total_reward += reward
        total_distance_reward += distance_reward

        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        score_info = frame_state.get("score_info", {})
        step_score = score_info.get("step_score", 0)
        treasure_score = score_info.get("treasure_score", 0)
        total_score = score_info.get("total_score", 0)

        is_flash = act >= 8

        hero_info, _ = get_hero_info_and_pos(frame_state)
        talent = hero_info.get('talent', {})
        flash_ready = talent.get('status', 0) == 1

        metrics.update(
            step=step,
            reward=reward,
            step_score=step_score,
            treasure_score=treasure_score,
            total_score=total_score,
            shaped_reward=shaped_reward,
            is_flash=is_flash,
            is_flash_legal=flash_ready,
            flash_ready=flash_ready,
            distance_reward=distance_reward
        )

        obs_data = _obs_data
        remain_info = _remain_info

    terminal_reward = -10.0 if terminated else 0.0

    metrics.set_terminal_state(terminated, truncated, terminal_reward)
    total_reward += terminal_reward

    summary = metrics.get_summary()
    summary['total_reward'] = round(total_reward, 4)

    logger.info(
        f"[VALIDATION] Episode {episode_cnt} | "
        f"Map:{map_id} | "
        f"Steps:{summary['steps']} | "
        f"Score:{summary['total_score']} | "
        f"Treasures:{summary['treasures']} | "
        f"TotalReward:{total_reward:.3f} | "
        f"DistanceReward:{total_distance_reward:.3f} | "
        f"Terminated:{summary['terminated']} | "
        f"Completed:{summary['completed']}"
    )

    return summary