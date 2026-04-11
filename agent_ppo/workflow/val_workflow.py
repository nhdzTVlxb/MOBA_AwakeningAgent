#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
验证流程

在验证地图上运行对局，评估模型泛化能力。
"""

import numpy as np
import random
from agent_ppo.conf.conf import Config, CurriculumConfig
from agent_ppo.feature.definition import ObsData
from agent_ppo.feature.preprocessor import get_hero_info_and_pos
from agent_ppo.metrics import EpisodeMetrics


def run_validation(agent, env, usr_conf, logger, episode_cnt=0):
    """
    运行一局验证对局
    
    Args:
        agent: Agent实例
        env: 环境实例
        usr_conf: 用户配置（会覆盖地图ID）
        logger: 日志器
        episode_cnt: 当前训练局数（用于日志）
    
    Returns:
        metrics: EpisodeMetrics实例的summary字典
    """
    # 使用验证地图ID
    valid_map_ids = CurriculumConfig.VALID_MAP_IDS
    map_id = random.choice(valid_map_ids)
    usr_conf['env_conf']['map_id'] = map_id
    
    logger.info(f"[VALIDATION] Episode {episode_cnt} on map {map_id}")
    
    # 重置环境
    env_obs = env.reset(usr_conf)
    
    # 重置Agent
    agent.reset(env_obs)
    
    # ========== 设置起始位置（与训练保持一致） ==========
    observation = env_obs.get("observation", {})
    frame_state = observation.get("frame_state", {})
    hero_info, hero_pos = get_hero_info_and_pos(frame_state)
    agent.preprocessor.set_start_pos(hero_pos)
    
    # 初始观测
    obs_data, remain_info = agent.observation_process(env_obs)
    
    # 获取加速步数配置
    speed_up_step = usr_conf.get('env_conf', {}).get('monster_speed_up_step', Config.MONSTER_SPEED_UP_STEP)
    
    # 初始化指标收集器
    metrics = EpisodeMetrics(speed_up_step)
    
    done = False
    step = 0
    total_reward = 0.0
    total_distance_reward = 0.0
    
    while not done:
        # 推理（使用贪心策略）
        act_data = agent.predict([obs_data])[0]
        act = agent.action_process(act_data, is_stochastic=False)
        
        # 与环境交互
        env_reward, env_obs = env.step(act)
        
        terminated = env_obs.get("terminated", False)
        truncated = env_obs.get("truncated", False)
        step += 1
        done = terminated or truncated
        
        # 下一帧观测
        _obs_data, _remain_info = agent.observation_process(env_obs)
        
        # ========== 关键修复：从 _remain_info 读取当前帧的奖励（与训练一致） ==========
        reward = _remain_info.get("reward", [0.0])[0] if isinstance(_remain_info.get("reward"), list) else _remain_info.get("reward", 0.0)
        shaped_reward = _remain_info.get("shaped_reward", 0.0)
        distance_reward = _remain_info.get("distance_reward", 0.0)
        
        total_reward += reward
        total_distance_reward += distance_reward
        
        # 获取得分信息
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        score_info = frame_state.get("score_info", {})
        step_score = score_info.get("step_score", 0)
        treasure_score = score_info.get("treasure_score", 0)
        total_score = score_info.get("total_score", 0)
        
        # 判断是否使用了闪现
        is_flash = act >= 8
        
        # 获取闪现是否可用
        hero_info, _ = get_hero_info_and_pos(frame_state)
        talent = hero_info.get('talent', {})
        flash_ready = talent.get('status', 0) == 1
        
        # 更新指标
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
        
        # 更新状态
        obs_data = _obs_data
        remain_info = _remain_info
    
    # 终局处理
    terminal_reward = 0.0
    if terminated:
        terminal_reward = -10.0
    elif not truncated:
        terminal_reward = 10.0
    
    metrics.set_terminal_state(terminated, truncated, terminal_reward)
    total_reward += terminal_reward
    
    # 获取最终指标
    summary = metrics.get_summary()
    summary['total_reward'] = round(total_reward, 4)
    
    # 输出验证日志
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
