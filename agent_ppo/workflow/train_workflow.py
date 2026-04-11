#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time
import random

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.feature.preprocessor import get_hero_info_and_pos
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

from agent_ppo.curriculum import CurriculumScheduler
from agent_ppo.metrics import EpisodeMetrics
from agent_ppo.workflow.val_workflow import run_validation
from agent_ppo.conf.conf import CurriculumConfig


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
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
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        
        self.curriculum = CurriculumScheduler(0)
        self.val_interval = CurriculumConfig.VALIDATION_INTERVAL
        
        self.total_terminated = 0
        self.total_completed = 0
        self.total_abnormal = 0

    def run_episodes(self):
        while True:
            # ========== 1. 获取课程配置 ==========
            stage_config = self.curriculum.get_current_config()
            self.logger.info(
                f"[CURRICULUM] Episode {self.episode_cnt} | "
                f"Stage:{stage_config['stage_name']} | "
                f"TreasureNum:{stage_config['treasure_num']} | "
                f"BuffNum:{stage_config['buff_num']} | "
                f"Monster2Spawn:{stage_config['monster2_spawn_step']} | "
                f"SpeedUp:{stage_config['monster_speed_up_step']}"
            )
            
            env_conf = self.usr_conf.get('env_conf', {})
            env_conf['treasure_num'] = stage_config['treasure_num']
            env_conf['buff_num'] = stage_config['buff_num']
            env_conf['monster2_spawn_step'] = stage_config['monster2_spawn_step']
            env_conf['monster_speed_up_step'] = stage_config['monster_speed_up_step']
            self.usr_conf['env_conf'] = env_conf
            
            train_map_ids = self.curriculum.get_train_map_ids()
            map_id = random.choice(train_map_ids)
            self.usr_conf['env_conf']['map_id'] = map_id
            
            # ========== 2. 重置环境 ==========
            env_obs = self.env.reset(self.usr_conf)

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # ========== 3. 重置Agent和指标 ==========
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)
            
            # ========== 设置起始位置（用于欧几里得距离探索奖励） ==========
            observation = env_obs.get("observation", {})
            frame_state = observation.get("frame_state", {})
            hero_info, hero_pos = get_hero_info_and_pos(frame_state)
            self.agent.preprocessor.set_start_pos(hero_pos)
            
            speed_up_step = stage_config['monster_speed_up_step']
            metrics = EpisodeMetrics(speed_up_step)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            total_shaped_reward = 0.0
            total_distance_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start on map {map_id}")

            # ========== 4. 对局循环 ==========
            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs.get("terminated", False)
                truncated = env_obs.get("truncated", False)
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward
                reward = _remain_info.get("reward", [0.0])[0] if isinstance(_remain_info.get("reward"), list) else _remain_info.get("reward", 0.0)
                shaped_reward = _remain_info.get("shaped_reward", 0.0)
                distance_reward = _remain_info.get("distance_reward", 0.0)
                total_reward += reward
                total_shaped_reward += shaped_reward
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
                is_flash_legal = flash_ready
                
                metrics.update(
                    step=step,
                    reward=reward,
                    step_score=step_score,
                    treasure_score=treasure_score,
                    total_score=total_score,
                    shaped_reward=shaped_reward,
                    is_flash=is_flash,
                    is_flash_legal=is_flash_legal,
                    flash_ready=flash_ready,
                    distance_reward=distance_reward
                )

                final_reward = 0.0
                if done:
                    total_score = env_obs.get("observation", {}).get("env_info", {}).get("total_score", 0)

                    if terminated:
                        final_reward = -10.0
                        result_str = "FAIL"
                        self.total_terminated += 1
                    elif not truncated:
                        final_reward = 10.0
                        result_str = "WIN"
                        self.total_completed += 1
                    else:
                        final_reward = 0.0
                        result_str = "TRUNC"
                        self.total_abnormal += 1
                    
                    total_reward += final_reward
                    
                    metrics.set_terminal_state(terminated, truncated, final_reward)
                    
                    visible_treasure_ratio = _remain_info.get("visible_treasure_ratio", 0.0)
                    metrics.set_final_visible_treasure(visible_treasure_ratio)
                    
                    summary = metrics.get_summary()
                    
                    total_episodes = self.total_terminated + self.total_completed + self.total_abnormal
                    terminated_rate = self.total_terminated / max(total_episodes, 1)
                    completed_rate = self.total_completed / max(total_episodes, 1)
                    
                    self.logger.info(
                        f"[GAMEOVER] Episode:{self.episode_cnt} | "
                        f"Result:{result_str} | "
                        f"Map:{map_id} | "
                        f"Stage:{stage_config['stage_name']} | "
                        f"Steps:{summary['steps']}(Pre:{summary['pre_steps']}/Post:{summary['post_steps']}) | "
                        f"Score:{summary['total_score']} | "
                        f"Treasures:{summary['treasures']} | "
                        f"TotalReward:{total_reward:.3f} | "
                        f"ShapedReward:{total_shaped_reward:.3f} | "
                        f"DistanceReward:{total_distance_reward:.3f} | "
                        f"FlashCount:{summary['flash_count']} | "
                        f"LastFlashReady:{summary['last_flash_ready']} | "
                        f"LastFlashLegal:{summary['last_flash_legal']} | "
                        f"FinalVisibleTre:{summary['final_visible_treasure']:.3f} | "
                        f"SpeedupReached:{summary['speedup_reached']} | "
                        f"TerminatedRate:{terminated_rate:.3f} | "
                        f"CompletedRate:{completed_rate:.3f}"
                    )
                    
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward, 4),
                            "total_score": summary['total_score'],
                            "treasures": summary['treasures'],
                            "steps": summary['steps'],
                            "pre_steps": summary['pre_steps'],
                            "post_steps": summary['post_steps'],
                            "pre_shaped_reward": summary['pre_shaped_reward'],
                            "post_shaped_reward": summary['post_shaped_reward'],
                            "pre_treasure_gain": summary['pre_treasure_gain'],
                            "post_treasure_gain": summary['post_treasure_gain'],
                            "pre_terminal": summary['pre_terminal'],
                            "post_terminal": summary['post_terminal'],
                            "flash_count": summary['flash_count'],
                            "last_flash_used": 1 if summary['last_flash_used'] else 0,
                            "last_flash_ready": 1 if summary['last_flash_ready'] else 0,
                            "last_flash_legal": 1 if summary['last_flash_legal'] else 0,
                            "final_danger": summary['final_danger'],
                            "final_treasure_dist": summary['final_treasure_dist'],
                            "final_visible_treasure": summary['final_visible_treasure'],
                            "speedup_reached": 1 if summary['speedup_reached'] else 0,
                            "terminated": 1 if summary['terminated'] else 0,
                            "completed": 1 if summary['completed'] else 0,
                            "terminated_rate": terminated_rate,
                            "completed_rate": completed_rate,
                            "total_distance_reward": summary.get('total_distance_reward', 0),
                            "pre_distance_reward": summary.get('pre_distance_reward', 0),
                            "post_distance_reward": summary.get('post_distance_reward', 0),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=np.array([reward], dtype=np.float32),
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

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
            
            self.curriculum.update_episode(self.episode_cnt)
            
            if self.episode_cnt % self.val_interval == 0:
                val_metrics = run_validation(
                    self.agent, self.env, self.usr_conf, 
                    self.logger, self.episode_cnt
                )
                if self.monitor:
                    val_monitor_data = {
                        "val_reward": val_metrics.get('total_reward', 0),
                        "val_total_score": val_metrics.get('total_score', 0),
                        "val_steps": val_metrics.get('steps', 0),
                    }
                    self.monitor.put_data({os.getpid(): val_monitor_data})
                
                self.logger.info(
                    f"[VALIDATION_COMPARE] Episode {self.episode_cnt} | "
                    f"Train Reward:{total_reward:.3f} | "
                    f"Val Reward:{val_metrics.get('total_reward', 0):.3f}"
                )
