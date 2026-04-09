#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
课程学习调度器

根据当前训练局数动态调整环境难度参数：
- 宝箱数量
- buff数量
- 第二只怪物出现时间
- 怪物加速时间
"""

import random
from agent_ppo.conf.conf import CurriculumConfig


class CurriculumScheduler:
    def __init__(self, episode_cnt=0):
        self.episode_cnt = episode_cnt
        self.current_stage = 0
        self._update_stage()

    def _update_stage(self):
        """根据episode数更新当前阶段"""
        if self.episode_cnt < CurriculumConfig.STAGE0_END:
            self.current_stage = 0
        elif self.episode_cnt < CurriculumConfig.STAGE1_END:
            self.current_stage = 1
        elif self.episode_cnt < CurriculumConfig.STAGE2_END:
            self.current_stage = 2
        else:
            self.current_stage = 3

    def update_episode(self, episode_cnt):
        """更新当前局数并重新计算阶段"""
        self.episode_cnt = episode_cnt
        self._update_stage()

    def _random_range(self, range_tuple):
        """在范围内随机采样整数"""
        if range_tuple[0] == range_tuple[1]:
            return range_tuple[0]
        return random.randint(range_tuple[0], range_tuple[1])

    def get_current_config(self):
        """获取当前阶段的环境配置参数"""
        if self.current_stage == 0:
            return {
                'stage': 0,
                'stage_name': 'warmup_stable',
                'treasure_num': self._random_range(CurriculumConfig.STAGE0_TREASURE_NUM),
                'buff_num': self._random_range(CurriculumConfig.STAGE0_BUFF_NUM),
                'monster2_spawn_step': self._random_range(CurriculumConfig.STAGE0_MONSTER2_SPAWN),
                'monster_speed_up_step': self._random_range(CurriculumConfig.STAGE0_SPEED_UP),
            }
        elif self.current_stage == 1:
            return {
                'stage': 1,
                'stage_name': 'mid_pressure',
                'treasure_num': self._random_range(CurriculumConfig.STAGE1_TREASURE_NUM),
                'buff_num': self._random_range(CurriculumConfig.STAGE1_BUFF_NUM),
                'monster2_spawn_step': self._random_range(CurriculumConfig.STAGE1_MONSTER2_SPAWN),
                'monster_speed_up_step': self._random_range(CurriculumConfig.STAGE1_SPEED_UP),
            }
        elif self.current_stage == 2:
            return {
                'stage': 2,
                'stage_name': 'late_speedup_survival',
                'treasure_num': self._random_range(CurriculumConfig.STAGE2_TREASURE_NUM),
                'buff_num': self._random_range(CurriculumConfig.STAGE2_BUFF_NUM),
                'monster2_spawn_step': self._random_range(CurriculumConfig.STAGE2_MONSTER2_SPAWN),
                'monster_speed_up_step': self._random_range(CurriculumConfig.STAGE2_SPEED_UP),
            }
        else:
            return {
                'stage': 3,
                'stage_name': 'hard_generalization',
                'treasure_num': self._random_range(CurriculumConfig.STAGE3_TREASURE_NUM),
                'buff_num': self._random_range(CurriculumConfig.STAGE3_BUFF_NUM),
                'monster2_spawn_step': self._random_range(CurriculumConfig.STAGE3_MONSTER2_SPAWN),
                'monster_speed_up_step': self._random_range(CurriculumConfig.STAGE3_SPEED_UP),
            }

    def get_train_map_ids(self):
        """获取训练地图ID列表"""
        return CurriculumConfig.TRAIN_MAP_IDS

    def get_valid_map_ids(self):
        """获取验证地图ID列表"""
        return CurriculumConfig.VALID_MAP_IDS

    def get_validation_interval(self):
        """获取验证间隔"""
        return CurriculumConfig.VALIDATION_INTERVAL
