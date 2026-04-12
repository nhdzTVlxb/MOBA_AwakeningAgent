#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
加速前后统计工具

记录并统计对局中加速前/后的各项指标：
- 步数、奖励、宝箱得分、总分
- 闪现使用次数
- 终局状态、危险度、最近宝箱距离
- 加速前后 shaping reward
- 最后一步闪现状态
- 视野内可见宝箱比例
- 距离探索奖励
"""


class EpisodeMetrics:
    def __init__(self, speed_up_step):
        """
        初始化指标收集器
        
        Args:
            speed_up_step: 怪物加速开始的步数阈值
        """
        self.speed_up_step = speed_up_step
        self.reset()

    def reset(self):
        """重置所有指标"""
        # 加速前指标
        self.pre_steps = 0
        self.pre_total_reward = 0.0
        self.pre_treasure_gain = 0
        self.pre_total_gain = 0
        self.pre_shaped_reward = 0.0
        self.pre_step_gain = 0
        self.pre_terminal = 0.0
        self.pre_distance_reward = 0.0
        
        # 加速后指标
        self.post_steps = 0
        self.post_total_reward = 0.0
        self.post_treasure_gain = 0
        self.post_total_gain = 0
        self.post_shaped_reward = 0.0
        self.post_step_gain = 0
        self.post_terminal = 0.0
        self.post_distance_reward = 0.0
        
        # 全局指标
        self.total_steps = 0
        self.total_reward = 0.0
        self.total_score = 0
        self.treasure_count = 0
        self.flash_count = 0
        self.last_flash_used = False
        self.last_flash_ready = False
        self.last_flash_legal = False
        self.final_danger = 0.0
        self.final_treasure_dist = 1000.0
        self.final_visible_treasure = 0.0
        self.total_distance_reward = 0.0
        
        # 终局状态
        self.terminated = False      # 阵亡
        self.completed = False       # 正常完成
        self.abnormal_trunc = False  # 异常截断
        
        # 是否进入加速阶段
        self.speedup_reached = False
        
        # 上一次的得分记录（用于计算增量）
        self._last_step_score = 0
        self._last_treasure_score = 0
        self._last_total_score = 0
        
        # 是否已进入加速阶段
        self._in_post = False

    def update(self, step, reward, step_score, treasure_score, total_score, 
               shaped_reward=0.0, is_flash=False, is_flash_legal=False,
               flash_ready=False, distance_reward=0.0):
        """
        每步更新指标
        
        Args:
            step: 当前步数
            reward: 当前步的即时奖励
            step_score: 当前步数得分
            treasure_score: 当前宝箱得分
            total_score: 当前总分
            shaped_reward: shaping reward累计值
            is_flash: 这一步是否使用了闪现
            is_flash_legal: 这一步闪现是否合法
            flash_ready: 当前闪现是否可用
            distance_reward: 距离探索奖励
        """
        self.total_steps = step
        self.total_reward += reward
        self.total_distance_reward += distance_reward
        
        # 判断是否在加速前/后
        is_pre = step < self.speed_up_step
        
        if is_pre and not self._in_post:
            # 加速前阶段
            self.pre_steps += 1
            self.pre_total_reward += reward
            self.pre_shaped_reward += shaped_reward
            self.pre_distance_reward += distance_reward
            
            # 计算得分增量
            step_gain = step_score - self._last_step_score
            if step_gain > 0:
                self.pre_step_gain += step_gain
                self.pre_total_gain += step_gain
            
            treasure_gain = treasure_score - self._last_treasure_score
            if treasure_gain > 0:
                self.pre_treasure_gain += treasure_gain
                self.pre_total_gain += treasure_gain
        else:
            # 加速后阶段
            if not self._in_post:
                self._in_post = True
                self.speedup_reached = True
            
            self.post_steps += 1
            self.post_total_reward += reward
            self.post_shaped_reward += shaped_reward
            self.post_distance_reward += distance_reward
            
            step_gain = step_score - self._last_step_score
            if step_gain > 0:
                self.post_step_gain += step_gain
                self.post_total_gain += step_gain
            
            treasure_gain = treasure_score - self._last_treasure_score
            if treasure_gain > 0:
                self.post_treasure_gain += treasure_gain
                self.post_total_gain += treasure_gain
        
        # 记录闪现
        if is_flash:
            self.flash_count += 1
            self.last_flash_used = True
        else:
            self.last_flash_used = False
        
        # 记录闪现状态
        self.last_flash_legal = is_flash_legal
        self.last_flash_ready = flash_ready
        
        # 更新上一次得分
        self._last_step_score = step_score
        self._last_treasure_score = treasure_score
        self._last_total_score = total_score
        self.total_score = total_score
        self.treasure_count = treasure_score // 100  # 假设每个宝箱100分

    def set_terminal_state(self, terminated, truncated, terminal_reward=0.0):
        """
        设置终局状态
        
        Args:
            terminated: 是否阵亡
            truncated: 是否截断（步数用完）
            terminal_reward: 终局奖励
        """
        if terminated:
            self.terminated = True
            self.completed = False
            self.abnormal_trunc = False
        else:
            # 时间到，按正常 TIMEUP 处理；这里只是复用旧字段
            self.terminated = False
            self.completed = False
            self.abnormal_trunc = True
        
        # 记录终局发生在加速前还是加速后
        if not self._in_post:
            self.pre_terminal = terminal_reward
        else:
            self.post_terminal = terminal_reward

    def set_final_danger(self, min_monster_dist):
        """设置最终危险度（最近怪物距离）"""
        self.final_danger = min_monster_dist

    def set_final_treasure_dist(self, dist):
        """设置最终最近宝箱距离"""
        self.final_treasure_dist = dist

    def set_final_visible_treasure(self, ratio):
        """设置最终视野内可见宝箱比例"""
        self.final_visible_treasure = ratio

    def get_summary(self):
        """获取完整的指标统计字典"""
        return {
            # 步数相关
            'steps': self.total_steps,
            'pre_steps': self.pre_steps,
            'post_steps': self.post_steps,
            'speedup_reached': self.speedup_reached,
            
            # 奖励相关
            'total_reward': round(self.total_reward, 4),
            'pre_total_reward': round(self.pre_total_reward, 4),
            'post_total_reward': round(self.post_total_reward, 4),
            'pre_shaped_reward': round(self.pre_shaped_reward, 4),
            'post_shaped_reward': round(self.post_shaped_reward, 4),
            
            # 距离探索奖励相关
            'total_distance_reward': round(self.total_distance_reward, 4),
            'pre_distance_reward': round(self.pre_distance_reward, 4),
            'post_distance_reward': round(self.post_distance_reward, 4),
            
            # 得分相关
            'total_score': self.total_score,
            'treasures': self.treasure_count,
            'pre_step_gain': self.pre_step_gain,
            'post_step_gain': self.post_step_gain,
            'pre_treasure_gain': self.pre_treasure_gain,
            'post_treasure_gain': self.post_treasure_gain,
            'pre_total_gain': self.pre_total_gain,
            'post_total_gain': self.post_total_gain,
            
            # 终局相关
            'pre_terminal': round(self.pre_terminal, 4),
            'post_terminal': round(self.post_terminal, 4),
            'terminated': self.terminated,
            'completed': self.completed,
            'abnormal_trunc': self.abnormal_trunc,
            
            # 危险度相关
            'final_danger': round(self.final_danger, 2),
            'final_treasure_dist': round(self.final_treasure_dist, 2),
            'final_visible_treasure': round(self.final_visible_treasure, 4),
            
            # 闪现相关
            'flash_count': self.flash_count,
            'last_flash_used': self.last_flash_used,
            'last_flash_ready': self.last_flash_ready,
            'last_flash_legal': self.last_flash_legal,
        }