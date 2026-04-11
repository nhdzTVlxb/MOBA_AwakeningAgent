#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。
"""

import os
import time

import torch
import numpy as np
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0
        
        # 存储额外指标
        self.last_clip_frac = 0.0
        self.last_explained_var = 0.0
        self.last_adv_mean = 0.0
        self.last_ret_mean = 0.0
        self.last_approx_kl = 0.0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData."""
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list, extra_metrics = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1
        
        # 存储额外指标
        self.last_clip_frac = extra_metrics['clip_frac']
        self.last_explained_var = extra_metrics['explained_var']
        self.last_adv_mean = extra_metrics['adv_mean']
        self.last_ret_mean = extra_metrics['ret_mean']
        self.last_approx_kl = extra_metrics['approx_kl']

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
                "clip_frac": round(self.last_clip_frac, 4),
                "explained_var": round(self.last_explained_var, 4),
                "adv_mean": round(self.last_adv_mean, 4),
                "ret_mean": round(self.last_ret_mean, 4),
                "approx_kl": round(self.last_approx_kl, 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"clip_frac:{results['clip_frac']} "
                f"explained_var:{results['explained_var']} "
                f"approx_kl:{results['approx_kl']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy)."""
        # Masked softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip)
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        
        # ========== 修复：advantage 标准化 ==========
        adv = advantage.view(-1, 1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Value loss (Clipped)
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        # ========== 计算额外指标 ==========
        with torch.no_grad():
            clip_low = 1 - self.clip_param
            clip_high = 1 + self.clip_param
            is_clipped = (ratio < clip_low) | (ratio > clip_high)
            clip_frac = is_clipped.float().mean().item()
            
            # ========== 修复：explained_var 使用当前 value_pred ==========
            pred = value_pred.detach()
            td_error = reward_sum - pred
            var_td = torch.var(td_error).item()
            var_return = torch.var(reward_sum).item()
            explained_var = 1 - var_td / (var_return + 1e-8)
            
            adv_mean = advantage.mean().item()
            ret_mean = reward_sum.mean().item()
            
            # ========== 新增：approx_kl 监控 ==========
            approx_kl = (old_action_prob.log() - new_prob.clamp(1e-9).log()).mean().item()
        
        extra_metrics = {
            'clip_frac': clip_frac,
            'explained_var': explained_var,
            'adv_mean': adv_mean,
            'ret_mean': ret_mean,
            'approx_kl': approx_kl,
        }

        return total_loss, [value_loss, policy_loss, entropy_loss], extra_metrics

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking."""
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
    
    def get_metrics(self):
        """获取最近一次训练的额外指标"""
        return {
            'clip_frac': self.last_clip_frac,
            'explained_var': self.last_explained_var,
            'adv_mean': self.last_adv_mean,
            'ret_mean': self.last_ret_mean,
            'approx_kl': self.last_approx_kl,
        }