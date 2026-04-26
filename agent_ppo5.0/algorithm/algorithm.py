#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
"""

import os
import time

import torch

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
        self.target_kl = Config.TARGET_KL

        self.last_report_monitor_time = 0
        self.train_step = 0

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

        raw_advantage = advantage.detach()
        if Config.USE_ADVANTAGE_NORM:
            advantage = self._normalize_advantage(advantage)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        self.var_beta = self._current_entropy_coef()
        total_loss, loss_info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
        )

        approx_kl = loss_info["approx_kl"]
        if self.target_kl > 0 and approx_kl.item() > self.target_kl:
            if self.logger:
                self.logger.info(
                    f"[train] skip update due to approx_kl={approx_kl.item():.6f} > "
                    f"target_kl={self.target_kl:.6f}"
                )
            return

        total_loss.backward()
        grad_clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "reward": round(reward.mean().item(), 4),
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(loss_info["value_loss"].item(), 4),
                "policy_loss": round(loss_info["policy_loss"].item(), 4),
                "entropy_loss": round(loss_info["entropy_loss"].item(), 4),
                "grad_clip_norm": round(float(grad_clip_norm), 4),
                "clip_frac": round(loss_info["clip_frac"].item(), 4),
                "explained_var": round(loss_info["explained_var"].item(), 4),
                "adv_mean": round(raw_advantage.mean().item(), 4),
                "ret_mean": round(reward_sum.mean().item(), 4),
            }
            if self.logger:
                self.logger.info(
                    f"[train] total_loss:{results['total_loss']} "
                    f"policy_loss:{results['policy_loss']} "
                    f"value_loss:{results['value_loss']} "
                    f"entropy:{results['entropy_loss']} "
                    f"clip_frac:{results['clip_frac']} "
                    f"explained_var:{results['explained_var']} "
                    f"ret_mean:{results['ret_mean']} "
                    f"approx_kl:{approx_kl.item():.6f} beta:{self.var_beta:.6f}"
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
    ):
        """Compute standard PPO loss."""

        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        value_clip = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(reward_sum - value_pred),
                torch.square(reward_sum - value_clip),
            ).mean()
        )

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()
        approx_kl = (old_action_prob.log() - new_prob.clamp(1e-9).log()).mean()

        returns = reward_sum.detach()
        value_pred_detached = value_pred.detach()
        returns_var = torch.var(returns, unbiased=False)
        if returns_var.item() <= 1e-8:
            explained_var = torch.zeros(1, device=value_pred.device).squeeze(0)
        else:
            explained_var = 1.0 - torch.var(
                returns - value_pred_detached,
                unbiased=False,
            ) / (returns_var + 1e-8)

        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "explained_var": explained_var,
        }

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking."""

        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)

    def _normalize_advantage(self, advantage):
        adv_mean = advantage.mean()
        adv_std = advantage.std(unbiased=False)
        return (advantage - adv_mean) / (adv_std + Config.ADVANTAGE_NORM_EPS)

    def _current_entropy_coef(self):
        if Config.BETA_DECAY_STEPS <= 0:
            return Config.BETA_END
        progress = min(self.train_step / Config.BETA_DECAY_STEPS, 1.0)
        return Config.BETA_START + (Config.BETA_END - Config.BETA_START) * progress
