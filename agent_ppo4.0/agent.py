#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import os

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.resume_utils import (
    extract_model_state_dict,
    extract_resume_metadata_from_checkpoint,
    load_checkpoint_object,
    normalize_resume_metadata,
    read_configured_resume_checkpoint,
    read_resume_metadata_sidecar,
    read_resume_progress_snapshot,
    resolve_model_checkpoint_file,
    write_resume_metadata_sidecar,
)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        self.resume_metadata = {}
        self.resume_checkpoint_state = read_configured_resume_checkpoint()
        self._auto_resume_loaded = False
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None, usr_conf=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset(usr_conf=usr_conf)
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward, extra_info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        if isinstance(extra_info, dict):
            remain_info.update(extra_info)
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        self._maybe_auto_resume_before_learning()
        return self.algorithm.learn(list_sample_data)

    def set_resume_metadata(self, metadata):
        self.resume_metadata = normalize_resume_metadata(metadata)

    def get_resume_metadata(self):
        return dict(self.resume_metadata)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        resume_metadata = self.get_resume_metadata()
        if not resume_metadata:
            resume_metadata = read_resume_progress_snapshot()
            if resume_metadata:
                self.resume_metadata = dict(resume_metadata)

        checkpoint_payload = {
            "model_state_dict": state_dict_cpu,
            "resume_metadata": resume_metadata,
        }
        torch.save(checkpoint_payload, model_file_path)
        if resume_metadata:
            write_resume_metadata_sidecar(model_file_path, resume_metadata)
        if self.logger:
            self.logger.info(
                f"save model {model_file_path} successfully, resume_metadata={resume_metadata}"
            )

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        requested_model_file = resolve_model_checkpoint_file(path, id)
        fallback_model_file = self.resume_checkpoint_state.get("model_file")
        requested_is_latest = str(id) == "latest"

        model_file_path = requested_model_file
        if not model_file_path or not os.path.isfile(model_file_path):
            can_use_configured_fallback = (
                not self._auto_resume_loaded
                and fallback_model_file
                and os.path.isfile(fallback_model_file)
            )
            if can_use_configured_fallback:
                model_file_path = fallback_model_file
                if self.logger:
                    self.logger.info(
                        f"requested model id={id} path={path} unavailable, "
                        f"fallback to configured checkpoint {model_file_path}"
                    )
            elif requested_is_latest:
                if self.logger:
                    self.logger.info(
                        f"latest model is not available yet, keep current model parameters in memory"
                    )
                return
            else:
                raise FileNotFoundError(
                    f"model checkpoint not found for id={id}, requested_path={requested_model_file}"
                )

        checkpoint_obj = load_checkpoint_object(model_file_path, map_location=self.device)
        state_dict = extract_model_state_dict(checkpoint_obj)
        self.model.load_state_dict(state_dict)

        resume_metadata = extract_resume_metadata_from_checkpoint(checkpoint_obj)
        if not resume_metadata:
            resume_metadata = read_resume_metadata_sidecar(model_file_path)
        self.resume_metadata = normalize_resume_metadata(resume_metadata)
        self._auto_resume_loaded = True

        if self.logger:
            self.logger.info(
                f"load model {model_file_path} successfully, resume_metadata={self.resume_metadata}"
            )

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _maybe_auto_resume_before_learning(self):
        if self._auto_resume_loaded:
            return

        if not self.resume_checkpoint_state.get("enabled", False):
            self._auto_resume_loaded = True
            return

        model_file_path = self.resume_checkpoint_state.get("model_file")
        checkpoint_dir = self.resume_checkpoint_state.get("preload_model_dir")
        checkpoint_id = self.resume_checkpoint_state.get("preload_model_id")
        if not (model_file_path and os.path.isfile(model_file_path)):
            self._auto_resume_loaded = True
            return

        if self.logger:
            self.logger.info(
                f"auto resume before learning from configured checkpoint {model_file_path}"
            )
        self.load_model(path=checkpoint_dir, id=checkpoint_id)
