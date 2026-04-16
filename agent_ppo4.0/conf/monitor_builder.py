#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Monitor configuration for Gorge Chase PPO."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    return (
        monitor.title("Gorge Chase PPO")
        .add_group(group_name="Progress", group_name_en="algorithm")
        .add_panel(name="EpisodeID", name_en="current_episode_id", type="line")
        .add_metric(metrics_name="current_episode_id", expr="avg(current_episode_id{})")
        .end_panel()
        .add_panel(name="CompletedEP", name_en="completed_episode_count", type="line")
        .add_metric(
            metrics_name="completed_episode_count",
            expr="avg(completed_episode_count{})",
        )
        .end_panel()
        .add_panel(name="EpisodeStep", name_en="current_episode_step", type="line")
        .add_metric(metrics_name="current_episode_step", expr="avg(current_episode_step{})")
        .end_panel()
        .add_panel(name="IsEval", name_en="current_episode_is_eval", type="line")
        .add_metric(
            metrics_name="current_episode_is_eval",
            expr="avg(current_episode_is_eval{})",
        )
        .end_panel()
        .add_panel(name="TrainTotalEP", name_en="train_episode_total", type="line")
        .add_metric(metrics_name="train_episode_total", expr="avg(train_episode_total{})")
        .end_panel()
        .end_group()
        .add_group(group_name="Train", group_name_en="algorithm")
        .add_panel(name="Reward", name_en="train_reward", type="line")
        .add_metric(metrics_name="train_reward", expr="avg(train_reward{})")
        .end_panel()
        .add_panel(name="TotalScore", name_en="train_total_score", type="line")
        .add_metric(metrics_name="train_total_score", expr="avg(train_total_score{})")
        .end_panel()
        .add_panel(name="StepScore", name_en="train_step_score", type="line")
        .add_metric(metrics_name="train_step_score", expr="avg(train_step_score{})")
        .end_panel()
        .add_panel(name="TreasureScore", name_en="train_treasure_score", type="line")
        .add_metric(metrics_name="train_treasure_score", expr="avg(train_treasure_score{})")
        .end_panel()
        .add_panel(name="Treasures", name_en="train_treasures_collected", type="line")
        .add_metric(metrics_name="train_treasures_collected", expr="avg(train_treasures_collected{})")
        .end_panel()
        .add_panel(name="Steps", name_en="train_episode_steps", type="line")
        .add_metric(metrics_name="train_episode_steps", expr="avg(train_episode_steps{})")
        .end_panel()
        .add_panel(name="SpeedupReached", name_en="train_speedup_reached", type="line")
        .add_metric(metrics_name="train_speedup_reached", expr="avg(train_speedup_reached{})")
        .end_panel()
        .add_panel(name="TimeToSpeedup", name_en="train_phase_time_to_speedup", type="line")
        .add_metric(
            metrics_name="train_phase_time_to_speedup",
            expr="avg(train_phase_time_to_speedup{})",
        )
        .end_panel()
        .add_panel(name="Pre_Steps", name_en="train_pre_speedup_steps", type="line")
        .add_metric(metrics_name="train_pre_speedup_steps", expr="avg(train_pre_speedup_steps{})")
        .end_panel()
        .add_panel(name="Post_Steps", name_en="train_post_speedup_steps", type="line")
        .add_metric(metrics_name="train_post_speedup_steps", expr="avg(train_post_speedup_steps{})")
        .end_panel()
        .add_panel(name="Pre_TotalR", name_en="train_pre_speedup_reward", type="line")
        .add_metric(metrics_name="train_pre_speedup_reward", expr="avg(train_pre_speedup_reward{})")
        .end_panel()
        .add_panel(name="Post_TotalR", name_en="train_post_speedup_reward", type="line")
        .add_metric(metrics_name="train_post_speedup_reward", expr="avg(train_post_speedup_reward{})")
        .end_panel()
        .add_panel(name="Pre_ShapedR", name_en="train_pre_speedup_shaped_reward", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_shaped_reward",
            expr="avg(train_pre_speedup_shaped_reward{})",
        )
        .end_panel()
        .add_panel(name="Post_ShapedR", name_en="train_post_speedup_shaped_reward", type="line")
        .add_metric(
            metrics_name="train_post_speedup_shaped_reward",
            expr="avg(train_post_speedup_shaped_reward{})",
        )
        .end_panel()
        .add_panel(name="EarlyLootB", name_en="train_early_loot_collection_bonus", type="line")
        .add_metric(
            metrics_name="train_early_loot_collection_bonus",
            expr="avg(train_early_loot_collection_bonus{})",
        )
        .end_panel()
        .add_panel(name="EarlyLootS", name_en="train_early_loot_stall_penalty", type="line")
        .add_metric(
            metrics_name="train_early_loot_stall_penalty",
            expr="avg(train_early_loot_stall_penalty{})",
        )
        .end_panel()
        .add_panel(name="Pre_BufferR", name_en="train_pre_speedup_buffer_reward", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_buffer_reward",
            expr="avg(train_pre_speedup_buffer_reward{})",
        )
        .end_panel()
        .add_panel(name="Monster2_P", name_en="train_second_monster_pressure_penalty", type="line")
        .add_metric(
            metrics_name="train_second_monster_pressure_penalty",
            expr="avg(train_second_monster_pressure_penalty{})",
        )
        .end_panel()
        .add_panel(name="FlashDir", name_en="train_flash_direction_reward", type="line")
        .add_metric(
            metrics_name="train_flash_direction_reward",
            expr="avg(train_flash_direction_reward{})",
        )
        .end_panel()
        .add_panel(name="FlashWall", name_en="train_flash_through_wall_reward", type="line")
        .add_metric(
            metrics_name="train_flash_through_wall_reward",
            expr="avg(train_flash_through_wall_reward{})",
        )
        .end_panel()
        .add_panel(name="FlashWaste", name_en="train_flash_waste_penalty", type="line")
        .add_metric(
            metrics_name="train_flash_waste_penalty",
            expr="avg(train_flash_waste_penalty{})",
        )
        .end_panel()
        .add_panel(name="HitWall", name_en="train_hit_wall_penalty", type="line")
        .add_metric(metrics_name="train_hit_wall_penalty", expr="avg(train_hit_wall_penalty{})")
        .end_panel()
        .add_panel(name="Stagnation", name_en="train_stagnation_penalty", type="line")
        .add_metric(
            metrics_name="train_stagnation_penalty",
            expr="avg(train_stagnation_penalty{})",
        )
        .end_panel()
        .add_panel(name="Oscillation", name_en="train_oscillation_penalty", type="line")
        .add_metric(
            metrics_name="train_oscillation_penalty",
            expr="avg(train_oscillation_penalty{})",
        )
        .end_panel()
        .add_panel(name="MissTreasure", name_en="train_treasure_miss_penalty", type="line")
        .add_metric(
            metrics_name="train_treasure_miss_penalty",
            expr="avg(train_treasure_miss_penalty{})",
        )
        .end_panel()
        .add_panel(name="NoVisionMove", name_en="train_no_vision_patrol_bonus", type="line")
        .add_metric(
            metrics_name="train_no_vision_patrol_bonus",
            expr="avg(train_no_vision_patrol_bonus{})",
        )
        .end_panel()
        .add_panel(name="FirstTrea", name_en="train_time_to_first_treasure", type="line")
        .add_metric(
            metrics_name="train_time_to_first_treasure",
            expr="avg(train_time_to_first_treasure{})",
        )
        .end_panel()
        .add_panel(name="Pre_StepGain", name_en="train_pre_speedup_step_score_gain", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_step_score_gain",
            expr="avg(train_pre_speedup_step_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_StepGain", name_en="train_post_speedup_step_score_gain", type="line")
        .add_metric(
            metrics_name="train_post_speedup_step_score_gain",
            expr="avg(train_post_speedup_step_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaGain", name_en="train_pre_speedup_treasure_gain", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_treasure_gain",
            expr="avg(train_pre_speedup_treasure_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_TreaGain", name_en="train_post_speedup_treasure_gain", type="line")
        .add_metric(
            metrics_name="train_post_speedup_treasure_gain",
            expr="avg(train_post_speedup_treasure_gain{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaCnt", name_en="train_pre_speedup_treasures_collected", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_treasures_collected",
            expr="avg(train_pre_speedup_treasures_collected{})",
        )
        .end_panel()
        .add_panel(name="Post_TreaCnt", name_en="train_post_speedup_treasures_collected", type="line")
        .add_metric(
            metrics_name="train_post_speedup_treasures_collected",
            expr="avg(train_post_speedup_treasures_collected{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaRate", name_en="train_pre_speedup_treasure_rate", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_treasure_rate",
            expr="avg(train_pre_speedup_treasure_rate{})",
        )
        .end_panel()
        .add_panel(name="Pre_TotalGain", name_en="train_pre_speedup_total_score_gain", type="line")
        .add_metric(
            metrics_name="train_pre_speedup_total_score_gain",
            expr="avg(train_pre_speedup_total_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_TotalGain", name_en="train_post_speedup_total_score_gain", type="line")
        .add_metric(
            metrics_name="train_post_speedup_total_score_gain",
            expr="avg(train_post_speedup_total_score_gain{})",
        )
        .end_panel()
        .end_group()
        .add_group(group_name="环境指标", group_name_en="env")
        .add_panel(name="得分", name_en="env_score", type="line")
        .add_metric(metrics_name="total_score", expr="avg(total_score{})")
        .add_metric(metrics_name="treasure_score", expr="avg(treasure_score{})")
        .add_metric(metrics_name="step_score", expr="avg(step_score{})")
        .end_panel()
        .add_panel(name="地图", name_en="env_map", type="line")
        .add_metric(metrics_name="total_map", expr="avg(total_map{})")
        .add_metric(metrics_name="map_random", expr="avg(map_random{})")
        .end_panel()
        .add_panel(name="步数", name_en="env_steps", type="line")
        .add_metric(metrics_name="max_steps", expr="avg(max_step{})")
        .add_metric(metrics_name="finished_steps", expr="avg(finished_steps{})")
        .end_panel()
        .add_panel(name="宝箱", name_en="env_treasure", type="line")
        .add_metric(metrics_name="total_treasure", expr="avg(total_treasure{})")
        .add_metric(metrics_name="treasures_collected", expr="avg(treasures_collected{})")
        .end_panel()
        .add_panel(name="闪现", name_en="env_flash", type="line")
        .add_metric(metrics_name="flash_count", expr="avg(flash_count{})")
        .add_metric(metrics_name="flash_cooldown", expr="avg(flash_cooldown{})")
        .end_panel()
        .add_panel(name="加速增益", name_en="env_buff", type="line")
        .add_metric(metrics_name="total_buff", expr="avg(total_buff{})")
        .add_metric(metrics_name="collected_buff", expr="avg(collected_buff{})")
        .add_metric(metrics_name="buff_refresh_time", expr="avg(buff_refresh_time{})")
        .end_panel()
        .add_panel(name="怪物移动速度", name_en="env_monster_speed", type="line")
        .add_metric(metrics_name="monster_speed", expr="avg(monster_speed{})")
        .end_panel()
        .add_panel(name="怪物出现间隔", name_en="env_monster_interval", type="line")
        .add_metric(metrics_name="monster_interval", expr="avg(monster_interval{})")
        .end_panel()
        .end_group()
        .add_group(group_name="训练损失", group_name_en="algorithm")
        .add_panel(name="CumReward", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="TotalLoss", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="ValueLoss", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="PolicyLoss", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="EntropyLoss", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="GradClipNorm", name_en="grad_clip_norm", type="line")
        .add_metric(metrics_name="grad_clip_norm", expr="avg(grad_clip_norm{})")
        .end_panel()
        .add_panel(name="ClipFrac", name_en="clip_frac", type="line")
        .add_metric(metrics_name="clip_frac", expr="avg(clip_frac{})")
        .end_panel()
        .add_panel(name="ExplainedVar", name_en="explained_var", type="line")
        .add_metric(metrics_name="explained_var", expr="avg(explained_var{})")
        .end_panel()
        .add_panel(name="AdvMean", name_en="adv_mean", type="line")
        .add_metric(metrics_name="adv_mean", expr="avg(adv_mean{})")
        .end_panel()
        .add_panel(name="RetMean", name_en="ret_mean", type="line")
        .add_metric(metrics_name="ret_mean", expr="avg(ret_mean{})")
        .end_panel()
        .end_group()
        .add_group(group_name="Val", group_name_en="algorithm")
        .add_panel(name="Reward", name_en="val_reward", type="line")
        .add_metric(metrics_name="val_reward", expr="avg(val_reward{})")
        .end_panel()
        .add_panel(name="TotalScore", name_en="val_total_score", type="line")
        .add_metric(metrics_name="val_total_score", expr="avg(val_total_score{})")
        .end_panel()
        .add_panel(name="StepScore", name_en="val_step_score", type="line")
        .add_metric(metrics_name="val_step_score", expr="avg(val_step_score{})")
        .end_panel()
        .add_panel(name="TreasureScore", name_en="val_treasure_score", type="line")
        .add_metric(metrics_name="val_treasure_score", expr="avg(val_treasure_score{})")
        .end_panel()
        .add_panel(name="Treasures", name_en="val_treasures_collected", type="line")
        .add_metric(metrics_name="val_treasures_collected", expr="avg(val_treasures_collected{})")
        .end_panel()
        .add_panel(name="Steps", name_en="val_episode_steps", type="line")
        .add_metric(metrics_name="val_episode_steps", expr="avg(val_episode_steps{})")
        .end_panel()
        .add_panel(name="SpeedupReached", name_en="val_speedup_reached", type="line")
        .add_metric(metrics_name="val_speedup_reached", expr="avg(val_speedup_reached{})")
        .end_panel()
        .add_panel(name="TimeToSpeedup", name_en="val_phase_time_to_speedup", type="line")
        .add_metric(
            metrics_name="val_phase_time_to_speedup",
            expr="avg(val_phase_time_to_speedup{})",
        )
        .end_panel()
        .add_panel(name="Pre_Steps", name_en="val_pre_speedup_steps", type="line")
        .add_metric(metrics_name="val_pre_speedup_steps", expr="avg(val_pre_speedup_steps{})")
        .end_panel()
        .add_panel(name="Post_Steps", name_en="val_post_speedup_steps", type="line")
        .add_metric(metrics_name="val_post_speedup_steps", expr="avg(val_post_speedup_steps{})")
        .end_panel()
        .add_panel(name="Pre_TotalR", name_en="val_pre_speedup_reward", type="line")
        .add_metric(metrics_name="val_pre_speedup_reward", expr="avg(val_pre_speedup_reward{})")
        .end_panel()
        .add_panel(name="Post_TotalR", name_en="val_post_speedup_reward", type="line")
        .add_metric(metrics_name="val_post_speedup_reward", expr="avg(val_post_speedup_reward{})")
        .end_panel()
        .add_panel(name="Pre_ShapedR", name_en="val_pre_speedup_shaped_reward", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_shaped_reward",
            expr="avg(val_pre_speedup_shaped_reward{})",
        )
        .end_panel()
        .add_panel(name="Post_ShapedR", name_en="val_post_speedup_shaped_reward", type="line")
        .add_metric(
            metrics_name="val_post_speedup_shaped_reward",
            expr="avg(val_post_speedup_shaped_reward{})",
        )
        .end_panel()
        .add_panel(name="EarlyLootB", name_en="val_early_loot_collection_bonus", type="line")
        .add_metric(
            metrics_name="val_early_loot_collection_bonus",
            expr="avg(val_early_loot_collection_bonus{})",
        )
        .end_panel()
        .add_panel(name="EarlyLootS", name_en="val_early_loot_stall_penalty", type="line")
        .add_metric(
            metrics_name="val_early_loot_stall_penalty",
            expr="avg(val_early_loot_stall_penalty{})",
        )
        .end_panel()
        .add_panel(name="Pre_BufferR", name_en="val_pre_speedup_buffer_reward", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_buffer_reward",
            expr="avg(val_pre_speedup_buffer_reward{})",
        )
        .end_panel()
        .add_panel(name="Monster2_P", name_en="val_second_monster_pressure_penalty", type="line")
        .add_metric(
            metrics_name="val_second_monster_pressure_penalty",
            expr="avg(val_second_monster_pressure_penalty{})",
        )
        .end_panel()
        .add_panel(name="FlashDir", name_en="val_flash_direction_reward", type="line")
        .add_metric(
            metrics_name="val_flash_direction_reward",
            expr="avg(val_flash_direction_reward{})",
        )
        .end_panel()
        .add_panel(name="FlashWall", name_en="val_flash_through_wall_reward", type="line")
        .add_metric(
            metrics_name="val_flash_through_wall_reward",
            expr="avg(val_flash_through_wall_reward{})",
        )
        .end_panel()
        .add_panel(name="FlashWaste", name_en="val_flash_waste_penalty", type="line")
        .add_metric(
            metrics_name="val_flash_waste_penalty",
            expr="avg(val_flash_waste_penalty{})",
        )
        .end_panel()
        .add_panel(name="HitWall", name_en="val_hit_wall_penalty", type="line")
        .add_metric(metrics_name="val_hit_wall_penalty", expr="avg(val_hit_wall_penalty{})")
        .end_panel()
        .add_panel(name="Stagnation", name_en="val_stagnation_penalty", type="line")
        .add_metric(
            metrics_name="val_stagnation_penalty",
            expr="avg(val_stagnation_penalty{})",
        )
        .end_panel()
        .add_panel(name="Oscillation", name_en="val_oscillation_penalty", type="line")
        .add_metric(
            metrics_name="val_oscillation_penalty",
            expr="avg(val_oscillation_penalty{})",
        )
        .end_panel()
        .add_panel(name="MissTreasure", name_en="val_treasure_miss_penalty", type="line")
        .add_metric(
            metrics_name="val_treasure_miss_penalty",
            expr="avg(val_treasure_miss_penalty{})",
        )
        .end_panel()
        .add_panel(name="NoVisionMove", name_en="val_no_vision_patrol_bonus", type="line")
        .add_metric(
            metrics_name="val_no_vision_patrol_bonus",
            expr="avg(val_no_vision_patrol_bonus{})",
        )
        .end_panel()
        .add_panel(name="FirstTrea", name_en="val_time_to_first_treasure", type="line")
        .add_metric(
            metrics_name="val_time_to_first_treasure",
            expr="avg(val_time_to_first_treasure{})",
        )
        .end_panel()
        .add_panel(name="Pre_StepGain", name_en="val_pre_speedup_step_score_gain", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_step_score_gain",
            expr="avg(val_pre_speedup_step_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_StepGain", name_en="val_post_speedup_step_score_gain", type="line")
        .add_metric(
            metrics_name="val_post_speedup_step_score_gain",
            expr="avg(val_post_speedup_step_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaGain", name_en="val_pre_speedup_treasure_gain", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_treasure_gain",
            expr="avg(val_pre_speedup_treasure_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_TreaGain", name_en="val_post_speedup_treasure_gain", type="line")
        .add_metric(
            metrics_name="val_post_speedup_treasure_gain",
            expr="avg(val_post_speedup_treasure_gain{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaCnt", name_en="val_pre_speedup_treasures_collected", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_treasures_collected",
            expr="avg(val_pre_speedup_treasures_collected{})",
        )
        .end_panel()
        .add_panel(name="Post_TreaCnt", name_en="val_post_speedup_treasures_collected", type="line")
        .add_metric(
            metrics_name="val_post_speedup_treasures_collected",
            expr="avg(val_post_speedup_treasures_collected{})",
        )
        .end_panel()
        .add_panel(name="Pre_TreaRate", name_en="val_pre_speedup_treasure_rate", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_treasure_rate",
            expr="avg(val_pre_speedup_treasure_rate{})",
        )
        .end_panel()
        .add_panel(name="Pre_TotalGain", name_en="val_pre_speedup_total_score_gain", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_total_score_gain",
            expr="avg(val_pre_speedup_total_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Post_TotalGain", name_en="val_post_speedup_total_score_gain", type="line")
        .add_metric(
            metrics_name="val_post_speedup_total_score_gain",
            expr="avg(val_post_speedup_total_score_gain{})",
        )
        .end_panel()
        .add_panel(name="Pre_Terminal", name_en="val_pre_speedup_terminal_bonus", type="line")
        .add_metric(
            metrics_name="val_pre_speedup_terminal_bonus",
            expr="avg(val_pre_speedup_terminal_bonus{})",
        )
        .end_panel()
        .add_panel(name="Post_Terminal", name_en="val_post_speedup_terminal_bonus", type="line")
        .add_metric(
            metrics_name="val_post_speedup_terminal_bonus",
            expr="avg(val_post_speedup_terminal_bonus{})",
        )
        .end_panel()
        .add_panel(name="Post_Terminated", name_en="val_post_speedup_terminated", type="line")
        .add_metric(
            metrics_name="val_post_speedup_terminated",
            expr="avg(val_post_speedup_terminated{})",
        )
        .end_panel()
        .add_panel(name="TerminatedRate", name_en="val_terminated_rate", type="line")
        .add_metric(metrics_name="val_terminated_rate", expr="avg(val_terminated_rate{})")
        .end_panel()
        .add_panel(name="CompletedRate", name_en="val_completed_rate", type="line")
        .add_metric(metrics_name="val_completed_rate", expr="avg(val_completed_rate{})")
        .end_panel()
        .add_panel(name="AbnormalTrunc", name_en="val_abnormal_truncated_rate", type="line")
        .add_metric(
            metrics_name="val_abnormal_truncated_rate",
            expr="avg(val_abnormal_truncated_rate{})",
        )
        .end_panel()
        .add_panel(name="Final_Danger", name_en="val_danger_level", type="line")
        .add_metric(metrics_name="val_danger_level", expr="avg(val_danger_level{})")
        .end_panel()
        .add_panel(name="Final_TreaDist", name_en="val_nearest_treasure_dist", type="line")
        .add_metric(
            metrics_name="val_nearest_treasure_dist",
            expr="avg(val_nearest_treasure_dist{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
