#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Checkpoint resume helpers for Gorge Chase PPO."""

import hashlib
import json
import os
import re
import tempfile
import time
import tomllib
import zipfile


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
RESUME_PROGRESS_SNAPSHOT_FILE = os.path.join("agent_ppo", "ckpt", "resume_progress.json")
RESUME_METADATA_KEYS = (
    "episode_cnt",
    "completed_episode_count",
    "train_episode_total",
    "train_episode_since_last_eval",
)
CHECKPOINT_WRAPPER_KEYS = {
    "model_state_dict",
    "state_dict",
    "model",
    "resume_metadata",
    "resume_state",
    "meta",
}
MODEL_CKPT_PATTERN = re.compile(r"model\.ckpt-(\d+)\.pkl$")


def _safe_str(value, default=""):
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_project_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, str(path)))


def normalize_resume_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}

    normalized = {
        key: _safe_int(metadata.get(key), 0)
        for key in RESUME_METADATA_KEYS
        if key in metadata
    }
    if not normalized:
        return {}

    normalized.setdefault("episode_cnt", normalized.get("completed_episode_count", 0))
    normalized.setdefault("completed_episode_count", normalized.get("episode_cnt", 0))
    normalized.setdefault("train_episode_total", normalized.get("completed_episode_count", 0))
    normalized.setdefault("train_episode_since_last_eval", 0)
    normalized["updated_at"] = _safe_int(metadata.get("updated_at"), int(time.time()))
    return normalized


def resolve_model_checkpoint_file(checkpoint_dir, checkpoint_id):
    if not checkpoint_dir or checkpoint_id in (None, ""):
        return None
    model_file = os.path.join(str(checkpoint_dir), f"model.ckpt-{str(checkpoint_id)}.pkl")
    return _resolve_project_path(model_file)


def resolve_preload_model_path(preload_model_path):
    if not preload_model_path:
        return None
    return _resolve_project_path(str(preload_model_path))


def resolve_checkpoint_resume_sidecar(model_file):
    if not model_file:
        return None
    root, _ = os.path.splitext(model_file)
    return f"{root}.resume.json"


def _read_zip_manifest(zip_file):
    manifest_file = f"{zip_file}.json"
    data = _read_json_file(manifest_file)
    return data if isinstance(data, dict) else {}


def _open_zip_file(zip_file):
    return zipfile.ZipFile(zip_file, "r", metadata_encoding="utf-8")


def _find_zip_checkpoint_member(zip_file):
    manifest = _read_zip_manifest(zip_file)
    model_file_paths = manifest.get("model_file_path")
    if isinstance(model_file_paths, list):
        for member in model_file_paths:
            member = _safe_str(member)
            if member.endswith(".pkl"):
                return member

    with _open_zip_file(zip_file) as zip_obj:
        members = [info.filename for info in zip_obj.infolist() if info.filename.endswith(".pkl")]

    for member in members:
        if MODEL_CKPT_PATTERN.search(os.path.basename(member)):
            return member
    return members[0] if members else None


def _derive_checkpoint_id_from_name(file_name):
    match = MODEL_CKPT_PATTERN.search(os.path.basename(_safe_str(file_name)))
    if match:
        return int(match.group(1))
    return None


def _extract_zip_checkpoint_member(zip_file, member_name):
    cache_root = _resolve_project_path(os.path.join("agent_ppo", "ckpt", ".resume_cache"))
    archive_hash = hashlib.sha256(os.path.abspath(zip_file).encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(cache_root, archive_hash)
    extracted_file = os.path.abspath(os.path.join(cache_dir, os.path.basename(member_name)))

    zip_mtime = os.path.getmtime(zip_file)
    needs_extract = not os.path.isfile(extracted_file) or os.path.getmtime(extracted_file) < zip_mtime
    if needs_extract:
        os.makedirs(cache_dir, exist_ok=True)
        with _open_zip_file(zip_file) as zip_obj:
            with zip_obj.open(member_name, "r") as src, open(extracted_file, "wb") as dst:
                dst.write(src.read())
    return extracted_file


def resolve_preload_checkpoint_source(preload_model_path=None, preload_model_dir=None, preload_model_id=None):
    if preload_model_path:
        source_path = resolve_preload_model_path(preload_model_path)
        if not source_path:
            return None
        suffix = os.path.splitext(source_path)[1].lower()
        if suffix == ".zip":
            if not os.path.isfile(source_path):
                return {
                    "source_type": "zip",
                    "configured_path": source_path,
                    "model_file": None,
                    "checkpoint_id": None,
                    "archive_member": None,
                    "metadata_file": f"{source_path}.json",
                }
            archive_member = _find_zip_checkpoint_member(source_path)
            model_file = _extract_zip_checkpoint_member(source_path, archive_member) if archive_member else None
            checkpoint_id = _derive_checkpoint_id_from_name(archive_member or source_path)
            return {
                "source_type": "zip",
                "configured_path": source_path,
                "model_file": model_file,
                "checkpoint_id": checkpoint_id,
                "archive_member": archive_member,
                "metadata_file": f"{source_path}.json",
            }

        checkpoint_id = _derive_checkpoint_id_from_name(source_path)
        return {
            "source_type": "file",
            "configured_path": source_path,
            "model_file": source_path if os.path.isfile(source_path) else None,
            "checkpoint_id": checkpoint_id,
            "archive_member": None,
            "metadata_file": resolve_checkpoint_resume_sidecar(source_path),
        }

    model_file = resolve_model_checkpoint_file(preload_model_dir, preload_model_id)
    return {
        "source_type": "dir_id",
        "configured_path": preload_model_dir,
        "model_file": model_file if (model_file and os.path.isfile(model_file)) else None,
        "checkpoint_id": _safe_int(preload_model_id, None) if preload_model_id not in (None, "") else None,
        "archive_member": None,
        "metadata_file": resolve_checkpoint_resume_sidecar(model_file) if model_file else None,
    }


def _read_json_file(file_path):
    resolved_path = _resolve_project_path(file_path)
    if not resolved_path or not os.path.isfile(resolved_path):
        return {}
    try:
        with open(resolved_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, ValueError, TypeError):
        return {}


def _write_json_file(file_path, payload):
    if not file_path:
        return
    resolved_path = _resolve_project_path(file_path)
    if not resolved_path:
        return
    target_dir = os.path.dirname(resolved_path)
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target_dir,
        prefix=f"{os.path.basename(resolved_path)}.",
        suffix=".tmp",
        delete=False,
    ) as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_file = file_obj.name
    os.replace(tmp_file, resolved_path)


def read_resume_progress_snapshot(snapshot_path=RESUME_PROGRESS_SNAPSHOT_FILE):
    return normalize_resume_metadata(_read_json_file(snapshot_path))


def write_resume_progress_snapshot(metadata, snapshot_path=RESUME_PROGRESS_SNAPSHOT_FILE):
    normalized = normalize_resume_metadata(metadata)
    if not normalized:
        return {}
    _write_json_file(snapshot_path, normalized)
    return normalized


def read_resume_metadata_sidecar(model_file):
    return normalize_resume_metadata(_read_json_file(resolve_checkpoint_resume_sidecar(model_file)))


def write_resume_metadata_sidecar(model_file, metadata):
    normalized = normalize_resume_metadata(metadata)
    if not normalized:
        return {}
    sidecar_file = resolve_checkpoint_resume_sidecar(model_file)
    _write_json_file(sidecar_file, normalized)
    return normalized


def _looks_like_state_dict(candidate):
    if not isinstance(candidate, dict) or not candidate:
        return False
    if any(key in CHECKPOINT_WRAPPER_KEYS for key in candidate):
        return False
    scalar_types = (str, bytes, int, float, bool, type(None))
    nested_types = (dict, list, tuple, set)
    return all(not isinstance(value, scalar_types + nested_types) for value in candidate.values())


def extract_model_state_dict(checkpoint_obj):
    if _looks_like_state_dict(checkpoint_obj):
        return checkpoint_obj
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            state_dict = checkpoint_obj.get(key)
            if _looks_like_state_dict(state_dict):
                return state_dict
    return checkpoint_obj


def extract_resume_metadata_from_checkpoint(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return {}

    for key in ("resume_metadata", "resume_state"):
        metadata = normalize_resume_metadata(checkpoint_obj.get(key))
        if metadata:
            return metadata

    meta = checkpoint_obj.get("meta")
    if isinstance(meta, dict):
        for key in ("resume_metadata", "resume_state"):
            metadata = normalize_resume_metadata(meta.get(key))
            if metadata:
                return metadata

    return {}


def load_checkpoint_object(model_file, map_location="cpu"):
    import torch

    return torch.load(model_file, map_location=map_location)


def load_resume_metadata_from_checkpoint_file(model_file):
    if not model_file or not os.path.isfile(model_file):
        return {}

    try:
        checkpoint_obj = load_checkpoint_object(model_file, map_location="cpu")
    except Exception:
        checkpoint_obj = None

    metadata = extract_resume_metadata_from_checkpoint(checkpoint_obj)
    if metadata:
        return metadata
    return read_resume_metadata_sidecar(model_file)


def read_configured_resume_checkpoint(config_path="conf/configure_app.toml"):
    state = {
        "configured": False,
        "enabled": False,
        "preload_model": False,
        "preload_model_path": None,
        "preload_model_dir": None,
        "preload_model_id": None,
        "model_file": None,
        "source_type": None,
        "archive_member": None,
        "configured_source": None,
        "manifest_metadata": {},
        "metadata": {},
    }

    try:
        with open(_resolve_project_path(config_path), "rb") as file_obj:
            app_conf = tomllib.load(file_obj).get("app", {})
    except (OSError, tomllib.TOMLDecodeError):
        return state

    preload_model_path = app_conf.get("preload_model_path")
    resolved_source = resolve_preload_checkpoint_source(preload_model_path=preload_model_path)
    model_file = resolved_source.get("model_file") if resolved_source else None
    resolved_checkpoint_id = resolved_source.get("checkpoint_id") if resolved_source else None
    configured = bool(preload_model_path)
    enabled = bool(configured and model_file and os.path.isfile(model_file))

    state.update(
        {
            "configured": configured,
            "enabled": enabled,
            "preload_model": bool(app_conf.get("preload_model", False)),
            "preload_model_path": preload_model_path,
            "preload_model_dir": None,
            "preload_model_id": resolved_checkpoint_id,
            "model_file": model_file,
            "source_type": resolved_source.get("source_type") if resolved_source else None,
            "archive_member": resolved_source.get("archive_member") if resolved_source else None,
            "configured_source": resolved_source.get("configured_path") if resolved_source else None,
            "manifest_metadata": (
                _read_zip_manifest(resolved_source.get("configured_path"))
                if resolved_source and resolved_source.get("source_type") == "zip"
                else {}
            ),
            "metadata": load_resume_metadata_from_checkpoint_file(model_file) if enabled else {},
        }
    )
    return state
