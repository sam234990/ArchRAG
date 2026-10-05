"""Explicit public experiment metadata for W&B.

New arguments are excluded until reviewed here. Credentials, paths, service
endpoints and arbitrary nested configuration must never be added to this list.
"""

import argparse

from src.graphrag.config_logging import safe_config


EXPERIMENT_FIELDS = frozenset({
    "dataset_name", "doc_idx", "eval_mode", "seed", "engine", "max_tokens",
    "temperature", "max_community_tokens", "max_retries", "embedding_local",
    "embedding_model_local", "embedding_model", "embedding_num_workers",
    "entity_second_embedding", "num_workers", "strategy", "k_each_level",
    "k_final", "topk_e", "all_k_inference", "only_entity", "wo_hierarchical",
    "ppr_refine", "involve_llm_res", "topk_chunk", "generate_strategy",
    "response_type", "range_level", "print_log", "debug_flag",
})


def experiment_tracking_settings(wandb):
    """Disable SDK capture that bypasses the explicit config allowlist.

    W&B 0.19.10 includes sys.argv in system metadata by default. The internal
    x_disable_meta setting is version-specific; recheck it when upgrading W&B.
    Console/code capture can also contain credentials unrelated to config.
    """
    return wandb.Settings(
        x_disable_meta=True,
        disable_git=True,
        disable_code=True,
        save_code=False,
        console="off",
    )


def experiment_config(args) -> dict:
    """Select scalar experiment fields without modifying the live arguments."""
    values = vars(args) if isinstance(args, argparse.Namespace) else args
    return safe_config({
        key: value
        for key, value in values.items()
        if key in EXPERIMENT_FIELDS and (value is None or type(value) in (str, int, float, bool))
    })
