"""Telemetría Weights & Biases para Etapa 3 sin acoplar entrenamiento/evaluación."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

DEFAULT_WANDB = {
    "enabled": False,
    "project": "pahm-rl-stage3",
    "entity": None,
    "mode": "disabled",
    "tags": ["stage3", "rl", "pahm"],
    "log_models": True,
    "log_evaluation": True,
}


class NullWandbRun:
    """Objeto nulo con la interfaz mínima usada por los flujos de Etapa 3."""

    enabled = False

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_artifact(self, artifact) -> None:
        return None

    def finish(self) -> None:
        return None


def normalize_wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_WANDB, **config.get("wandb", {})}


def build_wandb_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    rl_config = config.get("rl_training", {})
    domain_randomization = config.get("domain_randomization", {})
    reset_options = domain_randomization.get("reset_options", {})
    return {
        "training": {
            "mode": rl_config.get("mode"),
            "algorithm": rl_config.get("algorithm"),
            "learning_rate": rl_config.get("learning_rate"),
            "gamma": rl_config.get("gamma"),
            "n_steps": rl_config.get("n_steps"),
            "batch_size": rl_config.get("batch_size"),
            "total_timesteps": rl_config.get("total_timesteps"),
            "seed": rl_config.get("seed", config.get("seed")),
            "device": rl_config.get("device"),
        },
        "control": copy.deepcopy(config.get("control", {})),
        "reward": copy.deepcopy(config.get("reward", {})),
        "wind": copy.deepcopy(config.get("wind", {})),
        "environment": {
            "wind_enabled": bool(rl_config.get("wind_enabled", config.get("wind", {}).get("enabled", False))),
            "wind_pattern": rl_config.get("wind_pattern", config.get("wind", {}).get("default_pattern")),
            "reset_randomize": bool(reset_options.get("randomize", False)),
            "render": bool(rl_config.get("render", False)),
            "reset_angle_deg": rl_config.get("reset_angle_deg"),
        },
        "domain_randomization": copy.deepcopy(domain_randomization),
        "evaluation": copy.deepcopy(config.get("evaluation", {})),
    }


def init_wandb_run(
    config: dict[str, Any],
    *,
    run_name: str,
    tags: list[str] | None = None,
    job_type: str | None = None,
    wandb_module=None,
):
    wandb_config = normalize_wandb_config(config)
    if not wandb_config.get("enabled", False):
        return NullWandbRun()

    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError(
                "wandb es requerido cuando wandb.enabled=true. "
                "Instala requirements.txt o usa wandb.enabled=false."
            ) from exc

    merged_tags = list(wandb_config.get("tags", []))
    if tags:
        merged_tags.extend(tag for tag in tags if tag not in merged_tags)

    return wandb_module.init(
        project=wandb_config.get("project"),
        entity=wandb_config.get("entity"),
        mode=wandb_config.get("mode", "online"),
        tags=merged_tags,
        name=run_name,
        job_type=job_type,
        config=build_wandb_config_payload(config),
    )


def log_training_config(run, config: dict[str, Any]) -> None:
    if isinstance(run, NullWandbRun):
        return
    run.log({"config/stage3": build_wandb_config_payload(config)})


def log_training_metrics(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if isinstance(run, NullWandbRun):
        return
    run.log(metrics, step=step)


def build_evaluation_metrics_payload(comparison: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    summary = comparison.get("summary", {})
    for controller, metrics in summary.items():
        for metric_name, stats in metrics.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is not None:
                payload[f"eval/{controller}/{metric_name}_mean"] = float(mean)
            if std is not None:
                payload[f"eval/{controller}/{metric_name}_std"] = float(std)

    if "naive" in summary and "robust" in summary:
        for metric_name, naive_stats in summary["naive"].items():
            robust_stats = summary["robust"].get(metric_name, {})
            naive_mean = naive_stats.get("mean") if isinstance(naive_stats, dict) else None
            robust_mean = robust_stats.get("mean") if isinstance(robust_stats, dict) else None
            if naive_mean is not None and robust_mean is not None:
                payload[f"eval/comparison/{metric_name}_delta_robust_minus_naive"] = round(
                    float(robust_mean) - float(naive_mean),
                    12,
                )
    return payload


def log_evaluation_metrics(run, comparison: dict[str, Any]) -> None:
    if isinstance(run, NullWandbRun):
        return
    payload = build_evaluation_metrics_payload(comparison)
    if payload:
        run.log(payload)


def log_model_artifact(
    run,
    model_path: str | Path,
    artifact_name: str,
    config: dict[str, Any],
    *,
    wandb_module=None,
) -> None:
    wandb_config = normalize_wandb_config(config)
    log_file_artifact(
        run,
        model_path,
        artifact_name,
        artifact_type="model",
        enabled=bool(wandb_config.get("enabled", False)) and bool(wandb_config.get("log_models", True)),
        wandb_module=wandb_module,
    )


def log_file_artifact(
    run,
    file_path: str | Path,
    artifact_name: str,
    *,
    artifact_type: str,
    enabled: bool,
    wandb_module=None,
) -> None:
    if isinstance(run, NullWandbRun) or not enabled:
        return
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"W&B artifact file not found: {path}")

    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError("wandb es requerido para registrar artefactos.") from exc

    if not hasattr(wandb_module, "Artifact"):
        return
    artifact = wandb_module.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    run.finish()
