"""Evaluacion cuantitativa headless de controladores RL para PAHM."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gym_wrapper.learned_pahm_ode import LearnedPAHMODE
from gym_wrapper.rl_policy import load_rl_policy, predict_rl_action
from pahm_stage3.wandb_logger import (
    NullWandbRun,
    finish_wandb_run,
    init_wandb_run,
    log_evaluation_metrics,
    log_file_artifact,
    normalize_wandb_config,
)
from train_rl import load_config


DEFAULT_EVALUATION = {
    "enabled": True,
    "num_episodes": 10,
    "max_steps": 1000,
    "render": False,
    "controllers": ["naive", "robust"],
    "output_dir": "artifacts/stage3/evaluation",
    "metrics_path": "artifacts/stage3/evaluation/controller_metrics.json",
    "settling_tolerance": 0.05,
    "settling_window": 50,
    "unseen_wind_patterns": ["gust", "turbulent"],
    "wind_enabled": True,
    "deterministic_policy": True,
    "report_path": None,
}


def compute_tracking_mae(theta, theta_ref) -> float:
    error = _tracking_error(theta, theta_ref)
    return _rounded_float(np.mean(np.abs(error)))


def compute_tracking_mse(theta, theta_ref) -> float:
    error = _tracking_error(theta, theta_ref)
    return _rounded_float(np.mean(error**2))


def compute_overshoot(theta, theta_ref) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    theta_ref = np.asarray(theta_ref, dtype=np.float64)
    _validate_non_empty(theta, theta_ref)

    target = float(theta_ref[-1])
    if np.isclose(target, 0.0):
        overshoot = np.max(np.abs(theta))
    elif target > 0.0:
        overshoot = np.max(theta) - target
    else:
        overshoot = target - np.min(theta)
    return _rounded_float(max(0.0, float(overshoot)))


def compute_settling_time(
    theta,
    theta_ref,
    dt: float,
    tolerance: float,
    window: int,
) -> float | None:
    theta = np.asarray(theta, dtype=np.float64)
    theta_ref = np.asarray(theta_ref, dtype=np.float64)
    _validate_non_empty(theta, theta_ref)

    window = max(1, int(window))
    error_ok = np.abs(theta - theta_ref) <= float(tolerance)
    if len(error_ok) < window:
        return None

    for index in range(0, len(error_ok) - window + 1):
        if np.all(error_ok[index : index + window]):
            return _rounded_float(index * float(dt))
    return None


def get_evaluation_model_paths(config: dict[str, Any]) -> dict[str, Path]:
    evaluation = _evaluation_config(config)
    controllers = evaluation["controllers"]
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    models = evaluation.get("models") or config.get("demo", {}).get("models", {})

    if not models:
        models = _derive_model_paths_from_experiments(config)

    paths = {}
    for controller in controllers:
        if controller not in models:
            raise KeyError(f"Evaluation model path not configured for controller: {controller}")
        paths[controller] = _resolve_path(models[controller], config_dir)
    return paths


def make_evaluation_env(
    config: dict[str, Any],
    controller_name: str,
    episode_index: int | None = None,
) -> LearnedPAHMODE:
    evaluation = _evaluation_config(config)
    rl_config = config["rl_training"]
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    patterns = list(evaluation.get("unseen_wind_patterns", []))
    pattern = patterns[(episode_index or 0) % len(patterns)] if patterns else None

    return LearnedPAHMODE(
        render_mode=None,
        model_path=str(_resolve_path(rl_config["model_path"], config_dir)),
        reset_angle_deg=rl_config.get("reset_angle_deg", 720),
        enable_wind=bool(evaluation.get("wind_enabled", True)),
        wind_pattern=pattern,
        wind_seed=evaluation.get("wind_seed"),
        randomize_wind_pattern=bool(evaluation.get("randomize_wind_pattern", False)),
        config=config,
    )


def evaluate_controller(
    policy,
    env,
    config: dict[str, Any],
    *,
    controller_name: str,
) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    num_episodes = int(evaluation["num_episodes"])
    deterministic = bool(evaluation.get("deterministic_policy", True))
    reset_options = {
        **config.get("domain_randomization", {}).get("reset_options", {}),
        **evaluation.get("reset_options", {}),
    }
    reset_options.setdefault("randomize", True)

    episodes = []
    for episode_index in range(num_episodes):
        episodes.append(
            _run_evaluation_episode(
                policy,
                env,
                evaluation,
                reset_options,
                episode_index=episode_index,
                deterministic=deterministic,
            )
        )

    return {
        "controller": controller_name,
        "episodes": episodes,
        "summary": summarize_episode_metrics(episodes),
        "summary_by_wind_pattern": summarize_episode_metrics_by_key(
            episodes,
            "wind_pattern",
        ),
    }


def evaluate_controller_with_env_factory(
    policy,
    env_factory: Callable[[dict[str, Any], str, int | None], Any],
    config: dict[str, Any],
    *,
    controller_name: str,
) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    num_episodes = int(evaluation["num_episodes"])
    deterministic = bool(evaluation.get("deterministic_policy", True))
    reset_options = {
        **config.get("domain_randomization", {}).get("reset_options", {}),
        **evaluation.get("reset_options", {}),
    }
    reset_options.setdefault("randomize", True)

    episodes = []
    for episode_index in range(num_episodes):
        env = env_factory(config, controller_name, episode_index)
        try:
            episodes.append(
                _run_evaluation_episode(
                    policy,
                    env,
                    evaluation,
                    reset_options,
                    episode_index=episode_index,
                    deterministic=deterministic,
                )
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    return {
        "controller": controller_name,
        "episodes": episodes,
        "summary": summarize_episode_metrics(episodes),
        "summary_by_wind_pattern": summarize_episode_metrics_by_key(
            episodes,
            "wind_pattern",
        ),
    }


def compute_episode_metrics(
    trajectory: dict[str, list[float]],
    *,
    dt: float,
    tolerance: float,
    window: int,
    episode_return: float,
) -> dict[str, Any]:
    theta = trajectory["theta"]
    theta_ref = trajectory["theta_ref"]
    return {
        "mae_tracking_error": compute_tracking_mae(theta, theta_ref),
        "mse_tracking_error": compute_tracking_mse(theta, theta_ref),
        "settling_time": compute_settling_time(theta, theta_ref, dt, tolerance, window),
        "max_overshoot": compute_overshoot(theta, theta_ref),
        "episode_return": _rounded_float(episode_return),
    }


def summarize_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    metric_names = episodes[0]["metrics"].keys() if episodes else []
    summary = {}
    for name in metric_names:
        values = [episode["metrics"][name] for episode in episodes]
        finite_values = [float(value) for value in values if value is not None]
        if not finite_values:
            summary[name] = {"mean": None, "std": None}
        else:
            summary[name] = {
                "mean": _rounded_float(np.mean(finite_values)),
                "std": _rounded_float(np.std(finite_values)),
            }
    return summary


def summarize_episode_metrics_by_key(
    episodes: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, dict[str, float | None]]]:
    grouped = {}
    for episode in episodes:
        group_value = str(episode.get(key) or "unknown")
        grouped.setdefault(group_value, []).append(episode)
    return {
        group_value: summarize_episode_metrics(group_episodes)
        for group_value, group_episodes in grouped.items()
    }


def build_comparison_analysis(
    payload: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = evaluation or {}
    summary = payload.get("summary", {})
    controllers = list(payload.get("controllers", []))
    baseline = "naive" if "naive" in summary else (controllers[0] if controllers else None)
    candidate = "robust" if "robust" in summary else (controllers[1] if len(controllers) > 1 else None)
    metric_specs = _comparison_metric_specs()
    rows = []
    wins = {"baseline": 0, "candidate": 0, "tie": 0, "insufficient_data": 0}

    for metric_name, spec in metric_specs.items():
        baseline_mean = _summary_mean(summary, baseline, metric_name)
        candidate_mean = _summary_mean(summary, candidate, metric_name)
        winner = _metric_winner(baseline_mean, candidate_mean, spec["lower_is_better"])
        wins[winner] = wins.get(winner, 0) + 1
        rows.append(
            {
                "metric": metric_name,
                "label": spec["label"],
                "unit": spec["unit"],
                "lower_is_better": spec["lower_is_better"],
                "baseline_controller": baseline,
                "candidate_controller": candidate,
                "baseline_mean": baseline_mean,
                "candidate_mean": candidate_mean,
                "delta_candidate_minus_baseline": _delta(candidate_mean, baseline_mean),
                "relative_change_percent": _relative_change_percent(
                    candidate_mean,
                    baseline_mean,
                ),
                "winner": winner,
                "interpretation": spec["interpretation"],
            }
        )

    conclusion = _build_comparison_conclusion(
        baseline=baseline,
        candidate=candidate,
        wins=wins,
    )
    return {
        "baseline_controller": baseline,
        "candidate_controller": candidate,
        "unseen_wind_patterns": evaluation.get("unseen_wind_patterns", []),
        "settling_tolerance": evaluation.get("settling_tolerance"),
        "settling_window": evaluation.get("settling_window"),
        "comparison_rows": rows,
        "wins": wins,
        "conclusion": conclusion,
    }


def compare_controllers(
    config_path: str | Path,
    *,
    policy_loader: Callable[[Path, str], Any] | None = None,
    env_factory: Callable[[dict[str, Any], str, int | None], Any] | None = None,
    wandb_module=None,
) -> dict[str, Any]:
    config = load_config(config_path)
    evaluation = _evaluation_config(config)
    if not evaluation.get("enabled", True):
        raise ValueError("evaluation.enabled is false")

    model_paths = get_evaluation_model_paths(config)
    loader = policy_loader or (lambda path, algorithm: load_rl_policy(path, algorithm=algorithm))
    make_env = env_factory or make_evaluation_env
    algorithm = config["rl_training"].get("algorithm", "PPO")
    wandb_run = init_wandb_run(
        config,
        run_name="stage3-controller-evaluation",
        tags=["evaluation"],
        job_type="evaluate_controllers",
        wandb_module=wandb_module,
    )

    try:
        controller_results = {}
        for controller in evaluation["controllers"]:
            model_path = model_paths[controller]
            if not model_path.exists():
                raise FileNotFoundError(f"Evaluation model not found for {controller}: {model_path}")
            policy = loader(model_path, algorithm)
            result = evaluate_controller_with_env_factory(
                policy,
                make_env,
                config,
                controller_name=controller,
            )
            result["model_path"] = str(model_path)
            controller_results[controller] = result

        payload = {
            "controllers": list(evaluation["controllers"]),
            "summary": {
                controller: result["summary"]
                for controller, result in controller_results.items()
            },
            "results": controller_results,
            "artifacts": {},
        }
        payload["analysis"] = build_comparison_analysis(payload, evaluation)
        artifacts = _write_results(payload, config)
        payload["artifacts"] = artifacts
        _rewrite_json(payload, Path(artifacts["metrics_path"]))
        _log_evaluation_to_wandb(wandb_run, payload, config, wandb_module=wandb_module)
        return payload
    finally:
        finish_wandb_run(wandb_run)


def _evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_EVALUATION, **config.get("evaluation", {})}


def _derive_model_paths_from_experiments(config: dict[str, Any]) -> dict[str, str]:
    rl_config = config["rl_training"]
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    output_dir = _resolve_path(rl_config["model_output_dir"], config_dir)
    models = {}
    for controller in config.get("experiments", {}).get("modes", []):
        model_name = config["experiments"].get(controller, {}).get("model_name", f"pahm_ppo_{controller}")
        models[controller] = str(output_dir / f"{model_name}.zip")
    return models


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _tracking_error(theta, theta_ref) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    theta_ref = np.asarray(theta_ref, dtype=np.float64)
    _validate_non_empty(theta, theta_ref)
    return theta - theta_ref


def _validate_non_empty(theta: np.ndarray, theta_ref: np.ndarray) -> None:
    if theta.size == 0 or theta_ref.size == 0:
        raise ValueError("theta and theta_ref must not be empty")
    if theta.shape != theta_ref.shape:
        raise ValueError("theta and theta_ref must have the same shape")


def _rounded_float(value: float) -> float:
    return round(float(value), 12)


def _empty_trajectory() -> dict[str, list[float]]:
    return {
        "theta": [],
        "theta_ref": [],
        "actions": [],
        "rewards": [],
        "wind_torque": [],
        "tracking_error": [],
    }


def _append_step(
    trajectory: dict[str, list[float]],
    *,
    obs,
    action,
    reward,
    info: dict[str, Any],
) -> None:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    theta = float(obs[0])
    theta_ref = float(info.get("theta_ref", obs[2] if obs.size >= 3 else 0.0))
    tracking_error = float(info.get("tracking_error", theta - theta_ref))
    trajectory["theta"].append(_rounded_float(theta))
    trajectory["theta_ref"].append(_rounded_float(theta_ref))
    trajectory["actions"].append(_rounded_float(np.asarray(action).reshape(-1)[0]))
    trajectory["rewards"].append(_rounded_float(reward))
    trajectory["wind_torque"].append(_rounded_float(info.get("wind_torque", 0.0)))
    trajectory["tracking_error"].append(_rounded_float(tracking_error))


def _run_evaluation_episode(
    policy,
    env,
    evaluation: dict[str, Any],
    reset_options: dict[str, Any],
    *,
    episode_index: int,
    deterministic: bool,
) -> dict[str, Any]:
    obs, reset_info = env.reset(options=reset_options)
    trajectory = _empty_trajectory()
    episode_return = 0.0
    terminated = False
    truncated = False
    last_info = dict(reset_info)

    for step_index in range(int(evaluation["max_steps"])):
        action = predict_rl_action(policy, obs, env.action_space, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        episode_return += float(reward)
        _append_step(
            trajectory,
            obs=obs,
            action=action,
            reward=reward,
            info=info,
        )
        if terminated or truncated:
            break

    metrics = compute_episode_metrics(
        trajectory,
        dt=float(getattr(env, "dt", evaluation.get("dt", 0.02))),
        tolerance=float(evaluation["settling_tolerance"]),
        window=int(evaluation["settling_window"]),
        episode_return=episode_return,
    )
    wind_pattern = last_info.get("wind_pattern") or reset_info.get("wind_pattern")
    wind_source = last_info.get("wind_source") or reset_info.get("wind_source")
    return {
        "episode": episode_index,
        "steps": len(trajectory["theta"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "wind_pattern": wind_pattern,
        "wind_source": wind_source,
        "reset_info": _jsonable(reset_info),
        "metrics": metrics,
        "trajectory": trajectory,
    }


def _write_results(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    evaluation = _evaluation_config(config)
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    output_dir = _resolve_path(evaluation["output_dir"], config_dir)
    metrics_path = _resolve_path(evaluation["metrics_path"], config_dir)
    csv_path = output_dir / "controller_metrics.csv"
    report_path = (
        _resolve_path(evaluation["report_path"], config_dir)
        if evaluation.get("report_path")
        else output_dir / "informe.md"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _rewrite_json(payload, metrics_path)
    _write_csv(payload, csv_path)
    _write_report(payload, report_path)
    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


def _log_evaluation_to_wandb(
    wandb_run,
    payload: dict[str, Any],
    config: dict[str, Any],
    *,
    wandb_module=None,
) -> None:
    if isinstance(wandb_run, NullWandbRun):
        return
    wandb_config = normalize_wandb_config(config)
    if not wandb_config.get("log_evaluation", True):
        return

    log_evaluation_metrics(wandb_run, payload)
    artifacts = payload.get("artifacts", {})
    for key in ("metrics_path", "csv_path", "report_path"):
        path = artifacts.get(key)
        if not path:
            continue
        log_file_artifact(
            wandb_run,
            path,
            f"stage3-controller-{Path(path).stem}",
            artifact_type="metrics",
            enabled=True,
            wandb_module=wandb_module,
        )


def _rewrite_json(payload: dict[str, Any], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(payload: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for controller, result in payload["results"].items():
        for episode in result["episodes"]:
            rows.append(
                {
                    "controller": controller,
                    "episode": episode["episode"],
                    "wind_pattern": episode.get("wind_pattern"),
                    "wind_source": episode.get("wind_source"),
                    **episode["metrics"],
                }
            )
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")


def render_markdown_report(payload: dict[str, Any]) -> str:
    analysis = payload.get("analysis") or build_comparison_analysis(payload)
    lines = [
        "# Informe de comparacion de controladores RL",
        "",
        "## Contexto",
        "",
        f"- Controladores evaluados: {', '.join(payload.get('controllers', []))}.",
        "- Perturbaciones no vistas: "
        f"{', '.join(analysis.get('unseen_wind_patterns') or ['no especificadas'])}.",
        f"- Tolerancia de estabilizacion: {analysis.get('settling_tolerance')}.",
        f"- Ventana de estabilizacion: {analysis.get('settling_window')} pasos.",
        "",
        "## Tabla resumen naive vs robust",
        "",
        "| Metrica | Naive | Robust | Delta robust-naive | Cambio relativo | Mejor |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in analysis["comparison_rows"]:
        lines.append(
            "| {label} | {baseline} | {candidate} | {delta} | {relative} | {winner} |".format(
                label=row["label"],
                baseline=_format_optional_float(row["baseline_mean"]),
                candidate=_format_optional_float(row["candidate_mean"]),
                delta=_format_optional_float(row["delta_candidate_minus_baseline"]),
                relative=_format_percent(row["relative_change_percent"]),
                winner=_format_winner(row, analysis),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretacion de metricas",
            "",
        ]
    )
    for row in analysis["comparison_rows"]:
        direction = "menor es mejor" if row["lower_is_better"] else "mayor es mejor"
        lines.append(f"- **{row['label']}** ({direction}): {row['interpretation']}")

    lines.extend(
        [
            "",
            "## Resumen por perturbacion",
            "",
            "| Controlador | Perturbacion | MAE | MSE | Settling time | Overshoot | Retorno |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller, result in payload.get("results", {}).items():
        for wind_pattern, summary in result.get("summary_by_wind_pattern", {}).items():
            lines.append(
                "| {controller} | {wind_pattern} | {mae} | {mse} | {settling} | {overshoot} | {ret} |".format(
                    controller=controller,
                    wind_pattern=wind_pattern,
                    mae=_format_optional_float(_metric_mean(summary, "mae_tracking_error")),
                    mse=_format_optional_float(_metric_mean(summary, "mse_tracking_error")),
                    settling=_format_optional_float(_metric_mean(summary, "settling_time")),
                    overshoot=_format_optional_float(_metric_mean(summary, "max_overshoot")),
                    ret=_format_optional_float(_metric_mean(summary, "episode_return")),
                )
            )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            analysis["conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_metric_specs() -> dict[str, dict[str, Any]]:
    return {
        "mae_tracking_error": {
            "label": "MAE seguimiento",
            "unit": "rad",
            "lower_is_better": True,
            "interpretation": (
                "mide el error absoluto promedio respecto a theta_ref; valores "
                "menores indican seguimiento mas preciso."
            ),
        },
        "mse_tracking_error": {
            "label": "MSE seguimiento",
            "unit": "rad^2",
            "lower_is_better": True,
            "interpretation": (
                "penaliza errores grandes con mas fuerza que MAE; ayuda a detectar "
                "episodios con desviaciones severas."
            ),
        },
        "settling_time": {
            "label": "Tiempo de estabilizacion",
            "unit": "s",
            "lower_is_better": True,
            "interpretation": (
                "primer instante en que el error permanece dentro de la tolerancia "
                "durante la ventana configurada."
            ),
        },
        "max_overshoot": {
            "label": "Sobreimpulso maximo",
            "unit": "rad",
            "lower_is_better": True,
            "interpretation": (
                "exceso maximo sobre la referencia final; valores altos sugieren "
                "respuesta agresiva o poca amortiguacion."
            ),
        },
        "episode_return": {
            "label": "Retorno acumulado",
            "unit": "reward",
            "lower_is_better": False,
            "interpretation": (
                "suma de recompensas del episodio; valores mayores indican mejor "
                "desempeno bajo la funcion objetivo entrenada."
            ),
        },
    }


def _summary_mean(
    summary: dict[str, Any],
    controller: str | None,
    metric_name: str,
) -> float | None:
    if controller is None:
        return None
    metric = summary.get(controller, {}).get(metric_name, {})
    value = metric.get("mean")
    return None if value is None else float(value)


def _metric_mean(summary: dict[str, Any], metric_name: str) -> float | None:
    metric = summary.get(metric_name, {})
    value = metric.get("mean")
    return None if value is None else float(value)


def _metric_winner(
    baseline_mean: float | None,
    candidate_mean: float | None,
    lower_is_better: bool,
) -> str:
    if baseline_mean is None or candidate_mean is None:
        return "insufficient_data"
    if np.isclose(baseline_mean, candidate_mean):
        return "tie"
    candidate_wins = (
        candidate_mean < baseline_mean
        if lower_is_better
        else candidate_mean > baseline_mean
    )
    return "candidate" if candidate_wins else "baseline"


def _delta(candidate_mean: float | None, baseline_mean: float | None) -> float | None:
    if candidate_mean is None or baseline_mean is None:
        return None
    return _rounded_float(candidate_mean - baseline_mean)


def _relative_change_percent(
    candidate_mean: float | None,
    baseline_mean: float | None,
) -> float | None:
    if candidate_mean is None or baseline_mean is None or np.isclose(baseline_mean, 0.0):
        return None
    return _rounded_float(((candidate_mean - baseline_mean) / abs(baseline_mean)) * 100.0)


def _build_comparison_conclusion(
    *,
    baseline: str | None,
    candidate: str | None,
    wins: dict[str, int],
) -> str:
    if baseline is None or candidate is None:
        return (
            "No hay suficientes controladores para emitir una conclusion "
            "comparativa."
        )
    candidate_wins = wins.get("candidate", 0)
    baseline_wins = wins.get("baseline", 0)
    ties = wins.get("tie", 0)
    if candidate_wins > baseline_wins:
        return (
            f"El controlador {candidate} muestra superioridad global frente a "
            f"{baseline} en esta evaluacion: gana {candidate_wins} metricas, "
            f"pierde {baseline_wins} y empata {ties}. Esta conclusion debe "
            "defenderse junto con la tabla por perturbacion para confirmar que "
            "la mejora no proviene de un unico escenario."
        )
    if baseline_wins > candidate_wins:
        return (
            f"El controlador {candidate} no supera globalmente a {baseline} en "
            f"esta evaluacion: gana {candidate_wins} metricas, pierde "
            f"{baseline_wins} y empata {ties}. Conviene revisar entrenamiento, "
            "hiperparametros y severidad de perturbaciones antes de afirmar "
            "robustez superior."
        )
    return (
        f"La comparacion entre {candidate} y {baseline} queda empatada en esta "
        f"evaluacion: cada uno gana {candidate_wins} metricas y hay {ties} "
        "empates. La conclusion debe apoyarse en los resultados por "
        "perturbacion y en intervalos de confianza si se hacen corridas largas."
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def _format_winner(row: dict[str, Any], analysis: dict[str, Any]) -> str:
    winner = row["winner"]
    if winner == "candidate":
        return str(analysis.get("candidate_controller"))
    if winner == "baseline":
        return str(analysis.get("baseline_controller"))
    if winner == "tie":
        return "empate"
    return "datos insuficientes"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return copy.deepcopy(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua controladores RL PAHM en modo headless")
    parser.add_argument("--config", default="configs/stage3_config.json")
    args = parser.parse_args()

    result = compare_controllers(args.config)
    print(f"Resultados guardados en: {result['artifacts']['metrics_path']}")


if __name__ == "__main__":
    main()
