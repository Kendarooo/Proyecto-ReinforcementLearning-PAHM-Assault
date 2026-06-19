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
        randomize_wind_pattern=bool(evaluation.get("randomize_wind_pattern", len(patterns) > 1)),
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
    max_steps = int(evaluation["max_steps"])
    deterministic = bool(evaluation.get("deterministic_policy", True))
    reset_options = {
        **config.get("domain_randomization", {}).get("reset_options", {}),
        **evaluation.get("reset_options", {}),
    }
    reset_options.setdefault("randomize", True)

    episodes = []
    for episode_index in range(num_episodes):
        obs, reset_info = env.reset(options=reset_options)
        trajectory = _empty_trajectory()
        episode_return = 0.0
        terminated = False
        truncated = False

        for step_index in range(max_steps):
            action = predict_rl_action(policy, obs, env.action_space, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
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
        episodes.append(
            {
                "episode": episode_index,
                "steps": len(trajectory["theta"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reset_info": _jsonable(reset_info),
                "metrics": metrics,
                "trajectory": trajectory,
            }
        )

    return {
        "controller": controller_name,
        "episodes": episodes,
        "summary": summarize_episode_metrics(episodes),
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


def compare_controllers(
    config_path: str | Path,
    *,
    policy_loader: Callable[[Path, str], Any] | None = None,
    env_factory: Callable[[dict[str, Any], str, int | None], Any] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    evaluation = _evaluation_config(config)
    if not evaluation.get("enabled", True):
        raise ValueError("evaluation.enabled is false")

    model_paths = get_evaluation_model_paths(config)
    loader = policy_loader or (lambda path, algorithm: load_rl_policy(path, algorithm=algorithm))
    make_env = env_factory or make_evaluation_env
    algorithm = config["rl_training"].get("algorithm", "PPO")

    controller_results = {}
    for controller in evaluation["controllers"]:
        model_path = model_paths[controller]
        if not model_path.exists():
            raise FileNotFoundError(f"Evaluation model not found for {controller}: {model_path}")
        policy = loader(model_path, algorithm)
        env = make_env(config, controller, None)
        try:
            result = evaluate_controller(policy, env, config, controller_name=controller)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
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
    artifacts = _write_results(payload, config)
    payload["artifacts"] = artifacts
    _rewrite_json(payload, Path(artifacts["metrics_path"]))
    return payload


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


def _write_results(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    evaluation = _evaluation_config(config)
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    output_dir = _resolve_path(evaluation["output_dir"], config_dir)
    metrics_path = _resolve_path(evaluation["metrics_path"], config_dir)
    csv_path = output_dir / "controller_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    _rewrite_json(payload, metrics_path)
    _write_csv(payload, csv_path)
    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "csv_path": str(csv_path),
    }


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
                    **episode["metrics"],
                }
            )
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
