"""Self-play training loop.

Trains a single learner against an evolving pool of frozen past selves. This
script *only* trains: it builds the env via the factory, drives the SB3 model,
snapshots checkpoints into the opponent pool, and tracks ELO. Evaluation lives
entirely in ``evaluate.py`` (CLAUDE.md: never mix train and eval logic).

Run indirectly through ``run_experiment.py``; the ``train`` function is the
importable entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.opponent_pool import EloTracker, OpponentPool
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from agents.self_play import SelfPlaySingleAgentEnv
from envs.env_factory import make_env
from experiments.config_schema import ExperimentConfig

logger = logging.getLogger("racetrack_rl")


def _build_agent(algorithm: str, env, config: ExperimentConfig, tb_log: Path):
    """Instantiate the learner for the configured algorithm."""

    train = config.training
    if algorithm == "ppo":
        return PPOAgent(
            env,
            hyperparams=train.ppo.model_dump(),
            device=train.device,
            seed=train.seed,
            tensorboard_log=tb_log,
        )
    if algorithm == "sac":
        return SACAgent(
            env,
            hyperparams=train.sac.model_dump(),
            device=train.device,
            seed=train.seed,
            tensorboard_log=tb_log,
        )
    raise ValueError(f"unknown algorithm '{algorithm}' (expected 'ppo' or 'sac')")


def _make_self_play_callback(pool: OpponentPool, elo: EloTracker, config: ExperimentConfig):
    """Build the SB3 callback that runs the self-play bookkeeping.

    Snapshots the learner into the pool every ``checkpoint_interval`` steps and
    updates ELO at each episode end. Logs ``train/learner_elo`` and
    ``train/pool_mean_elo`` to TensorBoard.
    """

    from stable_baselines3.common.callbacks import BaseCallback

    class _SelfPlayCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._last_checkpoint = 0
            self.interval = config.training.checkpoint_interval

        def _on_step(self) -> bool:
            # Episode-end ELO updates (per env in the vec env).
            for i, done in enumerate(self.locals.get("dones", [])):
                if not done:
                    continue
                info = self.locals["infos"][i]
                opponent = self.training_env.envs[i].unwrapped.current_opponent
                if opponent is not None:
                    elo.update(opponent, learner_won=bool(info.get("won", False)))
                self.logger.record("train/learner_elo", elo.learner_elo)
                self.logger.record("train/pool_mean_elo", pool.mean_elo)

            # Periodic checkpoint snapshot into the pool.
            if self.num_timesteps - self._last_checkpoint >= self.interval:
                pool.add_checkpoint(
                    self.model.policy, elo=elo.learner_elo, step=self.num_timesteps
                )
                self._last_checkpoint = self.num_timesteps
            return True

    return _SelfPlayCallback()


def train(
    config: ExperimentConfig,
    output_dir: str | Path,
    seed: int | None = None,
) -> Path:
    """Run the full self-play training and return the final model path.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    output_dir : str or Path
        Directory to write checkpoints, TensorBoard logs, and the final model.
        Only this script (under ``experiments/``) writes to the filesystem.
    seed : int, optional
        Master seed; overrides the config seed when given.

    Returns
    -------
    Path
        Path to the saved final model (``final.zip``).
    """

    seed = seed if seed is not None else config.training.seed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log = output_dir / "tb"

    env_config = config.to_env_config(seed=seed)
    parallel_env = make_env(env_config)

    elo = EloTracker(learner_elo=1000.0)

    def learner_elo() -> float:
        return elo.learner_elo

    # The pool starts empty; the self-play env holds a reference to it, and we
    # seed slot 0 with the learner's freshly-initialised policy once the SB3
    # model exists. Both learner and opponent therefore begin identical.
    pool = OpponentPool(
        initial_policy=None,
        max_size=config.training.opponent_pool_max_size,
        temperature=config.training.pool_temperature,
        seed=seed,
    )
    sp_env = SelfPlaySingleAgentEnv(
        parallel_env, pool, learner_id="agent_0", learner_elo_fn=learner_elo
    )
    agent = _build_agent(config.training.algorithm, sp_env, config, tb_log)
    pool.add_checkpoint(agent.policy, elo=1000.0, step=0)

    callback = _make_self_play_callback(pool, elo, config)
    logger.info(
        "Training %s for %d steps (experiment=%s, seed=%d)",
        config.training.algorithm,
        config.training.total_timesteps,
        config.experiment_name,
        seed,
    )
    agent.learn(total_timesteps=config.training.total_timesteps, callback=callback)

    final_path = output_dir / "final.zip"
    agent.save(final_path)
    return final_path
