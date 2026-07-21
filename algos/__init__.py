"""PPO and GRPO. Both share algos/common/ so the only differences between them
are the advantage estimator and the presence of a value loss."""

from algos.grpo import GRPO
from algos.ppo import PPO

__all__ = ["GRPO", "PPO"]
