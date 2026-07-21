"""Racing-line overlays and tactic maps from logged trajectories.

    python analysis/plot_trajectories.py --trajectories results/.../trajectories.json

Supports Axis 4 (emergent behaviour taxonomy): plots agent racing lines, colours
segments by the discrete action taken, and overlays the opponent so blocking and
overtaking are visible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.racetrack_env import DISCRETE_ACTIONS

logger = logging.getLogger("racetrack_rl")

# Stable colour per discrete action for the tactic map.
_ACTION_COLOURS = {
    "LANE_LEFT": "tab:blue",
    "IDLE": "tab:gray",
    "LANE_RIGHT": "tab:green",
    "FASTER": "tab:red",
    "SLOWER": "tab:purple",
}


def plot_trajectory(
    trajectory: list[list[float]],
    actions: list[int] | None = None,
    opponent: list[list[float]] | None = None,
    out_path: str | Path = "trajectory.png",
) -> Path:
    """Plot one episode's racing line, coloured by action when provided.

    Parameters
    ----------
    trajectory : list of [x, y]
        Normalised agent positions over the episode.
    actions : list of int, optional
        Discrete action index per step; segments are coloured accordingly.
    opponent : list of [x, y], optional
        Opponent positions, drawn faint for context.
    out_path : str or Path
        Output figure path.

    Returns
    -------
    Path
        The saved figure path.
    """

    traj = np.asarray(trajectory, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 7))

    if actions is not None and len(actions) >= len(traj) - 1:
        for i in range(len(traj) - 1):
            label = DISCRETE_ACTIONS[actions[i]]
            ax.plot(
                traj[i : i + 2, 0], traj[i : i + 2, 1],
                color=_ACTION_COLOURS.get(label, "black"), linewidth=2,
            )
        handles = [
            plt.Line2D([], [], color=c, label=a) for a, c in _ACTION_COLOURS.items()
        ]
        ax.legend(handles=handles, fontsize=8, title="action")
    else:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:red", linewidth=2, label="agent")

    if opponent is not None:
        opp = np.asarray(opponent, dtype=float)
        ax.plot(opp[:, 0], opp[:, 1], color="black", alpha=0.4, linewidth=1.5,
                label="opponent")

    ax.set_aspect("equal")
    ax.set_xlabel("x (norm)")
    ax.set_ylabel("y (norm)")
    ax.set_title("Racing line / tactic map")
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved trajectory plot to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Plot episode trajectories.")
    p.add_argument("--trajectories", required=True, help="JSON file with a trajectory.")
    p.add_argument("--out", default="trajectory.png")
    args = p.parse_args()
    data = json.loads(Path(args.trajectories).read_text())
    plot_trajectory(
        trajectory=data["trajectory"],
        actions=data.get("actions"),
        opponent=data.get("opponent"),
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
