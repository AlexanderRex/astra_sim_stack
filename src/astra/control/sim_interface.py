"""Sim action interface (layer 6 boundary): sends one action per tick to the env.

Purpose
-------
Hide the env call details from the rest of the stack. What happens on the env side after ``step``
(for reference; implemented by Isaac Lab action terms in the env server, all applied on the same
physics step, physics 200 Hz, decimation 4):

    a[0:7]   left_arm_action    DifferentialInverseKinematicsAction (dls, command_type="pose",
                                use_relative_mode=False): absolute EEF pose in the robot root frame
                                → 7 left arm joint position targets
    a[7:14]  right_arm_action   same for the right arm
    a[14]    left_hand_action   BinaryJointPositionAction: value < 0 → 1.703 rad (close), else 0.0
    a[15]    right_hand_action  same
    a[16:19] base_velocity      GalbotBaseVelocityAction: body-frame [vx, vy, wz], smoothed, then
                                wheel velocity targets and root velocity
             (legacy 20-value actions carry wheel1..4 in a[16:20]; the env projects them to
             [vx, vy, wz] without smoothing)
    torso (leg_joint*) and head joints are not commanded and stay at their default pose.

Inputs:  SimAction (19,) or (20,), the task_env handle, num_envs (1 in evaluation).
Outputs: (raw observation dict, done flag, extras dict). The env gives the policy no reward, score
         or success flag; done only ends the rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from astra.types import SIM_ACTION_DIM, SimAction


class SimActionInterface:
    """Thin wrapper around ``task_env.step``.

    Attributes:
        task_env: env client handle passed into ``BasePolicy.eval``.
        num_envs: number of parallel envs (1 for evaluation; action is broadcast as (num_envs, D)).
        action_dim: the env's native action dimension (``usr_args["actions_dim"]``, 19).
        accepted_action_dims: every dimension the env takes (``usr_args["accepted_action_dims"]``,
            [19, 20]; only ``action_dim`` when the env does not report it).
        device: torch device of the action tensor. "cpu": ``RemoteEnv.step`` moves every action
            to the CPU before sending it, so a CUDA tensor only adds a round trip
            (``usr_args["action_device"]`` overrides).
        steps: env steps taken since reset.
        last_terminated: done flag (terminated or truncated) from the last step.
        last_extras: extras dict from the last step (logging, success info if provided).
    """

    def __init__(self, task_env: Any, usr_args: dict[str, Any]) -> None:
        self.task_env = task_env
        self.num_envs: int = int(usr_args.get("env_cfg", {}).get("num_envs", 1))
        self.action_dim: int = int(usr_args.get("actions_dim", SIM_ACTION_DIM))
        self.accepted_action_dims: list[int] = [
            int(d) for d in usr_args.get("accepted_action_dims", [self.action_dim])
        ]
        self.device: str = str(usr_args.get("action_device", "cpu"))
        self.steps: int = 0
        self.last_terminated: bool = False
        self.last_extras: dict[str, Any] = {}

    def check_contract(self, produced_dim: int) -> None:
        """Fail fast if the env does not accept the action dimension the codec produces.

        Args:
            produced_dim: result of ``ActionCodec.select_format`` (19 or 20).

        Raises:
            ValueError: ``produced_dim not in accepted_action_dims`` (wrong robot, task or format).
        """
        if int(produced_dim) not in self.accepted_action_dims:
            raise ValueError(
                f"the codec produces {produced_dim}-value actions, the env accepts {self.accepted_action_dims} "
                f"(native {self.action_dim}); check the robot, the task and schema.yaml sim_action_format"
            )

    def reset(self) -> None:
        """Episode start: clear counters."""
        self.steps = 0
        self.last_terminated = False
        self.last_extras = {}

    def step(self, action: SimAction) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """Send one action and advance the env by one control step.

        Functionality:
            Convert ``action.vector`` to a float32 torch tensor of shape (num_envs, D) on
            ``device`` and call ``task_env.step``. It returns ``(obs, terminated, truncated)``;
            env revisions before the restricted policy protocol return the 5-tuple
            ``(obs, reward, terminated, truncated, extras)``. Accept both, reduce
            ``terminated or truncated`` to a bool, update counters.

        Args:
            action: the sim action for this tick.

        Returns:
            (raw observation dict, done flag, extras; extras is empty when the env returns none).
        """
        import torch

        vec = np.asarray(action.vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] not in self.accepted_action_dims:
            raise ValueError(f"action has {vec.shape[0]} values, env accepts {self.accepted_action_dims}")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"non-finite sim action at tick {action.tick}: {vec}")
        tensor = torch.from_numpy(np.tile(vec, (self.num_envs, 1))).to(self.device)
        result = self.task_env.step(tensor)
        if isinstance(result, (tuple, list)) and len(result) == 3:
            obs, terminated, truncated = result
            extras: dict[str, Any] = {}
        elif isinstance(result, (tuple, list)) and len(result) >= 5:
            obs, _reward, terminated, truncated, extras = result[:5]
            extras = dict(extras) if isinstance(extras, dict) else {}
        else:
            raise RuntimeError(f"task_env.step returned {type(result).__name__} of unexpected length")
        done = bool(_any_true(terminated) or _any_true(truncated))
        self.steps += 1
        self.last_terminated = done
        self.last_extras = extras
        return obs, done, extras


def _any_true(flag: Any) -> bool:
    """Reduce a bool, numpy array or torch tensor flag to one bool."""
    if flag is None:
        return False
    if hasattr(flag, "detach"):
        flag = flag.detach().cpu().numpy()
    return bool(np.asarray(flag).any())
