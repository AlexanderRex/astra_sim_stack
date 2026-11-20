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

from astra.types import SIM_ACTION_DIM, SimAction


class SimActionInterface:
    """Thin wrapper around ``task_env.step``.

    Attributes:
        task_env: env client handle passed into ``BasePolicy.eval``.
        num_envs: number of parallel envs (1 for evaluation; action is broadcast as (num_envs, D)).
        action_dim: the env's native action dimension (``usr_args["actions_dim"]``, 19).
        accepted_action_dims: every dimension the env takes (``usr_args["accepted_action_dims"]``,
            [19, 20]; only ``action_dim`` when the env does not report it).
        device: torch device the env expects for the action tensor ("cuda" in BasePolicy).
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
        self.device: str = "cuda"
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
        raise NotImplementedError("SimActionInterface.check_contract")

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
        raise NotImplementedError("SimActionInterface.step")
