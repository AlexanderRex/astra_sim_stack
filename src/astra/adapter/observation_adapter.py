"""Observation adapter (layer 2): raw env observation → validated canonical observation → views.

Purpose
-------
Convert whatever the env server returns into one clean, checked representation (``CanonicalObs``)
and hand each consumer the format it needs. All indices come from the schema, never hard-coded.

Pipeline per tick
-----------------
1. **Ingest**: merge ``obs["policy"]`` and ``obs["embodiment_general_obs"]`` (as
   ``BasePolicy.encode_obs`` does), pick env 0, move tensors to numpy, keep images HWC uint8.
2. **Validate**: NaN/Inf, joint ranges, blank or constant images (renderer warm-up after reset),
   image shape/dtype. On failure reuse the last valid observation and count invalid ticks.
3. **Canonicalize**: build the 34-D state by **joint name** and **FK** (base-frame EEF poses):
   ``[left_eef_9d | right_eef_9d | left_arm_q | right_arm_q | gripper]``. Gripper positions are read
   by name (``left_gripper_joint``, ``right_gripper_joint``), not from the ``gripper_pos`` helper,
   which reads "the last two joints" of the articulation.
4. **Buffer**: push into the history buffer (sim-time indexed).
5. **Views**: ACT view, GR00T view, classifier window on request.

Inputs (raw observation from the env, per tick)
-----------------------------------------------
- ``joint_pos`` (77,): ABSOLUTE position of every joint (legs, head, arms, grippers, mimic fingers,
  wheels, ~40 passive rollers). Rollers are noise (std 2-3.8 rad in the baseline stats); excluded.
- ``joint_vel`` (77,): joint velocities.
  Both groups carry a ``joint_pos``: relative to the default pose in ``policy``, absolute in
  ``embodiment_general_obs``. The merge lets the second one win, so what arrives here is absolute
  (checked on the sim: 2.5, 1.571, 1.56 ... right after reset, where the relative one is all zeros).
- ``eef_pos`` (2, 3), ``eef_quat`` (2, 4): **world frame**; kept for logging only.
- ``gripper_pos`` (2,): ignored (see above).
- ``actions`` (19,): last action applied (20 on an env revision without the velocity interface).
- ``first_person_camera_rgb`` 480×640×3, ``left_hand_camera_rgb`` / ``right_hand_camera_rgb`` 224×224×3.

Outputs
-------
- ``CanonicalObs`` for the orchestrator, detectors and logging.
- ``ActView`` / ``Gr00tView`` / classifier tensors via the view builders.
- ``needs_hold``: True when too many consecutive observations were invalid (layer 5 sends a hold).
"""

from __future__ import annotations

from typing import Any

from astra.adapter.history import HistoryBuffer
from astra.adapter.kinematics import ForwardKinematics
from astra.adapter.views import ActViewBuilder, ClassifierViewBuilder, Gr00tViewBuilder
from astra.schema import Schema
from astra.types import CONTROL_HZ, ActView, CanonicalObs, Gr00tView, Stage


class ObservationAdapter:
    """Layer 2 entry point, one instance per policy, reset every episode.

    Attributes:
        schema: resolved schema.
        cfg: ``runtime.yaml: adapter`` section.
        fk: forward kinematics for base-frame EEF poses.
        history: ring buffer of recent CanonicalObs.
        act_views / groot_views / classifier_views: view builders.
        tick: control ticks since reset.
        last_valid: last observation that passed validation (fallback on invalid ticks).
        invalid_streak: consecutive invalid ticks.
        max_invalid_ticks: streak length after which ``needs_hold`` becomes True.
        blank_image_std_threshold: per-image pixel std below which the frame counts as blank.
    """

    def __init__(self, schema: Schema, cfg: dict[str, Any]) -> None:
        self.schema = schema
        self.cfg = cfg
        self.fk = ForwardKinematics(schema)
        self.history = HistoryBuffer(cfg.get("history_seconds", 3.0), schema.control_hz or CONTROL_HZ)
        self.act_views = ActViewBuilder(schema, tuple(cfg.get("act_image_hw", (224, 224))))
        self.groot_views = Gr00tViewBuilder(schema)
        self.classifier_views = ClassifierViewBuilder()
        self.tick: int = 0
        self.last_valid: CanonicalObs | None = None
        self.invalid_streak: int = 0
        self.max_invalid_ticks: int = int(cfg.get("max_invalid_ticks", 5))
        self.blank_image_std_threshold: float = float(cfg.get("blank_image_std_threshold", 2.0))

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """One-time setup: load the FK chains (``self.fk.load()``) and check camera specs.

        Raises:
            RuntimeError: if the robot description cannot be loaded.
        """
        raise NotImplementedError("ObservationAdapter.load")

    def reset(self) -> None:
        """Start of an episode: tick = 0, clear history, last_valid = None, invalid_streak = 0."""
        raise NotImplementedError("ObservationAdapter.reset")

    # ------------------------------------------------------------------ per tick
    def process(self, raw_obs: dict[str, Any]) -> CanonicalObs:
        """Run ingest → validate → canonicalize → buffer for one tick.

        Args:
            raw_obs: observation dict returned by ``task_env.step()`` / ``reset()``.

        Returns:
            CanonicalObs for this tick (or the last valid one, flagged ``valid=False``).
        """
        raise NotImplementedError("ObservationAdapter.process")

    def ingest(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Flatten the env dict and convert to numpy.

        Functionality:
            Merge ``raw_obs["policy"]`` and ``raw_obs["embodiment_general_obs"]``, select env 0,
            ``.cpu().numpy()`` tensors, keep images HWC uint8 (convert from float [0,1] if needed).

        Args:
            raw_obs: env observation dict.

        Returns:
            flat dict ``{key: np.ndarray}`` keyed by the env's observation names.
        """
        raise NotImplementedError("ObservationAdapter.ingest")

    def validate(self, flat: dict[str, Any]) -> list[str]:
        """Check one flattened observation.

        Checks:
            - all schema camera keys present, shape = schema hw, dtype uint8;
            - image not blank (pixel std > threshold) — the first frames after reset can be black;
            - no NaN/Inf in joint_pos, joint_vel, actions;
            - arm joint angles within limits (+ margin).

        Args:
            flat: output of ``ingest``.

        Returns:
            list of issue strings; empty means valid.
        """
        raise NotImplementedError("ObservationAdapter.validate")

    def canonicalize(self, flat: dict[str, Any]) -> CanonicalObs:
        """Build the CanonicalObs (34-D state, base-frame EEF, images by Astra camera name).

        Functionality:
            - arm q / gripper: ``joint_pos[schema.state_joint_groups[...]]``;
            - EEF: ``fk.compute(joint_pos)`` → positions + rotation matrices → xyz + rot6d (rows);
            - images: rename env keys to "head", "left_wrist", "right_wrist";
            - t = tick / control_hz.

        Args:
            flat: validated output of ``ingest``.

        Returns:
            CanonicalObs with ``valid=True``.
        """
        raise NotImplementedError("ObservationAdapter.canonicalize")

    @property
    def needs_hold(self) -> bool:
        """True when ``invalid_streak >= max_invalid_ticks`` (layer 5 should send a hold action)."""
        raise NotImplementedError("ObservationAdapter.needs_hold")

    # ------------------------------------------------------------------ views
    def act_view(self, obs: CanonicalObs, stage: Stage) -> ActView:
        """ACT input for this tick (delegates to ``ActViewBuilder.build``)."""
        raise NotImplementedError("ObservationAdapter.act_view")

    def groot_view(self, obs: CanonicalObs, prompt: str) -> Gr00tView:
        """GR00T input for this tick (delegates to ``Gr00tViewBuilder.build``)."""
        raise NotImplementedError("ObservationAdapter.groot_view")

    def classifier_input(self) -> dict[str, Any]:
        """Stage-classifier input from the history window (delegates to ``ClassifierViewBuilder``)."""
        raise NotImplementedError("ObservationAdapter.classifier_input")
