"""Observation adapter (layer 2): raw env observation → validated canonical observation → views.

Purpose
-------
Convert whatever the env server returns into one clean, checked representation (``CanonicalObs``)
and hand each consumer the format it needs. All indices come from the schema, never hard-coded.

Pipeline per tick
-----------------
1. **Ingest**: take ``obs["policy"]``, pick env 0, move tensors to numpy, keep images HWC uint8,
   turn the relative joint positions into absolute ones.
2. **Validate**: NaN/Inf, joint ranges, blank or constant images (renderer warm-up after reset),
   image shape/dtype. On failure reuse the last valid observation and count invalid ticks.
3. **Canonicalize**: build the 34-D state from joint angles and the observed EEF poses:
   ``[left_arm_q | right_arm_q | gripper | left_eef_world | right_eef_world]``. Gripper positions
   are read from ``joint_pos`` by index (``gripper_pos`` holds the same two joints). Forward kinematics is not needed for the state; it fills the base-frame EEF
   fields of ``CanonicalObs`` for the detectors and the hold action.
4. **Buffer**: push into the history buffer (sim-time indexed).
5. **Views**: ACT view, GR00T view, classifier window on request.

Inputs (raw observation from the env, per tick)
-----------------------------------------------
- ``joint_pos`` (77,): position of every joint (legs, head, arms, grippers, mimic fingers, wheels,
  ~40 passive rollers) RELATIVE to the default pose: all zeros right after reset. ``ingest`` adds
  ``JOINT_DEFAULT_POS`` (task repo, ``robofinals/distributed/chamber_schema.py``), so the rest of
  the stack sees absolute angles, as the dataset holds them. Rollers are noise (std 2-3.8 rad in the
  baseline stats); excluded.
- ``joint_vel`` (77,): joint velocities.
- ``eef_pos`` (2, 3), ``eef_quat`` (2, 4, xyzw): **world frame**; they become the two
  ``*_eef_world`` state groups (xyz + rot6d).
- ``gripper_pos`` (2,): left and right driven gripper joint, absolute; the same values as joints
  65 and 69 of ``joint_pos``.
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
            Take ``raw_obs["policy"]``, select env 0, ``.cpu().numpy()`` tensors, keep images HWC
            uint8, add ``JOINT_DEFAULT_POS`` to ``joint_pos``.

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
        """Build the CanonicalObs (34-D state, base-frame EEF from FK, images by Astra camera name).

        Functionality:
            - arm q / gripper: ``joint_pos[schema.state_joint_groups[...]]``;
            - world EEF groups: ``eef_pos`` + rot6d (rows) of ``eef_quat`` read as xyzw;
            - ``eef_base_*``: ``fk.compute(joint_pos)``, for the detectors and the hold action;
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
