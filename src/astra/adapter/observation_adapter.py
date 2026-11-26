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

Invalid observations
--------------------
If only the images fail (blank frames after reset, a missing camera), the fresh proprioception is
kept and the images of the last valid observation are reused. If the proprioception fails (NaN/Inf),
the whole last valid observation is repeated with the new tick. Either way the result is flagged
``valid=False`` and counts toward ``needs_hold``.

Without a URDF (``kinematics.urdf_path: null``) the base-frame EEF fields repeat the last commanded
pose from ``obs["actions"]`` (``eef_base_from_fk`` False); right after reset that command is all
zeros, so ``eef_base_valid`` is False until a pose has been sent.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS

from astra.adapter.history import HistoryBuffer
from astra.adapter.kinematics import ForwardKinematics
from astra.adapter.rotations import matrix_to_rot6d, quat_to_matrix
from astra.adapter.views import ActViewBuilder, ClassifierViewBuilder, Gr00tViewBuilder
from astra.schema import Schema
from astra.types import (
    CAMERA_NAMES,
    CONTROL_HZ,
    SIM_LEFT_EEF,
    SIM_RIGHT_EEF,
    STATE_DIM,
    ActView,
    CanonicalObs,
    Gr00tView,
    Stage,
)

log = logging.getLogger(__name__)

ARM_JOINT_ABS_LIMIT_RAD = 2.0 * np.pi  # sanity bound when no URDF limits are known
_JOINT_DEFAULT_POS = np.asarray(JOINT_DEFAULT_POS, dtype=np.float64)
PROPRIO_KEYS = ("joint_pos", "joint_vel", "eef_pos", "eef_quat", "actions")


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
        invalid_total: invalid ticks in this episode (logging).
    """

    def __init__(self, schema: Schema, cfg: dict[str, Any], use_stage_conditioning: bool = True) -> None:
        self.schema = schema
        self.cfg = cfg
        self.fk = ForwardKinematics(schema)
        self.history = HistoryBuffer(cfg.get("history_seconds", 3.0), schema.control_hz or CONTROL_HZ)
        self.act_views = ActViewBuilder(schema, tuple(cfg.get("act_image_hw", (224, 224))), use_stage_conditioning)
        self.act_views.interpolation = str(cfg.get("act_resize_interpolation", "area"))
        self.groot_views = Gr00tViewBuilder(schema)
        self.classifier_views = ClassifierViewBuilder()
        self.tick: int = 0
        self.last_valid: CanonicalObs | None = None
        self.invalid_streak: int = 0
        self.max_invalid_ticks: int = int(cfg.get("max_invalid_ticks", 5))
        self.blank_image_std_threshold: float = float(cfg.get("blank_image_std_threshold", 2.0))
        self.invalid_total: int = 0
        self._max_joint_id = max(
            (int(ids.max()) for ids in schema.state_joint_groups.values() if len(ids)), default=-1
        )

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """One-time setup: load the FK chains (``self.fk.load()``) and check camera specs.

        Raises:
            RuntimeError: if the robot description cannot be loaded.
        """
        try:
            self.fk.load()
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(f"forward kinematics: {exc}") from exc
        missing = [c for c in CAMERA_NAMES if c not in self.schema.cameras]
        if missing:
            raise RuntimeError(f"schema has no camera spec for {missing}")

    def reset(self) -> None:
        """Start of an episode: tick = 0, clear history, last_valid = None, invalid_streak = 0."""
        self.tick = 0
        self.history.reset()
        self.last_valid = None
        self.invalid_streak = 0
        self.invalid_total = 0

    # ------------------------------------------------------------------ per tick
    def process(self, raw_obs: dict[str, Any]) -> CanonicalObs:
        """Run ingest → validate → canonicalize → buffer for one tick.

        Args:
            raw_obs: observation dict returned by ``task_env.step()`` / ``reset()``.

        Returns:
            CanonicalObs for this tick (or the last valid one, flagged ``valid=False``).
        """
        flat = self.ingest(raw_obs)
        issues = self.validate(flat)
        if not issues:
            obs = self.canonicalize(flat)
            self.invalid_streak = 0
            self.last_valid = obs
        else:
            self.invalid_streak += 1
            self.invalid_total += 1
            proprio_bad = any(not i.startswith("image") for i in issues)
            if proprio_bad and self.last_valid is not None:
                obs = dataclasses.replace(
                    self.last_valid, tick=self.tick, t=self.tick / self.schema.control_hz, valid=False, issues=issues
                )
            else:
                obs = self.canonicalize(self._sanitize(flat))
                if self.last_valid is not None:
                    obs.images = dict(self.last_valid.images)
                obs.valid = False
                obs.issues = issues
            if self.invalid_streak == 1 or self.invalid_streak % 50 == 0:
                log.warning("tick %d: invalid observation (%d in a row): %s", self.tick, self.invalid_streak, issues)
        self.history.push(obs)
        self.tick += 1
        return obs

    def ingest(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Flatten the env dict and convert to numpy.

        Functionality:
            Take ``raw_obs["policy"]``, select env 0, ``.cpu().numpy()`` tensors, keep images HWC
            uint8 (convert from float [0,1] if needed), add ``JOINT_DEFAULT_POS`` to ``joint_pos``.
            The ``embodiment_general_obs`` group is ignored: the evaluation service sends it empty.

        Args:
            raw_obs: env observation dict.

        Returns:
            flat dict ``{key: np.ndarray}`` keyed by the env's observation names.
        """
        merged: dict[str, Any] = dict(raw_obs.get("policy") or {})
        image_keys = {spec.obs_key for spec in self.schema.cameras.values()}
        flat: dict[str, Any] = {}
        for key, value in merged.items():
            arr = _to_numpy(value)
            if arr is None:
                continue
            if arr.ndim >= 1 and arr.shape[0] >= 1 and _has_env_dim(key, arr, image_keys):
                arr = arr[0]
            if key in image_keys:
                arr = _to_hwc_uint8(arr)
            flat[key] = arr
        if "eef_pos" in flat:
            flat["eef_pos"] = np.asarray(flat["eef_pos"], dtype=np.float64).reshape(-1, 3)
        if "eef_quat" in flat:
            flat["eef_quat"] = np.asarray(flat["eef_quat"], dtype=np.float64).reshape(-1, 4)
        for key in ("joint_pos", "joint_vel", "actions"):
            if key in flat:
                flat[key] = np.asarray(flat[key], dtype=np.float64).reshape(-1)
        jp = flat.get("joint_pos")
        if jp is not None and jp.shape == _JOINT_DEFAULT_POS.shape:
            flat["joint_pos"] = jp + _JOINT_DEFAULT_POS  # relative to the default pose -> absolute
        return flat

    def validate(self, flat: dict[str, Any]) -> list[str]:
        """Check one flattened observation.

        Checks:
            - all schema camera keys present, shape = schema hw, dtype uint8;
            - image not blank (pixel std > threshold) — the first frames after reset can be black;
            - no NaN/Inf in joint_pos, joint_vel, eef pose, actions;
            - arm joint angles within a sanity bound (|q| < 2π; no URDF limits are applied).

        Issue strings start with ``image`` for camera problems and ``state`` for everything else.

        Args:
            flat: output of ``ingest``.

        Returns:
            list of issue strings; empty means valid.
        """
        issues: list[str] = []
        for name, spec in self.schema.cameras.items():
            img = flat.get(spec.obs_key)
            if img is None:
                issues.append(f"image {name}: missing ({spec.obs_key})")
                continue
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                issues.append(f"image {name}: dtype {img.dtype} shape {img.shape}, need uint8 HxWx3")
                continue
            if tuple(img.shape[:2]) != tuple(spec.hw):
                issues.append(f"image {name}: size {img.shape[:2]}, schema says {spec.hw}")
                continue
            # std over a strided subsample: cheap and enough to spot a black or constant frame
            if float(img[::4, ::4].std()) < self.blank_image_std_threshold:
                issues.append(f"image {name}: blank frame")
        for key in PROPRIO_KEYS:
            arr = flat.get(key)
            if arr is None:
                issues.append(f"state {key}: missing")
            elif not np.all(np.isfinite(arr)):
                issues.append(f"state {key}: NaN/Inf")
        jp = flat.get("joint_pos")
        if jp is not None:
            if jp.shape[0] <= self._max_joint_id:
                issues.append(f"state joint_pos: {jp.shape[0]} entries, schema indexes {self._max_joint_id}")
            else:
                for group in ("left_arm_q", "right_arm_q"):
                    ids = self.schema.state_joint_groups.get(group)
                    if ids is not None and np.any(np.abs(jp[ids]) > ARM_JOINT_ABS_LIMIT_RAD):
                        issues.append(f"state {group}: joint angle outside ±2π")
        for key, shape in (("eef_pos", (2, 3)), ("eef_quat", (2, 4))):
            arr = flat.get(key)
            if arr is not None and arr.shape != shape:
                issues.append(f"state {key}: shape {arr.shape}, need {shape}")
        return issues

    def canonicalize(self, flat: dict[str, Any]) -> CanonicalObs:
        """Build the CanonicalObs (34-D state, base-frame EEF from FK, images by Astra camera name).

        Functionality:
            - arm q / gripper: ``joint_pos[schema.state_joint_groups[...]]``;
            - world EEF groups: ``eef_pos`` + rot6d (rows) of ``eef_quat`` read in
              ``schema.obs_quat_order``;
            - ``eef_base_*``: ``fk.compute(joint_pos)``, for the detectors and the hold action
              (without FK: the last commanded pose);
            - images: rename env keys to "head", "left_wrist", "right_wrist";
            - t = tick / control_hz.

        Args:
            flat: validated output of ``ingest``.

        Returns:
            CanonicalObs with ``valid=True``.
        """
        jp = np.asarray(flat["joint_pos"], dtype=np.float64)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for name, ids in self.schema.state_joint_groups.items():
            state[self.schema.state_slices[name]] = jp[ids]
        eef_pos = np.asarray(flat["eef_pos"], dtype=np.float64).reshape(2, 3)
        eef_quat = np.asarray(flat["eef_quat"], dtype=np.float64).reshape(2, 4)
        for name, arm in self.schema.eef_world_groups.items():
            rot6d = matrix_to_rot6d(quat_to_matrix(eef_quat[arm], self.schema.obs_quat_order))
            state[self.schema.state_slices[name]] = np.concatenate([eef_pos[arm], rot6d])

        last_action = np.asarray(flat["actions"], dtype=np.float32).copy()
        gripper_ids = self.schema.state_joint_groups.get("gripper")
        gripper_raw = jp[gripper_ids].astype(np.float32) if gripper_ids is not None else np.zeros(2, np.float32)

        if self.fk.available:
            base_pos, base_rot = self.fk.compute(jp)
            base_valid, from_fk = True, True
        else:
            base_pos, base_rot, base_valid = _pose_from_command(last_action, self.schema.action_quat_order)
            from_fk = False

        images = {name: flat[spec.obs_key] for name, spec in self.schema.cameras.items() if spec.obs_key in flat}
        return CanonicalObs(
            tick=self.tick,
            t=self.tick / self.schema.control_hz,
            images=images,
            state=state,
            joint_pos=jp.astype(np.float32),
            joint_vel=np.asarray(flat["joint_vel"], dtype=np.float32),
            gripper_raw=gripper_raw,
            eef_base_pos=base_pos,
            eef_base_rotmat=base_rot,
            last_action=last_action,
            valid=True,
            issues=[],
            eef_base_valid=base_valid,
            eef_base_from_fk=from_fk,
        )

    def _sanitize(self, flat: dict[str, Any]) -> dict[str, Any]:
        """Fill missing or non-finite proprioception with zeros so ``canonicalize`` can still run."""
        out = dict(flat)
        n_joints = max(self._max_joint_id + 1, len(self.schema.joint_names), 1)
        defaults = {
            "joint_pos": np.zeros(n_joints),
            "joint_vel": np.zeros(n_joints),
            "eef_pos": np.zeros((2, 3)),
            "eef_quat": np.tile([0.0, 0.0, 0.0, 1.0] if self.schema.obs_quat_order == "xyzw" else [1.0, 0, 0, 0], (2, 1)),
            "actions": np.zeros(19),
        }
        for key, default in defaults.items():
            arr = out.get(key)
            if arr is None or np.shape(arr) == () or (key == "joint_pos" and arr.shape[0] <= self._max_joint_id):
                out[key] = default
            else:
                out[key] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @property
    def needs_hold(self) -> bool:
        """True when ``invalid_streak >= max_invalid_ticks`` (layer 5 should send a hold action)."""
        return self.invalid_streak >= self.max_invalid_ticks

    # ------------------------------------------------------------------ views
    def act_view(self, obs: CanonicalObs, stage: Stage) -> ActView:
        """ACT input for this tick (delegates to ``ActViewBuilder.build``)."""
        return self.act_views.build(obs, stage)

    def groot_view(self, obs: CanonicalObs, prompt: str) -> Gr00tView:
        """GR00T input for this tick (delegates to ``Gr00tViewBuilder.build``)."""
        return self.groot_views.build(obs, prompt)

    def classifier_input(self) -> dict[str, Any]:
        """Stage-classifier input from the history window (delegates to ``ClassifierViewBuilder``).

        Returns an empty dict while the history is empty.
        """
        if len(self.history) == 0:
            return {}
        cv = self.classifier_views
        return cv.build(self.history.window(cv.seconds, cv.rate_hz))


# ---------------------------------------------------------------------- helpers
def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        return np.asarray(value)
    except Exception:  # noqa: BLE001 - non-array extras are dropped
        return None


def _has_env_dim(key: str, arr: np.ndarray, image_keys: set[str]) -> bool:
    """True if ``arr`` still carries the leading num_envs dimension."""
    if key in image_keys:
        return arr.ndim == 4
    if key in ("eef_pos", "eef_quat"):
        return arr.ndim >= 2 and not (arr.ndim == 2 and arr.shape[-1] in (3, 4) and arr.shape[0] == 2)
    return arr.ndim >= 2


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Image to HWC uint8 (accepts CHW, float in [0, 1] or [0, 255], an alpha channel)."""
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        f = np.nan_to_num(img.astype(np.float32))
        if f.size and float(f.max()) <= 1.0 + 1e-6:
            f = f * 255.0
        img = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _pose_from_command(action: np.ndarray, quat_order: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Base-frame EEF poses from the last commanded sim action; invalid if a quaternion is ~0."""
    pos = np.zeros((2, 3), dtype=np.float64)
    rot = np.tile(np.eye(3), (2, 1, 1))
    if action.shape[0] < 14:
        return pos, rot, False
    valid = True
    for i, sl in enumerate((SIM_LEFT_EEF, SIM_RIGHT_EEF)):
        a = np.asarray(action[sl], dtype=np.float64)
        q = a[3:7]
        if not np.all(np.isfinite(a)) or np.linalg.norm(q) < 0.5:
            valid = False
            continue
        pos[i] = a[0:3]
        rot[i] = quat_to_matrix(q, quat_order)
    return pos, rot, valid
