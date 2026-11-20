"""Shared data types and fixed vector layouts for the Astra runtime loop.

Purpose
-------
Single place that defines the data passed between the runtime layers, so every subsystem agrees on
shapes, orderings and units. Nothing here has behaviour; it is the contract between modules.

Three vector layouts matter:

1. **Sim action (19-D)**, fixed by the env (``galbot.py`` ``MobileActionsCfg``, field order):

   ====== ============================================================================
   dims   meaning
   ====== ============================================================================
   0:7    left EEF pose target: x, y, z + quaternion, robot root (base) frame, absolute
   7:14   right EEF pose target: same
   14     left gripper: BinaryJointPositionAction, value < 0 closes, otherwise opens
   15     right gripper: same
   16:19  base velocity vx, vy, wz: body frame, m/s, m/s, rad/s (GalbotBaseVelocityAction)
   ====== ============================================================================

   Quaternion component order inside 3:7 / 10:14 must be verified against the pinned Isaac Lab
   (upstream names them quat_x, quat_y, quat_z, quat_w). See ``schema.yaml: action_quat_order``.

   The env also accepts the legacy 20-D form of the demos recorded before the velocity interface:
   dims 16:20 hold wheel1..wheel4 values, which the env projects to ``[vx, vy, wz]``
   (``BASE_VEL_FROM_WHEELS``). The env smooths the 19-D form (first order, alpha 0.1 per control
   step) and applies the 20-D form as is, because recorded wheel values are already smoothed.
   An env revision without the velocity interface takes only 20 values, and there the wheel values
   barely move the base. ``usr_args["actions_dim"]`` / ``["accepted_action_dims"]`` say what the
   env takes.

2. **Model action (23-D)**, shared by ACT and GR00T (rotation as rot6d to avoid quaternion sign flips):
   ``[left_eef xyz+rot6d 9 | right_eef 9 | gripper L,R 2 | base_vel vx,vy,wz 3]``.

3. **State (34-D)**, built by the observation adapter:
   ``[left_eef_9d 9 | right_eef_9d 9 | left_arm_q 7 | right_arm_q 7 | gripper 2]``.
   EEF poses are in the robot base frame (computed by forward kinematics), never the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np

# ----------------------------------------------------------------------------- timing
CONTROL_HZ: float = 50.0  # sim.dt = 1/200, decimation 4; asserted against usr_args["decimation"]
DATA_FPS: float = 30.0  # time base of the demonstration data and therefore of every action chunk
EPISODE_LIMIT_S: float = 900.0  # ChamberServiceTask.episode_length_s (15 min of sim time)

# ----------------------------------------------------------------------------- sim action (19)
SIM_ACTION_DIM: int = 19
SIM_LEFT_EEF = slice(0, 7)
SIM_RIGHT_EEF = slice(7, 14)
SIM_LEFT_GRIPPER: int = 14
SIM_RIGHT_GRIPPER: int = 15
SIM_BASE_VEL = slice(16, 19)  # vx, vy, wz in the body frame: m/s, m/s, rad/s
SIM_ACTION_DIM_LEGACY: int = 20  # recorded demos and env revisions without the velocity interface
SIM_WHEELS_LEGACY = slice(16, 20)  # wheel1..wheel4 values
# The env's own mapping: wheels = WHEELS_FROM_BASE_VEL @ [vx, vy, wz], and back.
WHEELS_FROM_BASE_VEL = np.array([[1, -1, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1]], dtype=np.float32)
BASE_VEL_FROM_WHEELS = np.array([[1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]], dtype=np.float32) / 4.0
GRIPPER_CLOSE: float = -1.0  # Isaac Lab BinaryJointPositionAction: close when action < 0
GRIPPER_OPEN: float = 1.0
GRIPPER_CLOSED_RAD: float = 1.703  # close_command_expr of left/right_gripper_joint

# ----------------------------------------------------------------------------- model action (23)
MODEL_ACTION_DIM: int = 23
MODEL_LEFT_EEF = slice(0, 9)
MODEL_RIGHT_EEF = slice(9, 18)
MODEL_GRIPPER = slice(18, 20)
MODEL_BASE_VEL = slice(20, 23)

# ----------------------------------------------------------------------------- state (34)
STATE_DIM: int = 34
STATE_LEFT_EEF = slice(0, 9)
STATE_RIGHT_EEF = slice(9, 18)
STATE_LEFT_ARM_Q = slice(18, 25)
STATE_RIGHT_ARM_Q = slice(25, 32)
STATE_GRIPPER = slice(32, 34)

# ----------------------------------------------------------------------------- chamber lid
# The lid is a hinged revolute joint on the chamber articulation (``Lid001_joint``, range -90°..0°),
# not a free object. The sim normalizes its angle to [0, 1] (0 = closed, 1 = open, 90°).
# Its state is NOT in the policy observation; Astra estimates it (kinematics while holding, vision).
LID_RANGE_RAD: float = float(np.pi / 2)  # full swing, 90°
LID_OPEN_SIM_TH: float = 0.4  # stage 1 sim condition: normalized angle >= 0.4 (≈ 36°), latched
LID_CLOSED_SIM_TH: float = 0.005  # stage 6 sim condition: normalized angle <= 0.005 (≈ 0.45°)

# Camera names used inside Astra → observation keys in the env (see schema.yaml).
CAMERA_NAMES: tuple[str, ...] = ("head", "left_wrist", "right_wrist")


class Stage(IntEnum):
    """Orchestrator stages; order = ChamberServiceTask.success_stages. DONE = all six finished."""

    LIFT_LID = 0
    EXTRACT_PART = 1
    PACK_PART = 2
    WIPE = 3
    PLACE_REPLACEMENT = 4
    CLOSE_LID = 5
    DONE = 6


class Phase(IntEnum):
    """Sub-phase inside a stage, derived from gripper events and predicted by the stage classifier."""

    APPROACH = 0
    GRASP = 1
    TRANSPORT = 2
    PLACE = 3
    RETRACT = 4


class PolicySource(str, Enum):
    """Which producer generated an action chunk (tag carried from layer 4 into layer 5)."""

    ACT = "act"
    GROOT = "groot"
    SCRIPT = "script"  # scripted wiping primitive
    HOLD = "hold"  # safety hold: repeat the current pose, grippers unchanged, base velocity zero


class OrchestratorEvent(str, Enum):
    """Event emitted by the orchestrator on a tick; consumed by the router and the chunk executor."""

    NONE = "none"
    ADVANCE = "advance"  # new stage, grippers free: flush queues, maybe switch model
    ADVANCE_KEEP_QUEUE = "advance_keep_queue"  # new stage while carrying (2→3): keep queue, blend
    RETRY = "retry"  # same stage again: flush queues, optional move to pre-grasp pose
    ENTER_SCRIPT = "enter_script"  # hand control to the scripted wiping primitive
    EXIT_SCRIPT = "exit_script"  # hand control back to the learned policy


@dataclass
class CanonicalObs:
    """Validated, model-agnostic observation for one control tick (output of layer 2).

    Attributes:
        tick: control-step counter since the last reset (0-based).
        t: sim time in seconds since reset (``tick / CONTROL_HZ``).
        images: camera name (``CAMERA_NAMES``) → RGB image, HWC uint8, native resolution
            (head 480×640, wrists 224×224).
        state: (34,) float32 state vector, layout ``STATE_*``.
        joint_pos_rel: (77,) raw ``joint_pos`` (position minus default) as received, for detectors/FK.
        joint_vel_rel: (77,) raw ``joint_vel``.
        gripper_raw: (2,) measured left/right gripper joint position in rad (0 open, 1.703 closed).
        eef_base_pos: (2, 3) left/right EEF position in the base frame (from FK).
        eef_base_rotmat: (2, 3, 3) left/right EEF rotation matrix in the base frame (from FK).
        last_action: (19,) the sim action sent on the previous tick (from obs["actions"]); (20,) on
            an env revision without the velocity interface.
        valid: False if validation failed and the observation was replaced by the last valid one.
        issues: human-readable validation messages for logging.
    """

    tick: int
    t: float
    images: dict[str, np.ndarray]
    state: np.ndarray
    joint_pos_rel: np.ndarray
    joint_vel_rel: np.ndarray
    gripper_raw: np.ndarray
    eef_base_pos: np.ndarray
    eef_base_rotmat: np.ndarray
    last_action: np.ndarray
    valid: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass
class ActView:
    """Inputs for the ACT policy (LeRobot format, before the checkpoint's own preprocessor).

    Attributes:
        images: LeRobot feature key (e.g. ``observation.images.first_person_camera_rgb``) →
            float32 CHW in [0, 1], 224×224. No ImageNet normalisation here.
        state: (34,) float32, raw units; the LeRobot preprocessor applies mean/std.
        stage_one_hot: (6,) float32 subtask conditioning (all zeros when stage conditioning is off).
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    stage_one_hot: np.ndarray


@dataclass
class Gr00tView:
    """Inputs for GR00T N1.7 in the ``Gr00tPolicy.get_action`` observation format.

    Attributes:
        video: ``{"head": (1,1,480,640,3), "left_wrist": (1,1,224,224,3), ...}`` uint8 (B=1, T=1).
        state: ``{"left_eef_9d": (1,1,9), "right_eef_9d": (1,1,9), "left_arm_q": (1,1,7),
            "right_arm_q": (1,1,7), "gripper": (1,1,2)}`` float32 raw units.
        language: ``{"task": [[prompt]]}``; the language key must match galbot_config.py.
    """

    video: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    language: dict[str, list[list[str]]]


@dataclass
class EventState:
    """Proprioceptive events for one tick (output of the event detectors in layer 3).

    Attributes:
        holding: (left, right) True if the gripper is commanded closed but stopped short of fully
            closed, i.e. an object is between the fingers.
        grasp: (left, right) rising edge of ``holding`` on this tick (debounced).
        release: (left, right) gripper re-opened after holding on this tick.
        drop: (left, right) ``holding`` fell while still commanded closed (object slipped).
        in_contact: True if the contact proxy (EEF tracking error while holding the sponge) is active.
        wipe_count: number of qualified wipe contacts (dwell ≥ threshold, separated by breaks)
            since the WIPE stage started.
        region: name of the workspace region the active EEF is in ("chamber", "empty_box",
            "full_box", "table", "lid_handle", "none"), from base-frame EEF position.
        lid_tracking: True while an arm holds the lid handle and the kinematic lid estimate is live.
        lid_angle_kin: normalized lid angle (0 closed … 1 open) estimated from the rotation of the
            holding hand since the handle grasp; None when not tracking.
        lid_angle_kin_peak: largest ``lid_angle_kin`` seen in the current stage (survives release).
    """

    holding: tuple[bool, bool] = (False, False)
    grasp: tuple[bool, bool] = (False, False)
    release: tuple[bool, bool] = (False, False)
    drop: tuple[bool, bool] = (False, False)
    in_contact: bool = False
    wipe_count: int = 0
    region: str = "none"
    lid_tracking: bool = False
    lid_angle_kin: float | None = None
    lid_angle_kin_peak: float = 0.0


@dataclass
class StagePrediction:
    """Output of the learned stage classifier for one tick.

    Attributes:
        stage_probs: (7,) probabilities over Stage 0..5 and DONE.
        phase_probs: (5,) probabilities over Phase.
        progress: estimated progress inside the current stage, 0..1.
        scene_ok: probability that the object state matches the stage goal (e.g. part in box).
        lid_open: regressed normalized lid angle from the cameras (0 closed … 1 open); the only lid
            signal once the hand has released the handle. Trained on the lid joint value per frame.
    """

    stage_probs: np.ndarray
    phase_probs: np.ndarray
    progress: float
    scene_ok: float
    lid_open: float = 0.0


@dataclass
class OrchestratorOutput:
    """Decision of the task orchestrator for one tick (output of layer 3).

    Attributes:
        stage: current stage.
        phase: current most-likely phase.
        prompt: fixed language prompt for the stage (GR00T ``language.task``).
        event: what changed on this tick (see ``OrchestratorEvent``).
        budget_left_s: remaining time budget of the current stage in seconds.
        retries: number of retries already used in this stage.
    """

    stage: Stage
    phase: Phase
    prompt: str
    event: OrchestratorEvent
    budget_left_s: float
    retries: int = 0


@dataclass
class ActionChunk:
    """A predicted sequence of model-space actions (output of layer 4).

    Attributes:
        actions: (T, 23) float32 in the model action layout, absolute targets, physical units,
            gripper dims in [0, 1] (1 = closed) or logits converted to probabilities.
        source: which producer generated it.
        fps: time base of the chunk (DATA_FPS for learned policies, CONTROL_HZ for scripts).
        start_tick: control tick of the observation the chunk was predicted from.
        stage: stage the chunk was produced for (used to drop stale chunks after a switch).
    """

    actions: np.ndarray
    source: PolicySource
    fps: float
    start_tick: int
    stage: Stage


@dataclass
class SimAction:
    """The 19-D action sent to the env on one tick (output of layer 5, input of layer 6).

    Attributes:
        vector: (19,) float32 in the sim action layout ``SIM_*``, or (20,) in the legacy wheel form.
        tick: control tick it is meant for.
        source: producer of the chunk this action came from (for logging).
        clamped: True if the safety filter modified the action.
    """

    vector: np.ndarray
    tick: int
    source: PolicySource
    clamped: bool = False
