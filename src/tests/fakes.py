"""Fake env, observations and a tiny URDF for the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS

from astra.adapter.rotations import matrix_to_quat, quat_to_matrix

N_JOINTS = 77
LEFT_ARM = [50, 53, 55, 57, 59, 61, 63]
RIGHT_ARM = [51, 54, 56, 58, 60, 62, 64]
GRIPPERS = [65, 69]


def make_raw_obs(
    rng: np.random.Generator,
    joint_pos: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    blank: bool = False,
    image_dtype: str = "uint8",
) -> dict:
    """Raw env observation as the evaluation service returns it, tensors with num_envs = 1.

    ``joint_pos`` is the absolute pose; it is sent relative to ``JOINT_DEFAULT_POS``.
    """
    jp = joint_pos if joint_pos is not None else rng.normal(scale=0.3, size=N_JOINTS)
    act = actions if actions is not None else np.zeros(19)
    eef_pos = np.array([[0.5, 0.3, 1.0], [0.5, -0.3, 1.0]])
    eef_quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

    def img(h, w):
        if blank:
            a = np.zeros((h, w, 3), np.uint8)
        else:
            a = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        if image_dtype == "float":
            return torch.from_numpy(a.astype(np.float32) / 255.0)[None]
        return torch.from_numpy(a)[None]

    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))[None]  # noqa: E731
    return {
        "policy": {
            "actions": t(act),
            "joint_pos": t(jp - np.asarray(JOINT_DEFAULT_POS)),  # relative to the default pose
            "joint_vel": t(np.zeros(N_JOINTS)),
            "eef_pos": t(eef_pos),
            "eef_quat": t(eef_quat),
            "gripper_pos": t(np.zeros(2)),
            "first_person_camera_rgb": img(480, 640),
            "left_hand_camera_rgb": img(224, 224),
            "right_hand_camera_rgb": img(224, 224),
        },
        "embodiment_general_obs": {},  # empty, as grpc_runner sends it
    }


class FakeEnv:
    """Steps synchronously; reports the last action back; ends after ``max_steps``.

    The arm "tracks" commands: the last commanded pose is echoed in ``actions``. Grippers move to the
    commanded side (closed 1.703, open 0) over a few ticks.
    """

    def __init__(self, rng: np.random.Generator, max_steps: int = 200, three_tuple: bool = True):
        self.rng = rng
        self.max_steps = max_steps
        self.three_tuple = three_tuple
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.jp = rng.normal(scale=0.3, size=N_JOINTS)
        self.jp[GRIPPERS] = 0.0

    def first_obs(self) -> dict:
        return make_raw_obs(self.rng, self.jp.copy())

    def step(self, action):
        a = action.detach().cpu().numpy()
        assert a.ndim == 2 and a.shape[0] == 1, a.shape
        self.actions.append(a[0].copy())
        for i, col in enumerate((14, 15)):
            target = 1.703 if a[0, col] < 0 else 0.0
            self.jp[GRIPPERS[i]] += np.clip(target - self.jp[GRIPPERS[i]], -0.3, 0.3)
        self.steps += 1
        obs = make_raw_obs(self.rng, self.jp.copy(), actions=a[0])
        done = self.steps >= self.max_steps
        term = torch.tensor([done])
        trunc = torch.tensor([False])
        if self.three_tuple:
            return obs, term, trunc
        return obs, torch.zeros(1), term, trunc, {}


def write_test_urdf(path: Path) -> Path:
    """Torso joint (leg_joint1) then two 7-joint arms; link offsets of 0.1 m along z / x."""
    def arm(side: str, y: float) -> str:
        out = []
        parent = "torso_link"
        for k in range(1, 8):
            child = f"{side}_arm_link{k}"
            axis = "0 0 1" if k % 2 else "0 1 0"
            xyz = f"0 {y} 0.1" if k == 1 else "0.1 0 0"
            out.append(
                f'<joint name="{side}_arm_joint{k}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                f'<origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="{axis}"/><limit lower="-3" upper="3"/></joint>'
                f'<link name="{child}"/>'
            )
            parent = child
        return "".join(out)

    urdf = (
        '<robot name="test"><link name="base_link"/><link name="torso_link"/>'
        '<joint name="leg_joint1" type="revolute"><parent link="base_link"/><child link="torso_link"/>'
        '<origin xyz="0 0 0.8" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>'
        + arm("left", 0.2)
        + arm("right", -0.2)
        + "</robot>"
    )
    path.write_text(urdf)
    return path


def ref_fk(jp: np.ndarray, side: str) -> np.ndarray:
    """Hand-written FK of ``write_test_urdf`` for comparison."""
    def rot(axis, q):
        x, y, z = axis
        c, s, C = np.cos(q), np.sin(q), 1 - np.cos(q)
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                         [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                         [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])

    def T(R=np.eye(3), p=(0, 0, 0)):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p
        return M

    ids = LEFT_ARM if side == "left" else RIGHT_ARM
    y = 0.2 if side == "left" else -0.2
    M = T(p=(0, 0, 0.8)) @ T(rot((0, 1, 0), jp[0]))
    for k in range(7):
        M = M @ T(p=(0, y, 0.1) if k == 0 else (0.1, 0, 0))
        M = M @ T(rot((0, 0, 1) if (k + 1) % 2 else (0, 1, 0), jp[ids[k]]))
    return M


__all__ = ["FakeEnv", "make_raw_obs", "write_test_urdf", "ref_fk", "quat_to_matrix", "matrix_to_quat"]
