"""Galbot camera extrinsics from calibrated links and absolute joint encoders."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from robofinals.distributed.chamber_schema import JOINT_DEFAULT_POS
from scipy.spatial.transform import Rotation


OBSERVATION_KEYS = {
    "head": "first_person_camera_rgb",
    "left_wrist": "left_hand_camera_rgb",
    "right_wrist": "right_hand_camera_rgb",
}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def transform(translation, quat_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    matrix[:3, 3] = translation
    return matrix


@dataclass(frozen=True)
class Camera:
    """Pinhole calibration. Transform maps optical (+X right, +Y down, +Z forward) into base."""

    name: str
    width: int
    height: int
    intrinsic: np.ndarray
    base_from_optical: np.ndarray
    distortion: np.ndarray | None = None

    def pixels_on_plane(self, pixels, z):
        """Intersect optical rays with a horizontal plane at base-frame height z."""
        uv = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(uv, self.intrinsic, self.distortion).reshape(-1, 2)
        rays = np.c_[normalized, np.ones(len(normalized))] @ self.base_from_optical[:3, :3].T
        origin = self.base_from_optical[:3, 3]
        vertical = np.where(np.abs(rays[:, 2]) > 1e-8, rays[:, 2], np.nan)
        distance = (z - origin[2]) / vertical
        return origin + rays * np.where(distance > 0, distance, np.nan)[:, None]

    def project(self, base_points):
        points = np.asarray(base_points, dtype=float).reshape(-1, 3)
        optical = (points - self.base_from_optical[:3, 3]) @ self.base_from_optical[:3, :3]
        pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3),
                                      self.intrinsic, self.distortion)
        pixels = pixels.reshape(-1, 2)
        pixels[optical[:, 2] <= 0.01] = np.nan
        return pixels


class GalbotCalibration:
    """Map the Galbot-Mobile encoder layout to optical calibration per view."""

    def __init__(self, path=None):
        calibration = Path(path) if path is not None else Path(__file__).with_name("galbot_calibration.json")
        data = json.loads(calibration.read_text())
        self.joint_count = data["joint_count"]
        self.joints = {
            name: (row[0], transform(row[1], row[2])) for name, row in data["joints"].items()
        }
        self.chains = data["chains"]
        self.optics = data["cameras"]

    def _joint_array(self, encoder_positions):
        joints = np.asarray(encoder_positions, dtype=float)
        if joints.shape != (self.joint_count,) or not np.isfinite(joints).all():
            raise ValueError(f"Expected {self.joint_count} finite absolute Galbot encoder angles")
        return joints

    def _chain_pose(self, joints, name):
        pose = np.eye(4)
        for joint in self.chains[name]:
            index, parent_from_joint = self.joints[joint]
            pose = pose @ parent_from_joint
            # These model joints all revolve around local Z with identity child anchor.
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([0., 0., joints[index]]).as_matrix()
        return pose

    def arm_poses(self, encoder_positions):
        joints = self._joint_array(encoder_positions)
        poses = [self._chain_pose(joints, name) for name in ("left_wrist", "right_wrist")]
        return np.asarray([np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
                           for pose in poses])

    def cameras(self, encoder_positions):
        joints = self._joint_array(encoder_positions)
        return {
            name: Camera(
                name, cfg["width"], cfg["height"], np.asarray(cfg["K"], dtype=float),
                self._chain_pose(joints, name) @ transform(*cfg["link_from_optical"]),
                np.asarray(cfg["distortion"], dtype=float) if "distortion" in cfg else None,
            )
            for name, cfg in self.optics.items()
        }


def read_sensors(observation):
    """Read only RGB and encoders from the stock Galbot policy observation; returns absolute joint angles."""
    policy = observation["policy"]
    images = {name: array(policy[key])[0] for name, key in OBSERVATION_KEYS.items() if key in policy}
    if not images:
        raise ValueError("Galbot observation does not contain RGB camera streams")
    joints = array(policy["joint_pos"])
    if joints.ndim != 2 or joints.shape[0] != 1:
        raise ValueError("Camera localization expects one robot's encoder vector")
    return images, joints[0] + JOINT_DEFAULT_POS  # the service sends joints relative to the default pose
