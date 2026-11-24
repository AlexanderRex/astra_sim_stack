"""Search viewpoints with RGB detections and body-frame velocity commands."""

import math

import cv2
import numpy as np

from .localization import TABLE_SIZE, _axes, _table_masks


TARGETS = frozenset(("table", "chamber", "sponge"))


class ViewSearch:
    def __init__(self, yaw_rate, sample_every):
        self.yaw_rate = yaw_rate
        self.sample_every = sample_every
        self.missing_steps = 0
        self.slide_sign = None

    def velocity(self, images, detections):
        if TARGETS.issubset(detections):
            return np.zeros(3)
        table = detections.get("table")
        if table is None:
            step = self.missing_steps
            self.missing_steps += self.sample_every
            hint = self._table_image_hint(images.get("head"))
            if hint:
                yaw = hint * self.yaw_rate
            else:
                # Sweep one side, then the other. Keep the base position fixed.
                yaw = self.yaw_rate if step % 356 < 90 or step % 356 >= 267 else -self.yaw_rate
            return np.array([0., 0., yaw])
        self.missing_steps = 0
        center = np.asarray(table.base_xyz_m[:2])
        heading = math.atan2(center[1], center[0])
        yaw = float(np.clip(1.65 * heading, -self.yaw_rate, self.yaw_rate))
        if abs(heading) > .34 or table.score < .24:
            return np.array([0., 0., yaw])

        axes = _axes(table.yaw_rad)
        side_from_center = -(axes @ center)
        gap = np.abs(side_from_center) - np.asarray(TABLE_SIZE) / 2
        side = int(np.argmax(gap))
        outward = math.copysign(1., side_from_center[side]) * axes[side]
        if gap[side] <= .55:
            # Establish clearance before translating parallel to the table.
            return np.r_[.20 * outward, yaw]
        tangent = np.array([outward[1], -outward[0]])
        if self.slide_sign is None:
            chamber = detections.get("chamber")
            if chamber is not None:
                occupied = np.asarray(chamber.base_xyz_m[:2]) - center
                self.slide_sign = -math.copysign(1., float(occupied @ tangent))
            else:
                self.slide_sign = 1.
        return np.r_[.36 * self.slide_sign * tangent, yaw]

    @staticmethod
    def _table_image_hint(rgb):
        if rgb is None:
            return 0.
        height, width = rgb.shape[:2]
        # Ignore the upper background and the lower corners occupied by Galbot's arms.
        bright, _ = _table_masks(rgb[height * 3 // 10:height * 7 // 10])
        mask = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            return 0.
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x = float(centroids[index, 0])
        if stats[index, cv2.CC_STAT_AREA] < 440 or abs(x - width / 2) < .12 * width:
            return 0.
        return math.copysign(1., width / 2 - x)
