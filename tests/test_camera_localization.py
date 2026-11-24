import unittest

import cv2
import numpy as np

from astra.perception.galbot import GalbotCalibration, OBSERVATION_KEYS, read_sensors
from astra.perception.localization import Detection, WorkbenchLocalizer
from astra.perception.view_search import ViewSearch


def nominal_encoders():
    joints = np.zeros(77)
    joints[[0, 5, 46, 52]] = [.93, 2.50, 1.56, .35]
    joints[[50, 53, 55, 57]] = [1.134464, -1.308997, -.52359878, -1.57079633]
    joints[[51, 54, 56, 58]] = -joints[[50, 53, 55, 57]]
    return joints


class CameraLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.joints = nominal_encoders()
        self.calibration = GalbotCalibration()

    def test_galbot_encoder_calibration(self):
        actual = self.calibration.arm_poses(self.joints)
        expected = [[.349326, .299017, .988982], [.349327, -.299017, .988982]]
        np.testing.assert_allclose(actual[:, :3], expected, atol=2e-6)
        cameras = self.calibration.cameras(self.joints)
        self.assertEqual(set(cameras), set(OBSERVATION_KEYS))
        baselines = cameras["left_wrist"].base_from_optical[:3, 3] - cameras["right_wrist"].base_from_optical[:3, 3]
        self.assertGreater(np.linalg.norm(baselines), .55)
        self.assertLess(np.linalg.norm(baselines), .60)

    def test_project_and_intersect_table_plane(self):
        camera = self.calibration.cameras(self.joints)["head"]
        points = np.array([[.9, 0., .72], [1.5, -.3, .72], [2., .4, .72]])
        np.testing.assert_allclose(camera.pixels_on_plane(camera.project(points), .72), points, atol=1e-10)

    def test_metric_scene_and_missing_objects(self):
        camera = self.calibration.cameras(self.joints)["head"]
        rgb = np.full((camera.height, camera.width, 3), (62, 88, 120), dtype=np.uint8)
        localizer = WorkbenchLocalizer(self.calibration)
        self.assertEqual(localizer.locate({"head": rgb}, self.joints), {})

        def paint(points, color):
            pixels = np.rint(camera.project(points)).astype(np.int32)
            cv2.fillConvexPoly(rgb, cv2.convexHull(pixels), color)

        for color in ((191, 191, 191), (53, 59, 66)):
            with self.subTest(chamber_rgb=color):
                paint([[.82, -.6, .72], [2.62, -.6, .72], [2.62, .6, .72], [.82, .6, .72]], (241, 241, 241))
                paint([[x, y, z] for x in (1.75, 2.45) for y in (-.361, .361) for z in (.72, 1.134)], color)
                paint([[x, y, 1.134] for x in (1.75, 2.45) for y in (-.361, .361)], (191, 191, 191))
                paint([[x, y, .731] for x in (1.05, 1.14) for y in (.25, .295)], (95, 187, 178))
                objects = localizer.locate({"head": rgb}, self.joints)
                self.assertTrue({"table", "chamber", "sponge"}.issubset(objects))
                np.testing.assert_allclose(objects["table"].base_xyz_m, [1.72, 0., .72], atol=.06)
                np.testing.assert_allclose(objects["chamber"].base_xyz_m, [2.10, 0., .95174], atol=.08)
                np.testing.assert_allclose(objects["sponge"].base_xyz_m, [1.095, .2725, .731], atol=.025)

    def test_view_search_uses_visible_table_to_translate(self):
        search = ViewSearch(.65, 10)
        np.testing.assert_array_equal(search.velocity({}, {}), [0., 0., .65])
        table = Detection("table", (1.70, 0., .72), 0., np.pi, .6, "head", [])
        chamber = Detection("chamber", (1.17, -.30, .952), 0., np.pi / 2, .7, "head", [])
        np.testing.assert_allclose(search.velocity({}, {"table": table, "chamber": chamber}), [0., .36, 0.])
        near_table = Detection("table", (1.30, 0., .72), 0., np.pi, .6, "head", [])
        np.testing.assert_allclose(search.velocity({}, {"table": near_table}), [-.2, 0., 0.])
        np.testing.assert_array_equal(search.velocity({}, {}), [0., 0., .65])

    def test_sensor_adapter_ignores_simulator_poses(self):
        image = np.zeros((1, 480, 640, 3), np.uint8)
        images, encoders = read_sensors({
            "policy": {OBSERVATION_KEYS["head"]: image, "joint_pos": None,
                       "lid_push_state": object(), "base_drive_state": object(), "eef_pos": object()},
            "embodiment_general_obs": {"joint_pos": self.joints[None]},
        })
        self.assertEqual(set(images), {"head"})
        np.testing.assert_array_equal(images["head"], image[0])
        np.testing.assert_array_equal(encoders, self.joints)
