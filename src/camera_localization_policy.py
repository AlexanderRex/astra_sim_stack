"""Locate the workbench and request another RGB viewpoint when targets are occluded."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
from policy.base import BasePolicy

from astra.perception.galbot import read_sensors
from astra.perception.localization import WorkbenchLocalizer
from astra.perception.view_search import TARGETS, ViewSearch


class CameraLocalizationPolicy(BasePolicy):
    def get_model(self, config):
        self.localizer = WorkbenchLocalizer(table_height_m=config["camera_localization"]["table_height_m"])
        return None

    def reset_model(self):
        pass

    def get_action(self):
        raise RuntimeError("CameraLocalizationPolicy runs through eval")

    def eval(self, env, observation, config, video_writer):
        if config["actions_dim"] != 19:
            raise ValueError("The camera scan requires Galbot-Mobile's native 19-value action")
        settings = config["camera_localization"]
        steps = int(settings["scan_steps"])
        period = int(settings["sample_every"])
        yaw_rate = float(settings["yaw_rate_rad_s"])
        if steps < 1 or not 1 <= period <= steps or not 0 < yaw_rate <= .7:
            raise ValueError("Invalid camera scan settings")
        _, encoders = read_sensors(observation)
        arm_hold = self.localizer.calibration.arm_poses(encoders)
        action = np.r_[arm_hold.ravel(), -1., -1., 0., 0., 0.][None].astype(np.float32)
        search = ViewSearch(yaw_rate, period)
        output = Path(config["save_path"])
        output.mkdir(parents=True, exist_ok=True)
        samples = []
        best = {}
        found_together = False
        confirming = False
        finished = False

        def advance():
            nonlocal observation, finished
            transition = env.step(action)
            observation = transition[0]
            term, trunc = transition[1:3] if len(transition) == 3 else transition[2:4]
            finished = bool(np.asarray(term).any() or np.asarray(trunc).any())

        # Allow all three RGB sensors to refresh at the new spawn before motion.
        for _ in range(8):
            advance()
            if finished:
                break
        for tick in range(steps + 1):
            if finished:
                break
            if tick % period == 0:
                images, encoders = read_sensors(observation)
                hits = self.localizer.locate(images, encoders)
                all_detected = TARGETS.issubset(hits)
                command = search.velocity(images, hits)
                samples.append({"tick": tick, "detections": {key: asdict(value) for key, value in hits.items()},
                                "base_velocity_command": command.tolist()})
                for name, hit in hits.items():
                    if (all_detected and confirming) or name not in best or hit.score > best[name][1].score:
                        best[name] = (tick, hit)
                        bgr = cv2.cvtColor(images[hit.camera], cv2.COLOR_RGB2BGR)
                        outline = np.round(hit.outline_px).astype(np.int32)
                        cv2.polylines(bgr, [outline], True, (0, 60, 220), 2, cv2.LINE_AA)
                        cv2.imwrite(str(output / f"{name}.png"), bgr)
                if all_detected and confirming:
                    found_together = True
                    break
                confirming = all_detected
                action[0, 16:] = command
            if tick == steps:
                break
            advance()
        if not finished:
            action[0, 16:] = 0.
            advance()
        report = {
            "coordinate_frame": "base_link at each sample",
            "sources": ["RGB cameras", "absolute joint encoders", "calibration"],
            "samples": samples,
            "all_targets_same_sample": found_together,
            "best": {name: {"tick": tick, **asdict(hit)} for name, (tick, hit) in best.items()},
        }
        (output / "localization.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name, (tick, hit) in best.items():
            print(f"{name}: {tuple(round(x, 3) for x in hit.base_xyz_m)} m from {hit.camera} at tick {tick}")
        return found_together
