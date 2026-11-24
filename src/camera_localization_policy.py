"""Run the RGB-only workbench detector during a short in-place Galbot scan."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np
from policy.base import BasePolicy

from astra.perception.galbot import read_sensors
from astra.perception.localization import WorkbenchLocalizer


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
        action = np.r_[arm_hold.ravel(), -1., -1., 0., 0., yaw_rate][None].astype(np.float32)
        output = Path(config["save_path"])
        output.mkdir(parents=True, exist_ok=True)
        samples = []
        best = {}
        for tick in range(steps + 1):
            if tick % period == 0:
                images, encoders = read_sensors(observation)
                hits = self.localizer.locate(images, encoders)
                samples.append({"tick": tick, "detections": {key: asdict(value) for key, value in hits.items()}})
                for name, hit in hits.items():
                    if name not in best or hit.score > best[name][1].score:
                        best[name] = (tick, hit)
                        bgr = cv2.cvtColor(images[hit.camera], cv2.COLOR_RGB2BGR)
                        outline = np.round(hit.outline_px).astype(np.int32)
                        cv2.polylines(bgr, [outline], True, (0, 60, 220), 2, cv2.LINE_AA)
                        cv2.imwrite(str(output / f"{name}.png"), bgr)
            if tick == steps:
                break
            transition = env.step(action)
            observation = transition[0]
            terminated, truncated = transition[1:3] if len(transition) == 3 else transition[2:4]
            if np.asarray(terminated).any() or np.asarray(truncated).any():
                break
        report = {
            "coordinate_frame": "base_link at each sample",
            "sources": ["RGB cameras", "absolute joint encoders", "calibration"],
            "samples": samples,
            "best": {name: {"tick": tick, **asdict(hit)} for name, (tick, hit) in best.items()},
        }
        (output / "localization.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name, (tick, hit) in best.items():
            print(f"{name}: {tuple(round(x, 3) for x in hit.base_xyz_m)} m from {hit.camera} at tick {tick}")
        return all(name in best for name in ("table", "chamber"))
