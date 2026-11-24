"""Export joint targets and planar base commands from a successful Stage 1 capture."""

import argparse
import json
import math
from pathlib import Path


def yaw(quat):
    x, y, z, w = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Directory containing the score and capture JSON")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.episode / "stage1_lid_push.json").read_text())
    samples = json.loads((args.episode / "manipulation_record.json").read_text())
    if report["score"] != 5 or not report["stayed_open"]:
        parser.error("The reference must have passed scoring and the open-lid observation")
    if report["steps"] != len(samples) or not samples:
        parser.error("The reference capture is incomplete")

    fixture = report["chamber_pose"]
    angle = yaw(fixture[3:])
    cos, sin = math.cos(angle), math.sin(angle)

    def fixture_base(sample):
        base = sample["base"]
        dx, dy = base[0] - fixture[0], base[1] - fixture[1]
        return [cos * dx + sin * dy, -sin * dx + cos * dy,
                math.remainder(yaw(base[3:]) - angle, math.tau)]

    def recorded_row(sample):
        return [*fixture_base(sample), *sample["base_velocity"], *sample["grippers"],
                *sample["joint_targets"]]

    approach_frame = next((i for i, sample in enumerate(samples) if sample["phase"] == "align_base"), None)
    push_frame = next((i for i, sample in enumerate(samples) if sample["phase"] == "push"), None)
    retract_frame = next((i for i, sample in enumerate(samples) if sample["phase"] == "retract"), None)
    observe_steps = 150
    if (None in (approach_frame, push_frame, retract_frame)
            or not 0 < approach_frame < push_frame < retract_frame < len(samples) - observe_steps
            or any(sample["phase"] != "observe" for sample in samples[-observe_steps:])):
        parser.error("The capture must include approach, push, retract, and three seconds of observation")
    data = {
        "source_run": args.episode.parent.name,
        "dt": .02,
        "approach_frame": approach_frame,
        "push_frame": push_frame,
        "retract_frame": retract_frame,
        "initial_joints": [round(value, 7) for value in samples[0]["joints"]],
        "observe_steps": observe_steps,
    }
    metadata = json.dumps(data, indent=2, allow_nan=False)
    serialized_rows = ",\n".join(
        "    " + json.dumps([round(value, 7) for value in recorded_row(sample)], allow_nan=False)
        for sample in samples
    )
    args.output.write_text(metadata[:-2] + ',\n  "frames": [\n' + serialized_rows + "\n  ]\n}\n")


if __name__ == "__main__":
    main()
