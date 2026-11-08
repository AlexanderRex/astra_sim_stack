"""One-step policy used to smoke-test the local simulator IPC container path."""

import numpy as np
import torch

from policy.base import BasePolicy


class SmokePolicy(BasePolicy):
    def get_model(self, usr_args):
        return None

    def get_action(self):
        return None

    def reset_model(self):
        pass

    def eval(self, task_env, observation, usr_args, video_writer):
        del observation, video_writer
        action = torch.zeros(
            (int(usr_args.get("env_cfg", {}).get("num_envs", 1)), int(usr_args["actions_dim"])),
            dtype=torch.float32,
            device="cpu",
        )
        _, _, terminated, truncated, _ = task_env.step(action)
        ended = np.asarray(terminated).any() or np.asarray(truncated).any()
        return bool(ended)
