"""Runs a policy from src/ against the evaluation service: python grpc_runner.py CONFIG.yml

The service speaks gRPC: it gives observations and takes actions, has no reset call and exchanges
no metadata. This wraps it as the env the policies here are written for (policy/base.py in the
task repo): step(action) returns (observation, terminated, truncated), and
observation["policy"][name] is a tensor of shape (num_envs, ...).
"""

import importlib
import os
import sys

import grpc
import numpy as np
import torch
import yaml
from robofinals.distributed import policy_v1_pb2 as pb
from robofinals.distributed.chamber_schema import OBSERVATION_SCHEMA
from robofinals.distributed.grpc_policy import PolicyClient

TIMEOUT_S = 300 * 60  # the service's own wall-clock limit for one episode


class GrpcEnv:
    def __init__(self, endpoint):
        channel = grpc.insecure_channel(endpoint, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
        grpc.channel_ready_future(channel).result(timeout=TIMEOUT_S)
        self._client = PolicyClient(channel)
        self.observation, _, _ = self._unpack(self._client.get_observation(pb.Empty(), timeout=TIMEOUT_S))

    @staticmethod
    def _unpack(transition):
        policy = {
            name: torch.from_numpy(
                np.stack([np.frombuffer(getattr(o, name), dtype=dtype).reshape(shape) for o in transition.observations])
            )
            for name, dtype, shape in OBSERVATION_SCHEMA
        }
        # The second group is what BasePolicy.encode_obs merges in; the service has only one.
        observation = {"policy": policy, "embodiment_general_obs": {}}
        return observation, torch.tensor(list(transition.terminated)), torch.tensor(list(transition.truncated))

    def step(self, action):
        rows = torch.as_tensor(action).detach().cpu().numpy().astype(np.float32)
        try:
            transition = self._client.step(pb.ActionBatch(actions=[pb.Action(values=row) for row in rows]), timeout=TIMEOUT_S)
        except grpc.RpcError as error:
            # The service reset an episode on its wall-clock limit and wants that read first.
            if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise
            transition = self._client.get_observation(pb.Empty(), timeout=TIMEOUT_S)
        self.observation, terminated, truncated = self._unpack(transition)
        return self.observation, terminated, truncated


def main(config_path):
    with open(config_path) as f:
        usr_args = yaml.safe_load(f)
    module, _, name = usr_args["policy_name"].partition(":")
    policy = getattr(importlib.import_module(module), name)(usr_args)
    env = GrpcEnv(os.environ.get("ECM27_EVAL_ENDPOINT", "127.0.0.1:50000"))
    num_envs = len(env.observation["policy"]["joint_pos"])
    # Fixed by the interface: 19 values per action (20 with legacy wheels), 0.02 s per step.
    usr_args.update(actions_dim=19, accepted_action_dims=[19, 20], decimation=4, env_cfg={"num_envs": num_envs})
    # shortcut: eval() runs one episode at a time, so with several envs it ends when the first one
    # does. Upgrade when the evaluator's env count is known to be above one.
    episodes = usr_args.get("test_num")  # absent: run until the evaluator stops the container
    done = 0
    while episodes is None or done < episodes:
        policy.reset_model()
        policy.eval(env, env.observation, usr_args, None)
        done += 1


if __name__ == "__main__":
    main(sys.argv[1])
