"""Astra: runtime policy stack for the ChamberServiceTask (Galbot-Mobile, Isaac Lab sim).

This package implements the runtime loop of the ACT / GR00T N1.7 switchable system, layers 2 to 6 of
``src/astra/diagrams/make_policy_diagram.py``. Every control tick (50 Hz) the data flows as follows:

    env server (layer 1, provided)
        │  raw observation dict (joint_pos, eef pose, cameras, ...)
        ▼
    astra.adapter          layer 2  Observation adapter: raw obs → canonical obs → model views
        ▼
    astra.orchestrator     layer 3  Task orchestrator: which of the 6 stages, is it done, retry?
        ▼
    astra.policies         layer 4  Router + policies (ACT, GR00T N1.7, scripted wipe)
        ▼                           → action chunk in the 23-D model action space
    astra.postprocess      layer 5  Chunk executor + safety + codec → one 19-D sim action per tick
        ▼
    astra.control          layer 6  Sim action interface: packs the action tensor, calls task_env.step();
                                    the actual low-level controllers (DiffIK, binary gripper, base
                                    velocity) run inside the env server.

``astra.policy.AstraPolicy`` is the entry point loaded by the task repo's evaluation harness
(``policy_name: astra.policy:AstraPolicy`` in ``astra/config/astra_eval.yml``). It owns all
subsystems and runs the loop inside ``BasePolicy.eval()``.

Status: implemented and unit-tested against a fake env (``src/tests``); not yet run against the sim
or with trained checkpoints. The offline training pipeline is not part of this package yet.
"""

__version__ = "0.1.0"
