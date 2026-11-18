"""Astra: runtime policy stack for the SGEAI ChamberServiceTask (Galbot-Mobile, Isaac Lab sim).

This package implements the runtime loop of the ACT / GR00T N1.7 switchable system, layers 2 to 6 of
``src/astra/diagrams/make_diagram.py``. Every control tick (50 Hz) the data flows as follows:

    env server (layer 1, provided)
        │  raw observation dict (joint_pos, eef pose, cameras, ...)
        ▼
    astra.adapter          layer 2  Observation adapter: raw obs → canonical obs → model views
        ▼
    astra.orchestrator     layer 3  Task orchestrator: which of the 6 stages, is it done, retry?
        ▼
    astra.policies         layer 4  Router + policies (ACT, GR00T N1.7, scripted wipe)
        ▼                           → action chunk in the 24-D model action space
    astra.postprocess      layer 5  Chunk executor + safety + codec → one 20-D sim action per tick
        ▼
    astra.control          layer 6  Sim action interface: packs the 20-D tensor, calls task_env.step();
                                    the actual low-level controllers (DiffIK, binary gripper, wheel
                                    velocity) run inside the env server.

``astra.policy.AstraPolicy`` is the entry point loaded by the task repo's evaluation harness
(``policy_name: astra.policy:AstraPolicy`` in ``astra/config/astra_eval.yml``). It owns all
subsystems and runs the loop inside ``BasePolicy.eval()``.

Status: skeleton. Classes initialise their state; behaviour methods raise ``NotImplementedError``
and document what they must do. The offline training pipeline is not part of this package yet.
"""

__version__ = "0.1.0"
