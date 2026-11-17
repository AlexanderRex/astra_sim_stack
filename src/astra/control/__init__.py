"""Layer 6 · Low-level controllers (boundary to the sim).

The low-level controllers themselves run inside the env server as Isaac Lab action terms
(``galbot.py`` ``MobileActionsCfg``); they are not part of the submission. This package is the
policy-side boundary: it packs the 20-D action tensor, steps the env and reports what came back.

Modules:
    sim_interface  SimActionInterface: tensor packing, task_env.step(), termination handling
"""
