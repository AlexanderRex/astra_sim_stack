"""Layer 3 · Task orchestrator.

Decides, every tick, which of the six ChamberServiceTask stages the robot is in, whether the current
stage is finished (work done **and** grippers free), and whether to retry. It never moves the robot.

Key facts from the sim scorer (``ChamberServiceTask._check_success``) that shape the design:
- Stage flags are chained: stage k only counts if stage k-1 already counted. Skipping a stage
  forfeits every later stage, so the only recovery is retry.
- The sim credits some stages before the robot is ready to move on (stage 1 fires when the hinged
  lid passes 36° while the hand still holds it), so the orchestrator uses its own stage boundaries.

Modules:
    stage_specs        per-stage prompt, budget, completion predicate, recovery rules
    event_detectors    holding / grasp / release / drop / contact / wipe counter / region (rules)
    stage_classifier   learned stage + phase + progress classifier (inference wrapper)
    task_orchestrator  state machine fusing classifier and detectors
"""
