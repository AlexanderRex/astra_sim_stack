---
name: astra-evaluation
description: Compare ACT, GR00T, and routed Astra policies using held-out closed-loop trials while separating development progress, official milestones, and intervention accounting.
---

# Policy evaluation with explicit score provenance

Read the pinned task README/checker and the latest official rules relevant to the requested evaluation. Store the exact evaluator revision and configuration with every result.

At task commit `6e3a66c1344e926ef66b3a804058918d146cd2d0`, the README explicitly distinguishes development `process_score` in `[0,6]` from the intended official score in `[0,12]`. Its stages are sequentially latched and do not implement two milestones per subtask or manual-intervention accounting. Verify the active checkout before assuming this is still true. `BasePolicy.format_eval_results` defaults to per-episode success booleans. Do not multiply the development score by two or present it as an official result.

Use explicit labels: `dev_process_score_0_6`, `episode_success`, `local_proxy_milestones`, and `official_milestone_score_0_12` only when produced by the corresponding evaluator. A project-owned proxy is useful but cannot finalize unpublished geometry, contact thresholds, locking criteria, or ranking rules.

For a milestone-aware evaluator or log, retain milestone ID, actor/provenance, achievement time, credit once, intervention records, skipped/failed/not-attempted status, duration, and termination reason. Manually supplied milestones receive no robot credit; later robot achievements may still count under the published rules. Do not turn this into automatic intervention capability in the submitted policy.

Compare ACT-only, GR00T-only, and per-subtask routing on the same fixed held-out seeds, permitted observation set, environment, and resource budget. Tune route maps and thresholds on validation trials; reserve a separate final test set. Report sample count and uncertainty, not a precise ranking from a handful of rollouts.

Record autonomous full completion, per-stage/milestone success, current score definition, interventions, timeouts, latency percentiles/deadline misses, and memory. Distinguish full-chain evaluation from isolated subtask resets, which change the initial-state distribution and task difficulty.

Check both `terminated` and `truncated`, independent per-env reset, and repeated-run reproducibility. Restart the one-run environment server between local runs. Aggregate per environment; the checker's `.max()` and first-env helpers are not multi-env episode statistics.

Validate evaluator behavior for duplicate credit, manual supply, skip then later achievement, boundary thresholds, missing results, and truncation when implementing an evaluator. Produce raw result files and an aggregate with hashes for weights, processor, schema, split/seed list, router, and container image. Clearly mark runtime trials not performed.

## Sources

- [Task specification and development-checker limitations](https://github.com/rocochallenge/ecm2027_sim_track/blob/6e3a66c1344e926ef66b3a804058918d146cd2d0/README.md)
- [Development checker](https://github.com/rocochallenge/ecm2027_sim_track/blob/6e3a66c1344e926ef66b3a804058918d146cd2d0/robofinals_tasks/chamber_service_task.py)
- [Published challenge rules](https://nrp.gov.sg/SGEAIchallenge/challenge/); recheck for updates when evaluating against official requirements.
