---
name: astra-demo-dataset
description: Collect, convert, or validate Galbot demonstration episodes and compatible ACT or GR00T LeRobot datasets with reproducible splits and statistics.
---

# Demonstrations to model-ready data

Inspect the actual recording format, robot contract, model checkout, and dataset loader first. At task commit `6e3a66c1344e926ef66b3a804058918d146cd2d0`, public demonstration data is pending, and generic LeRobot utilities reference SO100/SO101 modules absent from this subset. Verify the active checkout; do not treat those utilities as a ready Galbot converter.

Record episode/session/environment IDs, simulator and schema revisions, seeds and initial conditions, camera capture times, state measurement times, action command/application ticks, termination versus truncation, subtask labels, success/failure, interventions, and assistance. Separate requested action from controller-applied action and measured response.

Choose the dataset target representation deliberately: current EEF/wheel actions or a separately validated canonical representation. Do not call an EEF pose a seven-joint arm just because both have seven values. Align observations causally to actions; store timing conventions and resampling decisions. Reset boundaries must stop chunks and histories.

Split by complete episode and collection session/scene family before computing statistics or extracting windows. Keep near-duplicate demonstrations and recovery variants from the same source in one split. Fit statistics on training data only. Mask terminal padding and document whether failed/recovery segments participate in each training objective.

Produce a stable raw episode archive and model-specific exports. Pin LeRobot format/loader versions separately: current LeRobot defaults need not match the GR00T-flavored LeRobot v2 format described in the inspected GR00T guide. Generate feature metadata, video paths, episode indices, task annotations, and `meta/modality.json` to the chosen loader's real requirements.

Define modality slices and action configs from the schema. Distinguish absolute/relative and EEF/non-EEF fields; grippers usually remain absolute categorical/binary commands. A shared canonical dataset does not imply identical ACT and GR00T normalization.

Validation must open exported data through both intended loaders, decode representative frames, compare source/export state-action values, check monotonic timestamps and causal skew, finite values and action limits, terminal masks, split disjointness, and train-only statistics. Run a small training/inference load smoke check when those environments exist.

Deliver dataset/split manifests with hashes, schema and export versions, statistics provenance, an error/rejection report, and a replay example. Record downloads and redistribution terms for assets/data separately; source-code licenses do not establish data redistribution rights.

## Sources

- [Task dataset/export limitations](https://github.com/rocochallenge/ecm2027_sim_track/blob/6e3a66c1344e926ef66b3a804058918d146cd2d0/README.md)
- [GR00T dataset format at the inspected revision](https://github.com/NVIDIA/Isaac-GR00T/blob/3b49067943eb1ab2eb2203b93d461c4390c78da3/getting_started/data_preparation.md)
