"""Layer 2 · Observation adapter.

Turns the raw observation dict from the env server into a validated ``CanonicalObs`` and the
model-specific views (ACT, GR00T, stage classifier, event detectors). The same code is meant to be
imported by the offline dataset builder so training and runtime inputs are built identically.

Modules:
    observation_adapter  ObservationAdapter: ingest → validate → canonicalize → buffer → views
    kinematics           forward kinematics: arm joint angles → base-frame EEF poses
    rotations            quaternion / rotation-matrix / rot6d conversions (one convention)
    history              time-indexed ring buffer of CanonicalObs
    views                builders for ActView, Gr00tView and classifier inputs
"""
