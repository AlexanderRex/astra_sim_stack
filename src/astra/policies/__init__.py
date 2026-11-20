"""Layer 4 · Router + policies.

Produces action chunks in the 23-D model action space
``[left_eef xyz+rot6d 9 | right_eef 9 | gripper 2 | base_vel 3]`` (absolute targets, physical units).

Modules:
    base           PolicyBackend interface shared by all chunk producers
    act_policy     custom ACT (LeRobot) wrapper
    groot_policy   GR00T N1.7 (Isaac-GR00T Gr00tPolicy) wrapper
    scripted_wipe  scripted press-dwell-lift wiping primitive for the WIPE stage
    router         picks the producer per stage, manages switching and the latency/fault fallback
"""
