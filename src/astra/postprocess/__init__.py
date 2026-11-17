"""Layer 5 · Action post-processor.

Turns action chunks (24-D model space, 30 fps data time) into exactly one safe 20-D sim action per
50 Hz control tick.

Modules:
    chunk_executor  queue of chunks, 30 fps → 50 Hz time mapping, re-plan points, blending
    action_codec    24-D model action ↔ 20-D sim action (rot6d ↔ quaternion, gripper ↔ ±1)
    safety          workspace / step / wheel limits, gripper hysteresis, hold action
"""
