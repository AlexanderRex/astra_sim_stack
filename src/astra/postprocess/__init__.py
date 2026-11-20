"""Layer 5 · Action post-processor.

Turns action chunks (23-D model space, 30 fps data time) into exactly one safe 19-D sim action per
50 Hz control tick.

Modules:
    chunk_executor  queue of chunks, 30 fps → 50 Hz time mapping, re-plan points, blending
    action_codec    23-D model action ↔ 19-D sim action (rot6d ↔ quaternion, gripper ↔ ±1)
    safety          workspace / step / base velocity limits, gripper hysteresis, hold action
"""
