"""Layer 5 · Action post-processor.

Turns action chunks (23-D model space, each with its own time base) into exactly one safe 19-D sim action per
50 Hz control tick.

Modules:
    chunk_executor  queue of chunks, chunk time → control tick mapping, re-plan points, blending
    action_codec    23-D model action ↔ 19-D sim action (rot6d ↔ quaternion, gripper ↔ ±1)
    safety          workspace / step / base velocity limits, gripper hysteresis, hold action
"""
