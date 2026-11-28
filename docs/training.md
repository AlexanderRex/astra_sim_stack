# Fine-tune NVIDIA Isaac GR00T N1.7 (open vision-language-action (VLA) model)


Fine-tuning NVIDIA Isaac GR00T N1.7 VLA model is performed in RunPod environment using NVIDIA L40S GPU with 48GB VRAM, 94GB RAM and 250GB disk space.

## Prepare environment

```bash
cat >> ~/.bashrc <<'EOF'
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/hf_cache
export NO_ALBUMENTATIONS_UPDATE=1
EOF
```

```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_xxx
cd /workspace
```
## Download repository
```bash
apt-get update && apt-get install -y git-lfs ffmpeg tmux
git lfs install
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```
In case issue with fine-tuning check the following

```bash
echo "launch_finetune.py:  $(grep -c 'letter_box_transform = True' gr00t/experiment/launch_finetune.py)"
echo "processor override:  $(grep -c '"letter_box_transform",' gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py)"
```

and update if needed

Turn on letterbox padding in launch_finetune.py (only if missing)
```bash
grep -q "letter_box_transform = True" gr00t/experiment/launch_finetune.py || \
  sed -i 's/^    config.model.use_percentiles = ft_config.use_percentiles$/&\n    config.model.letter_box_transform = True/' gr00t/experiment/launch_finetune.py
```
Let the processor accept the letterbox override (only if missing)
```bash
grep -q '"letter_box_transform",' gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py || \
  sed -i 's/^                "strict_action_padding_mask",$/&\n                "letter_box_transform",/' gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
```

```bash
source ~/.bashrc
```

```bash
cd /workspace/Isaac-GR00T
mkdir -p examples/Galbot
```
```bash
cat > examples/Galbot/galbot_config.py <<'EOF'
import json

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Camera and language keys come straight from the dataset
_MOD = json.load(open("/workspace/data/galbot_gr00t_v21/meta/modality.json"))
VIDEO_KEYS = list(_MOD["video"].keys())          # head, left_wrist, right_wrist
LANG_KEY = "annotation." + list(_MOD["annotation"].keys())[0]   # per-frame stage instruction

# Layout written by dataset_conversion.py
STATE_KEYS = ["left_arm_q", "right_arm_q", "gripper", "left_eef_world", "right_eef_world"]
ACTION_KEYS = ["left_eef", "right_eef", "gripper", "base_vel"]
ACTION_HORIZON = 40   # steps at 50 Hz = 0.8 s; must match the base model's action_horizon

for _part, _keys in (("state", STATE_KEYS), ("action", ACTION_KEYS)):
    _missing = [k for k in _keys if k not in _MOD[_part]]
    if _missing:
        raise KeyError(f"{_part} keys {_missing} are not in modality.json {list(_MOD[_part])}")


def absolute():
    # Plain numbers, absolute: the arm targets are xyz + rot6d in the robot frame, with no state
    # group of the same name, so GR00T must not apply any EEF / rotation handling to them.
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


galbot_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[absolute() for _ in ACTION_KEYS],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=[LANG_KEY]),
}

register_modality_config(galbot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
EOF
```

```bash
export HF_TOKEN=hf_xxx
```

## Download data

```bash
uv run hf download atanasko-mitrev/galbot_gr00t_v21 --repo-type dataset \
  --local-dir /workspace/data/galbot_gr00t_v21
```

## Run the training
Training was performed on 30000 steps
```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /workspace/data/galbot_gr00t_v21 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/Galbot/galbot_config.py \
  --num-gpus 1 --output-dir /workspace/out/galbot_run1 \
  --max-steps 30000 --save-steps 2500 --save-total-limit 3 \
  --global-batch-size 32 --dataloader-num-workers 4 \
  2>&1 | tee /workspace/out/galbot_run1.log
```

## Download the result
```bash
mkdir -p src/checkpoints/galbot_run1
rsync -avP \
  --exclude='optimizer.pt' --exclude='rng_state.pth' --exclude='scheduler.pt' \
  -e "ssh -p PORT -i ~/.ssh/id_ID" \
  root@IP:/workspace/out/galbot_run1/checkpoint-30000 \
  src/checkpoints/galbot_run1/
```








