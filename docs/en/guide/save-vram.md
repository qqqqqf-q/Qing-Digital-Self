## VRAM Optimization Guide
### Important Parameters to Consider

| Parameter | Purpose | Recommended Adjustment |
| ------ | ------ | ------ |
| `--per_device_train_batch_size` | Training batch size per device | For low VRAM: 1-2, for high VRAM: 4 or higher |
| `--per_device_eval_batch_size` | Evaluation batch size per device | Same as above |
| `--gradient_accumulation_steps` | Gradient accumulation steps | Low VRAM: 4-16, high VRAM: reduce accordingly |