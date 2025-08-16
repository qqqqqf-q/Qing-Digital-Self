## 节省显存篇
### 注意以下命令

| 参数名 | 用处 | 建议调整 |
| ------ | ------ | ------ |
| `--per_device_train_batch_size` | 每设备训练批次大小 | 若显存小建议1-2，若大显存可以4及以上 |
| `--per_device_eval_batch_size` | 每设备验证批次大小 | 同上 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 显存小：4-16，显存大则降低 |