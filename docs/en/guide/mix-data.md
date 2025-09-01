## Mix Standard Dataset with Your Dataset!
> Without standard data mixed in, it's easy to cause `catastrophic forgetting`

Please go to the `merge_data` folder **and choose the script that matches your dataset structure**, I didn't make multi-structure adaptations, you can send the format to an LLM to help you adapt the format
> It's recommended to insert **20%-50%** of standard dataset into your dataset
### Usage
```bash
python merge_data/merge_data.py --qa-file ./data/qa_final.json --training-file training_data.jsonl --output-file merged_training_data.jsonl --use-new-prompt
```
You can use `--percentage` to set the insertion percentage (compared to original training data, e.g., if original has 200 entries, setting to 100 will insert 200 entries, resulting in 400 total entries)
**Please note that `qa_final.json` is just an AI training set downloaded from the internet, which can be replaced by training sets you find yourself. If there are structure incompatibility issues, you can use an LLM to help you adapt**
**This file has been uploaded to the repository, but the quality of the training set is uncertain**
~~Contacting me is also a good option~~
> It's recommended to use --use-new-prompt to prevent all system prompts from being the same role
> However, this prompt is hardcoded in the program and needs to be modified manually (not using config because it's too messy)
### Insert 20% data:
```bash
python merge_data/merge_data.py --percentage 20 --use-new-prompt --seed 123
```