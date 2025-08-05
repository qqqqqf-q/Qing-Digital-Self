## 2.3 Mix Standard Dataset with Your Dataset!
> Without standard data mixed in, it's easy to cause `catastrophic forgetting`

So we have the `merge_data` folder
Please go to the `merge` folder and choose a script with the same structure as your dataset. I haven't made multi-structure adaptation, you can send the format to LLM to help you adapt the format
> It's recommended to insert **20%-50%** of standard dataset into your dataset
### Usage
```bash
python merge_data/merge_data.py --qa_file qa_final.json --training_file training_data.jsonl --output_file merged_training_data.jsonl --use-new-prompt
```
> It's recommended to use --use-new-prompt to prevent all being role system prompts
>
### Insert 20% data:
```bash
python merge_training_data.py --percentage 20 --use-new-prompt --seed 123
```