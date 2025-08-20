## 将标准数据集和你的数据集混合在一起!
> 如果没有标准数据参杂在其中,很容易造成`灾难性遗忘`  

所以就有了 `merge_data`文件夹
请去`merge_data`文件夹**选择和你数据集结构相同的脚本**,我没有做多结构适配,可以将格式发给llm让他来帮你适配格式
> 建议往数据集插入**20%-50%**的标准数据集  
### 使用方法
```bash
python merge_data/merge_data.py --qa_file qa_final.json --training_file training_data.jsonl --output_file merged_training_data.jsonl --use-new-prompt
```
> 建议使用--use-new-prompt,防止全部都是角色system prompt
>
### 插入20%数据：
```bash
python merge_data/merge_data.py --percentage 20 --use-new-prompt --seed 123
```