## 将标准数据集和你的数据集混合在一起!
> 如果没有标准数据参杂在其中,很容易造成`灾难性遗忘`  

请去`merge_data`文件夹**选择和你数据集结构相同的脚本**,我没有做多结构适配,可以将格式发给llm让他来帮你适配格式
> 建议往数据集插入**20%-50%**的标准数据集  
### 使用方法
```bash
python merge_data/merge_data.py --qa-file ./data/qa_final.json --training-file training_data.jsonl --output-file merged_training_data.jsonl --use-new-prompt
```
可以使用`--percentage`来设置插入百分比(对比于原始训练数据,例如原始200条,设置为100则插入200条,最终有400条数据)
**请注意,`qa_final.json`只是一个在网上下载的ai训练集,可以由你自己寻找的训练集替换,若有结构不适配的问题可以使用llm来帮你适配**
**此文件已上传至仓库,但不确定训练集的质量**
~~联系我也是个不错的选择~~
> 建议使用--use-new-prompt,防止全部都是角色system prompt  
>不过这个prompt写死在程序中,需要自己修改(不用config是因为太乱了)
### 插入20%数据：
```bash
python merge_data/merge_data.py --percentage 20 --use-new-prompt --seed 123
```