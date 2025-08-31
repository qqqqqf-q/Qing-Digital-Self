##  获取 QQ 聊天数据

* 教程请参考：[NTQQ Windows 数据解密](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* 补充资料：[数据库解码参考](https://qq.sbcnm.top/decrypt/decode_db.html)
* 上面这两个是同一个教程的不同章节,耐心看完就好,不复杂(如果不会可以翻到最底下找我哦)
* 使用 DB Browser for SQLite，密码填写你获取到的 16 位密钥
* HMAC 算法一般为SHA1，也有人是SHA512和256,自行测试,算法错误了会打不开数据库（所以需要测试到打开为之,也可以用 AI 帮你适配）
* 在 DB Browser 里**导出 `c2c_msg_table` 的 SQL**
* 新建数据库，**导入刚才导出的 SQL 文件**
* 获得一个这样的数据库
* 结构如下图,是明文数据库(你能打开并且能看到数据就是正常的)
* 将数据库重命名为 `qq.db`并放在`dataset/original`文件夹下
> 或修改`setting.jsonc`中的`qq_db_path`



* <img src="https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp" alt="数据库图片">

## 获取Telegram(TG)聊天数据

* 请使用[Telegram Desktop](https://desktop.telegram.org/)导出聊天数据
* 点击`Export chat history`按钮
<img src="https://cdn.nodeimage.com/i/8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png" alt="8PmL1yOyWbk1tTUkoLSk14sqrXN1HhYS.png">
* 选择`JSON(Machine-readable JSON)`按钮
* 不必勾选其他按钮,因为此项目暂不支持多模态
<img src="https://cdn.nodeimage.com/i/ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png" alt="ZOx12BovPbYXo89k4xIF9yRlEamneq4g.png">

* 将`导出文件夹下的**ChatExport_**文件夹全部移至`dataset/original/`文件夹内,如下图所示
<img src="https://cdn.nodeimage.com/i/zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G.png" alt="zbc3iDHiqJrIOtWwrHkzX7TMONYatB8G">

* **重要**
* 修改`setting.jsonc`文件,将`telegram_chat_id`改为你的telegram聊天id
> **包含空格!!!**  
* 比如以下ID需要填写的是`qqqqq f`