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
* <img src="https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp" alt="数据库图片">