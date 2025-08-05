## Getting QQ Chat Data

* Tutorial reference: [NTQQ Windows Data Decryption](https://qq.sbcnm.top/decrypt/NTQQ%20%28Windows%29.html)
* Supplementary material: [Database Decoding Reference](https://qq.sbcnm.top/decrypt/decode_db.html)
* The above two are different chapters of the same tutorial, read them patiently, it's not complicated (if you don't know how, scroll to the bottom to find me)
* Use DB Browser for SQLite, enter the 16-digit key you obtained as the password
* HMAC algorithm is generally SHA1, some people use SHA512 and 256, test yourself, wrong algorithm will fail to open the database (so you need to test until it opens, you can also use AI to help you adapt)
* In DB Browser **export the SQL of `c2c_msg_table`**
* Create a new database, **import the SQL file you just exported**
* Get a database like this
* Structure as shown below, it's a plaintext database (you can open it and see the data, which means it's normal)
* <img src="https://cdn.nodeimage.com/i/oBfbWfVLhJI0CeZHTwwxq6G7XGO40Vy4.webp" alt="Database Image">