import sqlite3
import os
from typing import Optional
from config.config import get_config
from logger.logger import get_logger
# 获取配置实例
config = get_config()
logger = get_logger('Database')
class DatabaseConnector:
    # 连接sqlite数据库
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    
    def connect(self) -> sqlite3.Connection:
        # 检查文件是否存在
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"db file not found: {self.db_path}")
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            
            # 测试连接
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            
            logger.info(
                zhcn=f"成功连接到数据库: {self.db_path}",
                en=f"Successfully connected to database: {self.db_path}"
            )
            return self.conn
            
        except sqlite3.Error as e:
            if self.conn:
                self.conn.close()
                self.conn = None
            raise sqlite3.Error(f"connection failed: {str(e)}")
    
    def query(self, sql: str, params: tuple = ()) -> list:
        # 执行查询
        if not self.conn:
            raise RuntimeError("not connected, call connect() first")
        
        try:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall()
        except sqlite3.Error as e:
            raise sqlite3.Error(f"query failed: {str(e)}")
    
    def execute(self, cmd: str, params: tuple = ()) -> int:
        # 执行命令 (insert/update/delete)
        if not self.conn:
            raise RuntimeError("not connected")
        
        try:
            cur = self.conn.cursor()
            cur.execute(cmd, params)
            self.conn.commit()
            return cur.rowcount
        except sqlite3.Error as e:
            self.conn.rollback()
            raise sqlite3.Error(f"execute failed: {str(e)}")
    
    def get_tables(self) -> list:
        # 获取所有表名
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        results = self.query(sql)
        return [row[0] for row in results]
    
    def get_schema(self, table_name: str) -> list:
        # 获取表结构
        sql = f"PRAGMA table_info({table_name})"
        return self.query(sql)
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info(
                zhcn="数据库连接关闭",
                en="Database connection closed"
            )
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()