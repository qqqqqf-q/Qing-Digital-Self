# 数据库配置文件
# TODO: 可能需要从环境变量读取这些配置

DB_PATH = r"K:\shujufenxi\c2c.db"

# 一些常用的查询模板
COMMON_QUERIES = {
    'list_tables': "SELECT name FROM sqlite_master WHERE type='table'",
    'table_info': "PRAGMA table_info({})",
    'count_rows': "SELECT COUNT(*) FROM {}"
}