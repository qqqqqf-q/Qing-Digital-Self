import sqlite3
import os
import sys
import time
from pathlib import Path
from typing import Optional, Iterable, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.config.config import get_config
from utils.logger.logger import get_logger
# 获取配置实例
config = get_config()
logger = get_logger('Database')


def _detect_text_encoding(file_path: str, probe_bytes: int = 64 * 1024) -> str:
    """用较低成本探测SQL文件编码"""
    common_encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    with open(file_path, "rb") as f:
        head = f.read(probe_bytes)
    for encoding in common_encodings:
        try:
            head.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _iter_sql_statements(sql_file) -> Iterable[str]:
    """流式切分SQL语句，避免一次性读入内存"""
    buffer = ""
    for line in sql_file:
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            buffer = ""
            if statement:
                yield statement
    tail = buffer.strip()
    if tail:
        yield tail


def _is_sqlite_db_file(file_path: str) -> bool:
    """快速判断是否为SQLite文件"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        return header == b"SQLite format 3\x00"
    except OSError:
        return False


def _validate_sqlite_db(file_path: str) -> bool:
    """轻量验证缓存db可用，避免中断导入留下的坏文件被复用"""
    if not os.path.exists(file_path) or not _is_sqlite_db_file(file_path):
        return False

    try:
        conn = sqlite3.connect(file_path)
        try:
            conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False


class _StreamingInsertState:
    """将超大INSERT语句拆分成可执行的小批次，避免一次性缓存整条SQL"""

    def __init__(self, insert_prefix: str, batch_rows: int = 50, batch_chars: int = 256 * 1024):
        self.insert_prefix = insert_prefix.strip()
        self.batch_rows = max(1, int(batch_rows))
        self.batch_chars = max(1024, int(batch_chars))

        self._tuple_chars: List[str] = []
        self._tuples: List[str] = []
        self._tuples_chars = 0
        self._paren_depth = 0
        self._in_string = False
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def feed(self, text: str) -> List[tuple]:
        """喂入SQL片段，返回可执行批次[(sql, rows), ...]"""
        batches: List[tuple] = []
        i = 0
        while i < len(text):
            ch = text[i]

            if self._finished:
                break

            if self._in_string:
                if ch == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    self._tuple_chars.append("''")
                    i += 2
                    continue
                self._tuple_chars.append(ch)
                if ch == "'":
                    self._in_string = False
                i += 1
                continue

            if ch == "'":
                self._in_string = True
                self._tuple_chars.append(ch)
                i += 1
                continue

            if self._paren_depth == 0 and not self._tuple_chars and ch.isspace():
                i += 1
                continue

            if ch == "(":
                self._paren_depth += 1
                self._tuple_chars.append(ch)
                i += 1
                continue

            if ch == ")":
                self._paren_depth = max(0, self._paren_depth - 1)
                self._tuple_chars.append(ch)
                i += 1

                if self._paren_depth == 0:
                    tuple_sql = "".join(self._tuple_chars).strip()
                    self._tuple_chars = []
                    if tuple_sql:
                        self._tuples.append(tuple_sql)
                        self._tuples_chars += len(tuple_sql)

                    if self._tuples and (
                        len(self._tuples) >= self.batch_rows or self._tuples_chars >= self.batch_chars
                    ):
                        batches.append((self._build_statement(), len(self._tuples)))
                        self._tuples = []
                        self._tuples_chars = 0
                continue

            if self._paren_depth == 0:
                if ch == ";":
                    if self._tuples:
                        batches.append((self._build_statement(), len(self._tuples)))
                        self._tuples = []
                        self._tuples_chars = 0
                    self._finished = True
                    i += 1
                    continue
                if self._tuples and ch == ",":
                    i += 1
                    continue

            if self._paren_depth > 0 or self._tuple_chars:
                self._tuple_chars.append(ch)

            i += 1

        return batches

    def _build_statement(self) -> str:
        return f"{self.insert_prefix} {','.join(self._tuples)}"


def _import_sql_file_to_sqlite(sql_path: str, target_db_path: str) -> None:
    """将.sql导入为SQLite数据库（支持超大.sql）"""
    sql_path = os.path.abspath(sql_path)
    target_db_path = os.path.abspath(target_db_path)

    os.makedirs(os.path.dirname(target_db_path), exist_ok=True)
    tmp_db_path = f"{target_db_path}.tmp"
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)
    if os.path.exists(target_db_path):
        os.remove(target_db_path)

    conn = sqlite3.connect(tmp_db_path)
    try:
        # 让SQL里的 BEGIN/COMMIT 生效，也便于我们自己分批提交
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("PRAGMA cache_size=-65536")

        encoding = _detect_text_encoding(sql_path)

        def _has_explicit_transaction_prefix() -> bool:
            try:
                with open(sql_path, "r", encoding=encoding, errors="replace") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("--"):
                            continue
                        return stripped.upper().startswith("BEGIN")
            except OSError:
                return False
            return False

        use_file_transactions = _has_explicit_transaction_prefix()
        statement_count = 0
        inserted_rows = 0
        commit_every = 2000

        if not use_file_transactions:
            conn.execute("BEGIN")

        sql_size = os.path.getsize(sql_path)
        start_time = time.monotonic()
        last_log_time = start_time
        processed_bytes = 0

        buffer = ""
        insert_state: Optional[_StreamingInsertState] = None

        def _maybe_log_progress(force: bool = False):
            nonlocal last_log_time
            now = time.monotonic()
            if not force and now - last_log_time < 30:
                return
            elapsed = max(now - start_time, 1e-6)
            pct = (processed_bytes / sql_size) * 100 if sql_size else 0.0
            speed = processed_bytes / elapsed
            eta = int((sql_size - processed_bytes) / speed) if speed > 0 else -1
            logger.info(
                zhcn=f"SQL导入进度: {pct:.1f}% ({processed_bytes}/{sql_size} bytes)，语句:{statement_count}，行:{inserted_rows}，速度:{speed/1024/1024:.2f}MB/s，ETA:{eta}s",
                en=f"SQL import: {pct:.1f}% ({processed_bytes}/{sql_size} bytes), stmts:{statement_count}, rows:{inserted_rows}, speed:{speed/1024/1024:.2f}MB/s, ETA:{eta}s",
            )
            last_log_time = now

        with open(sql_path, "rb") as f:
            for raw_line in f:
                processed_bytes += len(raw_line)
                line = raw_line.decode(encoding, errors="replace")

                if insert_state is not None:
                    for stmt, rows in insert_state.feed(line):
                        conn.execute(stmt)
                        statement_count += 1
                        inserted_rows += int(rows)

                        if not use_file_transactions and statement_count % commit_every == 0:
                            conn.execute("COMMIT")
                            conn.execute("BEGIN")

                        _maybe_log_progress()

                    if insert_state.finished:
                        insert_state = None
                    continue

                buffer += line
                if sqlite3.complete_statement(buffer):
                    lines = buffer.splitlines()
                    while lines and lines[0].lstrip().startswith("--"):
                        lines.pop(0)
                    cleaned_statement = "\n".join(lines).strip()
                    buffer = ""
                    if cleaned_statement:
                        conn.execute(cleaned_statement)
                        statement_count += 1
                        if not use_file_transactions and statement_count % commit_every == 0:
                            conn.execute("COMMIT")
                            conn.execute("BEGIN")
                    continue

                upper = buffer.lstrip().upper()
                if upper.startswith("INSERT") and "VALUES" in upper:
                    upper_full = buffer.upper()
                    values_idx = upper_full.find("VALUES")
                    if values_idx != -1:
                        prefix = buffer[: values_idx + len("VALUES")].strip()
                        remainder = buffer[values_idx + len("VALUES") :]
                        insert_state = _StreamingInsertState(prefix)
                        buffer = ""
                        for stmt, rows in insert_state.feed(remainder):
                            conn.execute(stmt)
                            statement_count += 1
                            inserted_rows += int(rows)
                            if not use_file_transactions and statement_count % commit_every == 0:
                                conn.execute("COMMIT")
                                conn.execute("BEGIN")
                        _maybe_log_progress(force=True)

        if not use_file_transactions:
            conn.execute("COMMIT")
        else:
            try:
                conn.execute("COMMIT")
            except sqlite3.Error:
                pass

    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)
        except OSError:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    os.replace(tmp_db_path, target_db_path)


class DatabaseConnector:
    # 连接sqlite数据库
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.resolved_db_path: Optional[str] = None

    def _resolve_sqlite_path(self) -> str:
        """解析输入路径为可连接的SQLite文件路径（支持.sql -> .db）"""
        source_path = os.path.abspath(self.db_path)
        suffix = Path(source_path).suffix.lower()

        if suffix != ".sql":
            return source_path

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"sql file not found: {source_path}")

        stat = os.stat(source_path)
        cache_dir = Path(__file__).resolve().parents[2] / "cache" / "sqlite_imported"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_db_name = f"{Path(source_path).stem}-{int(stat.st_mtime)}-{stat.st_size}.db"
        cached_db_path = str(cache_dir / cached_db_name)

        if os.path.exists(cached_db_path) and _validate_sqlite_db(cached_db_path):
            return cached_db_path
        if os.path.exists(cached_db_path) and not _validate_sqlite_db(cached_db_path):
            try:
                os.remove(cached_db_path)
            except OSError:
                pass

        logger.info(
            zhcn=f"检测到SQL文件，开始导入为SQLite数据库: {source_path} -> {cached_db_path}",
            en=f"Detected SQL file, importing into SQLite: {source_path} -> {cached_db_path}",
        )
        _import_sql_file_to_sqlite(source_path, cached_db_path)
        return cached_db_path

    
    def connect(self) -> sqlite3.Connection:
        resolved_path = self._resolve_sqlite_path()
        self.resolved_db_path = resolved_path

        # 检查文件是否存在
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"db file not found: {resolved_path}")
        
        try:
            self.conn = sqlite3.connect(resolved_path)
            
            # 测试连接
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            
            logger.info(
                zhcn=f"成功连接到数据库: {resolved_path}",
                en=f"Successfully connected to database: {resolved_path}"
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
