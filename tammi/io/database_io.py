"""Database input/output handlers for SQL and NoSQL databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from tammi.io.base import InputReader, OutputWriter, ReaderFactory, WriterFactory, AVAILABLE_DRIVERS


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    db_type: str  # 'sqlite', 'mysql', 'postgresql', 'mongodb'
    host: str = "localhost"
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""
    table: str = "tammi_results"  # or collection for MongoDB
    text_column: str = "text_content"
    id_column: str = "text_id"
    
    def get_connection(self) -> Any:
        """Get a database connection based on config."""
        if self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(self.database)
        elif self.db_type == "mysql":
            try:
                import mysql.connector  # type: ignore[import-not-found]
                return mysql.connector.connect(
                    host=self.host,
                    port=self.port or 3306,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                )
            except ImportError:
                raise ImportError("mysql-connector-python is required for MySQL support. Install with: pip install mysql-connector-python")
        elif self.db_type == "postgresql":
            try:
                import psycopg2  # type: ignore[import-not-found]
                return psycopg2.connect(
                    host=self.host,
                    port=self.port or 5432,
                    dbname=self.database,
                    user=self.username,
                    password=self.password,
                )
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")
        elif self.db_type == "mongodb":
            try:
                import pymongo  # type: ignore[import-not-found]
                if self.username and self.password:
                    uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port or 27017}/{self.database}"
                else:
                    uri = f"mongodb://{self.host}:{self.port or 27017}/{self.database}"
                client = pymongo.MongoClient(uri)
                return client[self.database]
            except ImportError:
                raise ImportError("pymongo is required for MongoDB support. Install with: pip install pymongo")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    @classmethod
    def from_connection_string(
        cls,
        conn_str: str,
        table: str = "tammi_results",
        text_column: str = "text_content",
        id_column: str = "text_id",
    ) -> "DatabaseConfig":
        """
        Parse a connection string into a DatabaseConfig.
        
        Formats:
        - sqlite:path/to/db.db
        - mysql://user:pass@host:port/database
        - postgresql://user:pass@host:port/database
        - mongodb://user:pass@host:port/database
        """
        if conn_str.startswith("sqlite:"):
            db_path = conn_str[7:]  # Remove 'sqlite:'
            return cls(
                db_type="sqlite",
                database=db_path,
                table=table,
                text_column=text_column,
                id_column=id_column,
            )
        elif conn_str.startswith(("mysql://", "postgresql://", "mongodb://")):
            parsed = urlparse(conn_str)
            db_type = parsed.scheme
            return cls(
                db_type=db_type,
                host=parsed.hostname or "localhost",
                port=parsed.port or cls._default_port(db_type),
                database=parsed.path.lstrip("/") if parsed.path else "",
                username=parsed.username or "",
                password=parsed.password or "",
                table=table,
                text_column=text_column,
                id_column=id_column,
            )
        else:
            raise ValueError(
                f"Invalid connection string: {conn_str}. "
                "Use sqlite:path.db, mysql://..., postgresql://..., or mongodb://..."
            )
    
    @staticmethod
    def _default_port(db_type: str) -> int:
        """Get default port for database type."""
        ports = {
            "mysql": 3306,
            "postgresql": 5432,
            "mongodb": 27017,
        }
        return ports.get(db_type, 0)


# =============================================================================
# SQLite
# =============================================================================

class SQLiteReader(InputReader):
    """Read text records from SQLite database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        **kwargs: Any,
    ) -> None:
        self.config = db_config
        self._conn = None
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from SQLite."""
        import sqlite3
        conn = sqlite3.connect(self.config.database)
        cursor = conn.cursor()
        
        query = f'SELECT "{self.config.id_column}", "{self.config.text_column}" FROM "{self.config.table}"'
        cursor.execute(query)
        
        for row in cursor:
            text_id, text = str(row[0]), row[1] or ""
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}
        
        cursor.close()
        conn.close()
    
    def count(self) -> int:
        """Count records in table."""
        import sqlite3
        conn = sqlite3.connect(self.config.database)
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{self.config.table}"')
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def close(self) -> None:
        """Close connection if open."""
        if self._conn:
            self._conn.close()
            self._conn = None


class SQLiteWriter(OutputWriter):
    """Write result records to SQLite database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        columns: List[str],
        **kwargs: Any,
    ) -> None:
        import sqlite3
        self.config = db_config
        self.columns = columns
        self._conn = sqlite3.connect(self.config.database)
        self._cursor = self._conn.cursor()
        self._create_table()
    
    def _create_table(self) -> None:
        """Create results table if not exists."""
        columns_sql = ", ".join([f'"{col}" REAL' for col in self.columns])
        create_sql = f'''
            CREATE TABLE IF NOT EXISTS "{self.config.table}" (
                text_id TEXT PRIMARY KEY,
                {columns_sql}
            )
        '''
        self._cursor.execute(create_sql)
        if self._conn:
            self._conn.commit()
    
    def write_header(self, columns: List[str]) -> None:
        """Headers handled in table creation."""
        pass
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Write a single record."""
        quoted_columns = ", ".join([f'"{col}"' for col in self.columns])
        placeholders = ", ".join(["?"] * (len(self.columns) + 1))
        insert_sql = f'INSERT OR REPLACE INTO "{self.config.table}" (text_id, {quoted_columns}) VALUES ({placeholders})'
        self._cursor.execute(insert_sql, [text_id] + values)
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        """Write multiple records."""
        for text_id, values in records:
            self.write_record(text_id, values)
        if self._conn:
            self._conn.commit()
        return len(records)
    
    def close(self) -> None:
        """Commit and close connection."""
        if self._conn:
            self._conn.commit()
            self._cursor.close()
            self._conn.close()
            self._conn = None


# =============================================================================
# MySQL
# =============================================================================

class MySQLReader(InputReader):
    """Read text records from MySQL database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        **kwargs: Any,
    ) -> None:
        self.config = db_config
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from MySQL."""
        conn = self.config.get_connection()
        cursor = conn.cursor()
        
        query = f"SELECT `{self.config.id_column}`, `{self.config.text_column}` FROM `{self.config.table}`"
        cursor.execute(query)
        
        for row in cursor:
            text_id, text = str(row[0]), row[1] or ""
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}
        
        cursor.close()
        conn.close()
    
    def count(self) -> int:
        """Count records in table."""
        conn = self.config.get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM `{self.config.table}`")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def close(self) -> None:
        pass


class MySQLWriter(OutputWriter):
    """Write result records to MySQL database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        columns: List[str],
        **kwargs: Any,
    ) -> None:
        self.config = db_config
        self.columns = columns
        self._conn = self.config.get_connection()
        self._cursor = self._conn.cursor()
        self._create_table()
    
    def _create_table(self) -> None:
        """Create results table if not exists."""
        columns_sql = ", ".join([f"`{col}` DOUBLE" for col in self.columns])
        create_sql = f'''
            CREATE TABLE IF NOT EXISTS `{self.config.table}` (
                text_id VARCHAR(255) PRIMARY KEY,
                {columns_sql}
            )
        '''
        self._cursor.execute(create_sql)
        if self._conn:
            self._conn.commit()
    
    def write_header(self, columns: List[str]) -> None:
        pass
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        """Write a single record."""
        quoted_columns = ", ".join([f"`{col}`" for col in self.columns])
        placeholders = ", ".join(["%s"] * (len(self.columns) + 1))
        insert_sql = f"REPLACE INTO `{self.config.table}` (text_id, {quoted_columns}) VALUES ({placeholders})"
        self._cursor.execute(insert_sql, [text_id] + values)
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        for text_id, values in records:
            self.write_record(text_id, values)
        if self._conn:
            self._conn.commit()
        return len(records)
    
    def close(self) -> None:
        if self._conn:
            self._conn.commit()
            self._cursor.close()
            self._conn.close()
            self._conn = None


# =============================================================================
# PostgreSQL
# =============================================================================

class PostgreSQLReader(InputReader):
    """Read text records from PostgreSQL database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        **kwargs: Any,
    ) -> None:
        self.config = db_config
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from PostgreSQL."""
        conn = self.config.get_connection()
        cursor = conn.cursor()
        
        query = f'SELECT "{self.config.id_column}", "{self.config.text_column}" FROM "{self.config.table}"'
        cursor.execute(query)
        
        for row in cursor:
            text_id, text = str(row[0]), row[1] or ""
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}
        
        cursor.close()
        conn.close()
    
    def count(self) -> int:
        conn = self.config.get_connection()
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{self.config.table}"')
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def close(self) -> None:
        pass


class PostgreSQLWriter(OutputWriter):
    """Write result records to PostgreSQL database."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        columns: List[str],
        **kwargs: Any,
    ) -> None:
        self.config = db_config
        self.columns = columns
        self._conn = self.config.get_connection()
        self._cursor = self._conn.cursor()
        self._create_table()
    
    def _create_table(self) -> None:
        columns_sql = ", ".join([f'"{col}" DOUBLE PRECISION' for col in self.columns])
        create_sql = f'''
            CREATE TABLE IF NOT EXISTS "{self.config.table}" (
                text_id TEXT PRIMARY KEY,
                {columns_sql}
            )
        '''
        self._cursor.execute(create_sql)
        if self._conn:
            self._conn.commit()
    
    def write_header(self, columns: List[str]) -> None:
        pass
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        quoted_columns = ", ".join([f'"{col}"' for col in self.columns])
        placeholders = ", ".join(["%s"] * (len(self.columns) + 1))
        insert_sql = f'''
            INSERT INTO "{self.config.table}" (text_id, {quoted_columns}) 
            VALUES ({placeholders})
            ON CONFLICT (text_id) DO UPDATE SET
            {", ".join([f'"{col}" = EXCLUDED."{col}"' for col in self.columns])}
        '''
        self._cursor.execute(insert_sql, [text_id] + values)
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        for text_id, values in records:
            self.write_record(text_id, values)
        if self._conn:
            self._conn.commit()
        return len(records)
    
    def close(self) -> None:
        if self._conn:
            self._conn.commit()
            self._cursor.close()
            self._conn.close()
            self._conn = None


# =============================================================================
# MongoDB
# =============================================================================

class MongoDBReader(InputReader):
    """Read text records from MongoDB collection."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        **kwargs: Any,
    ) -> None:
        self.config = db_config
        self._db = None
    
    def stream(self, lowercase: bool = True) -> Iterator[Tuple[str, Dict[str, str]]]:
        """Stream text records from MongoDB."""
        db = self.config.get_connection()
        collection = db[self.config.table]
        
        projection = {self.config.id_column: 1, self.config.text_column: 1}
        
        for doc in collection.find({}, projection):
            text_id = str(doc.get(self.config.id_column, doc.get("_id", "")))
            text = doc.get(self.config.text_column, "")
            if lowercase:
                text = text.lower()
            yield text, {"text_id": text_id}
    
    def count(self) -> int:
        db = self.config.get_connection()
        collection = db[self.config.table]
        return collection.count_documents({})
    
    def close(self) -> None:
        pass


class MongoDBWriter(OutputWriter):
    """Write result records to MongoDB collection."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        columns: List[str],
        **kwargs: Any,
    ) -> None:
        self.config = db_config
        self.columns = columns
        self._db = self.config.get_connection()
        self._collection = self._db[self.config.table]
    
    def write_header(self, columns: List[str]) -> None:
        # MongoDB is schemaless
        pass
    
    def write_record(self, text_id: str, values: List[float]) -> None:
        doc: Dict[str, Any] = {"_id": text_id, "text_id": text_id}
        for col, val in zip(self.columns, values):
            doc[col] = val
        self._collection.replace_one({"_id": text_id}, doc, upsert=True)
    
    def write_batch(self, records: List[Tuple[str, List[float]]]) -> int:
        from pymongo import UpdateOne  # type: ignore[import-not-found]
        
        operations = []
        for text_id, values in records:
            doc: Dict[str, Any] = {"_id": text_id, "text_id": text_id}
            for col, val in zip(self.columns, values):
                doc[col] = val
            operations.append(
                UpdateOne({"_id": text_id}, {"$set": doc}, upsert=True)
            )
        
        if operations:
            self._collection.bulk_write(operations)
        return len(records)
    
    def close(self) -> None:
        # MongoDB client handles connection pooling
        pass


# Register with factories
if AVAILABLE_DRIVERS.get("sqlite", False):
    ReaderFactory.register("sqlite", SQLiteReader)
    WriterFactory.register("sqlite", SQLiteWriter)

if AVAILABLE_DRIVERS.get("mysql", False):
    ReaderFactory.register("mysql", MySQLReader)
    WriterFactory.register("mysql", MySQLWriter)

if AVAILABLE_DRIVERS.get("postgresql", False):
    ReaderFactory.register("postgresql", PostgreSQLReader)
    WriterFactory.register("postgresql", PostgreSQLWriter)

if AVAILABLE_DRIVERS.get("mongodb", False):
    ReaderFactory.register("mongodb", MongoDBReader)
    WriterFactory.register("mongodb", MongoDBWriter)
