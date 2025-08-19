import mysql.connector
from mysql.connector import pooling
import yaml
from pathlib import Path
from contextlib import contextmanager

class DatabaseManager:
    _connection_pool = None
    
    @classmethod
    def initialize_pool(cls):
        config_path = Path(__file__).parent.parent.parent / 'config' / 'database.yaml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['mysql']
        
        cls._connection_pool = pooling.MySQLConnectionPool(
            pool_name="sifs_pool",
            pool_size=db_config['pool_size'],
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            charset=db_config['charset'],
            autocommit=db_config['autocommit']
        )
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        if cls._connection_pool is None:
            cls.initialize_pool()
            
        connection = cls._connection_pool.get_connection()
        try:
            yield connection
        finally:
            connection.close()

def execute_query(query, params=None, fetch=False):
    with DatabaseManager.get_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise e
        finally:
            cursor.close()

def bulk_insert(table, data):
    if not data:
        return
        
    columns = ', '.join(data[0].keys())
    placeholders = ', '.join(['%s'] * len(data[0]))
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    
    with DatabaseManager.get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.executemany(query, [tuple(item.values()) for item in data])
            connection.commit()
            return cursor.rowcount
        except Exception as e:
            connection.rollback()
            raise e
        finally:
            cursor.close()