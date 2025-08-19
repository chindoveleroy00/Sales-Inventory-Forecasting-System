# config/database.py
import sqlite3
import yaml
from pathlib import Path


def load_db_config():
    config_path = Path(__file__).parent / "database.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['sqlite']


def get_db_connection():
    config = load_db_config()
    db_path = Path(__file__).parent.parent / "data" / config['database']

    # Ensure data directory exists
    db_path.parent.mkdir(exist_ok=True)

    return sqlite3.connect(
        db_path,
        timeout=config['timeout'],
        detect_types=config['detect_types'],
        isolation_level=config['isolation_level']
    )

DATABASE_URL = f"sqlite:///{(Path(__file__).parent.parent / 'data' / load_db_config()['database']).absolute()}"