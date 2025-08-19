# database/scripts/backup_db.py
import shutil
from pathlib import Path
import logging
from datetime import datetime
from config.database import load_db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backup_database():
    config = load_db_config()
    db_path = Path(__file__).parent.parent.parent / "data" / config['database']
    backup_dir = Path(__file__).parent.parent.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"sifs_backup_{timestamp}.db"

    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

        # Keep only last 7 backups
        backups = sorted(backup_dir.glob("*.db"))
        for old_backup in backups[:-7]:
            old_backup.unlink()
            logger.info(f"Removed old backup: {old_backup}")

    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        raise


if __name__ == "__main__":
    backup_database()