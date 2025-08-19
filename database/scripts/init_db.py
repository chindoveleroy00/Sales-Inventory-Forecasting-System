import sqlite3
import logging
from pathlib import Path
import sys
import os
import json
from werkzeug.security import generate_password_hash

# Fix import path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import your config
from config.database import load_db_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





def initialize_database():
    config = load_db_config()
    db_path = project_root / "data" / config['database']

    # Create data directory if it doesn't exist
    db_path.parent.mkdir(exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")

        # Create products table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            sku_id TEXT PRIMARY KEY,
            sku_name TEXT NOT NULL,
            category TEXT NOT NULL,
            base_price REAL NOT NULL,
            expiry_days INTEGER,
            peak_months TEXT,  -- JSON array stored as text
            peak_boost REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create sales table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            sku_id TEXT NOT NULL,
            quantity_sold INTEGER NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            promotion_flag INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sku_id) REFERENCES products(sku_id)
        )
        """)

        # Create inventory table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT NOT NULL,
            current_stock INTEGER NOT NULL,
            min_stock_level INTEGER NOT NULL,
            supplier_lead_time INTEGER NOT NULL,
            expiry_date TEXT,
            last_restock_date TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sku_id) REFERENCES products(sku_id)
        )
        """)

        # Create suppliers table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            contact TEXT NOT NULL,
            lead_time_days INTEGER NOT NULL,
            min_order_qty INTEGER NOT NULL,
            location TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create product-supplier mapping table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS product_supplier_mapping (
            mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku_id TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            is_primary_supplier INTEGER DEFAULT 1,
            contract_price_multiplier REAL NOT NULL,
            payment_terms TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sku_id) REFERENCES products(sku_id),
            FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
            UNIQUE(sku_id, supplier_id)
        )
        """)

        # Create events table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            name TEXT NOT NULL,
            impact REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            email TEXT UNIQUE,
            is_active BOOLEAN DEFAULT 1,
            role TEXT DEFAULT 'dataentry'
        )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales(sku_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mapping_sku ON product_supplier_mapping(sku_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mapping_supplier ON product_supplier_mapping(supplier_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        conn.commit()
        logger.info("SQLite database initialized successfully with all tables")

        # Insert sample Zimbabwean holidays if events table is empty
        cursor.execute("SELECT COUNT(*) FROM events")
        if cursor.fetchone()[0] == 0:
            zimbabwe_holidays = [
                ("2025-01-01", "New Year", 1.5),
                ("2025-04-18", "Independence Day", 1.8),
                ("2025-05-25", "Africa Day", 1.3),
                ("2025-08-12", "Heroes' Day", 1.2),
                ("2025-12-22", "Unity Day", 1.4),
                ("2025-12-25", "Christmas", 2.0)
            ]
            cursor.executemany(
                "INSERT INTO events (date, name, impact) VALUES (?, ?, ?)",
                zimbabwe_holidays
            )
            conn.commit()
            logger.info("Added Zimbabwean holidays to events table")

        # Insert initial admin user if users table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            hashed_password = generate_password_hash("adminpassword")  # Change this password

            cursor.execute(
                "INSERT INTO users (username, hashed_password, full_name, email, is_active, role) VALUES (?, ?, ?, ?, ?, ?)",
                ("admin@sifs.com", hashed_password, "Admin User", "admin@sifs.com", 1, "admin")
            )
            conn.commit()
            logger.info("Added initial admin user to users table")

    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    initialize_database()