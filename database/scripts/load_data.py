import sys
from pathlib import Path
import sqlite3
import pandas as pd
import logging
from tqdm import tqdm
import json

# Fix imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.database import load_db_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    config = load_db_config()
    db_path = PROJECT_ROOT / "data" / config['database']
    return sqlite3.connect(db_path)


def load_products(data_dir: Path, conn: sqlite3.Connection):
    """Load product data from CSV into database"""
    try:
        file_path = data_dir / "raw" / "sales" / "sales_2025.csv"
        logger.info(f"Loading unique products from {file_path}")

        # Get unique products from sales data (since it has all SKUs)
        sales_df = pd.read_csv(file_path)
        unique_products = sales_df[['sku_id', 'sku_name', 'category']].drop_duplicates()

        # Load product config from JSON
        config_path = data_dir / "external" / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create product records with full details
        product_data = []
        for _, row in unique_products.iterrows():
            product_config = next(
                (p for p in config['products'] if p['sku_id'] == row['sku_id']),
                None
            )
            if product_config:
                product_data.append({
                    'sku_id': row['sku_id'],
                    'sku_name': row['sku_name'],
                    'category': row['category'],
                    'base_price': product_config['base_price'],
                    'expiry_days': product_config['expiry_days'],
                    'peak_months': json.dumps(product_config['seasonality']['peak_months']),
                    'peak_boost': product_config['seasonality']['peak_boost']
                })

        # Insert products
        with conn:
            conn.executemany(
                """INSERT OR IGNORE INTO products 
                (sku_id, sku_name, category, base_price, expiry_days, peak_months, peak_boost)
                VALUES (:sku_id, :sku_name, :category, :base_price, 
                        :expiry_days, :peak_months, :peak_boost)""",
                product_data
            )
        logger.info(f"Loaded {len(product_data)} products")

    except Exception as e:
        logger.error(f"Error loading products: {str(e)}")
        raise


def load_sales(data_dir: Path, conn: sqlite3.Connection):
    """Load sales data from CSV into database"""
    try:
        file_path = data_dir / "raw" / "sales" / "sales_2025.csv"
        logger.info(f"Loading sales data from {file_path}")

        df = pd.read_csv(file_path)

        # Insert sales in batches
        batch_size = 1000
        with conn:
            for i in tqdm(range(0, len(df), batch_size), desc="Loading sales"):
                batch = df.iloc[i:i + batch_size]
                conn.executemany(
                    """INSERT INTO sales 
                    (date, sku_id, quantity_sold, price, category, promotion_flag)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    batch[['date', 'sku_id', 'quantity_sold', 'price', 'category', 'promotion_flag']].values.tolist()
                )
        logger.info(f"Loaded {len(df)} sales records")

    except Exception as e:
        logger.error(f"Error loading sales: {str(e)}")
        raise


def load_inventory(data_dir: Path, conn: sqlite3.Connection):
    """Load inventory data from CSV into database"""
    try:
        file_path = data_dir / "raw" / "inventory" / "inventory_2025.csv"
        logger.info(f"Loading inventory data from {file_path}")

        df = pd.read_csv(file_path)

        with conn:
            conn.executemany(
                """INSERT OR REPLACE INTO inventory
                (sku_id, current_stock, min_stock_level, supplier_lead_time, 
                 expiry_date, last_restock_date)
                VALUES (?, ?, ?, ?, ?, ?)""",
                df[['sku_id', 'current_stock', 'min_stock_level',
                    'supplier_lead_time', 'expiry_date', 'last_restock_date']].values.tolist()
            )
        logger.info(f"Loaded {len(df)} inventory records")

    except Exception as e:
        logger.error(f"Error loading inventory: {str(e)}")
        raise


def load_suppliers(data_dir: Path, conn: sqlite3.Connection):
    """Load supplier data from CSV into database"""
    try:
        file_path = data_dir / "raw" / "suppliers" / "suppliers.csv"
        logger.info(f"Loading suppliers from {file_path}")

        df = pd.read_csv(file_path)

        with conn:
            conn.executemany(
                """INSERT OR IGNORE INTO suppliers
                (supplier_id, name, contact, lead_time_days, min_order_qty, location)
                VALUES (?, ?, ?, ?, ?, ?)""",
                df[['supplier_id', 'name', 'contact',
                    'lead_time_days', 'min_order_qty', 'location']].values.tolist()
            )
        logger.info(f"Loaded {len(df)} suppliers")

    except Exception as e:
        logger.error(f"Error loading suppliers: {str(e)}")
        raise


def load_product_supplier_mappings(data_dir: Path, conn: sqlite3.Connection):
    """Load product-supplier mappings from CSV into database"""
    try:
        file_path = data_dir / "raw" / "suppliers" / "product_supplier_mapping.csv"
        logger.info(f"Loading product-supplier mappings from {file_path}")

        df = pd.read_csv(file_path)

        with conn:
            conn.executemany(
                """INSERT OR IGNORE INTO product_supplier_mapping
                (sku_id, supplier_id, is_primary_supplier, 
                 contract_price_multiplier, payment_terms)
                VALUES (?, ?, ?, ?, ?)""",
                df[['sku_id', 'supplier_id', 'is_primary_supplier',
                    'contract_price_multiplier', 'payment_terms']].values.tolist()
            )
        logger.info(f"Loaded {len(df)} product-supplier mappings")

    except Exception as e:
        logger.error(f"Error loading mappings: {str(e)}")
        raise


def load_events(data_dir: Path, conn: sqlite3.Connection):
    """Load events data from CSV into database"""
    try:
        file_path = data_dir / "external" / "events.csv"
        logger.info(f"Loading events from {file_path}")

        df = pd.read_csv(file_path)

        with conn:
            conn.executemany(
                """INSERT OR IGNORE INTO events
                (date, name, impact)
                VALUES (?, ?, ?)""",
                df[['date', 'name', 'impact']].values.tolist()
            )
        logger.info(f"Loaded {len(df)} events")

    except Exception as e:
        logger.error(f"Error loading events: {str(e)}")
        raise


def load_all_data():
    """Load all data into the database"""
    data_dir = PROJECT_ROOT / "data"
    conn = get_db_connection()

    try:
        # Load in proper order to respect foreign keys
        load_products(data_dir, conn)
        load_sales(data_dir, conn)
        load_inventory(data_dir, conn)
        load_suppliers(data_dir, conn)
        load_product_supplier_mappings(data_dir, conn)
        load_events(data_dir, conn)

        logger.info("All data loaded successfully")
    finally:
        conn.close()


if __name__ == "__main__":
    load_all_data()