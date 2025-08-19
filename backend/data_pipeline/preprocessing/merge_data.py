import pandas as pd
from pathlib import Path
import logging
from typing import Optional
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_with_inventory(sales_path: Path, inventory_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Merge sales data with inventory information
    Args:
        sales_path: Path to cleaned sales data
        inventory_path: Path to inventory data (data/raw/inventory/inventory_2025.csv)
        output_dir: Directory to save merged data
    Returns:
        Path to merged data file or None if failed
    """
    try:
        logger.info(f"Merging sales data with inventory...")
        logger.info(f"Sales data: {sales_path}")
        logger.info(f"Inventory data: {inventory_path}")

        # Load data
        sales = pd.read_parquet(sales_path)
        inventory = pd.read_csv(inventory_path)

        logger.info(f"Loaded sales data: {len(sales)} rows")
        logger.info(f"Loaded inventory data: {len(inventory)} rows")

        # Dynamically detect and convert all date-like columns in inventory
        date_cols = [col for col in inventory.columns if 'date' in col.lower()]
        for col in date_cols:
            inventory[col] = pd.to_datetime(inventory[col], dayfirst=True, errors='coerce')
            logger.info(f"Converted inventory column to datetime: {col}")

        # Merge on SKU_ID
        merged = pd.merge(
            sales,
            inventory,
            on='sku_id',
            how='left',
            suffixes=('', '_inv')
        )

        logger.info(f"Merged data: {len(merged)} rows")

        # Calculate days since last restock if applicable
        if 'last_restock_date' in merged.columns and 'date' in merged.columns:
            merged['days_since_restock'] = (merged['date'] - merged['last_restock_date']).dt.days
            logger.info("Calculated 'days_since_restock' column")

        # Save merged data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sales_inventory_merged_{datetime.now().strftime('%Y%m%d')}.parquet"
        merged.to_parquet(output_path)

        logger.info(f"Merged data saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        return None


def run_merge_pipeline():
    """End-to-end merge pipeline"""
    try:
        # Get the project root by going up from the script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent
        data_dir = project_root / "data"

        logger.info(f"Project root: {project_root}")
        logger.info(f"Data directory: {data_dir}")

        # Verify paths exist
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")

        # Find the most recent cleaned sales file
        cleaned_dir = data_dir / "processed/cleaned"
        if not cleaned_dir.exists():
            raise FileNotFoundError(f"Cleaned data directory not found at {cleaned_dir}")

        # Look for cleaned sales files
        cleaned_files = list(cleaned_dir.glob("cleaned_sales_*.parquet"))
        if not cleaned_files:
            raise FileNotFoundError(f"No cleaned sales files found in {cleaned_dir}")

        # Use the most recent file
        sales_path = max(cleaned_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using cleaned sales file: {sales_path}")

        # Inventory path
        inventory_path = data_dir / "raw/inventory/inventory_2025.csv"
        if not inventory_path.exists():
            raise FileNotFoundError(f"Inventory file not found at {inventory_path}")

        # Output directory
        output_dir = data_dir / "processed/features"

        # Execute merge
        merged_path = merge_with_inventory(sales_path, inventory_path, output_dir)

        if merged_path:
            logger.info(f"Merge pipeline completed successfully")
            return merged_path
        else:
            logger.error("Merge pipeline failed")
            return None

    except Exception as e:
        logger.error(f"Merge pipeline failed: {str(e)}")
        return None


if __name__ == "__main__":
    run_merge_pipeline()
