import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import logging
from scipy.stats import norm  # For Z-score calculation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SafetyStockCalculator:
    """
    Calculates safety stock levels for SKUs based on historical demand variability
    and supplier lead times.
    """

    def __init__(self,
                 product_supplier_mapping_path: Path,
                 suppliers_path: Path):
        """
        Initializes the SafetyStockCalculator.

        Args:
            product_supplier_mapping_path (Path): Path to the CSV file containing
                                                  SKU-to-supplier mapping.
            suppliers_path (Path): Path to the CSV file containing supplier details
                                   like lead times.
        """
        self.product_supplier_mapping_path = product_supplier_mapping_path
        self.suppliers_path = suppliers_path
        self._supplier_data = self._load_supplier_data()

    def _load_supplier_data(self) -> pd.DataFrame:
        """
        Loads and merges supplier data to get lead times for primary suppliers.
        """
        try:
            # Load product-supplier mapping
            product_map_df = pd.read_csv(self.product_supplier_mapping_path)
            # Filter for primary suppliers only
            primary_product_map = product_map_df[product_map_df['is_primary_supplier'] == True].copy()

            # Load suppliers data
            suppliers_df = pd.read_csv(self.suppliers_path)

            # Merge to get lead_time_days for each primary SKU-supplier pair
            merged_data = pd.merge(
                primary_product_map,
                suppliers_df[['supplier_id', 'lead_time_days']],
                on='supplier_id',
                how='left'
            )
            # Handle cases where lead_time_days might be missing after merge
            if merged_data['lead_time_days'].isnull().any():
                logger.warning("Some primary SKUs have missing lead_time_days after merging. Filling with 1 day.")
                merged_data['lead_time_days'] = merged_data['lead_time_days'].fillna(1)  # Default to 1 day if missing

            # Ensure lead_time_days is numeric
            merged_data['lead_time_days'] = pd.to_numeric(merged_data['lead_time_days'], errors='coerce')
            merged_data = merged_data.dropna(subset=['lead_time_days'])
            merged_data['lead_time_days'] = merged_data['lead_time_days'].astype(int)

            # Select relevant columns: sku_id and lead_time_days
            supplier_lead_times = merged_data[['sku_id', 'lead_time_days']].drop_duplicates(subset=['sku_id'])

            if supplier_lead_times.empty:
                raise ValueError("No valid primary supplier lead time data found after processing.")

            logger.info("Supplier lead time data loaded successfully.")
            return supplier_lead_times

        except FileNotFoundError as e:
            logger.error(f"Supplier data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading supplier data: {e}")
            raise

    def calculate_safety_stock(
            self,
            historical_sales: pd.DataFrame,
            service_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calculates the safety stock for each SKU.

        Formula used: Safety Stock = Z * sqrt(Lead Time) * Std Dev of Daily Demand

        Args:
            historical_sales (pd.DataFrame): DataFrame with historical sales data,
                                             must contain 'date', 'sku_id', and 'quantity_sold'.
            service_level (float): Desired service level (e.g., 0.95 for 95%).
                                   Used to determine the Z-score.

        Returns:
            pd.DataFrame: A DataFrame with 'sku_id' and 'safety_stock_qty'.
        """
        if historical_sales.empty:
            logger.warning("Historical sales data is empty. Cannot calculate safety stock.")
            return pd.DataFrame(columns=['sku_id', 'safety_stock_qty'])

        # Ensure date is datetime and quantity_sold is numeric
        historical_sales['date'] = pd.to_datetime(historical_sales['date'])
        historical_sales['quantity_sold'] = pd.to_numeric(historical_sales['quantity_sold'], errors='coerce')
        historical_sales = historical_sales.dropna(subset=['date', 'quantity_sold', 'sku_id'])

        if historical_sales.empty:
            logger.warning("Historical sales data is empty after cleaning. Cannot calculate safety stock.")
            return pd.DataFrame(columns=['sku_id', 'safety_stock_qty'])

        # Calculate daily demand (sum quantity_sold per SKU per day)
        daily_demand = historical_sales.groupby(['date', 'sku_id'])['quantity_sold'].sum().reset_index()

        # Calculate standard deviation of daily demand for each SKU
        std_dev_daily_demand = daily_demand.groupby('sku_id')['quantity_sold'].std().reset_index()
        std_dev_daily_demand.rename(columns={'quantity_sold': 'std_dev_demand'}, inplace=True)

        # Fill NaN std_dev (e.g., for SKUs with only one sales record) with 0
        std_dev_daily_demand['std_dev_demand'] = std_dev_daily_demand['std_dev_demand'].fillna(0)

        # Get the Z-score for the desired service level
        # norm.ppf(service_level) gives the inverse of the cumulative distribution function
        z_score = norm.ppf(service_level)
        logger.info(f"Using Z-score of {z_score:.2f} for service level {service_level * 100:.0f}%")

        # Merge with lead times
        # We need to ensure that _supplier_data is not empty
        if self._supplier_data.empty:
            logger.error("No supplier lead time data available. Cannot calculate safety stock.")
            return pd.DataFrame(columns=['sku_id', 'safety_stock_qty'])

        merged_data = pd.merge(
            std_dev_daily_demand,
            self._supplier_data,
            on='sku_id',
            how='left'
        )

        # Handle SKUs that don't have lead time data (e.g., fill with a default lead time or drop)
        if merged_data['lead_time_days'].isnull().any():
            logger.warning(
                "Some SKUs do not have lead time data. Filling their lead_time_days with 1 for safety stock calculation.")
            merged_data['lead_time_days'] = merged_data['lead_time_days'].fillna(1)

        # Ensure lead_time_days is numeric and positive
        merged_data['lead_time_days'] = pd.to_numeric(merged_data['lead_time_days'], errors='coerce').fillna(1)
        merged_data['lead_time_days'] = np.maximum(1, merged_data['lead_time_days'])  # Ensure at least 1 day lead time

        # Calculate safety stock
        # Safety Stock = Z * sqrt(Lead Time) * Std Dev of Daily Demand
        merged_data['safety_stock_qty'] = (
                z_score * np.sqrt(merged_data['lead_time_days']) * merged_data['std_dev_demand']
        )

        # Ensure safety stock is non-negative and round up to nearest integer
        merged_data['safety_stock_qty'] = np.ceil(np.maximum(0, merged_data['safety_stock_qty'])).astype(int)

        logger.info("Safety stock calculation complete.")
        return merged_data[['sku_id', 'safety_stock_qty']]


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Running SafetyStockCalculator Example Usage...")

    # FIXED PATH CALCULATION
    # Get the current file's directory (backend/inventory/)
    current_file_dir = Path(__file__).parent
    # Go up two levels to get to SIFS_Ultimate/
    project_root = current_file_dir.parent.parent  # SIFS_Ultimate/

    # Debug: Print the calculated paths to verify they're correct
    logger.info(f"Current file directory: {current_file_dir}")
    logger.info(f"Project root: {project_root}")

    product_supplier_mapping_path = project_root / 'data' / 'raw' / 'suppliers' / 'product_supplier_mapping.csv'
    suppliers_path = project_root / 'data' / 'raw' / 'suppliers' / 'suppliers.csv'

    # Debug: Print the full paths to verify they're correct
    logger.info(f"Looking for product mapping at: {product_supplier_mapping_path}")
    logger.info(f"Looking for suppliers at: {suppliers_path}")

    # Check if files exist before proceeding
    if not product_supplier_mapping_path.exists():
        logger.error(f"Product supplier mapping file not found: {product_supplier_mapping_path}")
        logger.info("Please ensure the file exists at the expected location.")
        exit(1)

    if not suppliers_path.exists():
        logger.error(f"Suppliers file not found: {suppliers_path}")
        logger.info("Please ensure the file exists at the expected location.")
        exit(1)

    # Create a mock historical sales DataFrame
    # This would typically come from your data preprocessing pipeline
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    skus = ['MEALIE_2KG', 'COOKOIL_2LT', 'SUGAR_2KG', 'BREAD_LOAF', 'SOAP_100G']

    # Generate more realistic sales data with some variability
    sales_data = []
    for sku in skus:
        # Base demand for each SKU
        if sku == 'MEALIE_2KG':
            base_demand = 15
        elif sku == 'COOKOIL_2LT':
            base_demand = 8
        elif sku == 'SUGAR_2KG':
            base_demand = 12
        elif sku == 'BREAD_LOAF':
            base_demand = 20  # Higher frequency
        else:
            base_demand = 5  # Lower frequency

        # Add seasonality and noise
        daily_sales = (
                base_demand
                + np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * (base_demand * 0.2)  # Weekly seasonality
                + np.random.normal(0, base_demand * 0.1, len(dates))  # Noise
        ).clip(min=0).astype(int)  # Ensure non-negative and integer quantities

        sales_data.append(pd.DataFrame({
            'date': dates,
            'sku_id': sku,
            'quantity_sold': daily_sales
        }))

    mock_historical_sales = pd.concat(sales_data, ignore_index=True)

    # Add some intermittent sales for a SKU to test robustness (e.g., for Croston-like behavior)
    intermittent_sku = 'SANITARY_10PK'
    intermittent_dates = pd.date_range('2024-01-01', periods=100, freq='D')
    intermittent_sales = np.zeros(100, dtype=int)
    # Simulate sales on ~10% of days
    sale_days = np.random.choice(100, size=10, replace=False)
    intermittent_sales[sale_days] = np.random.randint(1, 5, size=10)  # Small quantities

    mock_historical_sales = pd.concat([
        mock_historical_sales,
        pd.DataFrame({
            'date': intermittent_dates,
            'sku_id': intermittent_sku,
            'quantity_sold': intermittent_sales
        })
    ], ignore_index=True)

    try:
        # Initialize the calculator
        calculator = SafetyStockCalculator(
            product_supplier_mapping_path=product_supplier_mapping_path,
            suppliers_path=suppliers_path
        )

        # Calculate safety stock for a 95% service level
        safety_stocks = calculator.calculate_safety_stock(
            historical_sales=mock_historical_sales,
            service_level=0.95
        )

        print("\nCalculated Safety Stock Levels (95% Service Level):")
        print(safety_stocks)

        # Calculate safety stock for a 90% service level
        safety_stocks_90 = calculator.calculate_safety_stock(
            historical_sales=mock_historical_sales,
            service_level=0.90
        )
        print("\nCalculated Safety Stock Levels (90% Service Level):")
        print(safety_stocks_90)

    except Exception as e:
        logger.error(f"An error occurred during safety stock calculation: {e}", exc_info=True)

    logger.info("SafetyStockCalculator Example Usage Complete.")