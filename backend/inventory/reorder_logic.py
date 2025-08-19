import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Import the SafetyStockCalculator from the same directory
try:
    from .safety_stock_calc import SafetyStockCalculator
except ImportError:
    # Fallback for direct execution or different module structure
    import sys
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent # SIFS_Ultimate/
    sys.path.insert(0, str(project_root))
    from backend.inventory.safety_stock_calc import SafetyStockCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReorderLogic:
    """
    Calculates reorder points and quantities based on forecasts, current inventory,
    safety stock, and supplier lead times/MOQs.
    """

    def __init__(self,
                 product_supplier_mapping_path: Path,
                 suppliers_path: Path):
        """
        Initializes the ReorderLogic with paths to supplier data.
        """
        self.product_supplier_mapping_path = product_supplier_mapping_path
        self.suppliers_path = suppliers_path
        self._supplier_moq_lead_time_data = self._load_supplier_moq_lead_time_data()
        self.safety_stock_calculator = SafetyStockCalculator(
            product_supplier_mapping_path=product_supplier_mapping_path,
            suppliers_path=suppliers_path
        )

    def _load_supplier_moq_lead_time_data(self) -> pd.DataFrame:
        """
        Loads and merges supplier data to get lead times and MOQs for primary suppliers.
        """
        try:
            # Load product-supplier mapping
            product_map_df = pd.read_csv(self.product_supplier_mapping_path)
            # Convert to native Python bool
            primary_product_map = product_map_df[product_map_df['is_primary_supplier'].astype(bool)].copy()

            # Load suppliers data
            suppliers_df = pd.read_csv(self.suppliers_path)

            # Merge to get lead_time_days and min_order_qty
            merged_data = pd.merge(
                primary_product_map,
                suppliers_df[['supplier_id', 'lead_time_days', 'min_order_qty']],
                on='supplier_id',
                how='left'
            )

            # Handle missing values
            merged_data['lead_time_days'] = merged_data['lead_time_days'].fillna(1)
            merged_data['min_order_qty'] = merged_data['min_order_qty'].fillna(1)

            # Ensure numeric types
            merged_data['lead_time_days'] = pd.to_numeric(merged_data['lead_time_days'], errors='coerce').fillna(1).astype(int)
            merged_data['min_order_qty'] = pd.to_numeric(merged_data['min_order_qty'], errors='coerce').fillna(1).astype(int)

            # Select relevant columns
            supplier_info = merged_data[
                ['sku_id', 'lead_time_days', 'min_order_qty', 'supplier_id', 'supplier_name']
            ].drop_duplicates(subset=['sku_id'])

            if supplier_info.empty:
                raise ValueError("No valid primary supplier lead time or MOQ data found after processing.")

            logger.info("Supplier MOQ and lead time data loaded successfully.")
            return supplier_info

        except FileNotFoundError as e:
            logger.error(f"Supplier data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading supplier MOQ/lead time data: {e}")
            raise

    def calculate_reorder_needs(
            self,
            current_inventory: pd.DataFrame,
            forecast_df: pd.DataFrame,
            historical_sales: pd.DataFrame,
            service_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calculates reorder recommendations for each SKU with proper type conversion.
        """
        if current_inventory.empty or forecast_df.empty or historical_sales.empty:
            logger.warning("One or more input DataFrames are empty. Cannot calculate reorder needs.")
            return pd.DataFrame()

        # 1. Calculate Safety Stock
        safety_stocks = self.safety_stock_calculator.calculate_safety_stock(
            historical_sales=historical_sales,
            service_level=service_level
        )
        if safety_stocks.empty:
            logger.error("Safety stock calculation returned empty. Cannot proceed with reorder logic.")
            return pd.DataFrame()

        # 2. Prepare historical sales for average daily demand calculation
        historical_sales['date'] = pd.to_datetime(historical_sales['date'])
        historical_sales['quantity_sold'] = pd.to_numeric(historical_sales['quantity_sold'], errors='coerce')
        historical_sales_cleaned = historical_sales.dropna(subset=['date', 'quantity_sold', 'sku_id'])

        # Calculate average daily demand
        daily_demand_summary = historical_sales_cleaned.groupby('sku_id')['quantity_sold'].mean().reset_index()
        daily_demand_summary.rename(columns={'quantity_sold': 'avg_daily_demand'}, inplace=True)
        daily_demand_summary['avg_daily_demand'] = daily_demand_summary['avg_daily_demand'].fillna(0)

        # 3. Merge all necessary data
        reorder_data = current_inventory.copy()
        reorder_data = pd.merge(reorder_data, safety_stocks, on='sku_id', how='left')
        reorder_data = pd.merge(reorder_data, self._supplier_moq_lead_time_data, on='sku_id', how='left')
        reorder_data = pd.merge(reorder_data, daily_demand_summary, on='sku_id', how='left')

        # Drop rows with missing data
        initial_rows = len(reorder_data)
        reorder_data = reorder_data.dropna(
            subset=['safety_stock_qty', 'lead_time_days', 'min_order_qty', 'avg_daily_demand'])
        if len(reorder_data) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(reorder_data)} SKUs due to missing data.")
            if reorder_data.empty:
                logger.warning("No SKUs remaining after dropping rows with missing data.")
                return pd.DataFrame()

        # Ensure numeric types for calculations
        numeric_cols = ['current_stock', 'safety_stock_qty', 'lead_time_days', 
                       'min_order_qty', 'avg_daily_demand']
        reorder_data[numeric_cols] = reorder_data[numeric_cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce').fillna(0))

        # 4. Calculate Reorder Point (ROP)
        reorder_data['reorder_point'] = (
            reorder_data['avg_daily_demand'] * reorder_data['lead_time_days']
        ) + reorder_data['safety_stock_qty']
        reorder_data['reorder_point'] = np.maximum(0, reorder_data['reorder_point']).astype(int)

        # 5. Determine if Reorder is Needed with native Python bool
        reorder_data['reorder_needed'] = (
            reorder_data['current_stock'] <= reorder_data['reorder_point']
        ).astype(bool)

        # 6. Calculate Reorder Quantity (ROQ)
        reorder_quantities = []
        for index, row in reorder_data.iterrows():
            if row['reorder_needed']:
                sku_id = row['sku_id']
                current_stock = row['current_stock']
                min_order_qty = row['min_order_qty']
                safety_stock_qty = row['safety_stock_qty']
                lead_time_days = row['lead_time_days']

                # Get forecasted demand for lead time period
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                sku_forecast_for_lead_time = forecast_df[
                    (forecast_df['sku_id'] == sku_id)
                ].head(int(lead_time_days))
                forecasted_demand_during_lead_time = sku_forecast_for_lead_time['yhat'].sum()

                # Calculate reorder quantity
                target_inventory_level = forecasted_demand_during_lead_time + safety_stock_qty
                gross_roq = target_inventory_level - current_stock
                final_roq = max(min_order_qty, np.ceil(gross_roq).astype(int))
                final_roq = int(np.maximum(0, final_roq))
                reorder_quantities.append(final_roq)
            else:
                reorder_quantities.append(0)

        reorder_data['reorder_quantity'] = reorder_quantities

        # 7. Prepare final output with proper type conversion
        final_cols = [
            'sku_id', 'current_stock', 'safety_stock_qty', 'lead_time_days',
            'min_order_qty', 'reorder_point', 'reorder_needed', 'reorder_quantity',
            'supplier_id', 'supplier_name'
        ]
        
        # Ensure all columns exist
        for col in final_cols:
            if col not in reorder_data.columns:
                reorder_data[col] = None

        # Convert to proper types
        result_df = reorder_data[final_cols].copy()
        result_df['reorder_needed'] = result_df['reorder_needed'].astype(bool)
        
        int_cols = ['safety_stock_qty', 'lead_time_days', 'min_order_qty', 
                   'reorder_point', 'reorder_quantity']
        result_df[int_cols] = result_df[int_cols].astype(int)
        
        float_cols = ['current_stock']
        result_df[float_cols] = result_df[float_cols].astype(float)

        logger.info("Reorder logic calculation complete with proper type conversion.")
        return result_df


# Example Usage (unchanged from original)
if __name__ == "__main__":
    logger.info("Running ReorderLogic Example Usage...")