import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import holidays
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesPreprocessor:
    """Enhanced cleaning for Zimbabwean SME sales data"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.zw_holidays = holidays.Zimbabwe(years=[2025])
        self.sku_aliases = self._load_sku_aliases()

    def _load_sku_aliases(self) -> dict:
        """Load SKU normalization rules from config"""
        alias_path = self.data_dir / "external/sku_aliases.json"
        if alias_path.exists():
            return pd.read_json(alias_path, typ='series').to_dict()
        return {
            '2lt oil': 'COOKOIL_2LT',
            'mazoe 1l': 'MAZOE_1LT',
            'mealie 2': 'MEALIE_2KG'
        }

    def _handle_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Zimbabwean dollar values (handles hyperinflation and 'M' for million)"""
        if 'price' not in df.columns:
            logger.warning("Price column not found in DataFrame.")
            return df

        df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning

        # Ensure 'price' column is string type for string operations
        df['price'] = df['price'].astype(str)

        # Function to clean and convert price values, handling 'M' for million
        def safe_convert_price(price_str_val):
            price_str_val = str(price_str_val).strip().upper().replace(",", "")
            if 'M' in price_str_val and len(price_str_val) > 1:
                try:
                    # Remove 'M' and convert, then multiply by a million
                    return float(price_str_val.replace('M', '')) * 1_000_000
                except ValueError:
                    return np.nan  # Return NaN if conversion fails
            else:
                try:
                    # Just convert to float
                    return float(price_str_val)
                except ValueError:
                    return np.nan  # Return NaN if conversion fails

        # Apply the safe conversion function to the 'price' column
        df['price'] = df['price'].apply(safe_convert_price)

        # Remove rows where price conversion resulted in NaN, or handle as per your data policy
        if df['price'].isna().sum() > 0:
            logger.warning(f"Coerced {df['price'].isna().sum()} price values to NaN due to non-numeric content.")
            # Option 1: Drop rows with NaN prices (uncomment if this is desired behavior)
            # df = df.dropna(subset=['price'])
            # Option 2: Fill NaN prices with a default (e.g., 0, mean, or median)
            df['price'] = df['price'].fillna(0) # Example: fill with 0

        return df

    def _apply_common_cleaning_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies common cleaning and feature addition steps to a DataFrame."""
        df = df.copy()  # Ensure we're working on a copy

        # Currency normalization
        df = self._handle_currency(df)

        # SKU standardization
        # Ensure SKU is string before .str accessor
        df['sku_id'] = df['sku_id'].astype(str).str.lower().replace(self.sku_aliases)

        # Zimbabwe contextual features
        # Ensure 'date' is datetime type before accessing dt accessor
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['date'])  # Drop rows where date conversion failed

        df['is_payday'] = (df['date'].dt.day > 25) & (df['date'].dt.weekday == 4)
        df['is_border_day'] = df['date'].dt.day.isin([1, 15, 25])
        df['is_public_holiday'] = df['date'].isin(self.zw_holidays)

        # Validate and ensure quantity_sold is numeric
        if df['quantity_sold'].isna().sum() > 0:
            logger.warning(f"Dropping {df['quantity_sold'].isna().sum()} rows with missing quantities")
            df = df.dropna(subset=['quantity_sold'])

        df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce')
        df = df.dropna(subset=['quantity_sold'])  # Drop NaNs introduced by coercion

        return df

    def clean(self, sales_file_path: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Main cleaning orchestrator for sales data from a file.
        This method is kept for file-based processing.
        """
        try:
            logger.info(f"Starting sales data cleaning for {sales_file_path}")
            df = pd.read_csv(
                sales_file_path,
                parse_dates=['date'],
                dayfirst=True  # Zimbabwean date format dd/mm/yyyy
            )

            df = self._apply_common_cleaning_steps(df)

            if output_dir:
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"cleaned_sales_{datetime.now().strftime('%Y%m%d')}.parquet"
                df.to_parquet(output_path)
                logger.info(f"Saved cleaned data to {output_path}")

            return df

        except Exception as e:
            logger.error(f"Cleaning from file failed: {str(e)}", exc_info=True)
            raise  # Re-raise the exception after logging

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans sales data provided as a Pandas DataFrame.
        This method is for API integration.
        """
        logger.info("Starting sales data cleaning for in-memory DataFrame.")
        try:
            # Ensure 'date' column is handled correctly if it comes as string from API
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['date'])  # Drop rows where date conversion failed

            cleaned_df = self._apply_common_cleaning_steps(df)
            logger.info("In-memory DataFrame cleaning complete.")
            return cleaned_df
        except Exception as e:
            logger.error(f"Cleaning in-memory DataFrame failed: {str(e)}", exc_info=True)
            raise
