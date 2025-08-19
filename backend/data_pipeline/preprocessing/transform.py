import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Enhanced feature engineering for Zimbabwean retail"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.school_terms = {
            'Term1': (1, 4),  # Jan-Apr
            'Term2': (5, 8),  # May-Aug
            'Term3': (9, 12)  # Sep-Dec
        }

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge Zimbabwean economic indicators"""
        try:
            exchange_rates = pd.read_json(self.data_dir / "external/exchange_rates.json")

            # Convert the 'period' column in exchange_rates to Period type for proper merging
            exchange_rates['period'] = pd.to_datetime(exchange_rates['period']).dt.to_period('M')

            df = df.merge(
                exchange_rates,
                left_on=df['date'].dt.to_period('M'),
                right_on='period',
                how='left'
            )
            # Ensure 'price' and 'exchange_rate' are numeric before division
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['exchange_rate'] = pd.to_numeric(df['exchange_rate'], errors='coerce')

            df['price_usd'] = df['price'] / df['exchange_rate']
            # Fill NaN price_usd if exchange rate was missing for a period (optional, depending on desired behavior)
            # df['price_usd'] = df['price_usd'].fillna(df['price']) # Example: fall back to local price if conversion fails
        except Exception as e:
            logger.warning(f"Could not load exchange rates: {str(e)}")
        return df

    def _apply_common_feature_engineering_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies common feature engineering steps to a DataFrame."""
        df = df.copy()  # Ensure we're working on a copy

        # Ensure 'date' column is datetime type before accessing dt accessor
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        # 1. Time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

        # Time since last purchase for each SKU (requires sorting)
        df = df.sort_values(by=['sku_id', 'date'])
        df['time_since_last_sale'] = df.groupby('sku_id')['date'].diff().dt.days

        # 2. Zimbabwe-specific features
        df['school_term'] = pd.cut(
            df['date'].dt.month,
            bins=[0, 4, 8, 12],
            labels=list(self.school_terms.keys()),
            right=True,
            include_lowest=True
        )

        # 3. Demand patterns
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('sku_id')['quantity_sold'].shift(lag)

        # 4. Rolling stats (7-day window)
        df['rolling_avg'] = df.groupby('sku_id')['quantity_sold'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

        # 5. Macro features
        df = self._add_macro_features(df)

        # Create a unique identifier for daily SKU sales to handle potential duplicates for aggregation
        df['daily_sku_id_date'] = df['sku_id'].astype(str) + '_' + df['date'].dt.strftime('%Y%m%d')

        return df

    def engineer_features(self, cleaned_sales_path: Path) -> pd.DataFrame:
        """
        Loads cleaned sales data from a file and engineers relevant features.
        This method is kept for file-based processing.
        """
        if not cleaned_sales_path.exists():
            raise FileNotFoundError(f"Cleaned sales data not found at {cleaned_sales_path}")

        logger.info(f"Loading cleaned data from {cleaned_sales_path} for feature engineering.")
        df = pd.read_parquet(cleaned_sales_path)

        engineered_df = self._apply_common_feature_engineering_steps(df)
        logger.info("Feature engineering complete (from file).")
        return engineered_df

    def engineer_features_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features for sales data provided as a Pandas DataFrame.
        This method is for API integration.
        """
        logger.info("Starting feature engineering for in-memory DataFrame.")
        try:
            engineered_df = self._apply_common_feature_engineering_steps(df)
            logger.info("In-memory DataFrame feature engineering complete.")
            return engineered_df
        except Exception as e:
            logger.error(f"Feature engineering in-memory DataFrame failed: {str(e)}", exc_info=True)
            raise
