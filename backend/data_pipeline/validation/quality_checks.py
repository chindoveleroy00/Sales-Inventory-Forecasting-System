import pandas as pd
from typing import List


class DataQualityChecker:
    ZIM_CRITICAL_SKUS = ['MEALIE_2KG', 'COOKOIL_2LT', 'BREAD_LOAF']

    def run_checks(self, df: pd.DataFrame, data_type: str) -> List[str]:
        issues = []

        if data_type == 'sales':
            # Currency checks
            if (df['price'] > 100).any():
                issues.append("Prices exceed $100 USD - verify currency")

            # Month-end stock patterns
            if df['date'].dt.is_month_end.any():
                month_end_sales = df[df['date'].dt.is_month_end]['quantity_sold'].mean()
                if month_end_sales < df['quantity_sold'].mean() * 1.2:
                    issues.append("Low month-end sales - check payday replenishment")

            # Critical SKU availability
            for sku in self.ZIM_CRITICAL_SKUS:
                if sku in df['sku_id'].unique():
                    zero_sales = df[(df['sku_id'] == sku) & (df['quantity_sold'] == 0)]
                    if len(zero_sales) > 3:
                        issues.append(f"Multiple zero-sales days for {sku} - possible stockout")

        elif data_type == 'inventory':
            # Perishable goods
            perishables = df[df['sku_id'].isin(['BREAD_LOAF', 'MILK_1LT'])]
            if not perishables.empty:
                expired = perishables[pd.to_datetime(perishables['expiry_date']) < pd.Timestamp.now()]
                if not expired.empty:
                    issues.append(f"{len(expired)} perishable items expired")

            # Crisis items
            crisis_items = df[df['sku_id'].str.contains('COOKOIL|CANDLES|SALT')]
            if not crisis_items.empty:
                low_stock = crisis_items[crisis_items['current_stock'] < 5]
                if not low_stock.empty:
                    issues.append(f"Critical stock for {len(low_stock)} crisis-sensitive items")

        return issues