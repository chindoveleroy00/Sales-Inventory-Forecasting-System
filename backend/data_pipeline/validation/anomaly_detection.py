import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import List


class AnomalyDetector:
    def detect(self, df: pd.DataFrame, data_type: str) -> List[str]:
        anomalies = []

        if data_type == 'sales':
            anomalies.extend(self._detect_zim_sales_anomalies(df))

        return anomalies

    def _detect_zim_sales_anomalies(self, df: pd.DataFrame) -> List[str]:
        anomalies = []

        # 1. Holiday sales
        holiday_sales = df[df['is_holiday']]['quantity_sold'].mean()
        if holiday_sales < df['quantity_sold'].mean() * 1.5:
            anomalies.append("Abnormally low holiday sales")

        # 2. Mealie meal panic buying
        mealie_sales = df[df['sku_id'].str.contains('MEALIE')]
        if len(mealie_sales) > 0:
            spikes = mealie_sales[mealie_sales['quantity_sold'] > 3 * mealie_sales['quantity_sold'].median()]
            if not spikes.empty:
                anomalies.append(f"Mealie meal panic buying on {spikes['date'].dt.date.unique()}")

        # 3. Electricity token winter demand
        if 'ELECTRICITY_10' in df['sku_id'].unique():
            winter_sales = df[df['date'].dt.month.isin([5, 6, 7])]
            elec_sales = winter_sales[winter_sales['sku_id'] == 'ELECTRICITY_10']['quantity_sold'].sum()
            if elec_sales < winter_sales['quantity_sold'].mean() * 2:
                anomalies.append("Low electricity token sales in winter months")

        return anomalies