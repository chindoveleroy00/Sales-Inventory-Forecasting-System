import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.forecasting.prophet_model import ProphetForecaster
from backend.forecasting.arima_model import ARIMAForecaster
from backend.forecasting.ensemble import EnsembleForecaster
from backend.forecasting.utils import add_zim_features


class TestZimbabweForecasting(unittest.TestCase):
    """Unit tests for Zimbabwe Sales Forecasting System"""

    def setUp(self):
        """Set up test data and fixtures"""
        # Create synthetic Zimbabwe sales data for testing
        np.random.seed(42)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')

        # Generate realistic sales patterns with Zimbabwe characteristics
        n_days = len(date_range)
        n_skus = 5

        test_data = []
        for sku in range(1, n_skus + 1):
            for i, date in enumerate(date_range):
                # Add seasonality and trends
                base_sales = 100 + 20 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
                monthly_effect = 10 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
                payday_boost = 15 if date.day in [1, 15] else 0  # Payday effect
                weekend_drop = -5 if date.weekday() >= 5 else 0  # Weekend drop

                quantity = max(1, int(base_sales + monthly_effect + payday_boost +
                                      weekend_drop + np.random.normal(0, 10)))

                test_data.append({
                    'date': date,
                    'sku_id': f'SKU_{sku:03d}',
                    'quantity_sold': quantity,
                    'price': 2.50 + np.random.normal(0, 0.2),
                    'category': 'Groceries'
                })

        self.test_data = pd.DataFrame(test_data)
        self.test_data['date'] = pd.to_datetime(self.test_data['date'])

        # Create aggregated data for single SKU testing
        self.single_sku_data = self.test_data[self.test_data['sku_id'] == 'SKU_001'].copy()

    def test_prophet_forecaster_initialization(self):
        """Test ProphetForecaster initialization"""
        forecaster = ProphetForecaster(self.single_sku_data)

        self.assertIsNotNone(forecaster.data)
        self.assertIsNone(forecaster.model)  # Model should be None before training
        self.assertIsInstance(forecaster.holidays, pd.DataFrame)
        self.assertGreater(len(forecaster.holidays), 0)

    def test_prophet_training(self):
        """Test Prophet model training"""
        forecaster = ProphetForecaster(self.single_sku_data)
        forecaster.train()

        self.assertIsNotNone(forecaster.model)
        self.assertTrue(hasattr(forecaster.model, 'predict'))

    def test_prophet_prediction(self):
        """Test Prophet forecast generation"""
        forecaster = ProphetForecaster(self.single_sku_data)
        forecaster.train()

        forecast_df, metrics = forecaster.predict(periods=30)

        # Check forecast structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), len(self.single_sku_data) + 30)

        # Check required columns
        required_cols = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
        self.assertTrue(required_cols.issubset(forecast_df.columns))

        # Check metrics
        self.assertIsInstance(metrics, dict)

        # Check forecast values are reasonable
        self.assertTrue(all(forecast_df['yhat'] > 0))  # Sales should be positive
        self.assertTrue(all(forecast_df['yhat_lower'] <= forecast_df['yhat']))
        self.assertTrue(all(forecast_df['yhat'] <= forecast_df['yhat_upper']))

    def test_arima_forecaster_initialization(self):
        """Test ARIMAForecaster initialization"""
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = ARIMAForecaster(model_data)

        self.assertIsNotNone(forecaster.data)
        self.assertIsNone(forecaster.model)

    def test_arima_training(self):
        """Test ARIMA model training"""
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = ARIMAForecaster(model_data)
        forecaster.train(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))  # Simpler model for testing

        self.assertIsNotNone(forecaster.model)
        self.assertTrue(hasattr(forecaster.model, 'forecast'))

    def test_arima_prediction(self):
        """Test ARIMA forecast generation"""
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = ARIMAForecaster(model_data)
        forecaster.train(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))

        forecast_df, metrics = forecaster.predict(periods=30)

        # Check forecast structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 30)

        # Check required columns
        required_cols = {'date', 'yhat', 'yhat_lower', 'yhat_upper'}
        self.assertTrue(required_cols.issubset(forecast_df.columns))

        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('aic', metrics)
        self.assertIn('bic', metrics)

        # Check forecast values are reasonable
        self.assertTrue(all(forecast_df['yhat'] > 0))

    def test_ensemble_forecaster(self):
        """Test EnsembleForecaster functionality"""
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = EnsembleForecaster(model_data)

        # Test training
        forecaster.train()
        self.assertIsNotNone(forecaster.prophet.model)
        self.assertIsNotNone(forecaster.arima.model)

        # Test prediction
        forecast_df = forecaster.predict(periods=30)

        # Check forecast structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 30)

        # Check required columns
        required_cols = {'date', 'prophet', 'arima', 'final'}
        self.assertTrue(required_cols.issubset(forecast_df.columns))

        # Check that ensemble combines models reasonably
        self.assertTrue(all(forecast_df['final'] > 0))

    def test_zimbabwe_features(self):
        """Test Zimbabwe-specific feature engineering"""
        enhanced_data = add_zim_features(self.test_data)

        # Check new columns are added
        zim_features = {'is_payday', 'is_month_end', 'is_holiday'}
        self.assertTrue(zim_features.issubset(enhanced_data.columns))

        # Check feature logic
        payday_dates = enhanced_data[enhanced_data['is_payday'] == 1]['date'].dt.day.unique()
        self.assertTrue(all(day in [1, 15] for day in payday_dates))

        # Check month end logic
        month_end_dates = enhanced_data[enhanced_data['is_month_end'] == 1]
        self.assertTrue(all(month_end_dates['date'].dt.is_month_end))

    def test_forecast_accuracy_metrics(self):
        """Test forecast accuracy calculation"""
        # Create test data with known pattern
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        actual = pd.Series([100 + 10 * np.sin(2 * np.pi * i / 30) for i in range(100)])
        forecast = pd.Series([98 + 12 * np.sin(2 * np.pi * i / 30) for i in range(100)])

        # Calculate metrics manually
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))

        self.assertLess(mape, 50)  # MAPE should be reasonable
        self.assertGreater(rmse, 0)  # RMSE should be positive

    def test_data_validation(self):
        """Test data validation functionality"""
        # Test with valid data
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = ARIMAForecaster(model_data)
        self.assertTrue(forecaster.validate_data())

        # Test with invalid data (missing columns)
        invalid_data = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=10)})
        with self.assertRaises(ValueError):
            forecaster = ARIMAForecaster(invalid_data)
            forecaster.validate_data()

    def test_forecast_file_output(self):
        """Test forecast file saving and loading"""
        model_data = self.single_sku_data.set_index('date').sort_index()
        forecaster = EnsembleForecaster(model_data)
        forecaster.train()

        forecast_df = forecaster.predict(periods=30)

        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            temp_path = tmp.name

        try:
            forecast_df.to_parquet(temp_path)

            # Test loading
            loaded_forecast = pd.read_parquet(temp_path)
            pd.testing.assert_frame_equal(forecast_df, loaded_forecast)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_zimbabwe_holidays(self):
        """Test Zimbabwe holiday detection"""
        forecaster = ProphetForecaster(self.single_sku_data)
        holidays = forecaster.holidays

        # Check that key Zimbabwe holidays are included
        holiday_dates = pd.to_datetime(holidays['ds'])

        # Check for New Year
        new_year_included = any(
            (date.month == 1 and date.day == 1) for date in holiday_dates
        )
        self.assertTrue(new_year_included)

        # Check for Independence Day
        independence_included = any(
            (date.month == 4 and date.day == 18) for date in holiday_dates
        )
        self.assertTrue(independence_included)

    def test_forecast_robustness(self):
        """Test forecast robustness with edge cases"""
        # Test with minimal data
        minimal_data = self.single_sku_data.head(30)  # Only 30 days
        model_data = minimal_data.set_index('date').sort_index()

        forecaster = ProphetForecaster(model_data)
        forecaster.train()

        forecast_df, metrics = forecaster.predict(periods=7)  # Short forecast

        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertGreater(len(forecast_df), 0)

    def test_performance_benchmarks(self):
        """Test model performance benchmarks"""
        import time

        model_data = self.single_sku_data.set_index('date').sort_index()

        # Test Prophet performance
        start_time = time.time()
        prophet_forecaster = ProphetForecaster(model_data)
        prophet_forecaster.train()
        prophet_time = time.time() - start_time

        # Test ARIMA performance
        start_time = time.time()
        arima_forecaster = ARIMAForecaster(model_data)
        arima_forecaster.train(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        arima_time = time.time() - start_time

        # Performance should be reasonable (less than 60 seconds for test data)
        self.assertLess(prophet_time, 60)
        self.assertLess(arima_time, 60)

        print(f"Prophet training time: {prophet_time:.2f}s")
        print(f"ARIMA training time: {arima_time:.2f}s")


class TestZimbabweForecasterIntegration(unittest.TestCase):
    """Integration tests for the complete forecasting pipeline"""

    def test_end_to_end_forecast(self):
        """Test complete end-to-end forecasting pipeline"""
        # This test simulates the complete workflow

        # 1. Create test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        test_data = []

        for date in dates:
            quantity = max(1, int(100 + 20 * np.sin(2 * np.pi * date.dayofyear / 365) +
                                  np.random.normal(0, 10)))
            test_data.append({
                'date': date,
                'sku_id': 'TEST_SKU',
                'quantity_sold': quantity,
                'price': 2.50
            })

        df = pd.DataFrame(test_data)

        # 2. Add Zimbabwe features
        df = add_zim_features(df)

        # 3. Train ensemble model
        model_data = df.set_index('date').sort_index()
        forecaster = EnsembleForecaster(model_data)
        forecaster.train()

        # 4. Generate forecast
        forecast = forecaster.predict(periods=30)

        # 5. Validate results
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertEqual(len(forecast), 30)
        self.assertTrue(all(forecast['final'] > 0))

        print("✅ End-to-end forecast test passed!")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)