import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
import logging
import sys
import yaml # Added this import statement

# Try relative imports first, fall back to absolute if running directly
try:
    from .models.arima_model import ARIMAForecaster
    from .models.prophet_model import ProphetForecaster
    from .models.croston import CrostonTSB
    from .models.ensemble import EnsembleForecaster
    from .utils import get_zim_product_weights
except ImportError:
    # If relative imports fail, we're being run directly
    # Add project root to path and use absolute imports
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from backend.forecasting.models.arima_model import ARIMAForecaster
    from backend.forecasting.models.prophet_model import ProphetForecaster
    from backend.forecasting.models.croston import CrostonTSB
    from backend.forecasting.models.ensemble import EnsembleForecaster
    from backend.forecasting.utils import get_zim_product_weights

logger = logging.getLogger(__name__)

class SingleForecaster:
    """
    A simplified class to perform a single forecast for a given SKU
    using the existing forecasting models.
    """
    
    def __init__(self, model_type: str = 'ensemble', config_path: str = "config/default.yaml"):
        """
        Initializes the SingleForecaster with a specific model type.

        Args:
            model_type (str): The type of model to use for forecasting (e.g., 'arima', 'prophet', 'croston', 'ensemble').
            config_path (str): Path to the configuration file.
        """
        self.model_type = model_type.lower()
        self.config_path = config_path
        self.model = None
        self.config = self._load_config()
        self._initialize_model()

    def _load_config(self) -> Dict:
        """Load configuration file."""
        try:
            current_dir = Path(__file__).parent
            full_config_path = current_dir / self.config_path
            
            if not full_config_path.exists():
                project_root = current_dir.parent.parent
                full_config_path = project_root / self.config_path
            
            with open(full_config_path) as f:
                config = yaml.safe_load(f)
            
            config.setdefault('forecast_horizon', 30)
            config.setdefault('critical_skus', [])
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {'forecast_horizon': 30, 'critical_skus': []}

    def _initialize_model(self) -> None:
        """Initializes the specified forecasting model."""
        if self.model_type == 'arima':
            self.model = ARIMAForecaster(
                order=self.config.get('arima', {}).get('default_order', (1,1,1)),
                seasonal_order=self.config.get('arima', {}).get('seasonal_order', (1,1,1,7))
            )
        elif self.model_type == 'prophet':
            self.model = ProphetForecaster(
                growth='linear',
                seasonality_mode='multiplicative'
            )
        elif self.model_type == 'croston':
            self.model = CrostonTSB(
                alpha_d=0.1,
                alpha_p=0.1
            )
        elif self.model_type == 'ensemble':
            # For ensemble, we need to initialize its component models
            self.model = EnsembleForecaster(
                models=[
                    ARIMAForecaster(),
                    ProphetForecaster(),
                    CrostonTSB()
                ],
                # Weights can be dynamically adjusted during ensemble training,
                # or you can load them from config/product_weights if available.
                # For simplicity here, we'll let ensemble's internal logic handle initial weights.
                weights=None # Let ensemble determine initial weights
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Choose from 'arima', 'prophet', 'croston', 'ensemble'.")
        
        logger.info(f"Initialized {self.model_type} model.")

    def forecast(self, data: pd.DataFrame, sku_id: str, periods: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Generates a forecast for a specific SKU.

        Args:
            data (pd.DataFrame): The historical sales data.
            sku_id (str): The SKU ID for which to generate the forecast.
            periods (int, optional): The number of future periods to forecast. 
                                     Defaults to 'forecast_horizon' from config.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing the forecast DataFrame and metrics.
        """
        if data.empty:
            raise ValueError("Input data cannot be empty.")
        if not sku_id:
            raise ValueError("SKU ID must be provided for single prediction.")

        # Filter data for the specific SKU
        sku_data = data[data['sku_id'] == sku_id].copy()
        
        if sku_data.empty:
            raise ValueError(f"No data found for SKU: {sku_id}")

        # Ensure date column is datetime and sort
        sku_data['date'] = pd.to_datetime(sku_data['date'])
        sku_data = sku_data.sort_values('date')

        # Use periods from argument or config
        forecast_periods = periods if periods is not None else self.config['forecast_horizon']

        try:
            self.model.train(sku_data)
            forecast_df, metrics = self.model.predict(periods=forecast_periods)
            logger.info(f"Forecast generated successfully for SKU: {sku_id} using {self.model_type} model.")
            return forecast_df, metrics
        except Exception as e:
            logger.error(f"Failed to generate forecast for SKU {sku_id} using {self.model_type} model: {e}")
            raise RuntimeError(f"Single prediction failed for SKU {sku_id}: {str(e)}")


if __name__ == "__main__":
    # Configure logging for better visibility
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Running Single Prediction Test...")
    
    # Create sample data (similar to run_forecast.py)
    import numpy as np
    np.random.seed(42)
    
    dates_sku1 = pd.date_range('2025-07-18', periods=100, freq='D')
    base_sales_sku1 = 10 + np.sin(np.arange(100) * 2 * np.pi / 7) * 3
    noise_sku1 = np.random.normal(0, 2, 100)
    quantity_sold_sku1 = np.maximum(0, base_sales_sku1 + noise_sku1)
    
    dates_sku2 = pd.date_range('2025-07-18', periods=80, freq='D')
    base_sales_sku2 = 5 + np.cos(np.arange(80) * 2 * np.pi / 14) * 2
    noise_sku2 = np.random.normal(0, 1, 80)
    quantity_sold_sku2 = np.maximum(0, base_sales_sku2 + noise_sku2)

    sample_data = pd.DataFrame({
        'date': list(dates_sku1) + list(dates_sku2),
        'sku_id': ['MEALIE_2KG'] * 100 + ['COOKOIL_2LT'] * 80,
        'quantity_sold': list(quantity_sold_sku1) + list(quantity_sold_sku2)
    })

    # Example Usage:
    target_sku = 'MEALIE_2KG'
    forecast_periods = 30 # Forecast for 30 days

    # Test with Ensemble model
    print(f"\n--- Forecasting for {target_sku} using Ensemble model ---")
    try:
        forecaster = SingleForecaster(model_type='ensemble')
        forecast_df, metrics = forecaster.forecast(sample_data, target_sku, periods=forecast_periods)
        print(f"Forecast for {target_sku} (Ensemble):\n{forecast_df.head()}")
        print(f"Metrics for {target_sku} (Ensemble): {metrics}")
        print(f"Total forecast points: {len(forecast_df)}")
    except Exception as e:
        print(f"Error during Ensemble forecast: {e}")

    # Test with Prophet model
    print(f"\n--- Forecasting for {target_sku} using Prophet model ---")
    try:
        forecaster = SingleForecaster(model_type='prophet')
        forecast_df, metrics = forecaster.forecast(sample_data, target_sku, periods=forecast_periods)
        print(f"Forecast for {target_sku} (Prophet):\n{forecast_df.head()}")
        print(f"Metrics for {target_sku} (Prophet): {metrics}")
        print(f"Total forecast points: {len(forecast_df)}")
    except Exception as e:
        print(f"Error during Prophet forecast: {e}")

    # Test with ARIMA model
    print(f"\n--- Forecasting for {target_sku} using ARIMA model ---")
    try:
        forecaster = SingleForecaster(model_type='arima')
        forecast_df, metrics = forecaster.forecast(sample_data, target_sku, periods=forecast_periods)
        print(f"Forecast for {target_sku} (ARIMA):\n{forecast_df.head()}")
        print(f"Metrics for {target_sku} (ARIMA): {metrics}")
        print(f"Total forecast points: {len(forecast_df)}")
    except Exception as e:
        print(f"Error during ARIMA forecast: {e}")

    # Test with Croston model
    print(f"\n--- Forecasting for {target_sku} using Croston model ---")
    try:
        forecaster = SingleForecaster(model_type='croston')
        forecast_df, metrics = forecaster.forecast(sample_data, target_sku, periods=forecast_periods)
        print(f"Forecast for {target_sku} (Croston):\n{forecast_df.head()}")
        print(f"Metrics for {target_sku} (Croston): {metrics}")
        print(f"Total forecast points: {len(forecast_df)}")
    except Exception as e:
        print(f"Error during Croston forecast: {e}")

    # Example of forecasting for a different SKU
    target_sku_2 = 'COOKOIL_2LT'
    print(f"\n--- Forecasting for {target_sku_2} using Ensemble model ---")
    try:
        forecaster_sku2 = SingleForecaster(model_type='ensemble')
        forecast_df_sku2, metrics_sku2 = forecaster_sku2.forecast(sample_data, target_sku_2, periods=forecast_periods)
        print(f"Forecast for {target_sku_2} (Ensemble):\n{forecast_df_sku2.head()}")
        print(f"Metrics for {target_sku_2} (Ensemble): {metrics_sku2}")
        print(f"Total forecast points: {len(forecast_df_sku2)}")
    except Exception as e:
        print(f"Error during Ensemble forecast for {target_sku_2}: {e}")

