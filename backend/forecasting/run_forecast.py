import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
import yaml
import logging
import sys

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

class ForecastPipeline:
    """Main forecasting pipeline for SIFS"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
    
    def run(self, data: pd.DataFrame, sku_id: str = None) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """Run full forecasting pipeline"""
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # Prepare data
        processed_data = self._prepare_data(data, sku_id)
        
        # Train models
        for model in self.models.values():
            model.train(processed_data)
        
        # Generate forecasts
        results = {}
        for name, model in self.models.items():
            try:
                forecast, metrics = model.predict(periods=self.config['forecast_horizon'])
                results[name] = (forecast, metrics)
            except Exception as e:
                logger.error(f"Forecast failed for {name}: {e}")
                results[name] = (pd.DataFrame(), {})
        
        return results
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            # Try to find config file relative to this module
            current_dir = Path(__file__).parent
            full_config_path = current_dir / config_path
            
            if not full_config_path.exists():
                # Try relative to project root
                project_root = current_dir.parent.parent
                full_config_path = project_root / config_path
            
            with open(full_config_path) as f:
                config = yaml.safe_load(f)
            
            # Set defaults
            config.setdefault('forecast_horizon', 30)
            config.setdefault('critical_skus', [])
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {'forecast_horizon': 30, 'critical_skus': []}
    
    def _initialize_models(self) -> Dict:
        """Initialize forecasting models"""
        try:
            product_weights = get_zim_product_weights()
        except Exception as e:
            logger.warning(f"Failed to get product weights: {e}")
            product_weights = {'default': {'prophet': 0.6, 'arima': 0.4}}
        
        return {
            'arima': ARIMAForecaster(
                order=self.config.get('arima', {}).get('default_order', (1,1,1)),
                seasonal_order=self.config.get('arima', {}).get('seasonal_order', (1,1,1,7))
            ),
            'prophet': ProphetForecaster(
                growth='linear',
                seasonality_mode='multiplicative'
            ),
            'croston': CrostonTSB(
                alpha_d=0.1,
                alpha_p=0.1
            ),
            'ensemble': EnsembleForecaster(
                models=[
                    ARIMAForecaster(),
                    ProphetForecaster(),
                    CrostonTSB()
                ],
                weights=list(product_weights.get('default', {'prophet': 0.6, 'arima': 0.4}).values())
            )
        }
    
    def _prepare_data(self, data: pd.DataFrame, sku_id: str = None) -> pd.DataFrame:
        """Prepare data for forecasting"""
        # Filter for specific SKU if requested
        if sku_id:
            data = data[data['sku_id'] == sku_id].copy()
        
        # Ensure date column is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        return data


if __name__ == "__main__":
    print("Testing SIFS Forecasting Pipeline...")
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    base_sales = 10 + np.sin(np.arange(100) * 2 * np.pi / 7) * 3
    noise = np.random.normal(0, 2, 100)
    quantity_sold = np.maximum(0, base_sales + noise)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'sku_id': ['MEALIE_2KG'] * 100,
        'quantity_sold': quantity_sold
    })
    
    try:
        pipeline = ForecastPipeline()
        results = pipeline.run(sample_data)
        print(f"✓ Success! Generated forecasts with {len(results)} models")
        
        for model_name, (forecast, metrics) in results.items():
            print(f"  {model_name}: {len(forecast)} forecast points")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()