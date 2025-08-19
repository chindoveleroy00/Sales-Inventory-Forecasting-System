import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
from preprocessing.clean_sales import SalesPreprocessor
from preprocessing.transform import FeatureEngineer
from validation.validate import validate_data_pipeline
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sales_processor = SalesPreprocessor(data_dir)
        self.feature_engineer = FeatureEngineer(data_dir)

    def clean_and_transform(self, strict_validation: bool = False) -> Optional[Dict[str, Path]]:
        """Orchestrate full cleaning pipeline with Zimbabwean rules"""
        try:
            # 1. Clean raw sales data
            raw_sales_path = self.data_dir / "raw/sales/sales_2025.csv"
            if not raw_sales_path.exists():
                raise FileNotFoundError(f"Sales data not found at {raw_sales_path}")

            cleaned_df = self.sales_processor.clean(raw_sales_path)
            cleaned_path = self._save_data(cleaned_df, "cleaned", "cleaned_sales")

            # 2. Feature engineering with local seasonality
            features = self.feature_engineer.engineer_features(cleaned_path)
            features_path = self._save_data(features, "features", "modeling_features")

            # 3. Validate with Zimbabwean business rules
            output_paths = {
                'cleaned_sales': cleaned_path,
                'features': features_path
            }

            if not validate_data_pipeline(output_paths, strict=strict_validation):
                if strict_validation:
                    raise ValueError("Strict validation failed")
                logger.warning("Proceeding with validation warnings")

            return output_paths

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return None

    def _save_data(self, df: pd.DataFrame, subdir: str, prefix: str) -> Path:
        """Save DataFrame to parquet with timestamp"""
        output_dir = self.data_dir / "processed" / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(output_path)
        logger.info(f"Saved {prefix} data to {output_path}")
        return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', help='Enable strict validation mode')
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent.parent / "data"
    cleaner = DataCleaner(data_dir)
    result = cleaner.clean_and_transform(strict_validation=args.strict)

    if not result:
        exit(1)