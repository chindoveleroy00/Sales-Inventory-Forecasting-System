import pandas as pd
from typing import Dict, List
from validation.schema_check import SchemaValidator
from validation.quality_checks import DataQualityChecker
from validation.anomaly_detection import AnomalyDetector
from validation.policy_check import PolicyValidator
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, strict_mode: bool = False):
        self.schema_check = SchemaValidator()
        self.quality_check = DataQualityChecker()
        self.anomaly_detector = AnomalyDetector()
        self.policy_check = PolicyValidator()
        self.strict_mode = strict_mode

    def validate(self, df: pd.DataFrame, data_type: str) -> Dict[str, List[str]]:
        """Run all validation checks with Zimbabwean context"""
        report = {
            'schema_errors': self._validate_schema(df, data_type),
            'quality_issues': self.quality_check.run_checks(df, data_type),
            'anomalies': self.anomaly_detector.detect(df, data_type),
            'policy_violations': self.policy_check.check_price_controls(df)
        }

        if self.strict_mode and any(report.values()):
            logger.error("Strict validation failed - aborting pipeline")

        return report

    def _validate_schema(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """Wrapper with error handling for schema validation"""
        try:
            return self.schema_check.validate(df, data_type)
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return [f"Schema validation error: {str(e)}"]