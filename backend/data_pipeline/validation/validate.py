import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from policy_check import PolicyValidator
from price_enforcer import PriceEnforcer
import logging

logger = logging.getLogger(__name__)


def validate_data_pipeline(output_paths: Dict[str, Path],
                           strict: bool = False,
                           auto_correct: bool = True) -> bool:
    """Enhanced validation with correction capabilities"""
    validator = PolicyValidator()
    enforcer = PriceEnforcer()
    all_valid = True

    for data_type, path in output_paths.items():
        if not path.exists():
            logger.error(f"Missing file: {path}")
            return False

        try:
            df = pd.read_parquet(path)

            # Auto-correction if enabled
            if auto_correct:
                df = enforcer.enforce_controls(df)

            # Run validations
            report = validator.check_all_policies(df)

            if any(report.values()):
                log_validation_report(data_type, report)

                if strict:
                    all_valid = False
                    logger.error("Strict validation failed - pipeline halted")

            # Generate compliance documentation
            violations = df[df['sku_id'].isin(validator.ZIM_PRICE_CONTROLS.keys())]
            compliance_report = validator.generate_compliance_report(violations)
            logger.info(f"Compliance Report:\n{compliance_report}")

        except Exception as e:
            logger.error(f"Validation failed for {data_type}: {str(e)}")
            return False

    return all_valid


def log_validation_report(data_type: str, report: Dict[str, List[str]]) -> None:
    """Formatted validation output"""
    logger.warning(f"\nValidation Report for {data_type}:")

    for check_type, issues in report.items():
        if issues:
            logger.warning(f"  {check_type.upper()}:")
            for issue in issues[:5]:  # Show top 5 issues
                logger.warning(f"    • {issue}")
            if len(issues) > 5:
                logger.warning(f"    ... and {len(issues) - 5} more")