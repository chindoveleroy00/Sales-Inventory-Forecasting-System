import pandas as pd
from typing import Optional
from policy_check import PolicyValidator
import logging

logger = logging.getLogger(__name__)


class PriceEnforcer:
    """Automated price control enforcement system"""

    def __init__(self):
        self.validator = PolicyValidator()

    def enforce_controls(self, df: pd.DataFrame,
                         correction_mode: str = 'auto') -> pd.DataFrame:
        """
        Enforce price controls with multiple correction strategies

        Args:
            correction_mode:
                'auto' - automatically cap prices
                'log' - only log violations
                'null' - set violating prices to null
        """
        clean_df = df.copy()
        violations = []

        for sku, (min_price, max_price) in self.validator.ZIM_PRICE_CONTROLS.items():
            if sku not in clean_df['sku_id'].unique():
                continue

            sku_mask = clean_df['sku_id'] == sku

            # Handle maximum price violations
            over_max = clean_df.loc[sku_mask, 'price'] > max_price
            if over_max.any():
                violations.append(f"{sku} above {max_price} USD")

                if correction_mode == 'auto':
                    clean_df.loc[sku_mask & over_max, 'price'] = max_price
                elif correction_mode == 'null':
                    clean_df.loc[sku_mask & over_max, 'price'] = None

            # Handle minimum price violations
            under_min = clean_df.loc[sku_mask, 'price'] < min_price
            if under_min.any():
                violations.append(f"{sku} below {min_price} USD")

                if correction_mode == 'auto':
                    clean_df.loc[sku_mask & under_min, 'price'] = min_price
                elif correction_mode == 'null':
                    clean_df.loc[sku_mask & under_min, 'price'] = None

        if violations:
            logger.warning(f"Corrected {len(violations)} price violations")
            if correction_mode == 'log':
                logger.info("Violations logged but not modified")

        return clean_df

    def create_audit_log(self, original_df: pd.DataFrame,
                         corrected_df: pd.DataFrame) -> pd.DataFrame:
        """Generate audit trail of all corrections"""
        audit_log = pd.merge(
            original_df,
            corrected_df,
            on=['date', 'sku_id'],
            suffixes=('_original', '_corrected'),
            how='outer'
        )

        audit_log = audit_log[
            audit_log['price_original'] != audit_log['price_corrected']
            ]

        audit_log['change_type'] = audit_log.apply(
            lambda x: 'CAPPED' if x['price_corrected'] < x['price_original'] else 'FLOORED',
            axis=1
        )

        return audit_log