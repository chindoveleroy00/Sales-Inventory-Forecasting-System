import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PolicyValidator:
    """Zimbabwean regulatory compliance validator with enhanced reporting"""

    ZIM_PRICE_CONTROLS: Dict[str, Tuple[float, float]] = {
        'MEALIE_2KG': (1.50, 2.60),  # Meal-meal 2kg
        'COOKOIL_2LT': (3.00, 4.10),  # Cooking oil 2L
        'SUGAR_2KG': (2.00, 2.80),  # Sugar 2kg
        'BREAD_LOAF': (0.80, 1.20),  # Standard loaf
        'SALT_1KG': (0.50, 0.80)  # Salt 1kg
    }

    ZIM_TAX_EXEMPT: List[str] = [
        'MEALIE_2KG',
        'COOKOIL_2LT',
        'BREAD_LOAF'
    ]

    def check_price_controls(self, df: pd.DataFrame) -> List[str]:
        """Enhanced price control validation with detailed violation reporting"""
        violations = []

        # Ensure 'price' column is numeric for comparisons
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # Drop rows where price became NaN after coercion for price control checks
        df_valid_price = df.dropna(subset=['price'])

        for sku, (min_price, max_price) in self.ZIM_PRICE_CONTROLS.items():
            if sku not in df_valid_price['sku_id'].unique():
                continue

            sku_data = df_valid_price[df_valid_price['sku_id'] == sku]

            # Check for prices below minimum
            under_min = sku_data[sku_data['price'] < min_price]
            if not under_min.empty:
                first_violation_date = under_min['date'].min().strftime('%Y-%m-%d')
                violations.append(
                    f"SKU '{sku}' has {len(under_min)} instances below min price ({min_price:.2f} USD). "
                    f"First detected on {first_violation_date}. Example price: {under_min['price'].min():.2f} USD."
                )

            # Check for prices above maximum
            over_max = sku_data[sku_data['price'] > max_price]
            if not over_max.empty:
                first_violation_date = over_max['date'].min().strftime('%Y-%m-%d')
                violations.append(
                    f"SKU '{sku}' has {len(over_max)} instances above max price ({max_price:.2f} USD). "
                    f"First detected on {first_violation_date}. Example price: {over_max['price'].max():.2f} USD."
                )
        return violations

    def check_tax_compliance(self, df: pd.DataFrame) -> List[str]:
        """
        Check for tax compliance for products based on ZIM_TAX_EXEMPT.
        This is a placeholder and assumes 'price_usd' and a tax rate are available.
        You'll need to define how 'tax_applied' or 'effective_tax_rate' would be derived from your data.
        For demonstration, this assumes if a tax-exempt item has a price that suggests tax was added, it's an issue.
        """
        issues = []
        # This check requires 'price_usd' and an understanding of how taxes are represented in your data.
        # Placeholder logic: if 'price_usd' is used and you expect certain tax-exempt items to *not* have tax.

        # Example (conceptual, requires your data to have a clear 'tax_applied' or similar field):
        # if 'price_usd' in df.columns:
        #     for sku in self.ZIM_TAX_EXEMPT:
        #         # This is highly simplified and conceptual.
        #         # Actual implementation needs detailed tax data in DataFrame (e.g., 'item_tax_amount', 'total_amount').
        #         tax_discrepancies = df[(df['sku_id'] == sku) & (df['tax_rate_applied'] > 0.01)] # assuming tax_rate_applied column
        #         if not tax_discrepancies.empty:
        #             issues.append(
        #                 f"Tax compliance issue: Tax applied to exempt SKU '{sku}'. "
        #                 f"Found {len(tax_discrepancies)} instances."
        #             )
        return issues

    def check_all_policies(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Orchestrates all policy validation checks and returns a combined report.
        """
        report = {}

        logger.info("Running all policy checks...")

        # Run price control checks
        price_control_issues = self.check_price_controls(df)
        if price_control_issues:
            report['price_controls'] = price_control_issues
            logger.warning(f"Price control violations found: {len(price_control_issues)} issues.")
        else:
            logger.info("No price control violations detected.")

        # Run tax compliance checks (if implemented and relevant for your data)
        tax_compliance_issues = self.check_tax_compliance(df)
        if tax_compliance_issues:
            report['tax_compliance'] = tax_compliance_issues
            logger.warning(f"Tax compliance issues found: {len(tax_compliance_issues)} issues.")
        else:
            logger.info("No tax compliance issues detected.")

        return report

    def generate_compliance_report(self, violations: pd.DataFrame) -> str:
        """Generate formal compliance documentation"""
        if violations.empty:
            return "CLEAN REPORT: No regulatory violations detected"

        report = [
            "ZIMBABWE REGULATORY COMPLIANCE REPORT",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            "\nPRICE CONTROL VIOLATIONS:"
        ]

        # Ensure 'price' is numeric for report generation
        violations['price'] = pd.to_numeric(violations['price'], errors='coerce')
        violations = violations.dropna(subset=['price'])

        for sku in violations['sku_id'].unique():
            sku_violations = violations[violations['sku_id'] == sku]
            report.append(
                f"\n{sku}: {len(sku_violations)} violations\n"
                f"Max price: {sku_violations['price'].max():.2f} USD\n"
                f"First violation: {sku_violations['date'].min():%Y-%m-%d}"
            )

        report.extend([
            "\nLEGAL REQUIREMENTS:",
            "1. Violation of SI 123 of 2020 (Price Control) if prices exceed stipulated maximums for basic commodities.",
            "2. Violation of Value Added Tax Act (Chapter 23:12) if exempt goods are taxed or incorrect tax rates applied.",
            "\nRECOMMENDATIONS:",
            "1. Immediately adjust prices of violating SKUs to comply with regulations.",
            "2. Review and update POS system pricing rules based on current Statutory Instruments.",
            "3. Conduct regular internal audits of pricing and tax application.",
            "4. Provide staff training on regulatory compliance.",
            "\n--- END REPORT ---"
        ])

        return "\n".join(report)