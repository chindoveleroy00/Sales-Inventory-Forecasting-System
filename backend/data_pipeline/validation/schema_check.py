import pandas as pd
from typing import List, Dict


class SchemaValidator:
    SCHEMAS = {
        'sales': {
            'required': ['date', 'sku_id', 'quantity_sold', 'price', 'category'],
            'dtypes': {
                'date': 'datetime64[ns]',
                'sku_id': 'object',
                'quantity_sold': 'int32',
                'price': 'float32',
                'category': 'object'
            }
        },
        'cleaned_sales': {
            'required': ['date', 'sku_id', 'quantity_sold', 'price', 'is_border_day'],
            'dtypes': {
                'date': 'datetime64[ns]',
                'sku_id': 'object',
                'quantity_sold': 'int32',
                'price': 'float32',
                'is_border_day': 'bool'
            }
        },
        'features': {
            'required': ['date', 'sku_id', 'quantity_sold', 'price', 'lag_7'],
            'dtypes': {
                'date': 'datetime64[ns]',
                'sku_id': 'object',
                'quantity_sold': 'int32',
                'price': 'float32',
                'lag_7': 'float32'
            }
        }
    }

    def validate(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """Backward-compatible schema validation"""
        errors = []
        schema = self.SCHEMAS.get(data_type)

        if not schema:
            return [f"Unknown schema type: {data_type}"]

        # Check required columns
        missing_cols = [col for col in schema['required'] if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check data types
        for col, expected_type in schema['dtypes'].items():
            if col in df.columns:
                if not self._is_type_compatible(df[col].dtype, expected_type):
                    errors.append(f"Invalid dtype for {col}: expected {expected_type}, got {df[col].dtype}")

        return errors

    def _is_type_compatible(self, actual_dtype, expected_type: str) -> bool:
        """Flexible type checking for different pandas versions"""
        type_groups = {
            'object': ['object', 'string', 'category'],
            'int32': ['int32', 'int64', 'int'],
            'float32': ['float32', 'float64', 'float'],
            'bool': ['bool', 'boolean'],
            'datetime64[ns]': ['datetime64[ns]', 'datetime64']
        }

        actual_str = str(actual_dtype).lower()
        expected_str = expected_type.lower()

        if expected_str in type_groups:
            return any(t in actual_str for t in type_groups[expected_str])
        return expected_str in actual_str