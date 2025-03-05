from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re

class BaseType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()
    DATETIME = auto()
    TEXT = auto()
    UNKNOWN = auto()

@dataclass
class TypeMetadata:
    confidence: float
    statistics: Dict
    validation_info: Dict
    sample_values: List
    format_info: Optional[Dict] = None

class ValueTypeDetector:
    def __init__(self):
        self.datetime_formats = [
            '%Y-%m-%d', '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'
        ]
        
        self.binary_sets = [
            {'true', 'false'}, {'yes', 'no'}, {'0', '1'},
            {'active', 'inactive'}, {'pass', 'fail'}
        ]

    def infer_types(self, df: pd.DataFrame) -> Dict[str, Tuple[BaseType, TypeMetadata]]:
        """Infer types for all columns based purely on values"""
        results = {}
        for column in df.columns:
            results[column] = self._analyze_column(df[column])
        return results

    def _analyze_column(self, series: pd.Series) -> Tuple[BaseType, TypeMetadata]:
        """Analyze single column and return most likely type with metadata"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return self._unknown_type()

        # Try datetime first as it's most specific
        datetime_result = self._check_datetime(clean_series)
        if datetime_result[1].confidence > 0.8:
            return datetime_result

        # Check categorical (including binary)
        categorical_result = self._check_categorical(clean_series)
        if categorical_result[1].confidence > 0.8:
            return categorical_result

        # Try numeric
        numeric_result = self._check_numeric(clean_series)
        if numeric_result[1].confidence > 0.7:
            return numeric_result

        # Default to text with pattern analysis
        return self._check_text(clean_series)

    def _check_datetime(self, series: pd.Series) -> Tuple[BaseType, TypeMetadata]:
        """Check if series contains datetime values"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return BaseType.DATETIME, TypeMetadata(
                confidence=1.0,
                statistics=self._get_datetime_stats(series),
                validation_info={'native_datetime': True},
                sample_values=series.head().tolist(),
                format_info={'native_format': True}
            )

        # Try parsing with known formats
        best_format = None
        best_success = 0
        parsed_dates = None

        for fmt in self.datetime_formats:
            try:
                parsed = pd.to_datetime(series, format=fmt)
                success_rate = parsed.notna().mean()
                if success_rate > best_success:
                    best_success = success_rate
                    best_format = fmt
                    parsed_dates = parsed
            except ValueError:
                continue

        if best_success > 0.8:
            return BaseType.DATETIME, TypeMetadata(
                confidence=best_success,
                statistics=self._get_datetime_stats(parsed_dates),
                validation_info={'parsed_success_rate': best_success},
                sample_values=series.head().tolist(),
                format_info={'detected_format': best_format}
            )

        return BaseType.DATETIME, TypeMetadata(
            confidence=0.0,
            statistics={},
            validation_info={'is_datetime': False},
            sample_values=series.head().tolist()
        )

    def _check_categorical(self, series: pd.Series) -> Tuple[BaseType, TypeMetadata]:
        """Check if series is categorical (including binary)"""
        unique_ratio = series.nunique() / len(series)
        value_counts = series.value_counts(normalize=True)
        
        # Check binary first
        if series.nunique() == 2:
            unique_values = set(str(val).lower() for val in series.unique())
            is_known_binary = any(unique_values == binary_set 
                                for binary_set in self.binary_sets)
            
            # Consider any 2-value column as binary if distribution isn't extremely skewed
            if is_known_binary or max(value_counts) <= 0.95 or set(unique_values) == {'0', '1'}:  # Prevent extreme imbalance
                return BaseType.CATEGORICAL, TypeMetadata(
                    confidence=0.95 if is_known_binary else 0.85,
                    statistics={
                        'unique_ratio': unique_ratio,
                        'value_distribution': value_counts.to_dict()
                    },
                    validation_info={'is_binary': True},
                    sample_values=series.head().tolist(),
                    format_info={'binary_values': list(unique_values)}
                )

        # Check general categorical
        if unique_ratio <= 0.1 or (len(series) >= 1000 and series.nunique() <= 100):
            return BaseType.CATEGORICAL, TypeMetadata(
                confidence=0.9 if unique_ratio <= 0.05 else 0.8,
                statistics={
                    'unique_ratio': unique_ratio,
                    'value_distribution': value_counts.head(10).to_dict()
                },
                validation_info={'distinct_count': series.nunique()},
                sample_values=series.head().tolist()
            )

        return BaseType.CATEGORICAL, TypeMetadata(
            confidence=0.0,
            statistics={'unique_ratio': unique_ratio},
            validation_info={'is_categorical': False},
            sample_values=series.head().tolist()
        )

    def _check_numeric(self, series: pd.Series) -> Tuple[BaseType, TypeMetadata]:
        """Check if series contains numeric values"""
        # Try converting to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        success_rate = numeric_series.notna().mean()
        
        if success_rate > 0.7:
            clean_numeric = numeric_series.dropna()
            
            return BaseType.NUMERIC, TypeMetadata(
                confidence=success_rate,
                statistics={
                    'mean': clean_numeric.mean(),
                    'std': clean_numeric.std(),
                    'min': clean_numeric.min(),
                    'max': clean_numeric.max(),
                    'median': clean_numeric.median(),
                    'skew': clean_numeric.skew()
                },
                validation_info={
                    'conversion_success_rate': success_rate,
                    'contains_negatives': (clean_numeric < 0).any(),
                    'is_integer': all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in clean_numeric)
                },
                sample_values=series.head().tolist()
            )

        return BaseType.NUMERIC, TypeMetadata(
            confidence=0.0,
            statistics={},
            validation_info={'is_numeric': False},
            sample_values=series.head().tolist()
        )

    def _check_text(self, series: pd.Series) -> Tuple[BaseType, TypeMetadata]:
        """Analyze text patterns and structure"""
        str_series = series.astype(str)
        
        # Get length statistics
        lengths = str_series.str.len()
        
        # Check for common patterns
        patterns = {
            'alphanumeric': str_series.str.match(r'^[A-Za-z0-9]+$').mean(),
            'alpha_only': str_series.str.match(r'^[A-Za-z]+$').mean(),
            'numeric_only': str_series.str.match(r'^[0-9]+$').mean(),
            'has_spaces': str_series.str.contains(r'\s').mean(),
            'has_special': str_series.str.contains(r'[^A-Za-z0-9\s]').mean()
        }
        
        return BaseType.TEXT, TypeMetadata(
            confidence=1.0,  # Text is our fallback type
            statistics={
                'length_stats': {
                    'mean': lengths.mean(),
                    'std': lengths.std(),
                    'min': lengths.min(),
                    'max': lengths.max()
                },
                'pattern_matches': patterns
            },
            validation_info={
                'unique_ratio': series.nunique() / len(series),
                'patterns_detected': patterns
            },
            sample_values=series.head().tolist()
        )

    def _unknown_type(self) -> Tuple[BaseType, TypeMetadata]:
        """Return unknown type for empty or invalid columns"""
        return BaseType.UNKNOWN, TypeMetadata(
            confidence=1.0,
            statistics={},
            validation_info={'is_empty': True},
            sample_values=[]
        )

    def _get_datetime_stats(self, series: pd.Series) -> Dict:
        """Calculate datetime specific statistics"""
        return {
            'min_date': series.min(),
            'max_date': series.max(),
            'range_days': (series.max() - series.min()).days,
            'null_count': series.isna().sum(),
            'unique_count': series.nunique()
        }

# Example usage:
if __name__ == "__main__":
    # Create sample data
    data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'category': ['A', 'B', 'A'],
        'value': [100, 200, 300],
        'binary': [0,0,1],
        'text': ['abc123', 'def456', 'ghi789']
    }
    df = pd.read_csv("Data/ibrd_and_ida_net_flows_commitments_01-11-2025.csv")
    
    # Analyze types
    detector = ValueTypeDetector()
    results = detector.infer_types(df)
    
    # Print results
    for column, (base_type, metadata) in results.items():
        print(f"\nColumn: {column}")
        print(f"Base Type: {base_type}")
        print(f"Confidence: {metadata.confidence}")
        print(f"Statistics: {metadata.statistics}")