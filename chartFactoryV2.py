import os
import pandas as pd
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from analyserV2 import (
    UnifiedSemanticAnalyzer, 
    AxisType, 
    AxisScore, 
    BaseType
)

class ChartPattern:
    """Representation of a chart pattern with its configuration"""
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chart pattern with configuration
        
        Args:
            config (Dict): Configuration dictionary for the chart pattern
        """
        # Basic chart information
        self.chart_type = config.get('chart_type', 'generic')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # Required axes for the chart
        self.required_axes = config.get('required_axes', [])
        
        # Optional quality checks
        self.quality_checks = config.get('quality_checks', {})
        
        # Store full configuration for potential future use
        self.metadata = config

class ChartPatternManager:
    """Manages loading and storing of chart patterns"""
    def __init__(self, industry: Optional[str] = None):
        """
        Initialize pattern manager
        
        Args:
            industry (Optional[str]): Industry-specific configuration identifier
        """
        # Determine base config path
        self.config_base_path = Path('config')
        
        # Load default patterns
        self.default_patterns = self._load_patterns('default_patterns.yaml')
        
        # Load industry-specific patterns
        self.industry_patterns = {}
        if industry:
            industry_file = f'industries/{industry.lower()}.yaml'
            self.industry_patterns = self._load_patterns(industry_file)
        
        # Merge patterns with industry patterns taking precedence
        self.all_patterns = {**self.default_patterns, **self.industry_patterns}

    def _load_patterns(self, filename: str) -> Dict[str, ChartPattern]:
        """
        Load chart patterns from a YAML file
        
        Args:
            filename (str): Name of the YAML file to load
        
        Returns:
            Dict[str, ChartPattern]: Loaded chart patterns
        """
        try:
            # Construct full path
            full_path = self.config_base_path / filename
            
            # Ensure path exists
            if not full_path.exists():
                print(f"Pattern file not found: {full_path}")
                return {}
            
            # Load YAML content
            with open(full_path, 'r') as f:
                raw_patterns = yaml.safe_load(f)
            
            # Handle potential non-dictionary input
            if not isinstance(raw_patterns, dict):
                print(f"Invalid pattern format in {filename}")
                return {}
            
            # Special handling for industry-specific files
            if 'name' in raw_patterns:
                # Extract chart patterns, excluding metadata
                patterns = {
                    k: v for k, v in raw_patterns.items() 
                    if k not in ['name', 'axis_patterns']
                }
            else:
                patterns = raw_patterns
            
            # Convert to ChartPattern objects
            return {
                name: ChartPattern(pattern) 
                for name, pattern in patterns.items()
            }
        
        except Exception as e:
            print(f"Error loading pattern file {filename}: {e}")
            return {}

class ChartRecommender:
    """Recommends charts based on dataset characteristics"""
    def __init__(self, industry: Optional[str] = None):
        """
        Initialize chart recommender
        
        Args:
            industry (Optional[str]): Industry-specific configuration
        """
        self.semantic_analyzer = UnifiedSemanticAnalyzer(industry)
        self.pattern_manager = ChartPatternManager(industry)
        self.used_columns = set()

    def recommend_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate chart recommendations for a given DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame to analyze
        
        Returns:
            List[Dict[str, Any]]: Recommended charts with their details
        """
        # Reset used columns for each recommendation run
        self.used_columns.clear()

        # Analyze dataset semantics
        axis_mappings = self.semantic_analyzer.analyze_dataset(df)
        recommendations = []

        # Print debug information about axis mappings
        print("\nColumn Axis Mappings:")
        for column, scores in axis_mappings.items():
            print(f"\nColumn: {column}")
            for score in scores:
                print(f"  {score.axis_type.name}: {score.final_score:.3f}")

        # Print available patterns
        print("\nAvailable Patterns:")
        for chart_name, pattern in self.pattern_manager.all_patterns.items():
            print(f"- {chart_name}: {pattern.required_axes}")

        # Iterate through predefined chart patterns
        for chart_name, pattern in self.pattern_manager.all_patterns.items():
            try:
                # Debug: Print current pattern being processed
                print(f"\nProcessing pattern: {chart_name}")
                print(f"Required axes: {pattern.required_axes}")

                chart_rec = self._match_chart_pattern(df, axis_mappings, pattern)
                if chart_rec:
                    recommendations.append(chart_rec)
                else:
                    print(f"No match found for pattern: {chart_name}")
            except Exception as e:
                print(f"Error processing chart pattern {chart_name}: {e}")
                import traceback
                traceback.print_exc()

        # Handle case of no recommendations
        if not recommendations:
            print("\nNo chart recommendations found. Possible reasons:")
            print("1. No columns match required chart axes")
            print("2. Quality checks are too strict")
            print("3. Patterns do not match available columns")
            return []

        # Sort recommendations by score in descending order
        sorted_recommendations = sorted(recommendations, key=lambda x: x.get('score', 0), reverse=True)
        
        print("\nFinal Recommendations:")
        for rec in sorted_recommendations:
            print(f"Chart Type: {rec['chart_type']}")
            print(f"Columns: {rec['columns']}")
            print(f"Score: {rec['score']:.3f}")
        
        return sorted_recommendations

    def _match_chart_pattern(
        self, 
        df: pd.DataFrame, 
        axis_mappings: Dict[str, List[AxisScore]], 
        pattern: ChartPattern
    ) -> Optional[Dict[str, Any]]:
        """
        Match a specific chart pattern to available columns
        
        Args:
            df (pd.DataFrame): Input DataFrame
            axis_mappings (Dict): Semantic analysis of columns
            pattern (ChartPattern): Chart pattern to match
        
        Returns:
            Optional[Dict[str, Any]]: Matched chart recommendation or None
        """
        matched_columns = {}
        total_score = 0.0
        
        # Try to match each required axis
        for axis_spec in pattern.required_axes:
            axis_type = AxisType[axis_spec['type']]
            min_score = axis_spec.get('min_score', 0.5)
            
            # Find best matching column for this axis type
            best_match = self._find_best_column(axis_mappings, axis_type, min_score)
            
            if not best_match:
                return None  # Cannot match all required axes
            
            column = best_match['column']
            
            # Prevent using the same column twice
            if column in matched_columns.values():
                continue
            
            matched_columns[axis_type.name.lower()] = column
            total_score += best_match['score']
            
            # Mark column as used to prevent reuse
            self.used_columns.add(column)

        # Ensure we have matched the required number of axes
        if len(matched_columns) < len(pattern.required_axes):
            return None

        # Calculate average score, handling potential division by zero
        try:
            avg_score = total_score / len(pattern.required_axes) if pattern.required_axes else 0
        except ZeroDivisionError:
            avg_score = 0

        # Validate quality checks
        if not self._validate_quality_checks(df, matched_columns, pattern):
            return None

        return {
            'chart_type': pattern.chart_type,
            'columns': matched_columns,
            'score': avg_score,
            'pattern_metadata': pattern.metadata
        }

    def _find_best_column(
        self, 
        axis_mappings: Dict[str, List[AxisScore]], 
        axis_type: AxisType, 
        min_score: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best column matching a specific axis type with flexible type matching
        
        Args:
            axis_mappings (Dict): Semantic analysis results
            axis_type (AxisType): Desired axis type
            min_score (float): Minimum acceptable score
        
        Returns:
            Optional[Dict[str, Any]]: Best matching column details
        """
        best_match = None
        
        # Define type compatibility mapping
        type_compatibility = {
            AxisType.TIME: [AxisType.TIME],
            AxisType.METRIC: [AxisType.METRIC, AxisType.CURRENCY, AxisType.PERCENTAGE, AxisType.QUANTITY],
            AxisType.DIMENSION: [AxisType.DIMENSION, AxisType.GEOGRAPHIC],
            AxisType.CURRENCY: [AxisType.CURRENCY, AxisType.METRIC],
            AxisType.PERCENTAGE: [AxisType.PERCENTAGE, AxisType.METRIC]
        }
        
        for column, column_scores in axis_mappings.items():
            # Find scores matching the desired axis type or compatible types
            matching_scores = [
                score for score in column_scores 
                if (score.axis_type == axis_type or 
                    score.axis_type in type_compatibility.get(axis_type, []))
                and score.final_score >= min_score
            ]
            
            if matching_scores:
                # Take the highest scoring match
                current_match = max(matching_scores, key=lambda x: x.final_score)
                score = current_match.final_score
                
                # Penalize already used columns
                if column in self.used_columns:
                    score *= 0.7
                
                if not best_match or score > best_match['score']:
                    best_match = {
                        'column': column, 
                        'score': score,
                        'axis_score': current_match
                    }
        
        return best_match

    def _validate_quality_checks(
        self, 
        df: pd.DataFrame, 
        matched_columns: Dict[str, str], 
        pattern: ChartPattern
    ) -> bool:
        """
        Validate chart quality checks
        
        Args:
            df (pd.DataFrame): Input DataFrame
            matched_columns (Dict): Columns matched to axes
            pattern (ChartPattern): Chart pattern to validate
        
        Returns:
            bool: Whether the chart passes quality checks
        """
        quality_checks = pattern.quality_checks
        if not quality_checks:
            return True
        
        # Minimum data points check
        if 'min_data_points' in quality_checks:
            min_points = quality_checks['min_data_points']
            if len(df) < min_points:
                return False
        
        # Minimum categories check
        if 'min_categories' in quality_checks:
            min_categories = quality_checks['min_categories']
            category_column = matched_columns.get('dimension')
            if category_column:
                if df[category_column].nunique() < min_categories:
                    return False
        
        return True

def main():
    """
    Example usage of the chart recommendation engine
    """
    try:
        # Attempt to read the CSV file
        df = pd.read_csv("sample_financial_data.csv")
        
        # Initialize recommender with industry context
        recommender = ChartRecommender(industry="FINANCE")
        
        # Print loaded patterns for debugging
        print("Loaded Patterns:")
        for name, pattern in recommender.pattern_manager.all_patterns.items():
            print(f"- {name}: {pattern.required_axes}")
        
        # Generate recommendations
        recommendations = recommender.recommend_charts(df)
        
        # Print recommendations
        if recommendations:
            print("\nChart Recommendations:")
            for rec in recommendations:
                print("\n--- Recommendation ---")
                print(f"Chart Type: {rec['chart_type']}")
                print(f"Columns: {rec['columns']}")
                print(f"Score: {rec['score']:.2f}")
        else:
            print("\nNo recommendations found. Possible reasons:")
            print("1. No columns match required chart axes")
            print("2. Quality checks are too strict")
            print("3. No patterns defined in YAML")
    
    except FileNotFoundError:
        print("Error: Sample data file not found. Please check the file path.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        print("\nFull Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()