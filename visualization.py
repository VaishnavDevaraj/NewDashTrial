import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from analyserV2 import UnifiedSemanticAnalyzer
from chartFactoryV2 import ChartRecommender
from ingestionV2 import ValueTypeDetector

class VisualizationGenerator:
    def __init__(self, df=None, industry=None):
        self.df = df
        self.semantic_analyzer = UnifiedSemanticAnalyzer(industry)
        self.chart_recommender = ChartRecommender(industry)
        self.value_detector = ValueTypeDetector()
    
    def set_dataframe(self, df):
        """Set the dataframe to use for visualizations"""
        self.df = df
    
    def get_column_types(self):
        """Determine the data type of each column using semantic analysis"""
        if self.df is None:
            return {}
        
        type_analysis = self.value_detector.infer_types(self.df)
        column_types = {}
        
        for col, (base_type, metadata) in type_analysis.items():
            if base_type == 'NUMERIC':
                column_types[col] = 'numeric'
            elif base_type == 'CATEGORICAL':
                column_types[col] = 'categorical'
            elif base_type == 'DATETIME':
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'text'
        
        return column_types
    
    def get_data_summary(self):
        """Get a summary of the dataframe"""
        if self.df is None:
            return {}
        
        column_types = self.get_column_types()
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_types': column_types,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': [col for col, type in column_types.items() if type == 'numeric'],
            'categorical_columns': [col for col, type in column_types.items() if type == 'categorical'],
            'datetime_columns': [col for col, type in column_types.items() if type == 'datetime'],
            'text_columns': [col for col, type in column_types.items() if type == 'text']
        }
        return summary
    
    def suggest_visualizations(self):
        """Suggest appropriate visualizations based on semantic analysis"""
        if self.df is None:
            return []
        return self.chart_recommender.recommend_charts(self.df)
    
    def get_data_insights(self):
        """Generate insights using semantic analysis"""
        if self.df is None:
            return []
        return self.semantic_analyzer.analyze_dataset(self.df)
