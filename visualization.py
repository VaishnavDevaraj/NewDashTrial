import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from config import DEFAULT_CHART_COLORS, MAX_CATEGORIES_IN_CHART

class VisualizationGenerator:
    def __init__(self, df=None):
        self.df = df
    
    def set_dataframe(self, df):
        """Set the dataframe to use for visualizations"""
        self.df = df
    
    def get_column_types(self):
        """Determine the data type of each column"""
        if self.df is None:
            return {}
        
        column_types = {}
        for col in self.df.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].nunique() <= 10 and self.df[col].nunique() / len(self.df) < 0.05:
                    column_types[col] = 'categorical_numeric'
                else:
                    column_types[col] = 'numeric'
            # Check if column is datetime
            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                column_types[col] = 'datetime'
            # Check if column is categorical
            elif self.df[col].nunique() <= 20 or self.df[col].nunique() / len(self.df) < 0.1:
                column_types[col] = 'categorical'
            # Otherwise, treat as text
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
            'categorical_columns': [col for col, type in column_types.items() 
                                   if type in ['categorical', 'categorical_numeric']],
            'datetime_columns': [col for col, type in column_types.items() if type == 'datetime'],
            'text_columns': [col for col, type in column_types.items() if type == 'text']
        }
        
        # Add basic stats for numeric columns
        numeric_stats = {}
        for col in summary['numeric_columns']:
            numeric_stats[col] = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std())
            }
        summary['numeric_stats'] = numeric_stats
        
        # Add value counts for categorical columns (limited to top values)
        categorical_stats = {}
        for col in summary['categorical_columns']:
            value_counts = self.df[col].value_counts().head(10).to_dict()
            categorical_stats[col] = {
                'unique_values': self.df[col].nunique(),
                'top_values': value_counts
            }
        summary['categorical_stats'] = categorical_stats
        
        return summary
    
    def suggest_visualizations(self):
        """Suggest appropriate visualizations based on the data"""
        if self.df is None:
            return []
        
        suggestions = []
        column_types = self.get_column_types()
        
        # Get lists of different column types
        numeric_cols = [col for col, type in column_types.items() if type == 'numeric']
        categorical_cols = [col for col, type in column_types.items() 
                           if type in ['categorical', 'categorical_numeric']]
        datetime_cols = [col for col, type in column_types.items() if type == 'datetime']
        
        # Suggestion 1: Distribution of numeric variables
        for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many suggestions
            suggestions.append({
                'title': f'Distribution of {col}',
                'description': f'Histogram showing the distribution of {col}',
                'type': 'histogram',
                'config': {
                    'x': col,
                    'nbins': 20
                }
            })
        
        # Suggestion 2: Bar charts for categorical variables
        for col in categorical_cols[:3]:
            if self.df[col].nunique() <= MAX_CATEGORIES_IN_CHART:
                suggestions.append({
                    'title': f'Count of {col}',
                    'description': f'Bar chart showing the count of each {col} category',
                    'type': 'bar',
                    'config': {
                        'x': col
                    }
                })
        
        # Suggestion 3: Scatter plots between numeric variables
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols))):
                for j in range(i+1, min(4, len(numeric_cols))):
                    suggestions.append({
                        'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                        'description': f'Scatter plot showing relationship between {numeric_cols[i]} and {numeric_cols[j]}',
                        'type': 'scatter',
                        'config': {
                            'x': numeric_cols[i],
                            'y': numeric_cols[j]
                        }
                    })
        
        # Suggestion 4: Time series for datetime columns
        for date_col in datetime_cols[:2]:
            for num_col in numeric_cols[:3]:
                suggestions.append({
                    'title': f'{num_col} over time',
                    'description': f'Line chart showing {num_col} over {date_col}',
                    'type': 'line',
                    'config': {
                        'x': date_col,
                        'y': num_col
                    }
                })
        
        # Suggestion 5: Box plots for numeric variables grouped by categories
        for num_col in numeric_cols[:3]:
            for cat_col in categorical_cols[:2]:
                if self.df[cat_col].nunique() <= MAX_CATEGORIES_IN_CHART:
                    suggestions.append({
                        'title': f'{num_col} by {cat_col}',
                        'description': f'Box plot showing distribution of {num_col} for each {cat_col}',
                        'type': 'box',
                        'config': {
                            'x': cat_col,
                            'y': num_col
                        }
                    })
        
        # Suggestion 6: Heatmap for correlation between numeric variables
        if len(numeric_cols) >= 4:
            suggestions.append({
                'title': 'Correlation Heatmap',
                'description': 'Heatmap showing correlation between numeric variables',
                'type': 'heatmap',
                'config': {
                    'columns': numeric_cols[:8]  # Limit to 8 columns for readability
                }
            })
        
        # Suggestion 7: Pie chart for categorical variables with few categories
        for col in categorical_cols:
            if 2 <= self.df[col].nunique() <= 8:
                suggestions.append({
                    'title': f'Proportion of {col}',
                    'description': f'Pie chart showing the proportion of each {col} category',
                    'type': 'pie',
                    'config': {
                        'names': col,
                        'values': 'count'
                    }
                })
        
        return suggestions
    
    def create_visualization(self, viz_type, config):
        """Create a visualization based on the specified type and configuration"""
        if self.df is None:
            return None
        
        # Make a copy of the dataframe to avoid modifying the original
        df = self.df.copy()
        
        # Handle missing values for visualization
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        
        # Create the visualization based on the type
        if viz_type == 'histogram':
            fig = px.histogram(
                df, 
                x=config.get('x'),
                nbins=config.get('nbins', 20),
                title=config.get('title', f'Distribution of {config.get("x")}'),
                color_discrete_sequence=DEFAULT_CHART_COLORS
            )
            return fig.to_json()
        
        elif viz_type == 'bar':
            # Get value counts if needed
            if 'values' not in config:
                value_counts = df[config.get('x')].value_counts().reset_index()
                value_counts.columns = [config.get('x'), 'count']
                fig = px.bar(
                    value_counts,
                    x=config.get('x'),
                    y='count',
                    title=config.get('title', f'Count of {config.get("x")}'),
                    color_discrete_sequence=DEFAULT_CHART_COLORS
                )
            else:
                fig = px.bar(
                    df,
                    x=config.get('x'),
                    y=config.get('y'),
                    color=config.get('color'),
                    title=config.get('title', f'{config.get("y")} by {config.get("x")}'),
                    color_discrete_sequence=DEFAULT_CHART_COLORS
                )
            return fig.to_json()
        
        elif viz_type == 'scatter':
            fig = px.scatter(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                size=config.get('size'),
                title=config.get('title', f'{config.get("y")} vs {config.get("x")}'),
                color_discrete_sequence=DEFAULT_CHART_COLORS
            )
            return fig.to_json()
        
        elif viz_type == 'line':
            # Ensure datetime column is properly formatted
            if config.get('x') in df.columns and pd.api.types.is_datetime64_dtype(df[config.get('x')]):
                df[config.get('x')] = pd.to_datetime(df[config.get('x')])
            
            fig = px.line(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', f'{config.get("y")} over time'),
                color_discrete_sequence=DEFAULT_CHART_COLORS
            )
            return fig.to_json()
        
        elif viz_type == 'box':
            fig = px.box(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', f'Distribution of {config.get("y")} by {config.get("x")}'),
                color_discrete_sequence=DEFAULT_CHART_COLORS
            )
            return fig.to_json()
        
        elif viz_type == 'heatmap':
            columns = config.get('columns', [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
            corr_matrix = df[columns].corr()
            
            fig = px.imshow(
                corr_matrix,
                title=config.get('title', 'Correlation Heatmap'),
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            return fig.to_json()
        
        elif viz_type == 'pie':
            names_col = config.get('names')
            
            if config.get('values') == 'count':
                # Create a count of each category
                value_counts = df[names_col].value_counts().reset_index()
                value_counts.columns = [names_col, 'count']
                
                fig = px.pie(
                    value_counts,
                    names=names_col,
                    values='count',
                    title=config.get('title', f'Proportion of {names_col}'),
                    color_discrete_sequence=DEFAULT_CHART_COLORS
                )
            else:
                fig = px.pie(
                    df,
                    names=names_col,
                    values=config.get('values'),
                    title=config.get('title', f'Proportion of {names_col}'),
                    color_discrete_sequence=DEFAULT_CHART_COLORS
                )
            return fig.to_json()
        
        elif viz_type == 'pca':
            # Perform PCA on numeric columns
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) < 2:
                return None
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
                        # Create a new dataframe with PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            
            # Add color column if specified
            if config.get('color') and config.get('color') in df.columns:
                pca_df[config.get('color')] = df[config.get('color')].values
            
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=config.get('color'),
                title=config.get('title', 'PCA Visualization'),
                color_discrete_sequence=DEFAULT_CHART_COLORS
            )
            
            # Add explained variance as axis labels
            explained_var = pca.explained_variance_ratio_
            fig.update_xaxes(title_text=f"PC1 ({explained_var[0]:.2%} variance)")
            fig.update_yaxes(title_text=f"PC2 ({explained_var[1]:.2%} variance)")
            
            return fig.to_json()
        
        # Add more visualization types as needed
        
        return None
    
    def get_data_insights(self):
        """Generate automatic insights about the data"""
        if self.df is None:
            return []
        
        insights = []
        column_types = self.get_column_types()
        
        # Insight 1: Missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            missing_cols = missing_values[missing_values > 0]
            insights.append({
                'type': 'missing_values',
                'title': 'Missing Values Detected',
                'description': f'Found {missing_values.sum()} missing values across {len(missing_cols)} columns.',
                'details': missing_cols.to_dict()
            })
        
        # Insight 2: Highly correlated features
        numeric_cols = [col for col, type in column_types.items() if type == 'numeric']
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                         for col1 in upper_tri.index 
                         for col2 in upper_tri.columns 
                         if upper_tri.loc[col1, col2] > 0.8]
            
            if high_corr:
                insights.append({
                    'type': 'high_correlation',
                    'title': 'Highly Correlated Features',
                    'description': f'Found {len(high_corr)} pairs of highly correlated features (r > 0.8).',
                    'details': high_corr
                })
        
        # Insight 3: Skewed distributions
        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 1.5:
                direction = 'right' if skewness > 0 else 'left'
                insights.append({
                    'type': 'skewed_distribution',
                    'title': f'Skewed Distribution: {col}',
                    'description': f'{col} has a {direction}-skewed distribution (skewness = {skewness:.2f}).',
                    'details': {
                        'column': col,
                        'skewness': float(skewness),
                        'direction': direction
                    }
                })
        
        # Insight 4: Outliers
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            
            if len(outliers) > 0 and len(outliers) < len(self.df) * 0.05:  # Less than 5% are outliers
                insights.append({
                    'type': 'outliers',
                    'title': f'Outliers Detected: {col}',
                    'description': f'Found {len(outliers)} outliers in {col} ({len(outliers)/len(self.df):.1%} of data).',
                    'details': {
                        'column': col,
                        'outlier_count': int(len(outliers)),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                })
        
        # Insight 5: Imbalanced categories
        categorical_cols = [col for col, type in column_types.items() 
                           if type in ['categorical', 'categorical_numeric']]
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.8:  # If the most common category is more than 80%
                insights.append({
                    'type': 'imbalanced_category',
                    'title': f'Imbalanced Categories: {col}',
                    'description': f'The most common category in {col} represents {value_counts.iloc[0]:.1%} of the data.',
                    'details': {
                        'column': col,
                        'dominant_category': value_counts.index[0],
                        'dominant_percentage': float(value_counts.iloc[0])
                    }
                })
        
        return insights
