from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from pathlib import Path
import yaml
import re
from ingestionV2 import BaseType, ValueTypeDetector, TypeMetadata
class AxisType(Enum):
    TIME = auto()
    METRIC = auto()
    DIMENSION = auto()
    PERCENTAGE = auto()
    CURRENCY = auto()
    QUANTITY = auto()
    GEOGRAPHIC = auto()
    UNKNOWN = auto()

@dataclass
class AxisScore:
    axis_type: AxisType
    semantic_score: float
    value_score: float
    final_score: float
    metadata: Dict

class BasePatternMatcher:
    """Enhanced pattern matching with better compound term handling"""
    def __init__(self):
        self.patterns = {
            AxisType.TIME: {
                'keywords': ['date', 'time', 'year', 'month', 'quarter', 'period'],
                'boost_patterns': ['first', 'last', 'recent', 'approval'],
                'strong_indicators': ['date', 'timestamp'],
                'weight': 1.0
            },
            AxisType.METRIC: {
                'keywords': ['total', 'count', 'number', 'score', 'value', 'volume', 
                           'balance', 'net', 'aggregate', 'cumulative'],
                'exclude': ['currency', 'percent', 'ratio'],
                'weight': 1.1
            },
            AxisType.CURRENCY: {
                'keywords': ['amount', 'payment', 'commitment', 'principal', 'price', 
                           'cost', 'disbursement', 'repayment', 'obligation'],
                'strong_indicators': ['usd', 'eur', '$', '£', '€', 'dollars', '(us$)', '(usd)'],
                'compound_boost': ['amount_usd', 'principal_amount'],
                'weight': 1.2
            },
            AxisType.PERCENTAGE: {
                'keywords': ['rate', 'ratio', 'percent', 'proportion', 'share'],
                'strong_indicators': ['%', 'pct'],
                'weight': 0.85
            },
            AxisType.DIMENSION: {
                'keywords': ['type', 'category', 'group', 'class', 'name', 'level', 'grade'],
                'strong_indicators': ['type', 'status', 'category'],
                'boost_patterns': ['borrower', 'project', 'loan', 'customer', 'product'],
                'compound_boost': ['loan_type', 'project_status', 'borrower_category'],
                'weight': 0.9
            },
            AxisType.GEOGRAPHIC: {
                'keywords': ['city', 'state', 'location', 'area', 'zone', 'territory'],
                'strong_indicators': ['country', 'region'],
                'compound_indicators': ['country_code', 'region_name'],
                'boost_patterns': ['geographic', 'location', 'area'],
                'weight': 0.9
            },
            AxisType.QUANTITY: {
                'keywords': ['count', 'number', 'quantity', 'units'],
                'weight': 0.8
            }
        }
        
    def _tokenize_column_name(self, column_name: str) -> Set[str]:
        """Enhanced tokenization for compound terms"""
        # Convert camelCase to snake_case
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', column_name).lower()
        
        # Split on underscores and special characters
        tokens = set(re.split('[_\W]+', name))
        
        # Add original name for exact matches
        tokens.add(column_name.lower())
        
        # Add combined adjacent terms for partial matches
        parts = name.split('_')
        for i in range(len(parts)-1):
            tokens.add(f"{parts[i]}_{parts[i+1]}")
            
        return tokens

    def get_pattern(self, axis_type: AxisType) -> Dict:
        """Get pattern with defaults"""
        return self.patterns.get(axis_type, {})

    def match_pattern(self, column_name: str, pattern: Dict) -> Tuple[float, Dict]:
        """Enhanced pattern matching with detailed scoring"""
        tokens = self._tokenize_column_name(column_name)
        score = 0.0
        matches = {
            'keyword_matches': [],
            'strong_indicator_matches': [],
            'compound_matches': [],
            'exclusion_matches': []
        }
        
        # Check strong indicators first
        for ind in pattern.get('strong_indicators', []):
            if ind in tokens or ind in column_name.lower():
                matches['strong_indicator_matches'].append(ind)
                score = max(score, pattern.get('weight', 1.0))
        
        # Check keywords
        for kw in pattern.get('keywords', []):
            if kw in tokens:
                matches['keyword_matches'].append(kw)
                score = max(score, 0.8)
        
        # Check compound boosts
        for comp in pattern.get('compound_boost', []):
            if comp in column_name.lower():
                matches['compound_matches'].append(comp)
                score = max(score, 0.95)
        
        # Check exclusions
        for ex in pattern.get('exclude', []):
            if ex in tokens:
                matches['exclusion_matches'].append(ex)
                score *= 0.4
        
        # Apply boost patterns
        if any(bp in tokens for bp in pattern.get('boost_patterns', [])):
            score *= 1.2
        
        # Boost score for multiple matches
        if len(matches['keyword_matches']) > 1:
            score *= 1.1
        if len(matches['strong_indicator_matches']) > 0 and len(matches['keyword_matches']) > 0:
            score *= 1.2
            
        final_score = min(score * pattern.get('weight', 1.0), 1.0)
        
        return final_score, matches

class EmbeddingHandler:
    """Enhanced embedding handler with better compound term support"""
    def __init__(self):
        self.model = self._initialize_model()
        self.embedding_cache = {}
        self.fallback_threshold = 0.3
        
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Initialized embedding model successfully")
            return model
        except ImportError:
            print("Warning: Using fallback similarity - sentence-transformers not available")
            return None
        except Exception as e:
            print(f"Warning: Error initializing embedding model: {e}")
            return None

    def _preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing for compound terms"""
        # Convert camelCase to spaces
        text = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text)
        
        # Replace underscores and hyphens with spaces
        text = text.replace('_', ' ').replace('-', ' ')
        
        # Split into parts
        parts = text.lower().split()
        
        # Generate variations
        variations = [text.lower()]  # Original lowercased
        variations.extend(parts)  # Individual parts
        
        # Add adjacent pairs
        for i in range(len(parts) - 1):
            variations.append(f"{parts[i]} {parts[i+1]}")
            
        return list(set(variations))

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached or compute new embedding with error handling"""
        if text not in self.embedding_cache:
            try:
                if self.model:
                    self.embedding_cache[text] = self.model.encode([text])[0]
                else:
                    return None
            except Exception as e:
                print(f"Warning: Error computing embedding for '{text}': {e}")
                return None
        return self.embedding_cache[text]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation with compound term support"""
        if self.model:
            # Get variations of both texts
            variations1 = self._preprocess_text(text1)
            variations2 = self._preprocess_text(text2)
            
            # Calculate similarities between all variations
            max_similarity = 0.0
            for var1 in variations1:
                for var2 in variations2:
                    emb1 = self.get_embedding(var1)
                    emb2 = self.get_embedding(var2)
                    if emb1 is not None and emb2 is not None:
                        similarity = float(np.dot(emb1, emb2) / 
                                        (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                        max_similarity = max(max_similarity, similarity)
            
            return max_similarity if max_similarity > self.fallback_threshold else 0.0
        
        # Fallback similarity using token overlap
        tokens1 = set(self._preprocess_text(text1))
        tokens2 = set(self._preprocess_text(text2))
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def precompute_embeddings(self, texts: Set[str]):
        """Batch precompute embeddings"""
        if not self.model:
            return
            
        # Process all variations of texts
        all_variations = set()
        for text in texts:
            all_variations.update(self._preprocess_text(text))
            
        # Compute embeddings in batch where possible
        try:
            if all_variations:
                embeddings = self.model.encode(list(all_variations))
                for text, embedding in zip(all_variations, embeddings):
                    self.embedding_cache[text] = embedding
        except Exception as e:
            print(f"Warning: Error precomputing embeddings: {e}")

    def get_semantic_score(self, column_name: str, pattern: Dict) -> Tuple[float, Dict]:
        """Enhanced semantic scoring with detailed matching"""
        scores = []
        matches = {
            'keyword_matches': [],
            'strong_matches': [],
            'similarity_scores': {}
        }
        
        name_variations = self._preprocess_text(column_name)
        max_score = 0.0
        
        # Check strong indicators first
        for indicator in pattern.get('strong_indicators', []):
            for variation in name_variations:
                if indicator in variation:
                    matches['strong_matches'].append(indicator)
                    max_score = max(max_score, pattern.get('weight', 1.0))
        
        # Calculate similarity with keywords
        for keyword in pattern.get('keywords', []):
            best_variation_score = 0.0
            for variation in name_variations:
                similarity = self.calculate_similarity(variation, keyword)
                if similarity > best_variation_score:
                    best_variation_score = similarity
                    matches['similarity_scores'][keyword] = similarity
                    
            if best_variation_score > 0.6:  # Threshold for meaningful similarity
                matches['keyword_matches'].append(keyword)
                scores.append(best_variation_score)
        
        if scores:
            base_score = max(scores)
            # Apply pattern weights and boosts
            final_score = base_score * pattern.get('weight', 1.0)
            
            # Boost for multiple strong matches
            if len(matches['strong_matches']) > 0:
                final_score *= 1.2
            
            return min(final_score, 1.0), matches
            
        return max_score, matches

class UnifiedSemanticAnalyzer:
    """Enhanced semantic analyzer with improved scoring and debugging"""
    def __init__(self, industry: Optional[str] = None):
        self.value_detector = ValueTypeDetector()
        self.pattern_matcher = BasePatternMatcher()
        self.embedding_handler = EmbeddingHandler()
        self.industry = industry
        self._setup_type_mapping()
        
        # Initialize debug info
        self.debug_info = {}
        
        if industry:
            self._load_industry_patterns(industry)


    def _load_industry_patterns(self, industry: str):
        """Load and merge industry-specific patterns"""
        try:
            config_path = Path('config/industries') / f'{industry.lower()}.yaml'
            with open(config_path) as f:
                industry_config = yaml.safe_load(f)
                
            # Merge axis patterns if they exist
            if 'axis_patterns' in industry_config:
                for axis_type, patterns in industry_config['axis_patterns'].items():
                    try:
                        axis_enum = AxisType[axis_type.upper()]
                        current_patterns = self.pattern_matcher.patterns.get(axis_enum, {})
                        
                        # Deep merge of patterns
                        for key, value in patterns.items():
                            if isinstance(value, list):
                                # Append unique items
                                if key not in current_patterns:
                                    current_patterns[key] = []
                                current_patterns[key].extend(
                                    [v for v in value if v not in current_patterns[key]]
                                )
                            else:
                                current_patterns[key] = value
                                
                        self.pattern_matcher.patterns[axis_enum] = current_patterns
                        
                    except KeyError:
                        print(f"Invalid axis type in config: {axis_type}")
                        
        except Exception as e:
            print(f"Failed to load industry config: {e}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Attempted path: {config_path}")

    def _setup_type_mapping(self):
        """Enhanced type mapping with better validation"""
        self.type_mapping = {
            BaseType.NUMERIC: {
                AxisType.METRIC: {
                    'validation': lambda stats: True,  # Always consider numeric as potential metric
                    'boost': 1.1
                },
                AxisType.CURRENCY: {
                    'validation': lambda stats: stats.get('min', -1) >= 0,  # Non-negative
                    'boost': 1.3
                },
                AxisType.PERCENTAGE: {
                    'validation': lambda stats: 0 <= stats.get('max', 101) <= 100,
                    'boost': 1.15
                }
            },
            BaseType.DATETIME: {
                AxisType.TIME: {
                    'boost': 1.2,
                    'validation': lambda stats: True  # All datetime columns are time axes
                }
            },
            BaseType.CATEGORICAL: {
                AxisType.DIMENSION: {
                    'boost': 1.15,
                    'validation': lambda stats: stats.get('unique_ratio', 1) < 0.2,
                    'keywords': ['type', 'status', 'category']
                },
                AxisType.GEOGRAPHIC: {
                    'boost': 1.2,
                    'validation': lambda stats: stats.get('unique_ratio', 1) < 0.3,
                    'keywords': ['country', 'region', 'city']
                }
            }
        }

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, List[AxisScore]]:
        """Analyze entire dataset with detailed results"""
        results = {}
        for column in df.columns:
            print(f"\nAnalyzing column: {column}")
            scores = self.analyze_column(df[column], column)
            
            # Print debug info
            for score in scores:
                if score.final_score > 0.3:
                    print(f"  {score.axis_type.name}: {score.final_score:.3f}")
                    print(f"  Semantic: {score.semantic_score:.3f}")
                    print(f"  Value: {score.value_score:.3f}")
            
            results[column] = scores
        return results

    def analyze_column(self, series: pd.Series, column_name: str) -> List[AxisScore]:
        """Enhanced column analysis with better scoring"""
        # Get value-based type information
        base_type, type_metadata = self.value_detector._analyze_column(series)
        
        # Get pattern and semantic scores
        pattern_scores = {}
        semantic_scores = {}
        match_info = {}
        
        for axis_type in AxisType:
            if axis_type == AxisType.UNKNOWN:
                continue
                
            # Get pattern scores
            pattern = self.pattern_matcher.get_pattern(axis_type)
            pattern_score, pattern_matches = self.pattern_matcher.match_pattern(
                column_name, pattern
            )
            
            # Get semantic scores
            semantic_score, semantic_matches = self.embedding_handler.get_semantic_score(
                column_name, pattern
            )
            
            # Take max of pattern and semantic scores
            pattern_scores[axis_type] = pattern_score
            semantic_scores[axis_type] = semantic_score
            match_info[axis_type] = {
                'pattern_matches': pattern_matches,
                'semantic_matches': semantic_matches
            }
        
        # Combine scores and generate results
        return self._combine_scores(
            base_type,
            type_metadata,
            pattern_scores,
            semantic_scores,
            match_info,
            column_name
        )

    def _combine_scores(
        self,
        base_type: BaseType,
        type_metadata: TypeMetadata,
        pattern_scores: Dict[AxisType, float],
        semantic_scores: Dict[AxisType, float],
        match_info: Dict[AxisType, Dict],
        column_name: str
    ) -> List[AxisScore]:
        """Enhanced score combination with better normalization"""
        results = []
        
        for axis_type in AxisType:
            if axis_type == AxisType.UNKNOWN:
                continue
                
            # Get base scores
            pattern_score = pattern_scores.get(axis_type, 0)
            semantic_score = semantic_scores.get(axis_type, 0)
            
            # Take maximum of pattern and semantic scores
            base_score = max(pattern_score, semantic_score)
            
            # Get type compatibility score
            type_compat = self.type_mapping.get(base_type, {}).get(axis_type, {})
            type_boost = type_compat.get('boost', 1.0)
            
            # Additional scoring adjustments
            if axis_type == AxisType.CURRENCY and ('rate' in column_name.lower() or 'ratio' in column_name.lower()):
                base_score *= 0.4
                
            if axis_type == AxisType.DIMENSION and any(k in column_name.lower() for k in type_compat.get('keywords', [])):
                base_score *= 1.2
                
            if axis_type == AxisType.GEOGRAPHIC and any(k in column_name.lower() for k in type_compat.get('keywords', [])):
                base_score *= 1.2
            
            # Calculate final score
            final_score = base_score * type_boost
            
            # Apply validation if available
            if type_compat.get('validation'):
                if type_compat['validation'](type_metadata.statistics):
                    final_score *= 1.1
                else:
                    final_score *= 0.7  # Stronger penalty for validation failure
            
            # Add to results if score is significant
            if final_score > 0.3:
                results.append(AxisScore(
                    axis_type=axis_type,
                    semantic_score=semantic_score,
                    value_score=type_boost,
                    final_score=final_score,
                    metadata={
                        'base_type': base_type,
                        'type_metadata': type_metadata.statistics,
                        'matches': match_info[axis_type],
                        'validation': type_metadata.validation_info
                    }
                ))
        
        return sorted(results, key=lambda x: x.final_score, reverse=True)