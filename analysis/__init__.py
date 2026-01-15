"""
Analysis module for Arsenal Intelligence Brief.

This module contains analytical tools for:
- Odds conversion and analysis
- ML predictions
- Sentiment analysis
- Value betting calculations
"""

from .odds_analyzer import (
    OddsAnalyzer,
    OddsData,
    MatchOdds,
    decimal_to_probability,
    fractional_to_decimal,
    american_to_decimal,
    remove_overround,
    calculate_expected_value,
    convert_odds_batch,
)

from .value_calculator import (
    ValueCalculator,
    ValueOpportunity,
    ValueAnalysisResult,
    MLPrediction,
    BookmakerOddsInput,
    MarketType,
    RiskLevel,
    ConfidenceLevel,
    rank_opportunities,
    filter_by_risk,
    filter_high_value_only,
)

from .ml_predictor import (
    MatchPredictor,
    MatchFeatures,
    FormStats,
    H2HStats,
    PredictionResult,
    CrossValidationResults,
    generate_synthetic_training_data,
    MODEL_VERSION,
)

from .sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    ArticleSentiment,
    CommentSentiment,
    ThemeExtraction,
    SentimentSummaryReport,
    analyze_news_data,
    calculate_sentiment_score,
    batch_sentiment_scores,
)

__all__ = [
    # Odds Analyzer
    'OddsAnalyzer',
    'OddsData',
    'MatchOdds',
    'decimal_to_probability',
    'fractional_to_decimal',
    'american_to_decimal',
    'remove_overround',
    'calculate_expected_value',
    'convert_odds_batch',
    # Value Calculator
    'ValueCalculator',
    'ValueOpportunity',
    'ValueAnalysisResult',
    'MLPrediction',
    'BookmakerOddsInput',
    'MarketType',
    'RiskLevel',
    'ConfidenceLevel',
    'rank_opportunities',
    'filter_by_risk',
    'filter_high_value_only',
    # ML Predictor
    'MatchPredictor',
    'MatchFeatures',
    'FormStats',
    'H2HStats',
    'PredictionResult',
    'CrossValidationResults',
    'generate_synthetic_training_data',
    'MODEL_VERSION',
    # Sentiment Analyzer
    'SentimentAnalyzer',
    'SentimentResult',
    'ArticleSentiment',
    'CommentSentiment',
    'ThemeExtraction',
    'SentimentSummaryReport',
    'analyze_news_data',
    'calculate_sentiment_score',
    'batch_sentiment_scores',
]
