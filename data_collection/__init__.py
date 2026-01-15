"""
Arsenal Intelligence Brief - Data Collection Module

This module provides tools for collecting betting odds, lineup information,
historical match data, and news data for Arsenal matches.
"""

from .odds_data import OddsData, OddsConverter
from .odds_fetcher import OddsFetcher, APIError, RateLimitExceeded
from .historical_data import HistoricalDataCollector, MatchResult
from .news_scraper import NewsScraper, NewsData, Article, Quote, QuoteExtractor
from .lineup_scraper import (
    LineupScraper,
    LineupData,
    LineupPrediction,
    InjuryInfo,
    InjuryParser,
    LineupExtractor,
    RedditScraper,
    TwitterScraper,
    SourceReliability,
    PlayerPosition,
)

__all__ = [
    'OddsData',
    'OddsConverter',
    'OddsFetcher',
    'APIError',
    'RateLimitExceeded',
    'HistoricalDataCollector',
    'MatchResult',
    'NewsScraper',
    'NewsData',
    'Article',
    'Quote',
    'QuoteExtractor',
    # Lineup scraper exports
    'LineupScraper',
    'LineupData',
    'LineupPrediction',
    'InjuryInfo',
    'InjuryParser',
    'LineupExtractor',
    'RedditScraper',
    'TwitterScraper',
    'SourceReliability',
    'PlayerPosition',
]
