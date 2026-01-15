#!/usr/bin/env python3
"""
Integration Tests for Arsenal Intelligence Brief

This module provides comprehensive integration tests for the complete pipeline:
fetch -> analyze -> report

Test scenarios covered:
1. Happy path: All data sources available
2. Partial data: Some sources fail but report still generates
3. No data: All sources fail, graceful error handling
4. Invalid data: Malformed API responses
5. Edge cases: Empty arrays, missing fields, null values

Task: arsenalScript-vqp.46 - Integration tests
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules under test
from data_collection.odds_fetcher import OddsFetcher, APIError, RateLimitExceeded
from data_collection.odds_data import OddsData
from data_collection.historical_data import HistoricalDataCollector, MatchResult
from analysis.value_calculator import (
    ValueCalculator, MLPrediction, BookmakerOddsInput,
    ValueOpportunity, ValueAnalysisResult, MarketType, RiskLevel, ConfidenceLevel
)
from analysis.odds_analyzer import OddsAnalyzer
from reporting.report_builder import (
    ReportBuilder, IntelligenceBrief, DataCompleteness, ReportSection, DataSource
)


# =============================================================================
# FIXTURES - Mock Data Conforming to Schemas
# =============================================================================

@pytest.fixture
def mock_match_id():
    """Standard match ID for testing."""
    return "20260120_ARS_CHE"


@pytest.fixture
def mock_odds_data():
    """
    Mock odds data conforming to data/schemas/odds.json schema.
    """
    return {
        "match_id": "20260120_ARS_CHE",
        "timestamp": "2026-01-18T14:30:00Z",
        "bookmaker_odds": [
            {
                "bookmaker": "bet365",
                "home_win": 2.10,
                "draw": 3.40,
                "away_win": 3.50,
                "over_2_5": 1.85,
                "under_2_5": 1.95
            },
            {
                "bookmaker": "william_hill",
                "home_win": 2.15,
                "draw": 3.30,
                "away_win": 3.45,
                "over_2_5": 1.80,
                "under_2_5": 2.00
            },
            {
                "bookmaker": "paddy_power",
                "home_win": 2.08,
                "draw": 3.50,
                "away_win": 3.40,
                "over_2_5": 1.87,
                "under_2_5": 1.93
            }
        ],
        "best_value": {
            "home_win": {"bookmaker": "william_hill", "odds": 2.15},
            "draw": {"bookmaker": "paddy_power", "odds": 3.50},
            "away_win": {"bookmaker": "bet365", "odds": 3.50},
            "over_2_5": {"bookmaker": "paddy_power", "odds": 1.87},
            "under_2_5": {"bookmaker": "william_hill", "odds": 2.00}
        },
        "metadata": {
            "source": "the-odds-api",
            "fetch_timestamp": "2026-01-18T14:29:45Z"
        }
    }


@pytest.fixture
def mock_news_data():
    """
    Mock news data conforming to data/schemas/news.json schema.
    """
    return {
        "match_id": "20260120_ARS_CHE",
        "timestamp": "2026-01-19T12:00:00Z",
        "articles": [
            {
                "title": "Arsenal vs Chelsea: Key Battles to Watch in London Derby",
                "url": "https://www.bbc.co.uk/sport/football/arsenal-chelsea-preview",
                "source": "BBC Sport",
                "author": "Phil McNulty",
                "publish_date": "2026-01-19T08:30:00Z",
                "full_text": "The North London club faces their West London rivals in what promises to be a thrilling encounter. Arsenal's recent form has been impressive, with Arteta's side winning their last five matches. The Gunners will be confident heading into this clash.",
                "summary": "Analysis of key tactical matchups ahead of the Arsenal vs Chelsea fixture."
            },
            {
                "title": "Arteta: We Must Be at Our Best Against Chelsea",
                "url": "https://www.theguardian.com/football/arteta-chelsea-preview",
                "source": "The Guardian",
                "author": "Jacob Steinberg",
                "publish_date": "2026-01-18T16:45:00Z",
                "full_text": "Mikel Arteta has warned his players they must be at their absolute best to overcome Chelsea at the Emirates Stadium. The Arsenal manager praised his opponent's quality.",
                "summary": "Arteta previews the Chelsea match and discusses team news."
            },
            {
                "title": "Chelsea's Maresca expects tough test against Arsenal",
                "url": "https://www.skysports.com/chelsea-arsenal-preview",
                "source": "Sky Sports",
                "author": None,
                "publish_date": "2026-01-18T14:00:00Z",
                "full_text": "Chelsea manager Enzo Maresca has spoken about the difficulty of facing Arsenal at the Emirates, calling them one of the best teams in Europe.",
                "summary": None
            }
        ],
        "quotes": [
            {
                "speaker": "Mikel Arteta (Arsenal Manager)",
                "quote": "Chelsea are a top side with excellent players. We need to be at our absolute best to get the result we want.",
                "context": "pre-match press conference",
                "source_url": "https://www.arsenal.com/news/arteta-press-conference"
            },
            {
                "speaker": "Enzo Maresca (Chelsea Manager)",
                "quote": "Arsenal is one of the best teams in Europe right now. It will be a great test for us.",
                "context": "pre-match press conference",
                "source_url": "https://www.chelseafc.com/news/maresca-speaks"
            }
        ],
        "sentiment_scores": {
            "overall": 0.35,
            "by_source": [
                {"source": "BBC Sport", "score": 0.40},
                {"source": "The Guardian", "score": 0.30},
                {"source": "Sky Sports", "score": 0.35}
            ]
        },
        "metadata": {
            "sources_scraped": ["BBC Sport", "The Guardian", "Sky Sports", "Arsenal.com", "ChelseaFC.com"],
            "fetch_timestamp": "2026-01-19T11:55:00Z"
        }
    }


@pytest.fixture
def mock_lineup_data():
    """
    Mock lineup data conforming to data/schemas/lineups.json schema.
    """
    return {
        "match_id": "20260120_ARS_CHE",
        "timestamp": "2026-01-19T10:00:00Z",
        "injuries": [
            {
                "player": "Bukayo Saka",
                "status": "doubtful",
                "return_date": "2026-01-22",
                "source": "press_conference"
            },
            {
                "player": "Takehiro Tomiyasu",
                "status": "out",
                "return_date": None,
                "source": "official_club"
            }
        ],
        "rumored_lineup": {
            "formation": "4-3-3",
            "players": [
                {"position": "GK", "name": "David Raya", "confidence": 0.95},
                {"position": "RB", "name": "Ben White", "confidence": 0.90},
                {"position": "CB", "name": "William Saliba", "confidence": 0.98},
                {"position": "CB", "name": "Gabriel Magalhaes", "confidence": 0.98},
                {"position": "LB", "name": "Oleksandr Zinchenko", "confidence": 0.75},
                {"position": "CM", "name": "Declan Rice", "confidence": 0.95},
                {"position": "CM", "name": "Martin Odegaard", "confidence": 0.92},
                {"position": "CM", "name": "Thomas Partey", "confidence": 0.80},
                {"position": "RW", "name": "Gabriel Martinelli", "confidence": 0.70},
                {"position": "ST", "name": "Kai Havertz", "confidence": 0.88},
                {"position": "LW", "name": "Leandro Trossard", "confidence": 0.65}
            ]
        },
        "confirmed_lineup": None,
        "source_reliability": {
            "sources": [
                {
                    "name": "David Ornstein",
                    "type": "journalist",
                    "reliability_score": 0.92,
                    "historical_accuracy": {
                        "correct_predictions": 46,
                        "total_predictions": 50
                    }
                },
                {
                    "name": "Arsenal Official",
                    "type": "official",
                    "reliability_score": 1.0,
                    "historical_accuracy": {
                        "correct_predictions": 100,
                        "total_predictions": 100
                    }
                }
            ],
            "last_updated": "2026-01-17T00:00:00Z"
        }
    }


@pytest.fixture
def mock_historical_matches():
    """Mock historical match results."""
    return [
        MatchResult(
            date="2025-12-15",
            home_team="Arsenal FC",
            away_team="Manchester City FC",
            competition="Premier League",
            home_score=2,
            away_score=1,
            result="W",
            venue="home",
            match_id=1001
        ),
        MatchResult(
            date="2025-12-22",
            home_team="Aston Villa FC",
            away_team="Arsenal FC",
            competition="Premier League",
            home_score=0,
            away_score=2,
            result="W",
            venue="away",
            match_id=1002
        ),
        MatchResult(
            date="2025-12-28",
            home_team="Arsenal FC",
            away_team="Liverpool FC",
            competition="Premier League",
            home_score=2,
            away_score=2,
            result="D",
            venue="home",
            match_id=1003
        ),
        MatchResult(
            date="2026-01-04",
            home_team="Arsenal FC",
            away_team="Newcastle United FC",
            competition="Premier League",
            home_score=3,
            away_score=0,
            result="W",
            venue="home",
            match_id=1004
        ),
        MatchResult(
            date="2026-01-11",
            home_team="Tottenham Hotspur FC",
            away_team="Arsenal FC",
            competition="Premier League",
            home_score=0,
            away_score=2,
            result="W",
            venue="away",
            match_id=1005
        )
    ]


@pytest.fixture
def mock_ml_prediction():
    """Mock ML prediction data."""
    return {
        "probabilities": {
            "win": 52.0,
            "draw": 26.0,
            "loss": 22.0
        },
        "confidence_intervals": {
            "win": {"lower": 45.0, "upper": 59.0},
            "draw": {"lower": 20.0, "upper": 32.0},
            "loss": {"lower": 16.0, "upper": 28.0}
        },
        "predicted_outcome": "W",
        "prediction_confidence": 85.0,
        "model_version": "1.0.0",
        "timestamp": "2026-01-19T10:00:00Z"
    }


@pytest.fixture
def mock_sentiment_data():
    """Mock sentiment analysis data."""
    return {
        "match_id": "20260120_ARS_CHE",
        "generated_at": "2026-01-19T12:00:00Z",
        "overall_sentiment": 0.35,
        "overall_label": "positive",
        "sentiment_distribution": {
            "positive": 8,
            "neutral": 4,
            "negative": 2
        },
        "articles_analyzed": 14,
        "comments_analyzed": 0,
        "sentiment_by_source": {
            "Arsenal.com": 0.6,
            "BBC Sport": 0.2,
            "The Guardian": 0.1
        },
        "key_insights": [
            "Positive coverage dominated by Arsenal's strong form",
            "Minor concerns about defensive depth",
            "Arteta praised for tactical flexibility"
        ],
        "themes": {
            "keywords": [("arsenal", 45), ("chelsea", 38), ("derby", 22)],
            "bigrams": [("london derby", 15), ("premier league", 12)],
            "entities": [("Mikel Arteta", "PERSON", 28), ("Bukayo Saka", "PERSON", 24)],
            "trending_concerns": ["injury concerns for saka"],
            "trending_optimism": ["strong recent form", "home advantage"]
        },
        "articles": []
    }


@pytest.fixture
def mock_value_analysis():
    """Mock value betting analysis data."""
    return {
        "match_id": "20260120_ARS_CHE",
        "analysis_timestamp": "2026-01-19T12:00:00Z",
        "opportunities": [
            {
                "market": "home_win",
                "expected_value": 0.092,
                "expected_value_percentage": 9.2,
                "ml_probability": 0.52,
                "implied_probability": 0.476,
                "edge": 0.044,
                "edge_percentage": 4.4,
                "decimal_odds": 2.15,
                "bookmaker": "William Hill",
                "confidence_level": "high",
                "risk_level": "low",
                "kelly_fraction": 0.085,
                "is_high_value": True
            },
            {
                "market": "over_2_5",
                "expected_value": 0.058,
                "expected_value_percentage": 5.8,
                "ml_probability": 0.55,
                "implied_probability": 0.520,
                "edge": 0.030,
                "edge_percentage": 3.0,
                "decimal_odds": 1.87,
                "bookmaker": "Paddy Power",
                "confidence_level": "medium",
                "risk_level": "medium",
                "kelly_fraction": 0.062,
                "is_high_value": True
            }
        ],
        "high_value_count": 2,
        "total_markets_analyzed": 5,
        "best_opportunity": {
            "market": "home_win",
            "expected_value_percentage": 9.2
        },
        "summary": {
            "total_opportunities": 2,
            "markets_analyzed": 5,
            "average_ev": 7.5,
            "max_ev": 9.2,
            "min_ev": 5.8,
            "high_value_markets": ["home_win", "over_2_5"],
            "recommendation": "Strong value identified in home win market. Consider positions in: home_win, over_2_5."
        }
    }


@pytest.fixture
def mock_api_response():
    """Mock response from The Odds API."""
    return [
        {
            "id": "abc123def456",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-01-20T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.10},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Chelsea", "price": 3.50}
                            ]
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 2.5, "price": 1.85},
                                {"name": "Under", "point": 2.5, "price": 1.95}
                            ]
                        }
                    ]
                },
                {
                    "key": "william_hill",
                    "title": "William Hill",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.15},
                                {"name": "Draw", "price": 3.30},
                                {"name": "Chelsea", "price": 3.45}
                            ]
                        }
                    ]
                }
            ]
        }
    ]


# =============================================================================
# TEST CLASSES - Happy Path
# =============================================================================

class TestHappyPath:
    """Test scenarios where all data sources are available and working."""

    def test_full_pipeline_with_all_data(
        self,
        mock_match_id,
        mock_odds_data,
        mock_news_data,
        mock_ml_prediction,
        mock_sentiment_data,
        mock_value_analysis,
        mock_lineup_data
    ):
        """Test complete pipeline with all data sources available."""
        # Create report builder
        builder = ReportBuilder(include_charts=False)

        # Build report with all data
        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            match_date="2026-01-20T15:00:00Z",
            competition="Premier League",
            odds_data=mock_odds_data,
            ml_prediction=mock_ml_prediction,
            sentiment_data=mock_sentiment_data,
            value_analysis=mock_value_analysis,
            news_data=mock_news_data,
            lineup_data=mock_lineup_data
        )

        # Verify report structure
        assert isinstance(report, IntelligenceBrief)
        assert report.match_id == mock_match_id
        assert report.home_team == "Arsenal"
        assert report.away_team == "Chelsea"

        # Verify data completeness
        assert report.data_completeness.odds_available is True
        assert report.data_completeness.ml_prediction_available is True
        assert report.data_completeness.sentiment_available is True
        assert report.data_completeness.value_available is True
        assert report.data_completeness.news_available is True
        assert report.data_completeness.lineup_available is True

        # Verify completeness score (5 core sources available = 100%)
        assert report.data_completeness.completeness_score == 100.0

        # Verify sections were created
        assert len(report.sections) > 0

        # Verify no errors
        assert len(report.data_completeness.errors) == 0

    def test_value_calculator_with_valid_predictions(self, mock_odds_data):
        """Test value calculator with valid ML predictions and odds."""
        calculator = ValueCalculator()

        # Create predictions
        predictions = {
            MarketType.HOME_WIN: MLPrediction(probability=0.52, confidence=0.85),
            MarketType.DRAW: MLPrediction(probability=0.26, confidence=0.78),
            MarketType.AWAY_WIN: MLPrediction(probability=0.22, confidence=0.80)
        }

        # Convert bookmaker odds to BookmakerOddsInput
        bookmaker_odds = [
            BookmakerOddsInput(
                bookmaker=bm["bookmaker"],
                home_win=bm["home_win"],
                draw=bm["draw"],
                away_win=bm["away_win"],
                over_2_5=bm.get("over_2_5"),
                under_2_5=bm.get("under_2_5")
            )
            for bm in mock_odds_data["bookmaker_odds"]
        ]

        # Analyze match
        result = calculator.analyze_match(
            match_id="20260120_ARS_CHE",
            predictions=predictions,
            bookmaker_odds=bookmaker_odds
        )

        # Verify result structure
        assert isinstance(result, ValueAnalysisResult)
        assert result.match_id == "20260120_ARS_CHE"
        assert result.total_markets_analyzed >= 0
        assert isinstance(result.opportunities, list)

    def test_report_outputs_html_and_markdown(
        self,
        mock_match_id,
        mock_odds_data,
        mock_ml_prediction
    ):
        """Test that reports can be rendered to HTML and Markdown."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=mock_odds_data,
            ml_prediction=mock_ml_prediction
        )

        # Test Markdown output
        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert "Arsenal" in markdown
        assert "Chelsea" in markdown
        assert mock_match_id in markdown

        # Test HTML output
        html = report.to_html()
        assert isinstance(html, str)
        assert "<html" in html.lower() or "<!doctype" in html.lower()

    def test_report_json_serialization(
        self,
        mock_match_id,
        mock_odds_data,
        mock_ml_prediction
    ):
        """Test that reports can be serialized to JSON."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=mock_odds_data,
            ml_prediction=mock_ml_prediction
        )

        # Test JSON output
        json_str = report.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["match_id"] == mock_match_id
        assert parsed["home_team"] == "Arsenal"
        assert "sections" in parsed


# =============================================================================
# TEST CLASSES - Partial Data
# =============================================================================

class TestPartialData:
    """Test scenarios where some data sources fail but report still generates."""

    def test_report_with_only_odds_data(self, mock_match_id, mock_odds_data):
        """Test report generation with only odds data available."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=mock_odds_data
        )

        # Verify report still generates
        assert isinstance(report, IntelligenceBrief)
        assert report.data_completeness.odds_available is True
        assert report.data_completeness.ml_prediction_available is False
        assert report.data_completeness.sentiment_available is False

        # Verify warnings are generated for missing data
        assert len(report.data_completeness.warnings) > 0

    def test_report_with_only_ml_prediction(self, mock_match_id, mock_ml_prediction):
        """Test report generation with only ML prediction available."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            ml_prediction=mock_ml_prediction
        )

        # Verify report still generates
        assert isinstance(report, IntelligenceBrief)
        assert report.data_completeness.ml_prediction_available is True
        assert report.data_completeness.odds_available is False

    def test_report_with_odds_and_news_only(
        self,
        mock_match_id,
        mock_odds_data,
        mock_news_data
    ):
        """Test report with odds and news but missing ML predictions."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=mock_odds_data,
            news_data=mock_news_data
        )

        assert report.data_completeness.odds_available is True
        assert report.data_completeness.news_available is True
        assert report.data_completeness.ml_prediction_available is False
        assert report.data_completeness.value_available is False

    def test_value_calculator_with_single_bookmaker(self):
        """Test value calculator works with only one bookmaker."""
        calculator = ValueCalculator()

        predictions = {
            MarketType.HOME_WIN: MLPrediction(probability=0.55, confidence=0.80)
        }

        bookmaker_odds = [
            BookmakerOddsInput(
                bookmaker="bet365",
                home_win=2.00,
                draw=3.50,
                away_win=4.00
            )
        ]

        result = calculator.analyze_match(
            match_id="test_match",
            predictions=predictions,
            bookmaker_odds=bookmaker_odds
        )

        assert isinstance(result, ValueAnalysisResult)
        assert result.total_markets_analyzed >= 0


# =============================================================================
# TEST CLASSES - No Data
# =============================================================================

class TestNoData:
    """Test scenarios where all data sources fail."""

    def test_report_with_no_data(self, mock_match_id):
        """Test report generation with no data sources available."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea"
        )

        # Verify report still generates (graceful degradation)
        assert isinstance(report, IntelligenceBrief)
        assert report.match_id == mock_match_id

        # Verify data completeness reflects missing data
        assert report.data_completeness.odds_available is False
        assert report.data_completeness.ml_prediction_available is False
        assert report.data_completeness.news_available is False
        assert report.data_completeness.sentiment_available is False
        assert report.data_completeness.value_available is False

        # Verify completeness score is 0
        assert report.data_completeness.completeness_score == 0.0

        # Verify warnings are present
        assert len(report.data_completeness.warnings) > 0

    def test_report_markdown_with_no_data(self, mock_match_id):
        """Test Markdown generation still works with no data."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea"
        )

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert mock_match_id in markdown

    def test_value_calculator_with_no_predictions(self):
        """Test value calculator handles empty predictions gracefully."""
        calculator = ValueCalculator()

        predictions = {}
        bookmaker_odds = []

        result = calculator.analyze_match(
            match_id="test_match",
            predictions=predictions,
            bookmaker_odds=bookmaker_odds
        )

        assert isinstance(result, ValueAnalysisResult)
        assert result.total_markets_analyzed == 0
        assert len(result.opportunities) == 0
        assert result.best_opportunity is None

    def test_data_completeness_critical_data_missing(self):
        """Test critical data missing flag works correctly."""
        completeness = DataCompleteness()

        # Both odds and ML missing - critical
        assert completeness.is_critical_data_missing is True

        # Only odds available
        completeness.odds_available = True
        assert completeness.is_critical_data_missing is False

        # Reset and only ML available
        completeness.odds_available = False
        completeness.ml_prediction_available = True
        assert completeness.is_critical_data_missing is False


# =============================================================================
# TEST CLASSES - Invalid Data
# =============================================================================

class TestInvalidData:
    """Test scenarios with malformed API responses and invalid data."""

    def test_report_with_empty_odds_array(self, mock_match_id):
        """Test report handles empty bookmaker odds array."""
        builder = ReportBuilder(include_charts=False)

        invalid_odds = {
            "match_id": mock_match_id,
            "timestamp": "2026-01-18T14:30:00Z",
            "bookmaker_odds": [],  # Empty array
            "best_value": {},
            "metadata": {"source": "test"}
        }

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=invalid_odds
        )

        # Should still generate report
        assert isinstance(report, IntelligenceBrief)

    def test_report_with_missing_fields(self, mock_match_id):
        """Test report handles odds data with missing fields."""
        builder = ReportBuilder(include_charts=False)

        incomplete_odds = {
            "match_id": mock_match_id,
            # Missing timestamp, bookmaker_odds, best_value, metadata
        }

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=incomplete_odds
        )

        # Should still generate report
        assert isinstance(report, IntelligenceBrief)

    def test_ml_prediction_invalid_probability(self):
        """Test MLPrediction validates probability bounds."""
        # Probability > 1 should raise ValueError
        with pytest.raises(ValueError):
            MLPrediction(probability=1.5, confidence=0.8)

        # Probability < 0 should raise ValueError
        with pytest.raises(ValueError):
            MLPrediction(probability=-0.1, confidence=0.8)

    def test_ml_prediction_invalid_confidence(self):
        """Test MLPrediction validates confidence bounds."""
        # Confidence > 1 should raise ValueError
        with pytest.raises(ValueError):
            MLPrediction(probability=0.5, confidence=1.5)

        # Confidence < 0 should raise ValueError
        with pytest.raises(ValueError):
            MLPrediction(probability=0.5, confidence=-0.1)

    def test_value_calculator_with_invalid_odds(self):
        """Test value calculator handles invalid odds values."""
        calculator = ValueCalculator()

        # Odds < 1.0 should raise error
        with pytest.raises(ValueError):
            calculator.calculate_expected_value(0.5, 0.5)

    def test_value_calculator_invalid_probability(self):
        """Test value calculator validates probability."""
        calculator = ValueCalculator()

        with pytest.raises(ValueError):
            calculator.calculate_expected_value(1.5, 2.0)  # prob > 1

        with pytest.raises(ValueError):
            calculator.calculate_expected_value(-0.1, 2.0)  # prob < 0

    def test_report_with_null_values(self, mock_match_id):
        """Test report handles null values in data."""
        builder = ReportBuilder(include_charts=False)

        news_with_nulls = {
            "match_id": mock_match_id,
            "timestamp": "2026-01-18T12:00:00Z",
            "articles": [
                {
                    "title": "Test Article",
                    "url": "https://example.com",
                    "source": "Test Source",
                    "author": None,  # null value
                    "publish_date": "2026-01-18T10:00:00Z",
                    "full_text": None,  # null value
                    "summary": None  # null value
                }
            ],
            "quotes": [],
            "sentiment_scores": {"overall": 0.0, "by_source": []},
            "metadata": {"sources_scraped": [], "fetch_timestamp": "2026-01-18T12:00:00Z"}
        }

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            news_data=news_with_nulls
        )

        assert isinstance(report, IntelligenceBrief)
        assert report.data_completeness.news_available is True


# =============================================================================
# TEST CLASSES - Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_value_calculator_zero_probability(self):
        """Test Kelly criterion with edge probability values."""
        calculator = ValueCalculator()

        # Edge case: very low probability
        kelly = calculator.calculate_kelly_criterion(0.01, 50.0)
        assert kelly >= 0  # Should be non-negative

        # Edge case: very high probability
        kelly = calculator.calculate_kelly_criterion(0.99, 1.01)
        assert kelly >= 0 and kelly <= calculator.max_kelly_fraction

    def test_value_calculator_boundary_odds(self):
        """Test EV calculation with boundary odds values."""
        calculator = ValueCalculator()

        # Minimum valid odds (1.0)
        ev, ev_pct = calculator.calculate_expected_value(0.5, 1.0)
        assert ev == -0.5  # 0.5 * 1.0 - 1 = -0.5

        # High odds
        ev, ev_pct = calculator.calculate_expected_value(0.1, 20.0)
        assert ev == 1.0  # 0.1 * 20 - 1 = 1.0

    def test_report_with_very_long_text(self, mock_match_id):
        """Test report handles very long article text."""
        builder = ReportBuilder(include_charts=False)

        long_text = "This is a test article. " * 1000  # Very long text

        news_data = {
            "match_id": mock_match_id,
            "timestamp": "2026-01-18T12:00:00Z",
            "articles": [
                {
                    "title": "A" * 500,  # Long title
                    "url": "https://example.com",
                    "source": "Test",
                    "author": None,
                    "publish_date": "2026-01-18T10:00:00Z",
                    "full_text": long_text,
                    "summary": None
                }
            ],
            "quotes": [],
            "sentiment_scores": {"overall": 0.0, "by_source": []},
            "metadata": {"sources_scraped": [], "fetch_timestamp": "2026-01-18T12:00:00Z"}
        }

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            news_data=news_data
        )

        assert isinstance(report, IntelligenceBrief)

    def test_special_characters_in_team_names(self):
        """Test report handles special characters in team names."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id="test_match",
            home_team="Arsenal FC (London)",
            away_team="Chelsea FC - The Blues",
            competition="Premier League 2025/26"
        )

        assert isinstance(report, IntelligenceBrief)
        markdown = report.to_markdown()
        assert "Arsenal FC (London)" in markdown

    def test_unicode_characters(self, mock_match_id):
        """Test report handles unicode characters properly."""
        builder = ReportBuilder(include_charts=False)

        news_data = {
            "match_id": mock_match_id,
            "timestamp": "2026-01-18T12:00:00Z",
            "articles": [
                {
                    "title": "Arsenal vs Chelsea: Odegaard scores!",  # Norwegian character
                    "url": "https://example.com",
                    "source": "Test",
                    "author": "Jorgen Muller",  # German umlauts
                    "publish_date": "2026-01-18T10:00:00Z",
                    "full_text": "Zinchenko played well.",
                    "summary": None
                }
            ],
            "quotes": [
                {
                    "speaker": "Mikel Arteta",
                    "quote": "Cest magnifique!",  # French
                    "context": "post-match",
                    "source_url": "https://example.com"
                }
            ],
            "sentiment_scores": {"overall": 0.5, "by_source": []},
            "metadata": {"sources_scraped": [], "fetch_timestamp": "2026-01-18T12:00:00Z"}
        }

        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            news_data=news_data
        )

        json_output = report.to_json()
        assert isinstance(json_output, str)

        # Verify it parses back correctly
        parsed = json.loads(json_output)
        assert "match_id" in parsed

    def test_multiple_high_value_opportunities(self):
        """Test value calculator with many high value opportunities."""
        calculator = ValueCalculator()

        # Create predictions for all market types
        predictions = {
            MarketType.HOME_WIN: MLPrediction(probability=0.55, confidence=0.90),
            MarketType.DRAW: MLPrediction(probability=0.28, confidence=0.85),
            MarketType.AWAY_WIN: MLPrediction(probability=0.17, confidence=0.82),
            MarketType.OVER_2_5: MLPrediction(probability=0.58, confidence=0.78),
            MarketType.UNDER_2_5: MLPrediction(probability=0.42, confidence=0.78),
        }

        # Odds that create multiple value opportunities
        bookmaker_odds = [
            BookmakerOddsInput(
                bookmaker="bet365",
                home_win=2.20,  # Implied 45.5%, ML 55%
                draw=4.00,      # Implied 25%, ML 28%
                away_win=6.00,  # Implied 16.7%, ML 17%
                over_2_5=1.85,  # Implied 54%, ML 58%
                under_2_5=2.10  # Implied 47.6%, ML 42%
            )
        ]

        result = calculator.analyze_match(
            match_id="test_match",
            predictions=predictions,
            bookmaker_odds=bookmaker_odds
        )

        assert isinstance(result, ValueAnalysisResult)
        # Opportunities are sorted by EV
        if len(result.opportunities) > 1:
            assert result.opportunities[0].expected_value >= result.opportunities[1].expected_value

    def test_confidence_level_categorization(self):
        """Test MLPrediction confidence level categorization."""
        # Very high confidence
        pred = MLPrediction(probability=0.5, confidence=0.95)
        assert pred.get_confidence_level() == ConfidenceLevel.VERY_HIGH

        # High confidence
        pred = MLPrediction(probability=0.5, confidence=0.80)
        assert pred.get_confidence_level() == ConfidenceLevel.HIGH

        # Medium confidence
        pred = MLPrediction(probability=0.5, confidence=0.65)
        assert pred.get_confidence_level() == ConfidenceLevel.MEDIUM

        # Low confidence
        pred = MLPrediction(probability=0.5, confidence=0.50)
        assert pred.get_confidence_level() == ConfidenceLevel.LOW

        # Very low confidence
        pred = MLPrediction(probability=0.5, confidence=0.30)
        assert pred.get_confidence_level() == ConfidenceLevel.VERY_LOW

    def test_risk_level_assessment(self):
        """Test value calculator risk assessment."""
        calculator = ValueCalculator()

        # Low risk: low odds, high confidence, high probability
        risk = calculator.assess_risk_level(
            ml_probability=0.55,
            implied_probability=0.50,
            confidence=0.90,
            decimal_odds=1.80
        )
        assert risk in [RiskLevel.VERY_LOW, RiskLevel.LOW]

        # High risk: high odds, low confidence, low probability
        risk = calculator.assess_risk_level(
            ml_probability=0.15,
            implied_probability=0.10,
            confidence=0.40,
            decimal_odds=8.00
        )
        assert risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]


# =============================================================================
# TEST CLASSES - Data Completeness
# =============================================================================

class TestDataCompleteness:
    """Test DataCompleteness tracking functionality."""

    def test_completeness_score_calculation(self):
        """Test completeness score is calculated correctly."""
        completeness = DataCompleteness()

        # 0 sources
        assert completeness.completeness_score == 0.0

        # 1 of 5 sources
        completeness.odds_available = True
        assert completeness.completeness_score == 20.0

        # 2 of 5 sources
        completeness.ml_prediction_available = True
        assert completeness.completeness_score == 40.0

        # 5 of 5 sources
        completeness.news_available = True
        completeness.sentiment_available = True
        completeness.value_available = True
        assert completeness.completeness_score == 100.0

    def test_warnings_and_errors(self):
        """Test warning and error tracking."""
        completeness = DataCompleteness()

        completeness.add_warning("Test warning 1")
        completeness.add_warning("Test warning 2")

        assert len(completeness.warnings) == 2
        assert "Test warning 1" in completeness.warnings

        completeness.add_error("Test error")
        assert len(completeness.errors) == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        completeness = DataCompleteness()
        completeness.odds_available = True
        completeness.add_warning("Test warning")

        result = completeness.to_dict()

        assert "completeness_score" in result
        assert "sources" in result
        assert result["sources"]["odds"] is True
        assert result["sources"]["ml_prediction"] is False
        assert "Test warning" in result["warnings"]


# =============================================================================
# TEST CLASSES - Integration with Mocked External APIs
# =============================================================================

class TestMockedAPIIntegration:
    """Test integration with mocked external APIs."""

    @patch('data_collection.odds_fetcher.urllib.request.urlopen')
    def test_odds_fetcher_with_mocked_api(self, mock_urlopen, mock_api_response):
        """Test OddsFetcher with mocked API response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_response.headers = {"x-requests-remaining": "499"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Create fetcher with test API key
        with patch.dict('os.environ', {'THE_ODDS_API_KEY': 'test_key'}):
            fetcher = OddsFetcher()

            # This would normally make an API call
            matches = fetcher.fetch_upcoming_matches(use_cache=False)

            # Verify the mock was called
            assert mock_urlopen.called

            # Verify response structure
            assert isinstance(matches, list)

    def test_historical_data_collector_form_stats(self, mock_historical_matches):
        """Test form statistics calculation from historical data."""
        collector = HistoricalDataCollector.__new__(HistoricalDataCollector)
        collector._matches = mock_historical_matches
        collector.api_key = "test"
        collector.session = MagicMock()

        form = collector.get_form_stats(n_games=5)

        assert form["n_games"] == 5
        assert form["wins"] == 4
        assert form["draws"] == 1
        assert form["losses"] == 0
        assert form["points"] == 13
        assert form["max_points"] == 15
        assert form["form_string"] == "WWDWW"

    def test_historical_data_collector_h2h(self, mock_historical_matches):
        """Test head-to-head record calculation."""
        collector = HistoricalDataCollector.__new__(HistoricalDataCollector)
        collector._matches = mock_historical_matches
        collector.api_key = "test"
        collector.session = MagicMock()

        h2h = collector.get_head_to_head("Liverpool")

        assert h2h["matches_played"] == 1
        assert h2h["draws"] == 1
        assert h2h["win_rate"] == 0.0

    def test_odds_data_conversion(self, mock_api_response):
        """Test conversion from API response to OddsData format."""
        match = mock_api_response[0]

        # Generate match ID
        match_id = OddsData.generate_match_id(
            match["commence_time"],
            match["home_team"],
            match["away_team"]
        )

        assert "20260120" in match_id
        assert "ARS" in match_id or "Arsenal" in match_id


# =============================================================================
# TEST CLASSES - Report Section Generation
# =============================================================================

class TestReportSections:
    """Test individual report section generation."""

    def test_fixture_section_generation(self):
        """Test fixture information section is generated correctly."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id="test_match",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date="2026-01-20T15:00:00Z",
            competition="Premier League"
        )

        fixture_section = report.get_section("Fixture Information")
        assert fixture_section is not None
        assert fixture_section.is_available is True
        assert "Arsenal" in fixture_section.content
        assert "Chelsea" in fixture_section.content

    def test_missing_section_placeholder(self):
        """Test missing data sections show placeholder."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id="test_match",
            home_team="Arsenal",
            away_team="Chelsea"
            # No data provided
        )

        # Find odds section (should be unavailable)
        odds_section = report.get_section("Odds Comparison")
        assert odds_section is not None
        assert odds_section.is_available is False
        assert "not available" in odds_section.content.lower()

    def test_odds_section_with_data(self, mock_odds_data):
        """Test odds section is properly generated with data."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id="test_match",
            home_team="Arsenal",
            away_team="Chelsea",
            odds_data=mock_odds_data
        )

        odds_section = report.get_section("Odds Comparison")
        assert odds_section is not None
        assert odds_section.is_available is True
        assert odds_section.source == DataSource.ODDS

    def test_value_section_with_opportunities(self, mock_value_analysis):
        """Test value betting section shows opportunities."""
        builder = ReportBuilder(include_charts=False)

        report = builder.build_report(
            match_id="test_match",
            home_team="Arsenal",
            away_team="Chelsea",
            value_analysis=mock_value_analysis
        )

        value_section = report.get_section("Value Betting")
        assert value_section is not None
        assert value_section.is_available is True
        # Should mention home_win as high value
        assert "home" in value_section.content.lower() or "opportunity" in value_section.content.lower()


# =============================================================================
# TEST CLASSES - Value Analysis
# =============================================================================

class TestValueAnalysis:
    """Test value betting analysis functionality."""

    def test_expected_value_calculation(self):
        """Test EV calculation formula: EV = (ML prob x odds) - 1."""
        calculator = ValueCalculator()

        # Test case: 52% ML prob, 2.10 odds
        # EV = 0.52 * 2.10 - 1 = 0.092 (9.2%)
        ev, ev_pct = calculator.calculate_expected_value(0.52, 2.10)

        assert abs(ev - 0.092) < 0.001
        assert abs(ev_pct - 9.2) < 0.1

    def test_edge_calculation(self):
        """Test edge calculation: ML prob - implied prob."""
        calculator = ValueCalculator()

        edge, edge_pct = calculator.calculate_edge(0.52, 0.476)  # implied from 2.10 odds

        assert abs(edge - 0.044) < 0.001
        assert abs(edge_pct - 4.4) < 0.1

    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion stake calculation."""
        calculator = ValueCalculator()

        # With positive edge
        kelly = calculator.calculate_kelly_criterion(0.55, 2.00)
        assert kelly > 0
        assert kelly <= calculator.max_kelly_fraction

        # No edge (fair odds)
        kelly = calculator.calculate_kelly_criterion(0.50, 2.00)
        assert kelly == 0.0

        # Negative edge
        kelly = calculator.calculate_kelly_criterion(0.40, 2.00)
        assert kelly == 0.0

    def test_value_opportunity_summary(self):
        """Test value opportunity generates correct summary."""
        opportunity = ValueOpportunity(
            market=MarketType.HOME_WIN,
            expected_value=0.092,
            expected_value_percentage=9.2,
            ml_probability=0.52,
            implied_probability=0.476,
            edge=0.044,
            edge_percentage=4.4,
            decimal_odds=2.10,
            bookmaker="bet365",
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.LOW,
            kelly_fraction=0.085,
            is_high_value=True
        )

        summary = opportunity.to_brief_summary()

        assert "Home Win" in summary
        assert "9.2%" in summary
        assert "bet365" in summary

    def test_analyze_from_dict_convenience_method(self):
        """Test analyze_from_dict convenience method."""
        calculator = ValueCalculator()

        ml_predictions = {
            "home_win": {"probability": 0.52, "confidence": 0.85},
            "draw": {"probability": 0.26, "confidence": 0.80},
            "away_win": {"probability": 0.22, "confidence": 0.82}
        }

        bookmaker_odds_list = [
            {
                "bookmaker": "bet365",
                "home_win": 2.10,
                "draw": 3.40,
                "away_win": 3.50
            }
        ]

        result = calculator.analyze_from_dict(
            match_id="test_match",
            ml_predictions=ml_predictions,
            bookmaker_odds_list=bookmaker_odds_list
        )

        assert isinstance(result, ValueAnalysisResult)
        assert result.match_id == "test_match"


# =============================================================================
# TEST CLASSES - Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling throughout the pipeline."""

    def test_odds_fetcher_handles_api_error(self):
        """Test OddsFetcher handles API errors gracefully."""
        with patch.dict('os.environ', {'THE_ODDS_API_KEY': 'test_key'}):
            fetcher = OddsFetcher()

            with patch('data_collection.odds_fetcher.urllib.request.urlopen') as mock_urlopen:
                from urllib.error import HTTPError
                mock_urlopen.side_effect = HTTPError(
                    url="http://test.com",
                    code=500,
                    msg="Internal Server Error",
                    hdrs={},
                    fp=None
                )

                with pytest.raises(APIError):
                    fetcher.fetch_upcoming_matches(use_cache=False)

    def test_odds_fetcher_handles_rate_limit(self):
        """Test OddsFetcher handles rate limit errors."""
        with patch.dict('os.environ', {'THE_ODDS_API_KEY': 'test_key'}):
            fetcher = OddsFetcher()
            fetcher.daily_request_count = 100  # Exceed limit

            with pytest.raises(RateLimitExceeded):
                fetcher._enforce_rate_limit()

    def test_report_builder_handles_malformed_sentiment(self, mock_match_id):
        """Test ReportBuilder handles malformed sentiment data."""
        builder = ReportBuilder(include_charts=False)

        malformed_sentiment = {
            "match_id": mock_match_id,
            # Missing required fields
            "overall_sentiment": "not_a_number",  # Wrong type
        }

        # Should not raise exception
        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            sentiment_data=malformed_sentiment
        )

        assert isinstance(report, IntelligenceBrief)


# =============================================================================
# TEST CLASSES - End to End
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_workflow(
        self,
        mock_match_id,
        mock_odds_data,
        mock_news_data,
        mock_ml_prediction,
        mock_sentiment_data,
        mock_value_analysis,
        mock_lineup_data,
        mock_historical_matches
    ):
        """
        Test complete workflow from data to final report.

        This test simulates the full pipeline:
        1. Collect data (mocked)
        2. Run analysis (value calculator)
        3. Generate report
        4. Output in multiple formats
        """
        # Step 1: Value Analysis
        calculator = ValueCalculator()
        value_result = calculator.analyze_from_dict(
            match_id=mock_match_id,
            ml_predictions={
                "home_win": {"probability": 0.52, "confidence": 0.85},
                "draw": {"probability": 0.26, "confidence": 0.80},
                "away_win": {"probability": 0.22, "confidence": 0.82}
            },
            bookmaker_odds_list=[
                {
                    "bookmaker": bm["bookmaker"],
                    "home_win": bm["home_win"],
                    "draw": bm["draw"],
                    "away_win": bm["away_win"],
                    "over_2_5": bm.get("over_2_5"),
                    "under_2_5": bm.get("under_2_5")
                }
                for bm in mock_odds_data["bookmaker_odds"]
            ]
        )

        # Step 2: Build Report
        builder = ReportBuilder(include_charts=False)
        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            match_date="2026-01-20T15:00:00Z",
            competition="Premier League",
            odds_data=mock_odds_data,
            ml_prediction=mock_ml_prediction,
            sentiment_data=mock_sentiment_data,
            value_analysis=value_result.to_dict(),
            news_data=mock_news_data,
            lineup_data=mock_lineup_data
        )

        # Step 3: Verify Report
        assert isinstance(report, IntelligenceBrief)
        assert report.data_completeness.completeness_score >= 80.0

        # Step 4: Output Formats
        markdown = report.to_markdown()
        html = report.to_html()
        json_output = report.to_json()

        assert len(markdown) > 1000
        assert len(html) > 1000
        assert len(json_output) > 500

        # Verify JSON is valid
        parsed_json = json.loads(json_output)
        assert parsed_json["match_id"] == mock_match_id

    def test_graceful_degradation_workflow(self, mock_match_id, mock_odds_data):
        """
        Test workflow continues with partial data.

        Simulates real-world scenario where some APIs fail.
        """
        builder = ReportBuilder(include_charts=False)

        # Only odds available - simulate other API failures
        report = builder.build_report(
            match_id=mock_match_id,
            home_team="Arsenal",
            away_team="Chelsea",
            match_date="2026-01-20T15:00:00Z",
            competition="Premier League",
            odds_data=mock_odds_data,
            ml_prediction=None,  # API failed
            sentiment_data=None,  # API failed
            news_data=None,  # Scraper failed
        )

        # Report should still be generated
        assert isinstance(report, IntelligenceBrief)
        assert report.data_completeness.odds_available is True

        # Should have warnings for missing data
        assert len(report.data_completeness.warnings) >= 3

        # Can still produce output
        markdown = report.to_markdown()
        assert "Arsenal" in markdown
        assert "Data Completeness" in markdown


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
