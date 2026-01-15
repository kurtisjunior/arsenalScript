#!/usr/bin/env python3
"""
Arsenal Intelligence Brief Orchestrator

This module runs the full Intelligence Brief pipeline for Arsenal FC matches:

1. Data Collection Phase:
   - Fetch odds from The Odds API
   - Scrape news from Arsenal.com and BBC Sport
   - Load historical data

2. Analysis Phase:
   - Run ML predictions
   - Run sentiment analysis
   - Calculate value opportunities

3. Report Generation Phase:
   - Aggregate all data
   - Generate HTML intelligence brief
   - Send notifications (email)

Usage:
    # Basic usage
    python orchestrator.py --opponent "Chelsea" --match-date "2026-01-20"

    # With all options
    python orchestrator.py --opponent "Chelsea" --match-date "2026-01-20" \
        --venue home --send-email --output-dir ./reports

    # Environment variables:
    # THE_ODDS_API_KEY - Required for odds fetching
    # FOOTBALL_DATA_API_KEY - Required for historical data
    # SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD - For email notifications
    # EMAIL_RECIPIENTS - Comma-separated list of email addresses

Task: arsenalScript-vqp.47 - Orchestrator for full pipeline
"""

import argparse
import asyncio
import json
import logging
import os
import smtplib
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the orchestrator pipeline."""
    # Match info
    match_id: str = ""
    opponent: str = ""
    match_date: str = ""
    venue: str = "home"  # "home" or "away"
    opposing_manager: Optional[str] = None

    # API keys (read from environment)
    odds_api_key: Optional[str] = None
    football_data_api_key: Optional[str] = None

    # Output options
    output_dir: Path = REPORTS_DIR
    output_format: str = "html"  # "html", "json", or "both"

    # Email notification
    send_email: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)

    # Pipeline options
    skip_odds: bool = False
    skip_news: bool = False
    skip_historical: bool = False
    skip_ml: bool = False
    skip_sentiment: bool = False
    skip_value: bool = False

    # Error handling
    continue_on_error: bool = True  # Continue pipeline on non-critical failures

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> 'PipelineConfig':
        """Create config from environment variables and command-line args."""
        # Generate match_id if not provided
        match_id = args.match_id
        if not match_id and args.match_date and args.opponent:
            # Generate match ID
            match_date = args.match_date
            home_team = "Arsenal" if args.venue == "home" else args.opponent
            away_team = args.opponent if args.venue == "home" else "Arsenal"

            # Simple abbreviation
            team_abbrev = {
                "arsenal": "ARS", "chelsea": "CHE", "liverpool": "LIV",
                "manchester city": "MCI", "manchester united": "MUN",
                "tottenham": "TOT", "aston villa": "AVL", "newcastle": "NEW",
                "brighton": "BHA", "west ham": "WHU", "bournemouth": "BOU",
                "crystal palace": "CRY", "fulham": "FUL", "brentford": "BRE",
                "everton": "EVE", "nottingham forest": "NFO", "wolves": "WOL",
                "ipswich": "IPS", "leicester": "LEI", "southampton": "SOU"
            }

            def get_abbrev(name: str) -> str:
                return team_abbrev.get(name.lower(), name[:3].upper())

            date_str = match_date.replace("-", "")[:8]
            match_id = f"{date_str}_{get_abbrev(home_team)}_{get_abbrev(away_team)}"

        # Parse email recipients
        email_recipients = []
        env_recipients = os.environ.get("EMAIL_RECIPIENTS", "")
        if env_recipients:
            email_recipients = [e.strip() for e in env_recipients.split(",") if e.strip()]
        if hasattr(args, 'email_recipients') and args.email_recipients:
            email_recipients.extend(args.email_recipients)

        return cls(
            match_id=match_id,
            opponent=args.opponent,
            match_date=args.match_date,
            venue=args.venue,
            opposing_manager=getattr(args, 'opposing_manager', None),
            odds_api_key=os.environ.get("THE_ODDS_API_KEY"),
            football_data_api_key=os.environ.get("FOOTBALL_DATA_API_KEY"),
            output_dir=Path(args.output_dir) if args.output_dir else REPORTS_DIR,
            output_format=getattr(args, 'output_format', 'html'),
            send_email=getattr(args, 'send_email', False),
            smtp_host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER"),
            smtp_password=os.environ.get("SMTP_PASSWORD"),
            email_recipients=email_recipients,
            skip_odds=getattr(args, 'skip_odds', False),
            skip_news=getattr(args, 'skip_news', False),
            skip_historical=getattr(args, 'skip_historical', False),
            skip_ml=getattr(args, 'skip_ml', False),
            skip_sentiment=getattr(args, 'skip_sentiment', False),
            skip_value=getattr(args, 'skip_value', False),
            continue_on_error=getattr(args, 'continue_on_error', True),
        )


# =============================================================================
# PIPELINE RESULTS
# =============================================================================

@dataclass
class PhaseResult:
    """Result of a pipeline phase."""
    phase_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class PipelineResult:
    """Complete result of the pipeline execution."""
    match_id: str
    config: PipelineConfig
    phases: List[PhaseResult] = field(default_factory=list)
    report_path: Optional[str] = None
    email_sent: bool = False
    total_duration_seconds: float = 0.0
    success: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def add_phase(self, phase: PhaseResult):
        """Add a phase result."""
        self.phases.append(phase)
        if not phase.success:
            self.success = False

    def get_phase(self, name: str) -> Optional[PhaseResult]:
        """Get a phase result by name."""
        for phase in self.phases:
            if phase.phase_name == name:
                return phase
        return None


# =============================================================================
# DATA COLLECTION PHASE
# =============================================================================

class DataCollector:
    """Handles all data collection operations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataCollector")

    def fetch_odds(self) -> PhaseResult:
        """Fetch odds from The Odds API."""
        start_time = datetime.utcnow()
        self.logger.info("Starting odds collection...")

        if self.config.skip_odds:
            return PhaseResult(
                phase_name="odds_collection",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        if not self.config.odds_api_key:
            return PhaseResult(
                phase_name="odds_collection",
                success=False,
                error="THE_ODDS_API_KEY environment variable not set",
                duration_seconds=0.0
            )

        try:
            from data_collection.odds_fetcher import OddsFetcher

            fetcher = OddsFetcher(api_key=self.config.odds_api_key)
            arsenal_matches = fetcher.fetch_arsenal_matches()

            # Find the specific match if possible
            target_match = None
            for match in arsenal_matches:
                if self.config.opponent.lower() in match.get('home_team', '').lower() or \
                   self.config.opponent.lower() in match.get('away_team', '').lower():
                    target_match = match
                    break

            if target_match:
                odds_data = fetcher.convert_to_odds_data(target_match)
                result_data = {
                    'match': target_match,
                    'odds_data': odds_data,
                    'bookmaker_odds': odds_data.bookmaker_odds,
                    'best_value': odds_data.best_value,
                }
            else:
                # Return all Arsenal matches if specific one not found
                result_data = {
                    'all_matches': arsenal_matches,
                    'bookmaker_odds': [],
                    'best_value': {},
                }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Odds collection completed in {duration:.2f}s")

            return PhaseResult(
                phase_name="odds_collection",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Odds collection failed: {e}")
            return PhaseResult(
                phase_name="odds_collection",
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    async def scrape_news(self) -> PhaseResult:
        """Scrape news from Arsenal.com and BBC Sport."""
        start_time = datetime.utcnow()
        self.logger.info("Starting news scraping...")

        if self.config.skip_news:
            return PhaseResult(
                phase_name="news_scraping",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        try:
            from data_collection.news_scraper import NewsScraper

            async with NewsScraper() as scraper:
                news_data = await scraper.scrape_match_news(
                    match_id=self.config.match_id,
                    opponent=self.config.opponent,
                    opposing_manager=self.config.opposing_manager,
                    max_articles_per_source=5
                )

                # Save to file
                news_data.save_to_file()

                result_data = {
                    'news_data': news_data,
                    'articles': [a.to_dict() for a in news_data.articles],
                    'quotes': [q.to_dict() for q in news_data.quotes],
                    'article_count': len(news_data.articles),
                    'quote_count': len(news_data.quotes),
                }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"News scraping completed in {duration:.2f}s - {result_data['article_count']} articles, {result_data['quote_count']} quotes")

            return PhaseResult(
                phase_name="news_scraping",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"News scraping failed: {e}")
            return PhaseResult(
                phase_name="news_scraping",
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    def load_historical_data(self) -> PhaseResult:
        """Load historical match data."""
        start_time = datetime.utcnow()
        self.logger.info("Loading historical data...")

        if self.config.skip_historical:
            return PhaseResult(
                phase_name="historical_data",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        try:
            from data_collection.historical_data import HistoricalDataCollector

            collector = HistoricalDataCollector(api_key=self.config.football_data_api_key)

            # Try to fetch real data, or use existing CSV
            historical_csv = DATA_DIR / "historical" / "arsenal_matches.csv"

            if self.config.football_data_api_key:
                # Fetch from API
                matches = collector.fetch_multiple_seasons()
                collector.save_to_csv()
            elif historical_csv.exists():
                # Load from existing CSV
                import csv
                matches = []
                with open(historical_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        from data_collection.historical_data import MatchResult
                        matches.append(MatchResult(
                            date=row['date'],
                            home_team=row['home_team'],
                            away_team=row['away_team'],
                            competition=row['competition'],
                            home_score=int(row['home_score']),
                            away_score=int(row['away_score']),
                            result=row['result'],
                            venue=row['venue'],
                        ))
                collector._matches = matches
            else:
                matches = []

            # Get form stats and H2H
            form_stats = collector.get_form_stats(n_games=5) if matches else {}
            h2h = collector.get_head_to_head(self.config.opponent) if matches else {}

            result_data = {
                'matches': matches,
                'match_count': len(matches) if matches else 0,
                'form_stats': form_stats,
                'h2h': h2h,
            }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Historical data loaded in {duration:.2f}s - {result_data['match_count']} matches")

            return PhaseResult(
                phase_name="historical_data",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Historical data loading failed: {e}")
            return PhaseResult(
                phase_name="historical_data",
                success=False,
                error=str(e),
                duration_seconds=duration
            )


# =============================================================================
# ANALYSIS PHASE
# =============================================================================

class Analyzer:
    """Handles all analysis operations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Analyzer")

    def run_ml_predictions(
        self,
        historical_data: Optional[Dict],
        odds_data: Optional[Dict]
    ) -> PhaseResult:
        """Run ML predictions."""
        start_time = datetime.utcnow()
        self.logger.info("Running ML predictions...")

        if self.config.skip_ml:
            return PhaseResult(
                phase_name="ml_predictions",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        try:
            from analysis.ml_predictor import (
                MatchPredictor, generate_synthetic_training_data,
                FormStats, H2HStats, MatchFeatures
            )

            # Check if trained model exists
            model_path = MODELS_DIR / "match_predictor.pkl"

            if model_path.exists():
                predictor = MatchPredictor.load_model(str(model_path))
            else:
                # Train on synthetic data for demo
                self.logger.warning("No trained model found, using synthetic training data")
                X, y = generate_synthetic_training_data(n_samples=200)
                predictor = MatchPredictor()
                predictor.train(X, y)

            # Build features from collected data
            arsenal_form = {}
            opponent_form = {}
            h2h = {}
            bookmaker_odds = {}

            if historical_data and historical_data.get('form_stats'):
                form = historical_data['form_stats']
                arsenal_form = {
                    'n_games': form.get('n_games', 5),
                    'wins': form.get('wins', 2),
                    'draws': form.get('draws', 2),
                    'losses': form.get('losses', 1),
                    'goals_scored': form.get('goals_scored', 8),
                    'goals_conceded': form.get('goals_conceded', 5),
                    'clean_sheets': form.get('clean_sheets', 1),
                    'failed_to_score': form.get('failed_to_score', 0),
                }
            else:
                # Default form (neutral)
                arsenal_form = {
                    'n_games': 5, 'wins': 2, 'draws': 2, 'losses': 1,
                    'goals_scored': 8, 'goals_conceded': 5,
                    'clean_sheets': 1, 'failed_to_score': 0,
                }

            # Default opponent form
            opponent_form = {
                'n_games': 5, 'wins': 2, 'draws': 1, 'losses': 2,
                'goals_scored': 6, 'goals_conceded': 7,
                'clean_sheets': 1, 'failed_to_score': 1,
            }

            if historical_data and historical_data.get('h2h'):
                h2h_data = historical_data['h2h']
                h2h = {
                    'opponent': self.config.opponent,
                    'matches_played': h2h_data.get('matches_played', 5),
                    'wins': h2h_data.get('wins', 2),
                    'draws': h2h_data.get('draws', 2),
                    'losses': h2h_data.get('losses', 1),
                    'goals_scored': 10,
                    'goals_conceded': 8,
                }
            else:
                h2h = {
                    'opponent': self.config.opponent,
                    'matches_played': 5, 'wins': 2, 'draws': 2, 'losses': 1,
                    'goals_scored': 10, 'goals_conceded': 8,
                }

            # Extract bookmaker odds for features
            if odds_data and odds_data.get('best_value'):
                best_odds = odds_data['best_value']
                home_odds = best_odds.get('home_win', {}).get('odds', 2.0)
                draw_odds = best_odds.get('draw', {}).get('odds', 3.5)
                away_odds = best_odds.get('away_win', {}).get('odds', 3.5)

                total_implied = (1/home_odds + 1/draw_odds + 1/away_odds)
                bookmaker_odds = {
                    'win_prob': (1/home_odds) / total_implied,
                    'draw_prob': (1/draw_odds) / total_implied,
                    'loss_prob': (1/away_odds) / total_implied,
                }
            else:
                bookmaker_odds = {'win_prob': 0.40, 'draw_prob': 0.30, 'loss_prob': 0.30}

            # Run prediction
            is_home = self.config.venue == "home"
            prediction = predictor.predict_from_raw_features(
                arsenal_form=arsenal_form,
                opponent_form=opponent_form,
                h2h=h2h,
                is_home=is_home,
                arsenal_injury_factor=0.1,
                opponent_injury_factor=0.1,
                bookmaker_odds=bookmaker_odds
            )

            result_data = {
                'prediction': prediction,
                'probabilities': {
                    'win': prediction.win_prob,
                    'draw': prediction.draw_prob,
                    'loss': prediction.loss_prob,
                },
                'confidence_intervals': {
                    'win': (prediction.win_ci_lower, prediction.win_ci_upper),
                    'draw': (prediction.draw_ci_lower, prediction.draw_ci_upper),
                    'loss': (prediction.loss_ci_lower, prediction.loss_ci_upper),
                },
                'predicted_outcome': prediction.predicted_outcome,
                'prediction_confidence': prediction.prediction_confidence,
                'model_version': prediction.model_version,
            }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"ML predictions completed in {duration:.2f}s - Predicted: {prediction.predicted_outcome} ({prediction.prediction_confidence:.1%})")

            return PhaseResult(
                phase_name="ml_predictions",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"ML predictions failed: {e}")
            self.logger.error(traceback.format_exc())
            return PhaseResult(
                phase_name="ml_predictions",
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    def run_sentiment_analysis(self, news_data: Optional[Dict]) -> PhaseResult:
        """Run sentiment analysis on news articles."""
        start_time = datetime.utcnow()
        self.logger.info("Running sentiment analysis...")

        if self.config.skip_sentiment:
            return PhaseResult(
                phase_name="sentiment_analysis",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        if not news_data or not news_data.get('articles'):
            return PhaseResult(
                phase_name="sentiment_analysis",
                success=True,
                data={'overall_sentiment': 0.0, 'overall_label': 'neutral'},
                error="No news articles to analyze",
                duration_seconds=0.0
            )

        try:
            from analysis.sentiment_analyzer import SentimentAnalyzer

            analyzer = SentimentAnalyzer()

            articles = news_data.get('articles', [])

            report = analyzer.generate_sentiment_report(
                match_id=self.config.match_id,
                articles=articles,
                comments=None  # Could add Reddit comments here
            )

            result_data = {
                'overall_sentiment': report.overall_sentiment,
                'overall_label': report.overall_label,
                'sentiment_distribution': report.sentiment_distribution,
                'sentiment_by_source': report.sentiment_by_source,
                'key_insights': report.key_insights,
                'themes': {
                    'keywords': report.themes.keywords[:10],
                    'trending_concerns': report.themes.trending_concerns,
                    'trending_optimism': report.themes.trending_optimism,
                },
                'articles_analyzed': report.articles_analyzed,
            }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Sentiment analysis completed in {duration:.2f}s - Overall: {report.overall_label} ({report.overall_sentiment:+.2f})")

            return PhaseResult(
                phase_name="sentiment_analysis",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Sentiment analysis failed: {e}")
            return PhaseResult(
                phase_name="sentiment_analysis",
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    def calculate_value_opportunities(
        self,
        ml_predictions: Optional[Dict],
        odds_data: Optional[Dict]
    ) -> PhaseResult:
        """Calculate value betting opportunities."""
        start_time = datetime.utcnow()
        self.logger.info("Calculating value opportunities...")

        if self.config.skip_value:
            return PhaseResult(
                phase_name="value_calculation",
                success=True,
                data=None,
                error="Skipped by configuration",
                duration_seconds=0.0
            )

        if not ml_predictions or not odds_data:
            return PhaseResult(
                phase_name="value_calculation",
                success=True,
                data={'opportunities': []},
                error="Missing ML predictions or odds data",
                duration_seconds=0.0
            )

        try:
            from analysis.value_calculator import ValueCalculator, MarketType

            calculator = ValueCalculator()

            # Prepare ML predictions
            probs = ml_predictions.get('probabilities', {})
            confidence = ml_predictions.get('prediction_confidence', 0.7)

            ml_preds = {
                'home_win': {'probability': probs.get('win', 0.33), 'confidence': confidence},
                'draw': {'probability': probs.get('draw', 0.33), 'confidence': confidence * 0.9},
                'away_win': {'probability': probs.get('loss', 0.33), 'confidence': confidence * 0.9},
            }

            # Prepare bookmaker odds
            bookmaker_odds = odds_data.get('bookmaker_odds', [])
            if not bookmaker_odds and odds_data.get('best_value'):
                # Create from best_value
                best = odds_data['best_value']
                bookmaker_odds = [{
                    'bookmaker': 'best_available',
                    'home_win': best.get('home_win', {}).get('odds', 2.0),
                    'draw': best.get('draw', {}).get('odds', 3.5),
                    'away_win': best.get('away_win', {}).get('odds', 3.5),
                }]

            # Run analysis
            analysis = calculator.analyze_from_dict(
                match_id=self.config.match_id,
                ml_predictions=ml_preds,
                bookmaker_odds_list=bookmaker_odds
            )

            result_data = {
                'opportunities': [opp.to_dict() for opp in analysis.opportunities],
                'high_value_count': analysis.high_value_count,
                'total_markets_analyzed': analysis.total_markets_analyzed,
                'best_opportunity': analysis.best_opportunity.to_dict() if analysis.best_opportunity else None,
                'summary': analysis.summary,
                'intelligence_brief': analysis.to_intelligence_brief(),
            }

            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Value calculation completed in {duration:.2f}s - {len(analysis.opportunities)} opportunities found")

            return PhaseResult(
                phase_name="value_calculation",
                success=True,
                data=result_data,
                duration_seconds=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Value calculation failed: {e}")
            return PhaseResult(
                phase_name="value_calculation",
                success=False,
                error=str(e),
                duration_seconds=duration
            )


# =============================================================================
# REPORT GENERATION
# =============================================================================

class ReportGenerator:
    """Generates intelligence brief reports."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ReportGenerator")

    def generate_html_report(self, pipeline_result: PipelineResult) -> str:
        """Generate HTML intelligence brief."""
        self.logger.info("Generating HTML report...")

        # Extract data from phases
        odds_phase = pipeline_result.get_phase("odds_collection")
        news_phase = pipeline_result.get_phase("news_scraping")
        historical_phase = pipeline_result.get_phase("historical_data")
        ml_phase = pipeline_result.get_phase("ml_predictions")
        sentiment_phase = pipeline_result.get_phase("sentiment_analysis")
        value_phase = pipeline_result.get_phase("value_calculation")

        odds_data = odds_phase.data if odds_phase and odds_phase.success and odds_phase.data else {}
        news_data = news_phase.data if news_phase and news_phase.success and news_phase.data else {}
        historical_data = historical_phase.data if historical_phase and historical_phase.success and historical_phase.data else {}
        ml_data = ml_phase.data if ml_phase and ml_phase.success and ml_phase.data else {}
        sentiment_data = sentiment_phase.data if sentiment_phase and sentiment_phase.success and sentiment_phase.data else {}
        value_data = value_phase.data if value_phase and value_phase.success and value_phase.data else {}

        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arsenal Intelligence Brief - {self.config.opponent}</title>
    <style>
        :root {{
            --arsenal-red: #EF0107;
            --arsenal-gold: #9C824A;
            --arsenal-navy: #063672;
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --text-light: #eaeaea;
            --text-muted: #a0a0a0;
            --success: #00c853;
            --warning: #ffc107;
            --danger: #ff5252;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text-light);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--arsenal-red), var(--arsenal-navy));
            padding: 30px 20px;
            text-align: center;
            border-bottom: 4px solid var(--arsenal-gold);
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        header .match-info {{
            font-size: 1.2rem;
            color: var(--text-muted);
        }}

        .section {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            border-left: 4px solid var(--arsenal-red);
        }}

        .section h2 {{
            color: var(--arsenal-gold);
            margin-bottom: 15px;
            font-size: 1.5rem;
            border-bottom: 1px solid var(--arsenal-gold);
            padding-bottom: 10px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}

        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 20px;
        }}

        .card h3 {{
            color: var(--arsenal-red);
            margin-bottom: 10px;
        }}

        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .stat-label {{
            color: var(--text-muted);
        }}

        .stat-value {{
            font-weight: bold;
        }}

        .prediction-box {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, rgba(239,1,7,0.2), rgba(6,54,114,0.2));
            border-radius: 10px;
            margin: 20px 0;
        }}

        .prediction-outcome {{
            font-size: 3rem;
            font-weight: bold;
            color: var(--arsenal-gold);
        }}

        .prediction-confidence {{
            font-size: 1.5rem;
            color: var(--text-muted);
        }}

        .probability-bar {{
            display: flex;
            height: 40px;
            border-radius: 5px;
            overflow: hidden;
            margin: 15px 0;
        }}

        .prob-win {{
            background: var(--success);
        }}

        .prob-draw {{
            background: var(--warning);
        }}

        .prob-loss {{
            background: var(--danger);
        }}

        .prob-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            font-weight: bold;
            font-size: 0.9rem;
        }}

        .sentiment-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}

        .sentiment-positive {{
            background: var(--success);
            color: #000;
        }}

        .sentiment-neutral {{
            background: var(--warning);
            color: #000;
        }}

        .sentiment-negative {{
            background: var(--danger);
            color: #fff;
        }}

        .value-opportunity {{
            background: linear-gradient(135deg, rgba(0,200,83,0.2), rgba(0,200,83,0.1));
            border-left: 4px solid var(--success);
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}

        .value-opportunity.high-value {{
            border-left-color: var(--arsenal-gold);
            background: linear-gradient(135deg, rgba(156,130,74,0.3), rgba(156,130,74,0.1));
        }}

        .quote {{
            background: rgba(255,255,255,0.05);
            padding: 15px 20px;
            border-left: 3px solid var(--arsenal-gold);
            margin: 10px 0;
            font-style: italic;
        }}

        .quote-author {{
            text-align: right;
            color: var(--text-muted);
            margin-top: 10px;
            font-style: normal;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        th {{
            background: rgba(239,1,7,0.2);
            color: var(--arsenal-gold);
        }}

        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}

        .status-success {{ color: var(--success); }}
        .status-warning {{ color: var(--warning); }}
        .status-error {{ color: var(--danger); }}
    </style>
</head>
<body>
    <header>
        <h1>Arsenal Intelligence Brief</h1>
        <div class="match-info">
            Arsenal vs {self.config.opponent} | {self.config.match_date} | {self.config.venue.title()}
        </div>
        <div class="match-info" style="margin-top: 10px;">
            Match ID: {self.config.match_id}
        </div>
    </header>

    <div class="container">
        <!-- Executive Summary -->
        <section class="section">
            <h2>Executive Summary</h2>
            <div class="prediction-box">
                <div class="prediction-outcome">
                    {ml_data.get('predicted_outcome', 'N/A')}
                </div>
                <div class="prediction-confidence">
                    Confidence: {ml_data.get('prediction_confidence', 0)*100:.1f}%
                </div>
            </div>

            {self._generate_probability_bar(ml_data)}

            <div class="grid">
                <div class="card">
                    <h3>Sentiment</h3>
                    <p>
                        <span class="sentiment-badge sentiment-{sentiment_data.get('overall_label', 'neutral')}">
                            {sentiment_data.get('overall_label', 'Unknown').upper()}
                        </span>
                    </p>
                    <p style="margin-top: 10px; color: var(--text-muted);">
                        Score: {sentiment_data.get('overall_sentiment', 0):+.2f}
                    </p>
                </div>

                <div class="card">
                    <h3>Value Opportunities</h3>
                    <p style="font-size: 2rem; color: var(--arsenal-gold);">
                        {value_data.get('high_value_count', 0)}
                    </p>
                    <p style="color: var(--text-muted);">High value bets identified</p>
                </div>

                <div class="card">
                    <h3>Recent Form</h3>
                    <p style="font-size: 1.5rem; letter-spacing: 3px;">
                        {historical_data.get('form_stats', {}).get('form_string', 'N/A')[:5]}
                    </p>
                    <p style="color: var(--text-muted);">
                        {historical_data.get('form_stats', {}).get('points', 0)}/{historical_data.get('form_stats', {}).get('max_points', 15)} points
                    </p>
                </div>
            </div>
        </section>

        <!-- ML Predictions -->
        <section class="section">
            <h2>ML Predictions</h2>
            {self._generate_ml_section(ml_data)}
        </section>

        <!-- Value Opportunities -->
        <section class="section">
            <h2>Value Betting Opportunities</h2>
            {self._generate_value_section(value_data)}
        </section>

        <!-- Odds Analysis -->
        <section class="section">
            <h2>Odds Analysis</h2>
            {self._generate_odds_section(odds_data)}
        </section>

        <!-- Sentiment Analysis -->
        <section class="section">
            <h2>Sentiment Analysis</h2>
            {self._generate_sentiment_section(sentiment_data)}
        </section>

        <!-- News & Quotes -->
        <section class="section">
            <h2>News & Quotes</h2>
            {self._generate_news_section(news_data)}
        </section>

        <!-- Historical Data -->
        <section class="section">
            <h2>Historical Analysis</h2>
            {self._generate_historical_section(historical_data)}
        </section>

        <!-- Pipeline Status -->
        <section class="section">
            <h2>Pipeline Status</h2>
            {self._generate_pipeline_status(pipeline_result)}
        </section>

        <div class="footer">
            <p>Arsenal Intelligence Brief | Generated: {pipeline_result.timestamp}</p>
            <p>Total Pipeline Duration: {pipeline_result.total_duration_seconds:.2f} seconds</p>
            <p style="margin-top: 10px; font-size: 0.8rem;">
                This report is for informational purposes only. Not financial advice.
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_probability_bar(self, ml_data: Dict) -> str:
        """Generate probability bar HTML."""
        probs = ml_data.get('probabilities', {})
        win = probs.get('win', 0.33) * 100
        draw = probs.get('draw', 0.33) * 100
        loss = probs.get('loss', 0.33) * 100

        return f"""
        <div class="probability-bar">
            <div class="prob-segment prob-win" style="width: {win}%;">W {win:.0f}%</div>
            <div class="prob-segment prob-draw" style="width: {draw}%;">D {draw:.0f}%</div>
            <div class="prob-segment prob-loss" style="width: {loss}%;">L {loss:.0f}%</div>
        </div>
        """

    def _generate_ml_section(self, ml_data: Dict) -> str:
        """Generate ML predictions section."""
        if not ml_data:
            return "<p>ML predictions not available.</p>"

        probs = ml_data.get('probabilities', {})
        cis = ml_data.get('confidence_intervals', {})

        return f"""
        <div class="grid">
            <div class="card">
                <h3>Win Probability</h3>
                <div class="stat">
                    <span class="stat-label">Probability</span>
                    <span class="stat-value">{probs.get('win', 0)*100:.1f}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">95% CI</span>
                    <span class="stat-value">{cis.get('win', (0,1))[0]*100:.1f}% - {cis.get('win', (0,1))[1]*100:.1f}%</span>
                </div>
            </div>
            <div class="card">
                <h3>Draw Probability</h3>
                <div class="stat">
                    <span class="stat-label">Probability</span>
                    <span class="stat-value">{probs.get('draw', 0)*100:.1f}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">95% CI</span>
                    <span class="stat-value">{cis.get('draw', (0,1))[0]*100:.1f}% - {cis.get('draw', (0,1))[1]*100:.1f}%</span>
                </div>
            </div>
            <div class="card">
                <h3>Loss Probability</h3>
                <div class="stat">
                    <span class="stat-label">Probability</span>
                    <span class="stat-value">{probs.get('loss', 0)*100:.1f}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">95% CI</span>
                    <span class="stat-value">{cis.get('loss', (0,1))[0]*100:.1f}% - {cis.get('loss', (0,1))[1]*100:.1f}%</span>
                </div>
            </div>
        </div>
        <p style="margin-top: 15px; color: var(--text-muted);">
            Model Version: {ml_data.get('model_version', 'N/A')}
        </p>
        """

    def _generate_value_section(self, value_data: Dict) -> str:
        """Generate value opportunities section."""
        if not value_data or not value_data.get('opportunities'):
            return "<p>No value opportunities identified.</p>"

        opportunities = value_data.get('opportunities', [])
        summary = value_data.get('summary', {})

        html = f"""
        <div class="card" style="margin-bottom: 20px;">
            <h3>Summary</h3>
            <div class="stat">
                <span class="stat-label">Markets Analyzed</span>
                <span class="stat-value">{summary.get('markets_analyzed', 0)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">High Value Opportunities</span>
                <span class="stat-value">{value_data.get('high_value_count', 0)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Max EV</span>
                <span class="stat-value">{summary.get('max_ev', 0):.1f}%</span>
            </div>
            <p style="margin-top: 15px;">{summary.get('recommendation', '')}</p>
        </div>
        """

        for opp in opportunities[:5]:  # Show top 5
            is_high = "high-value" if opp.get('is_high_value') else ""
            html += f"""
            <div class="value-opportunity {is_high}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>{opp.get('market', '').replace('_', ' ').title()}</strong>
                    <span>EV: <strong>{opp.get('expected_value_percentage', 0):+.1f}%</strong></span>
                </div>
                <div style="margin-top: 10px; color: var(--text-muted);">
                    Edge: {opp.get('edge_percentage', 0):.1f}% |
                    Odds: {opp.get('decimal_odds', 0):.2f} ({opp.get('bookmaker', 'N/A')}) |
                    Kelly: {opp.get('kelly_fraction', 0)*100:.1f}%
                </div>
            </div>
            """

        return html

    def _generate_odds_section(self, odds_data: Dict) -> str:
        """Generate odds analysis section."""
        if not odds_data:
            return "<p>Odds data not available.</p>"

        best_value = odds_data.get('best_value', {})
        bookmaker_odds = odds_data.get('bookmaker_odds', [])

        html = "<h3>Best Available Odds</h3>"

        if best_value:
            html += """<table>
                <tr><th>Market</th><th>Odds</th><th>Bookmaker</th><th>Implied Prob</th></tr>
            """
            for market, data in best_value.items():
                odds = data.get('odds', 0)
                implied = (1/odds * 100) if odds > 0 else 0
                html += f"""
                <tr>
                    <td>{market.replace('_', ' ').title()}</td>
                    <td>{odds:.2f}</td>
                    <td>{data.get('bookmaker', 'N/A')}</td>
                    <td>{implied:.1f}%</td>
                </tr>
                """
            html += "</table>"

        if bookmaker_odds:
            html += "<h3 style='margin-top: 20px;'>All Bookmaker Odds</h3>"
            html += """<table>
                <tr><th>Bookmaker</th><th>Home</th><th>Draw</th><th>Away</th></tr>
            """
            for bm in bookmaker_odds[:5]:
                html += f"""
                <tr>
                    <td>{bm.get('bookmaker', 'N/A')}</td>
                    <td>{bm.get('home_win', 0):.2f}</td>
                    <td>{bm.get('draw', 0):.2f}</td>
                    <td>{bm.get('away_win', 0):.2f}</td>
                </tr>
                """
            html += "</table>"

        return html

    def _generate_sentiment_section(self, sentiment_data: Dict) -> str:
        """Generate sentiment analysis section."""
        if not sentiment_data:
            return "<p>Sentiment analysis not available.</p>"

        html = f"""
        <div class="grid">
            <div class="card">
                <h3>Overall Sentiment</h3>
                <p style="font-size: 1.5rem;">
                    <span class="sentiment-badge sentiment-{sentiment_data.get('overall_label', 'neutral')}">
                        {sentiment_data.get('overall_label', 'Unknown').upper()}
                    </span>
                </p>
                <p style="margin-top: 10px;">Score: {sentiment_data.get('overall_sentiment', 0):+.2f}</p>
            </div>
            <div class="card">
                <h3>Distribution</h3>
                <div class="stat">
                    <span class="stat-label">Positive</span>
                    <span class="stat-value status-success">{sentiment_data.get('sentiment_distribution', {}).get('positive', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Neutral</span>
                    <span class="stat-value status-warning">{sentiment_data.get('sentiment_distribution', {}).get('neutral', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Negative</span>
                    <span class="stat-value status-error">{sentiment_data.get('sentiment_distribution', {}).get('negative', 0)}</span>
                </div>
            </div>
        </div>
        """

        # Key insights
        insights = sentiment_data.get('key_insights', [])
        if insights:
            html += "<h3 style='margin-top: 20px;'>Key Insights</h3><ul>"
            for insight in insights:
                html += f"<li style='margin: 10px 0;'>{insight}</li>"
            html += "</ul>"

        # Themes
        themes = sentiment_data.get('themes', {})
        if themes:
            html += "<div class='grid' style='margin-top: 20px;'>"

            if themes.get('trending_concerns'):
                html += "<div class='card'><h3>Trending Concerns</h3><ul>"
                for concern in themes['trending_concerns'][:3]:
                    html += f"<li style='color: var(--danger);'>{concern}</li>"
                html += "</ul></div>"

            if themes.get('trending_optimism'):
                html += "<div class='card'><h3>Trending Optimism</h3><ul>"
                for opt in themes['trending_optimism'][:3]:
                    html += f"<li style='color: var(--success);'>{opt}</li>"
                html += "</ul></div>"

            html += "</div>"

        return html

    def _generate_news_section(self, news_data: Dict) -> str:
        """Generate news and quotes section."""
        if not news_data:
            return "<p>News data not available.</p>"

        html = f"""
        <p style="margin-bottom: 15px;">
            {news_data.get('article_count', 0)} articles and {news_data.get('quote_count', 0)} quotes collected.
        </p>
        """

        # Show quotes
        quotes = news_data.get('quotes', [])[:5]
        if quotes:
            html += "<h3>Key Quotes</h3>"
            for quote in quotes:
                html += f"""
                <div class="quote">
                    "{quote.get('quote', '')}"
                    <div class="quote-author">- {quote.get('speaker', 'Unknown')}</div>
                </div>
                """

        # Show article titles
        articles = news_data.get('articles', [])[:5]
        if articles:
            html += "<h3 style='margin-top: 20px;'>Recent Articles</h3><ul>"
            for article in articles:
                html += f"""
                <li style="margin: 10px 0;">
                    <a href="{article.get('url', '#')}" style="color: var(--arsenal-gold);" target="_blank">
                        {article.get('title', 'Untitled')}
                    </a>
                    <span style="color: var(--text-muted);"> - {article.get('source', 'Unknown')}</span>
                </li>
                """
            html += "</ul>"

        return html

    def _generate_historical_section(self, historical_data: Dict) -> str:
        """Generate historical analysis section."""
        if not historical_data:
            return "<p>Historical data not available.</p>"

        form = historical_data.get('form_stats', {})
        h2h = historical_data.get('h2h', {})

        html = """<div class="grid">"""

        # Form stats
        if form:
            html += f"""
            <div class="card">
                <h3>Recent Form (Last {form.get('n_games', 5)} Games)</h3>
                <div class="stat">
                    <span class="stat-label">Form</span>
                    <span class="stat-value" style="letter-spacing: 3px;">{form.get('form_string', 'N/A')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Points</span>
                    <span class="stat-value">{form.get('points', 0)}/{form.get('max_points', 15)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Goals Scored</span>
                    <span class="stat-value">{form.get('goals_scored', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Goals Conceded</span>
                    <span class="stat-value">{form.get('goals_conceded', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Clean Sheets</span>
                    <span class="stat-value">{form.get('clean_sheets', 0)}</span>
                </div>
            </div>
            """

        # H2H
        if h2h and h2h.get('matches_played', 0) > 0:
            html += f"""
            <div class="card">
                <h3>Head-to-Head vs {h2h.get('opponent', self.config.opponent)}</h3>
                <div class="stat">
                    <span class="stat-label">Matches</span>
                    <span class="stat-value">{h2h.get('matches_played', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Wins</span>
                    <span class="stat-value status-success">{h2h.get('wins', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Draws</span>
                    <span class="stat-value status-warning">{h2h.get('draws', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Losses</span>
                    <span class="stat-value status-error">{h2h.get('losses', 0)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Win Rate</span>
                    <span class="stat-value">{h2h.get('win_rate', 0)*100:.0f}%</span>
                </div>
            </div>
            """

        html += "</div>"

        return html

    def _generate_pipeline_status(self, result: PipelineResult) -> str:
        """Generate pipeline status section."""
        html = """<table>
            <tr><th>Phase</th><th>Status</th><th>Duration</th><th>Details</th></tr>
        """

        for phase in result.phases:
            status_class = "status-success" if phase.success else "status-error"
            status_text = "Success" if phase.success else "Failed"
            error_text = phase.error if phase.error else "-"

            html += f"""
            <tr>
                <td>{phase.phase_name.replace('_', ' ').title()}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{phase.duration_seconds:.2f}s</td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">{error_text}</td>
            </tr>
            """

        html += "</table>"

        return html

    def save_report(self, html_content: str, pipeline_result: PipelineResult) -> str:
        """Save HTML report to file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"intelligence_brief_{self.config.match_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.config.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"Report saved to {filepath}")
        return str(filepath)

    def save_json_report(self, pipeline_result: PipelineResult) -> str:
        """Save JSON report to file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"intelligence_brief_{self.config.match_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.config.output_dir / filename

        # Build JSON structure
        report_data = {
            'match_id': pipeline_result.match_id,
            'timestamp': pipeline_result.timestamp,
            'config': {
                'opponent': self.config.opponent,
                'match_date': self.config.match_date,
                'venue': self.config.venue,
            },
            'phases': {},
            'success': pipeline_result.success,
            'total_duration_seconds': pipeline_result.total_duration_seconds,
        }

        for phase in pipeline_result.phases:
            report_data['phases'][phase.phase_name] = {
                'success': phase.success,
                'duration_seconds': phase.duration_seconds,
                'error': phase.error,
                'data': phase.data if phase.data else None,
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"JSON report saved to {filepath}")
        return str(filepath)


# =============================================================================
# EMAIL NOTIFICATION
# =============================================================================

class EmailNotifier:
    """Handles email notifications."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EmailNotifier")

    def send_report(self, report_path: str, pipeline_result: PipelineResult) -> bool:
        """Send report via email."""
        if not self.config.send_email:
            return False

        if not self.config.email_recipients:
            self.logger.warning("No email recipients configured")
            return False

        if not self.config.smtp_user or not self.config.smtp_password:
            self.logger.warning("SMTP credentials not configured")
            return False

        try:
            # Read report content
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Arsenal Intelligence Brief - {self.config.opponent} ({self.config.match_date})"
            msg['From'] = self.config.smtp_user
            msg['To'] = ', '.join(self.config.email_recipients)

            # Plain text version
            text_content = f"""
Arsenal Intelligence Brief

Match: Arsenal vs {self.config.opponent}
Date: {self.config.match_date}
Venue: {self.config.venue.title()}

Pipeline Status: {'Success' if pipeline_result.success else 'Partial Failure'}

Please view the attached HTML report for full details.

Generated: {pipeline_result.timestamp}
            """

            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            self.logger.info(f"Report sent to {len(self.config.email_recipients)} recipients")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class IntelligenceBriefOrchestrator:
    """
    Main orchestrator for the Arsenal Intelligence Brief pipeline.

    Coordinates all phases:
    1. Data Collection (odds, news, historical)
    2. Analysis (ML predictions, sentiment, value calculation)
    3. Report Generation (HTML, JSON)
    4. Notification (email)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")

        self.collector = DataCollector(config)
        self.analyzer = Analyzer(config)
        self.reporter = ReportGenerator(config)
        self.notifier = EmailNotifier(config)

    async def run(self) -> PipelineResult:
        """Run the full pipeline."""
        start_time = datetime.utcnow()

        self.logger.info("=" * 60)
        self.logger.info("Arsenal Intelligence Brief Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Match: Arsenal vs {self.config.opponent}")
        self.logger.info(f"Date: {self.config.match_date}")
        self.logger.info(f"Venue: {self.config.venue}")
        self.logger.info(f"Match ID: {self.config.match_id}")
        self.logger.info("=" * 60)

        result = PipelineResult(
            match_id=self.config.match_id,
            config=self.config
        )

        # =============================================================
        # PHASE 1: DATA COLLECTION
        # =============================================================
        self.logger.info("\n[PHASE 1] DATA COLLECTION")
        self.logger.info("-" * 40)

        # 1a. Fetch odds
        odds_result = self.collector.fetch_odds()
        result.add_phase(odds_result)
        if not odds_result.success and not self.config.continue_on_error:
            return self._finalize_result(result, start_time)

        # 1b. Scrape news (async)
        news_result = await self.collector.scrape_news()
        result.add_phase(news_result)
        if not news_result.success and not self.config.continue_on_error:
            return self._finalize_result(result, start_time)

        # 1c. Load historical data
        historical_result = self.collector.load_historical_data()
        result.add_phase(historical_result)
        if not historical_result.success and not self.config.continue_on_error:
            return self._finalize_result(result, start_time)

        # =============================================================
        # PHASE 2: ANALYSIS
        # =============================================================
        self.logger.info("\n[PHASE 2] ANALYSIS")
        self.logger.info("-" * 40)

        # 2a. ML predictions
        ml_result = self.analyzer.run_ml_predictions(
            historical_data=historical_result.data if historical_result.success else None,
            odds_data=odds_result.data if odds_result.success else None
        )
        result.add_phase(ml_result)

        # 2b. Sentiment analysis
        sentiment_result = self.analyzer.run_sentiment_analysis(
            news_data=news_result.data if news_result.success else None
        )
        result.add_phase(sentiment_result)

        # 2c. Value calculation
        value_result = self.analyzer.calculate_value_opportunities(
            ml_predictions=ml_result.data if ml_result.success else None,
            odds_data=odds_result.data if odds_result.success else None
        )
        result.add_phase(value_result)

        # =============================================================
        # PHASE 3: REPORT GENERATION
        # =============================================================
        self.logger.info("\n[PHASE 3] REPORT GENERATION")
        self.logger.info("-" * 40)

        report_start = datetime.utcnow()
        try:
            # Generate HTML report
            html_content = self.reporter.generate_html_report(result)
            report_path = self.reporter.save_report(html_content, result)
            result.report_path = report_path

            # Optionally save JSON report
            if self.config.output_format in ['json', 'both']:
                self.reporter.save_json_report(result)

            report_duration = (datetime.utcnow() - report_start).total_seconds()
            result.add_phase(PhaseResult(
                phase_name="report_generation",
                success=True,
                data={'report_path': report_path},
                duration_seconds=report_duration
            ))
        except Exception as e:
            report_duration = (datetime.utcnow() - report_start).total_seconds()
            self.logger.error(f"Report generation failed: {e}")
            result.add_phase(PhaseResult(
                phase_name="report_generation",
                success=False,
                error=str(e),
                duration_seconds=report_duration
            ))

        # =============================================================
        # PHASE 4: NOTIFICATION
        # =============================================================
        if self.config.send_email and result.report_path:
            self.logger.info("\n[PHASE 4] NOTIFICATION")
            self.logger.info("-" * 40)

            email_start = datetime.utcnow()
            email_sent = self.notifier.send_report(result.report_path, result)
            result.email_sent = email_sent

            email_duration = (datetime.utcnow() - email_start).total_seconds()
            result.add_phase(PhaseResult(
                phase_name="email_notification",
                success=email_sent,
                error=None if email_sent else "Email sending failed",
                duration_seconds=email_duration
            ))

        return self._finalize_result(result, start_time)

    def _finalize_result(
        self,
        result: PipelineResult,
        start_time: datetime
    ) -> PipelineResult:
        """Finalize the pipeline result."""
        result.total_duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        # Determine overall success
        critical_phases = ['ml_predictions', 'report_generation']
        for phase in result.phases:
            if phase.phase_name in critical_phases and not phase.success:
                result.success = False
                break

        # Log summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {'SUCCESS' if result.success else 'PARTIAL FAILURE'}")
        self.logger.info(f"Duration: {result.total_duration_seconds:.2f}s")
        if result.report_path:
            self.logger.info(f"Report: {result.report_path}")
        self.logger.info("=" * 60)

        return result


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Arsenal Intelligence Brief Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --opponent "Chelsea" --match-date "2026-01-20"
  python orchestrator.py --opponent "Liverpool" --match-date "2026-01-25" --venue away
  python orchestrator.py --opponent "Tottenham" --match-date "2026-02-01" --send-email

Environment Variables:
  THE_ODDS_API_KEY       - API key for The Odds API
  FOOTBALL_DATA_API_KEY  - API key for football-data.org
  SMTP_HOST             - SMTP server hostname
  SMTP_PORT             - SMTP server port
  SMTP_USER             - SMTP username
  SMTP_PASSWORD         - SMTP password
  EMAIL_RECIPIENTS      - Comma-separated list of email addresses
        """
    )

    # Required arguments
    parser.add_argument(
        '--opponent', '-o',
        required=True,
        help='Name of the opposing team (e.g., "Chelsea")'
    )
    parser.add_argument(
        '--match-date', '-d',
        required=True,
        help='Match date in YYYY-MM-DD format'
    )

    # Optional arguments
    parser.add_argument(
        '--match-id', '-m',
        help='Match ID (auto-generated if not provided)'
    )
    parser.add_argument(
        '--venue', '-v',
        choices=['home', 'away'],
        default='home',
        help='Venue (home or away, default: home)'
    )
    parser.add_argument(
        '--opposing-manager',
        help='Name of the opposing manager (for quote extraction)'
    )
    parser.add_argument(
        '--output-dir',
        default=str(REPORTS_DIR),
        help='Output directory for reports'
    )
    parser.add_argument(
        '--output-format',
        choices=['html', 'json', 'both'],
        default='html',
        help='Output format (default: html)'
    )

    # Email options
    parser.add_argument(
        '--send-email',
        action='store_true',
        help='Send report via email'
    )
    parser.add_argument(
        '--email-recipients',
        nargs='+',
        help='Additional email recipients'
    )

    # Skip options
    parser.add_argument('--skip-odds', action='store_true', help='Skip odds collection')
    parser.add_argument('--skip-news', action='store_true', help='Skip news scraping')
    parser.add_argument('--skip-historical', action='store_true', help='Skip historical data')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML predictions')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--skip-value', action='store_true', help='Skip value calculation')

    # Error handling
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop pipeline on any error (default: continue)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle --stop-on-error
    args.continue_on_error = not args.stop_on_error

    # Create configuration
    config = PipelineConfig.from_env_and_args(args)

    # Validate
    if not config.match_id:
        logger.error("Could not generate match_id. Provide --match-id or both --opponent and --match-date")
        sys.exit(1)

    # Run orchestrator
    orchestrator = IntelligenceBriefOrchestrator(config)
    result = await orchestrator.run()

    # Exit code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
