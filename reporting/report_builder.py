#!/usr/bin/env python3
"""
Report Builder for Arsenal Intelligence Brief

Main report generator that aggregates all data sources (odds, lineup, news,
ML predictions, sentiment, value) and generates comprehensive intelligence
briefs in HTML and Markdown formats.

Features:
- Graceful handling of missing data
- Data completeness validation
- Jinja2 template-based HTML output
- Clean Markdown output
- Embedded base64 charts

Task: arsenalScript-vqp.37-41 - Reporting module
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import Jinja2
try:
    from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("jinja2 not installed. HTML template rendering will be limited.")

# Import chart generator
try:
    from .chart_generator import ChartGenerator, ChartConfig
except ImportError:
    try:
        from chart_generator import ChartGenerator, ChartConfig
    except ImportError:
        ChartGenerator = None
        ChartConfig = None
        logger.warning("ChartGenerator not available")

# Constants
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = Path(__file__).parent / "templates"
DATA_DIR = BASE_DIR / "data"


class DataSource(Enum):
    """Enumeration of data sources for the report."""
    ODDS = "odds"
    NEWS = "news"
    ML_PREDICTION = "ml_prediction"
    SENTIMENT = "sentiment"
    VALUE = "value"
    LINEUP = "lineup"
    HISTORICAL = "historical"


@dataclass
class DataCompleteness:
    """
    Tracks completeness of data for report generation.

    Used to validate and report on which data sources are available
    and provide warnings for missing critical data.
    """
    odds_available: bool = False
    news_available: bool = False
    ml_prediction_available: bool = False
    sentiment_available: bool = False
    value_available: bool = False
    lineup_available: bool = False
    historical_available: bool = False

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def completeness_score(self) -> float:
        """Calculate data completeness as percentage (0-100)."""
        sources = [
            self.odds_available,
            self.news_available,
            self.ml_prediction_available,
            self.sentiment_available,
            self.value_available,
        ]
        return sum(sources) / len(sources) * 100

    @property
    def is_critical_data_missing(self) -> bool:
        """Check if critical data (odds or ML) is missing."""
        return not (self.odds_available or self.ml_prediction_available)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "completeness_score": round(self.completeness_score, 1),
            "sources": {
                "odds": self.odds_available,
                "news": self.news_available,
                "ml_prediction": self.ml_prediction_available,
                "sentiment": self.sentiment_available,
                "value": self.value_available,
                "lineup": self.lineup_available,
                "historical": self.historical_available,
            },
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ReportSection:
    """
    Represents a section of the intelligence brief.

    Each section has a title, content, and optional metadata
    about its data source and quality.
    """
    title: str
    content: str
    source: Optional[DataSource] = None
    is_available: bool = True
    confidence: Optional[float] = None
    html_content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source.value if self.source else None,
            "is_available": self.is_available,
            "confidence": self.confidence,
            "data": self.data,
        }


@dataclass
class IntelligenceBrief:
    """
    Complete Intelligence Brief report.

    Contains all sections, charts, and metadata for generating
    the final HTML or Markdown output.
    """
    match_id: str
    generated_at: str
    home_team: str
    away_team: str
    match_date: Optional[str]
    competition: str

    sections: List[ReportSection] = field(default_factory=list)
    charts: Dict[str, str] = field(default_factory=dict)  # name -> base64
    data_completeness: DataCompleteness = field(default_factory=DataCompleteness)

    raw_data: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def get_section(self, title: str) -> Optional[ReportSection]:
        """Get a section by title."""
        for section in self.sections:
            if section.title.lower() == title.lower():
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match_id": self.match_id,
            "generated_at": self.generated_at,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "competition": self.competition,
            "sections": [s.to_dict() for s in self.sections],
            "data_completeness": self.data_completeness.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_html(self) -> str:
        """
        Generate HTML output using Jinja2 template.

        Returns:
            HTML string for email/web display
        """
        if not JINJA2_AVAILABLE:
            return self._fallback_html()

        # Try to load template from file
        template_path = TEMPLATES_DIR / "intelligence_brief.html"
        if template_path.exists():
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATES_DIR)),
                autoescape=select_autoescape(['html', 'xml'])
            )
            template = env.get_template("intelligence_brief.html")
        else:
            # Use inline template
            template = Template(INLINE_HTML_TEMPLATE)

        return template.render(report=self)

    def _fallback_html(self) -> str:
        """Generate simple HTML when Jinja2 is not available."""
        sections_html = ""
        for section in self.sections:
            if section.is_available:
                sections_html += f"""
                <div class="section">
                    <h2>{section.title}</h2>
                    <div class="content">{section.html_content or section.content}</div>
                </div>
                """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Arsenal Intelligence Brief - {self.match_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }}
        h1 {{ color: #EF0107; }}
        h2 {{ color: #063672; }}
    </style>
</head>
<body>
    <h1>Arsenal Intelligence Brief</h1>
    <p>{self.home_team} vs {self.away_team}</p>
    <p>Generated: {self.generated_at}</p>
    {sections_html}
</body>
</html>"""

    def to_markdown(self) -> str:
        """
        Generate Markdown output.

        Returns:
            Markdown string for documentation/display
        """
        lines = [
            f"# Arsenal Intelligence Brief",
            f"",
            f"## {self.home_team} vs {self.away_team}",
            f"",
            f"**Match ID:** {self.match_id}",
            f"**Competition:** {self.competition}",
            f"**Match Date:** {self.match_date or 'TBD'}",
            f"**Generated:** {self.generated_at}",
            f"",
            f"---",
            f"",
        ]

        # Data completeness summary
        dc = self.data_completeness
        lines.extend([
            f"### Data Completeness: {dc.completeness_score:.0f}%",
            f"",
        ])

        if dc.warnings:
            lines.append("**Warnings:**")
            for warning in dc.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Add each section
        for section in self.sections:
            if section.is_available:
                lines.append(f"## {section.title}")
                lines.append("")
                lines.append(section.content)
                lines.append("")

                if section.confidence is not None:
                    lines.append(f"*Confidence: {section.confidence:.0%}*")
                    lines.append("")

                lines.append("---")
                lines.append("")

        # Footer
        lines.extend([
            "",
            "---",
            "",
            "*This report was generated by Arsenal Intelligence Brief system.*",
            "*Data sources may include odds APIs, news scrapers, and ML predictions.*",
        ])

        return "\n".join(lines)


class ReportBuilder:
    """
    Main report builder for Arsenal Intelligence Brief.

    Aggregates data from multiple sources and generates comprehensive
    reports in HTML and Markdown formats.

    Usage:
        builder = ReportBuilder()
        report = builder.build_report(
            match_id="20260120_ARS_CHE",
            odds_data=odds_dict,
            ml_prediction=prediction_dict,
            sentiment_data=sentiment_dict,
            value_analysis=value_dict,
            news_data=news_dict
        )

        html_output = report.to_html()
        markdown_output = report.to_markdown()
    """

    def __init__(
        self,
        chart_generator: Optional['ChartGenerator'] = None,
        include_charts: bool = True
    ):
        """
        Initialize the report builder.

        Args:
            chart_generator: Optional ChartGenerator instance
            include_charts: Whether to generate and include charts
        """
        self.include_charts = include_charts
        if include_charts and ChartGenerator is not None:
            self.chart_generator = chart_generator or ChartGenerator()
        else:
            self.chart_generator = None
            if include_charts:
                logger.warning("ChartGenerator not available. Charts will be skipped.")

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def build_report(
        self,
        match_id: str,
        home_team: str = "Arsenal",
        away_team: str = "Opponent",
        match_date: Optional[str] = None,
        competition: str = "Premier League",
        odds_data: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
        value_analysis: Optional[Dict[str, Any]] = None,
        news_data: Optional[Dict[str, Any]] = None,
        lineup_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceBrief:
        """
        Build a complete intelligence brief report.

        Aggregates all available data sources and generates sections
        for each component. Missing data is handled gracefully with
        appropriate warnings.

        Args:
            match_id: Unique match identifier
            home_team: Home team name
            away_team: Away team name
            match_date: Match date (ISO format)
            competition: Competition name
            odds_data: Odds data dictionary
            ml_prediction: ML prediction dictionary
            sentiment_data: Sentiment analysis dictionary
            value_analysis: Value betting analysis dictionary
            news_data: News and quotes dictionary
            lineup_data: Team lineup data dictionary
            historical_data: Historical match data dictionary

        Returns:
            IntelligenceBrief object ready for rendering
        """
        self.logger.info(f"Building report for match: {match_id}")

        # Initialize report
        report = IntelligenceBrief(
            match_id=match_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            competition=competition,
        )

        # Store raw data for template access
        report.raw_data = {
            "odds": odds_data,
            "ml_prediction": ml_prediction,
            "sentiment": sentiment_data,
            "value": value_analysis,
            "news": news_data,
            "lineup": lineup_data,
            "historical": historical_data,
        }

        # Validate and build sections
        completeness = DataCompleteness()

        # Section 1: Fixture Information
        report.add_section(self._build_fixture_section(
            home_team, away_team, match_date, competition
        ))

        # Section 2: Odds Comparison
        if odds_data:
            completeness.odds_available = True
            section = self._build_odds_section(odds_data, home_team, away_team)
            report.add_section(section)

            # Generate odds chart
            if self.chart_generator and self.include_charts:
                bookmaker_odds = odds_data.get('bookmaker_odds', [])
                if bookmaker_odds:
                    chart = self.chart_generator.create_odds_comparison_chart(
                        bookmaker_odds, home_team, away_team
                    )
                    report.charts['odds_comparison'] = chart
        else:
            completeness.add_warning("Odds data not available")
            report.add_section(self._build_missing_section(
                "Odds Comparison", DataSource.ODDS
            ))

        # Section 3: ML Prediction
        if ml_prediction:
            completeness.ml_prediction_available = True
            section = self._build_ml_prediction_section(ml_prediction, home_team, away_team)
            report.add_section(section)

            # Generate probability gauge
            if self.chart_generator and self.include_charts:
                probs = ml_prediction.get('probabilities', {})
                win = probs.get('win', probs.get('win_prob', 0)) / 100 if probs.get('win', probs.get('win_prob', 0)) > 1 else probs.get('win', probs.get('win_prob', 0))
                draw = probs.get('draw', probs.get('draw_prob', 0)) / 100 if probs.get('draw', probs.get('draw_prob', 0)) > 1 else probs.get('draw', probs.get('draw_prob', 0))
                loss = probs.get('loss', probs.get('loss_prob', 0)) / 100 if probs.get('loss', probs.get('loss_prob', 0)) > 1 else probs.get('loss', probs.get('loss_prob', 0))

                if win + draw + loss > 0:
                    confidence = ml_prediction.get('prediction_confidence', ml_prediction.get('confidence'))
                    if confidence and confidence > 1:
                        confidence = confidence / 100
                    chart = self.chart_generator.create_win_probability_gauge(
                        win, draw, loss, confidence
                    )
                    report.charts['probability_gauge'] = chart
        else:
            completeness.add_warning("ML prediction not available")
            report.add_section(self._build_missing_section(
                "ML Prediction", DataSource.ML_PREDICTION
            ))

        # Section 4: News and Injuries
        if news_data:
            completeness.news_available = True
            section = self._build_news_section(news_data)
            report.add_section(section)
        else:
            completeness.add_warning("News data not available")
            report.add_section(self._build_missing_section(
                "News & Injuries", DataSource.NEWS
            ))

        # Section 5: Sentiment Analysis
        if sentiment_data:
            completeness.sentiment_available = True
            section = self._build_sentiment_section(sentiment_data)
            report.add_section(section)

            # Generate sentiment timeline if article data available
            if self.chart_generator and self.include_charts:
                articles = sentiment_data.get('articles', [])
                if articles:
                    sentiment_points = []
                    for article in articles:
                        date = article.get('publish_date', '')
                        score = article.get('combined_score', 0)
                        if isinstance(article, dict) and hasattr(article, 'combined_score'):
                            score = article.combined_score
                        elif isinstance(article, dict):
                            score = article.get('combined_score', 0)
                        sentiment_points.append({
                            'date': date,
                            'score': score,
                            'source': article.get('source', 'Unknown')
                        })
                    if sentiment_points:
                        chart = self.chart_generator.create_sentiment_timeline(sentiment_points)
                        report.charts['sentiment_timeline'] = chart
        else:
            completeness.add_warning("Sentiment analysis not available")
            report.add_section(self._build_missing_section(
                "Sentiment Analysis", DataSource.SENTIMENT
            ))

        # Section 6: Value Betting
        if value_analysis:
            completeness.value_available = True
            section = self._build_value_section(value_analysis)
            report.add_section(section)

            # Generate value opportunities chart
            if self.chart_generator and self.include_charts:
                opportunities = value_analysis.get('opportunities', [])
                if opportunities:
                    chart = self.chart_generator.create_value_opportunities_chart(opportunities)
                    report.charts['value_opportunities'] = chart
        else:
            completeness.add_warning("Value analysis not available")
            report.add_section(self._build_missing_section(
                "Value Betting", DataSource.VALUE
            ))

        # Section 7: Lineup (if available)
        if lineup_data:
            completeness.lineup_available = True
            section = self._build_lineup_section(lineup_data, home_team, away_team)
            report.add_section(section)

        # Section 8: Historical Data (if available)
        if historical_data:
            completeness.historical_available = True
            section = self._build_historical_section(historical_data, away_team)
            report.add_section(section)

        # Set completeness
        report.data_completeness = completeness

        self.logger.info(
            f"Report built with {len(report.sections)} sections, "
            f"{len(report.charts)} charts, "
            f"{completeness.completeness_score:.0f}% data completeness"
        )

        return report

    def _build_fixture_section(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str],
        competition: str
    ) -> ReportSection:
        """Build the fixture information section."""
        if match_date:
            try:
                dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%A, %B %d, %Y at %H:%M UTC')
            except (ValueError, TypeError):
                formatted_date = match_date
        else:
            formatted_date = "Date TBD"

        content = f"""**{home_team}** vs **{away_team}**

**Competition:** {competition}
**Date:** {formatted_date}
**Venue:** {"Emirates Stadium" if "Arsenal" in home_team else "Away"}
"""

        html_content = f"""
<div class="fixture-info">
    <h3>{home_team} vs {away_team}</h3>
    <p><strong>Competition:</strong> {competition}</p>
    <p><strong>Date:</strong> {formatted_date}</p>
    <p><strong>Venue:</strong> {"Emirates Stadium" if "Arsenal" in home_team else "Away"}</p>
</div>
"""

        return ReportSection(
            title="Fixture Information",
            content=content,
            html_content=html_content,
            is_available=True
        )

    def _build_odds_section(
        self,
        odds_data: Dict[str, Any],
        home_team: str,
        away_team: str
    ) -> ReportSection:
        """Build the odds comparison section."""
        best_value = odds_data.get('best_value', {})
        bookmaker_odds = odds_data.get('bookmaker_odds', [])

        # Extract best odds
        home_best = best_value.get('home_win', {})
        draw_best = best_value.get('draw', {})
        away_best = best_value.get('away_win', {})

        content_lines = [
            "### Best Available Odds",
            "",
            f"| Outcome | Odds | Bookmaker |",
            f"|---------|------|-----------|",
            f"| {home_team} Win | {home_best.get('odds', 'N/A')} | {home_best.get('bookmaker', 'N/A')} |",
            f"| Draw | {draw_best.get('odds', 'N/A')} | {draw_best.get('bookmaker', 'N/A')} |",
            f"| {away_team} Win | {away_best.get('odds', 'N/A')} | {away_best.get('bookmaker', 'N/A')} |",
            "",
        ]

        # Add bookmaker comparison
        if bookmaker_odds:
            content_lines.append("### All Bookmaker Odds")
            content_lines.append("")
            content_lines.append(f"| Bookmaker | {home_team} | Draw | {away_team} |")
            content_lines.append("|-----------|----------|------|----------|")
            for bm in bookmaker_odds[:5]:  # Limit to 5 bookmakers
                content_lines.append(
                    f"| {bm.get('bookmaker', 'Unknown')} | "
                    f"{bm.get('home_win', 'N/A')} | "
                    f"{bm.get('draw', 'N/A')} | "
                    f"{bm.get('away_win', 'N/A')} |"
                )
            content_lines.append("")

        html_content = f"""
<div class="odds-section">
    <h4>Best Available Odds</h4>
    <table class="odds-table">
        <tr>
            <th>Outcome</th>
            <th>Best Odds</th>
            <th>Bookmaker</th>
        </tr>
        <tr class="home-win">
            <td>{home_team} Win</td>
            <td><strong>{home_best.get('odds', 'N/A')}</strong></td>
            <td>{home_best.get('bookmaker', 'N/A')}</td>
        </tr>
        <tr class="draw">
            <td>Draw</td>
            <td><strong>{draw_best.get('odds', 'N/A')}</strong></td>
            <td>{draw_best.get('bookmaker', 'N/A')}</td>
        </tr>
        <tr class="away-win">
            <td>{away_team} Win</td>
            <td><strong>{away_best.get('odds', 'N/A')}</strong></td>
            <td>{away_best.get('bookmaker', 'N/A')}</td>
        </tr>
    </table>
</div>
"""

        return ReportSection(
            title="Odds Comparison",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.ODDS,
            is_available=True,
            data=odds_data
        )

    def _build_ml_prediction_section(
        self,
        ml_prediction: Dict[str, Any],
        home_team: str,
        away_team: str
    ) -> ReportSection:
        """Build the ML prediction section."""
        probs = ml_prediction.get('probabilities', {})
        predicted_outcome = ml_prediction.get('predicted_outcome', 'Unknown')
        confidence = ml_prediction.get('prediction_confidence', ml_prediction.get('confidence'))
        model_version = ml_prediction.get('model_version', 'Unknown')

        # Handle both percentage and decimal formats
        win_prob = probs.get('win', probs.get('win_prob', 0))
        draw_prob = probs.get('draw', probs.get('draw_prob', 0))
        loss_prob = probs.get('loss', probs.get('loss_prob', 0))

        # Ensure values are percentages
        if win_prob <= 1:
            win_prob *= 100
        if draw_prob <= 1:
            draw_prob *= 100
        if loss_prob <= 1:
            loss_prob *= 100
        if confidence and confidence <= 1:
            confidence *= 100

        # Confidence intervals if available
        ci = ml_prediction.get('confidence_intervals', {})

        content_lines = [
            f"### Predicted Outcome: **{predicted_outcome}**",
            "",
            f"**Model Confidence:** {confidence:.1f}%" if confidence else "",
            f"**Model Version:** {model_version}",
            "",
            "### Win Probabilities",
            "",
            f"| Outcome | Probability | 95% CI |",
            f"|---------|-------------|--------|",
            f"| {home_team} Win | {win_prob:.1f}% | {self._format_ci(ci.get('win', {}))} |",
            f"| Draw | {draw_prob:.1f}% | {self._format_ci(ci.get('draw', {}))} |",
            f"| {away_team} Win | {loss_prob:.1f}% | {self._format_ci(ci.get('loss', {}))} |",
            "",
        ]

        outcome_class = "prediction-win" if predicted_outcome == "W" else \
                       "prediction-draw" if predicted_outcome == "D" else "prediction-loss"

        html_content = f"""
<div class="ml-prediction-section">
    <div class="prediction-highlight {outcome_class}">
        <h4>Predicted Outcome</h4>
        <span class="outcome">{predicted_outcome}</span>
        <span class="confidence">{confidence:.1f}% confidence</span>
    </div>

    <h4>Win Probabilities</h4>
    <div class="probability-bars">
        <div class="prob-bar">
            <span class="label">{home_team} Win</span>
            <div class="bar-container">
                <div class="bar win-bar" style="width: {win_prob}%"></div>
            </div>
            <span class="value">{win_prob:.1f}%</span>
        </div>
        <div class="prob-bar">
            <span class="label">Draw</span>
            <div class="bar-container">
                <div class="bar draw-bar" style="width: {draw_prob}%"></div>
            </div>
            <span class="value">{draw_prob:.1f}%</span>
        </div>
        <div class="prob-bar">
            <span class="label">{away_team} Win</span>
            <div class="bar-container">
                <div class="bar loss-bar" style="width: {loss_prob}%"></div>
            </div>
            <span class="value">{loss_prob:.1f}%</span>
        </div>
    </div>

    <p class="model-info">Model: {model_version}</p>
</div>
"""

        return ReportSection(
            title="ML Prediction",
            content="\n".join([l for l in content_lines if l]),
            html_content=html_content,
            source=DataSource.ML_PREDICTION,
            is_available=True,
            confidence=confidence / 100 if confidence else None,
            data=ml_prediction
        )

    def _format_ci(self, ci_data: Dict[str, Any]) -> str:
        """Format confidence interval for display."""
        if not ci_data:
            return "N/A"
        lower = ci_data.get('lower', 0)
        upper = ci_data.get('upper', 0)
        if lower <= 1:
            lower *= 100
        if upper <= 1:
            upper *= 100
        return f"{lower:.1f}% - {upper:.1f}%"

    def _build_news_section(self, news_data: Dict[str, Any]) -> ReportSection:
        """Build the news and injuries section."""
        articles = news_data.get('articles', [])
        quotes = news_data.get('quotes', [])

        content_lines = ["### Recent News", ""]

        if articles:
            for article in articles[:5]:  # Limit to 5 articles
                title = article.get('title', 'Untitled')
                source = article.get('source', 'Unknown')
                url = article.get('url', '#')
                content_lines.append(f"- **{title}** ({source}) [Link]({url})")
            content_lines.append("")
        else:
            content_lines.append("*No recent news articles available.*")
            content_lines.append("")

        if quotes:
            content_lines.append("### Key Quotes")
            content_lines.append("")
            for quote in quotes[:3]:  # Limit to 3 quotes
                speaker = quote.get('speaker', 'Unknown')
                text = quote.get('quote', '')
                if len(text) > 200:
                    text = text[:200] + "..."
                content_lines.append(f"> \"{text}\"")
                content_lines.append(f"> *- {speaker}*")
                content_lines.append("")

        # Build HTML
        articles_html = ""
        if articles:
            for article in articles[:5]:
                articles_html += f"""
                <div class="news-item">
                    <h5><a href="{article.get('url', '#')}">{article.get('title', 'Untitled')}</a></h5>
                    <span class="source">{article.get('source', 'Unknown')}</span>
                    <p class="summary">{article.get('summary', '')[:200] if article.get('summary') else ''}</p>
                </div>
                """

        quotes_html = ""
        if quotes:
            for quote in quotes[:3]:
                quotes_html += f"""
                <blockquote class="quote">
                    <p>"{quote.get('quote', '')[:200]}..."</p>
                    <cite>- {quote.get('speaker', 'Unknown')}</cite>
                </blockquote>
                """

        html_content = f"""
<div class="news-section">
    <h4>Recent News</h4>
    <div class="news-list">
        {articles_html or '<p>No recent news available.</p>'}
    </div>

    {f'<h4>Key Quotes</h4><div class="quotes-list">{quotes_html}</div>' if quotes_html else ''}
</div>
"""

        return ReportSection(
            title="News & Injuries",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.NEWS,
            is_available=True,
            data=news_data
        )

    def _build_sentiment_section(self, sentiment_data: Dict[str, Any]) -> ReportSection:
        """Build the sentiment analysis section."""
        try:
            overall_sentiment = float(sentiment_data.get('overall_sentiment', 0))
        except (ValueError, TypeError):
            overall_sentiment = 0.0
        overall_label = sentiment_data.get('overall_label', 'neutral')
        distribution = sentiment_data.get('sentiment_distribution', {})
        by_source = sentiment_data.get('sentiment_by_source', {})
        key_insights = sentiment_data.get('key_insights', [])
        themes = sentiment_data.get('themes', {})

        content_lines = [
            f"### Overall Sentiment: **{overall_label.upper()}** (Score: {overall_sentiment:+.2f})",
            "",
            "### Sentiment Distribution",
            "",
            f"- Positive: {distribution.get('positive', 0)} articles",
            f"- Neutral: {distribution.get('neutral', 0)} articles",
            f"- Negative: {distribution.get('negative', 0)} articles",
            "",
        ]

        if by_source:
            content_lines.append("### Sentiment by Source")
            content_lines.append("")
            for source, score in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
                sentiment_emoji = "+" if score > 0 else ""
                content_lines.append(f"- {source}: {sentiment_emoji}{score:.2f}")
            content_lines.append("")

        if key_insights:
            content_lines.append("### Key Insights")
            content_lines.append("")
            for insight in key_insights[:5]:
                content_lines.append(f"- {insight}")
            content_lines.append("")

        # Determine sentiment class
        if overall_sentiment > 0.2:
            sentiment_class = "sentiment-positive"
        elif overall_sentiment < -0.2:
            sentiment_class = "sentiment-negative"
        else:
            sentiment_class = "sentiment-neutral"

        html_content = f"""
<div class="sentiment-section">
    <div class="sentiment-summary {sentiment_class}">
        <h4>Overall Sentiment</h4>
        <span class="sentiment-label">{overall_label.upper()}</span>
        <span class="sentiment-score">{overall_sentiment:+.2f}</span>
    </div>

    <div class="sentiment-distribution">
        <h4>Distribution</h4>
        <div class="dist-bar">
            <span class="positive" style="width: {distribution.get('positive', 0) * 10}%">
                {distribution.get('positive', 0)} Positive
            </span>
            <span class="neutral" style="width: {distribution.get('neutral', 0) * 10}%">
                {distribution.get('neutral', 0)} Neutral
            </span>
            <span class="negative" style="width: {distribution.get('negative', 0) * 10}%">
                {distribution.get('negative', 0)} Negative
            </span>
        </div>
    </div>

    {'<h4>Key Insights</h4><ul>' + ''.join(f'<li>{i}</li>' for i in key_insights[:5]) + '</ul>' if key_insights else ''}
</div>
"""

        return ReportSection(
            title="Sentiment Analysis",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.SENTIMENT,
            is_available=True,
            data=sentiment_data
        )

    def _build_value_section(self, value_analysis: Dict[str, Any]) -> ReportSection:
        """Build the value betting section."""
        opportunities = value_analysis.get('opportunities', [])
        summary = value_analysis.get('summary', {})
        best_opportunity = value_analysis.get('best_opportunity', {})

        content_lines = ["### Value Betting Analysis", ""]

        if not opportunities:
            content_lines.append("*No value betting opportunities identified.*")
        else:
            content_lines.extend([
                f"**Total Opportunities:** {len(opportunities)}",
                f"**High Value Opportunities (EV > 5%):** {summary.get('high_value_count', value_analysis.get('high_value_count', 0))}",
                "",
                "### Top Opportunities",
                "",
                "| Market | EV | Edge | Odds | Bookmaker | Risk |",
                "|--------|----|----- |------|-----------|------|",
            ])

            for opp in opportunities[:5]:
                market = opp.get('market', 'unknown').replace('_', ' ').title()
                ev = opp.get('expected_value_percentage', 0)
                edge = opp.get('edge_percentage', 0)
                odds = opp.get('decimal_odds', 0)
                bookmaker = opp.get('bookmaker', 'Unknown')
                risk = opp.get('risk_level', 'unknown').replace('_', ' ').title()
                is_hv = "**" if opp.get('is_high_value', False) else ""

                content_lines.append(
                    f"| {is_hv}{market}{is_hv} | {ev:.1f}% | {edge:.1f}% | {odds:.2f} | {bookmaker} | {risk} |"
                )

            content_lines.append("")

            if summary.get('recommendation'):
                content_lines.append(f"**Recommendation:** {summary.get('recommendation')}")
                content_lines.append("")

        # HTML content
        opps_html = ""
        if opportunities:
            for opp in opportunities[:5]:
                is_high = "high-value" if opp.get('is_high_value', False) else ""
                opps_html += f"""
                <tr class="{is_high}">
                    <td>{opp.get('market', 'unknown').replace('_', ' ').title()}</td>
                    <td class="ev">{opp.get('expected_value_percentage', 0):.1f}%</td>
                    <td>{opp.get('edge_percentage', 0):.1f}%</td>
                    <td>{opp.get('decimal_odds', 0):.2f}</td>
                    <td>{opp.get('bookmaker', 'Unknown')}</td>
                    <td class="risk-{opp.get('risk_level', 'unknown')}">{opp.get('risk_level', 'unknown').replace('_', ' ').title()}</td>
                </tr>
                """

        html_content = f"""
<div class="value-section">
    <h4>Value Betting Analysis</h4>

    {'<p class="no-value">No value betting opportunities identified.</p>' if not opportunities else f'''
    <div class="value-summary">
        <span class="stat">
            <strong>{len(opportunities)}</strong> Opportunities
        </span>
        <span class="stat high-value">
            <strong>{summary.get('high_value_count', value_analysis.get('high_value_count', 0))}</strong> High Value (EV > 5%)
        </span>
    </div>

    <table class="value-table">
        <tr>
            <th>Market</th>
            <th>EV</th>
            <th>Edge</th>
            <th>Odds</th>
            <th>Bookmaker</th>
            <th>Risk</th>
        </tr>
        {opps_html}
    </table>

    {f'<p class="recommendation"><strong>Recommendation:</strong> {summary.get("recommendation", "")}</p>' if summary.get('recommendation') else ''}
    '''}
</div>
"""

        return ReportSection(
            title="Value Betting",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.VALUE,
            is_available=True,
            data=value_analysis
        )

    def _build_lineup_section(
        self,
        lineup_data: Dict[str, Any],
        home_team: str,
        away_team: str
    ) -> ReportSection:
        """Build the lineup section."""
        home_lineup = lineup_data.get('home', lineup_data.get('arsenal', {}))
        away_lineup = lineup_data.get('away', lineup_data.get('opponent', {}))
        injuries = lineup_data.get('injuries', [])
        suspensions = lineup_data.get('suspensions', [])

        content_lines = ["### Team Lineups", ""]

        if injuries:
            content_lines.append("### Injuries")
            content_lines.append("")
            for injury in injuries:
                player = injury.get('player', 'Unknown')
                status = injury.get('status', 'Unknown')
                reason = injury.get('reason', '')
                content_lines.append(f"- **{player}**: {status} ({reason})")
            content_lines.append("")

        if suspensions:
            content_lines.append("### Suspensions")
            content_lines.append("")
            for suspension in suspensions:
                player = suspension.get('player', 'Unknown')
                reason = suspension.get('reason', '')
                content_lines.append(f"- **{player}**: {reason}")
            content_lines.append("")

        html_content = f"""
<div class="lineup-section">
    <h4>Team News</h4>

    {'<div class="injuries"><h5>Injuries</h5><ul>' + ''.join(f'<li><strong>{i.get("player", "Unknown")}</strong>: {i.get("status", "Unknown")} ({i.get("reason", "")})</li>' for i in injuries) + '</ul></div>' if injuries else ''}

    {'<div class="suspensions"><h5>Suspensions</h5><ul>' + ''.join(f'<li><strong>{s.get("player", "Unknown")}</strong>: {s.get("reason", "")}</li>' for s in suspensions) + '</ul></div>' if suspensions else ''}
</div>
"""

        return ReportSection(
            title="Lineups & Team News",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.LINEUP,
            is_available=True,
            data=lineup_data
        )

    def _build_historical_section(
        self,
        historical_data: Dict[str, Any],
        opponent: str
    ) -> ReportSection:
        """Build the historical data section."""
        form = historical_data.get('form', {})
        h2h = historical_data.get('h2h', {})

        content_lines = ["### Recent Form", ""]

        if form:
            form_string = form.get('form_string', '')
            points = form.get('points', 0)
            max_points = form.get('max_points', 0)

            content_lines.extend([
                f"**Last 5 Games:** {form_string}",
                f"**Points:** {points}/{max_points}",
                f"**Goals Scored:** {form.get('goals_scored', 0)}",
                f"**Goals Conceded:** {form.get('goals_conceded', 0)}",
                f"**Clean Sheets:** {form.get('clean_sheets', 0)}",
                "",
            ])

        if h2h and h2h.get('matches_played', 0) > 0:
            content_lines.extend([
                f"### Head-to-Head vs {opponent}",
                "",
                f"**Matches Played:** {h2h.get('matches_played', 0)}",
                f"**Wins:** {h2h.get('wins', 0)}",
                f"**Draws:** {h2h.get('draws', 0)}",
                f"**Losses:** {h2h.get('losses', 0)}",
                "",
            ])

        html_content = f"""
<div class="historical-section">
    {'<div class="form"><h4>Recent Form</h4><span class="form-string">' + form.get('form_string', '') + '</span><p>{} points from {} games</p></div>'.format(form.get('points', 0), form.get('n_games', 5)) if form else ''}

    {'<div class="h2h"><h4>Head-to-Head vs ' + opponent + '</h4><p>W{}-D{}-L{} in {} matches</p></div>'.format(h2h.get('wins', 0), h2h.get('draws', 0), h2h.get('losses', 0), h2h.get('matches_played', 0)) if h2h and h2h.get('matches_played', 0) > 0 else ''}
</div>
"""

        return ReportSection(
            title="Historical Data",
            content="\n".join(content_lines),
            html_content=html_content,
            source=DataSource.HISTORICAL,
            is_available=True,
            data=historical_data
        )

    def _build_missing_section(
        self,
        title: str,
        source: DataSource
    ) -> ReportSection:
        """Build a placeholder section for missing data."""
        content = f"*{title} data is not available for this report.*"

        html_content = f"""
<div class="missing-section">
    <p class="unavailable">{title} data is not available for this report.</p>
</div>
"""

        return ReportSection(
            title=title,
            content=content,
            html_content=html_content,
            source=source,
            is_available=False
        )


# Inline HTML template for when template file is not available
INLINE_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arsenal Intelligence Brief - {{ report.match_id }}</title>
    <style>
        /* Reset and base styles */
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #EF0107 0%, #9C0104 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 8px 8px 0 0;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }

        .header .match-info {
            font-size: 20px;
            font-weight: bold;
        }

        .header .meta {
            font-size: 12px;
            opacity: 0.9;
            margin-top: 10px;
        }

        /* Sections */
        .section {
            padding: 20px;
            border-bottom: 1px solid #eee;
        }

        .section h2 {
            color: #063672;
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #EF0107;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        th {
            background-color: #063672;
            color: white;
            font-weight: 600;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        /* Prediction highlight */
        .prediction-highlight {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .prediction-win { background-color: #d4edda; border: 2px solid #28a745; }
        .prediction-draw { background-color: #fff3cd; border: 2px solid #ffc107; }
        .prediction-loss { background-color: #f8d7da; border: 2px solid #dc3545; }

        .prediction-highlight .outcome {
            font-size: 36px;
            font-weight: bold;
            display: block;
        }

        /* Probability bars */
        .probability-bars { margin: 20px 0; }

        .prob-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .prob-bar .label {
            width: 120px;
            font-weight: 500;
        }

        .prob-bar .bar-container {
            flex: 1;
            height: 24px;
            background-color: #eee;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 10px;
        }

        .prob-bar .bar {
            height: 100%;
            border-radius: 4px;
        }

        .win-bar { background-color: #28a745; }
        .draw-bar { background-color: #ffc107; }
        .loss-bar { background-color: #dc3545; }

        /* Value betting */
        .high-value { background-color: #d4edda !important; }

        .ev { color: #28a745; font-weight: bold; }

        /* Sentiment */
        .sentiment-positive { background-color: #d4edda; border-color: #28a745; }
        .sentiment-neutral { background-color: #fff3cd; border-color: #ffc107; }
        .sentiment-negative { background-color: #f8d7da; border-color: #dc3545; }

        .sentiment-summary {
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid;
            margin-bottom: 20px;
        }

        /* Charts */
        .chart {
            text-align: center;
            margin: 20px 0;
        }

        .chart img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Quotes */
        blockquote {
            border-left: 4px solid #EF0107;
            padding: 10px 20px;
            margin: 15px 0;
            background-color: #f9f9f9;
            font-style: italic;
        }

        blockquote cite {
            display: block;
            margin-top: 10px;
            font-style: normal;
            color: #666;
        }

        /* Data completeness */
        .completeness {
            padding: 10px 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
            margin-bottom: 20px;
        }

        /* Missing section */
        .unavailable {
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
            background-color: #f5f5f5;
            border-radius: 4px;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #eee;
        }

        /* Mobile responsive */
        @media (max-width: 600px) {
            .container { padding: 10px; }
            .header { padding: 20px; }
            .header h1 { font-size: 20px; }
            .section { padding: 15px; }
            .prob-bar .label { width: 80px; font-size: 12px; }
            table { font-size: 12px; }
            th, td { padding: 6px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Arsenal Intelligence Brief</h1>
            <div class="match-info">{{ report.home_team }} vs {{ report.away_team }}</div>
            <div class="meta">
                {{ report.competition }} | {{ report.match_date or 'Date TBD' }}<br>
                Generated: {{ report.generated_at }}
            </div>
        </div>

        <div class="completeness">
            Data Completeness: {{ report.data_completeness.completeness_score|round(0) }}%
            {% if report.data_completeness.warnings %}
            <br><small>Warnings: {{ report.data_completeness.warnings|join(', ') }}</small>
            {% endif %}
        </div>

        {% for section in report.sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            {% if section.is_available %}
                {{ section.html_content|safe if section.html_content else section.content }}

                {% if section.title == 'Odds Comparison' and report.charts.get('odds_comparison') %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ report.charts.odds_comparison }}" alt="Odds Comparison Chart">
                </div>
                {% endif %}

                {% if section.title == 'ML Prediction' and report.charts.get('probability_gauge') %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ report.charts.probability_gauge }}" alt="Win Probability Gauge">
                </div>
                {% endif %}

                {% if section.title == 'Sentiment Analysis' and report.charts.get('sentiment_timeline') %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ report.charts.sentiment_timeline }}" alt="Sentiment Timeline">
                </div>
                {% endif %}

                {% if section.title == 'Value Betting' and report.charts.get('value_opportunities') %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ report.charts.value_opportunities }}" alt="Value Opportunities">
                </div>
                {% endif %}
            {% else %}
                <p class="unavailable">{{ section.title }} data is not available for this report.</p>
            {% endif %}
        </div>
        {% endfor %}

        <div class="footer">
            <p>Arsenal Intelligence Brief | Match ID: {{ report.match_id }}</p>
            <p>This report was generated automatically. Data accuracy depends on source availability.</p>
        </div>
    </div>
</body>
</html>
'''


# ==========================================================================
# DEMO / TEST
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Arsenal Intelligence Brief - Report Builder Demo")
    print("=" * 60)

    # Create sample data
    sample_odds = {
        "match_id": "20260120_ARS_CHE",
        "bookmaker_odds": [
            {"bookmaker": "Bet365", "home_win": 2.10, "draw": 3.40, "away_win": 3.50},
            {"bookmaker": "William Hill", "home_win": 2.15, "draw": 3.30, "away_win": 3.45},
        ],
        "best_value": {
            "home_win": {"bookmaker": "William Hill", "odds": 2.15},
            "draw": {"bookmaker": "Bet365", "odds": 3.40},
            "away_win": {"bookmaker": "Bet365", "odds": 3.50},
        }
    }

    sample_prediction = {
        "probabilities": {"win": 52.0, "draw": 26.0, "loss": 22.0},
        "predicted_outcome": "W",
        "prediction_confidence": 85.0,
        "model_version": "1.0.0",
        "confidence_intervals": {
            "win": {"lower": 45.0, "upper": 59.0},
            "draw": {"lower": 20.0, "upper": 32.0},
            "loss": {"lower": 16.0, "upper": 28.0},
        }
    }

    sample_sentiment = {
        "overall_sentiment": 0.35,
        "overall_label": "positive",
        "sentiment_distribution": {"positive": 8, "neutral": 4, "negative": 2},
        "sentiment_by_source": {
            "Arsenal.com": 0.6,
            "BBC Sport": 0.2,
            "The Guardian": 0.1,
        },
        "key_insights": [
            "Positive coverage dominated by Arsenal's strong form",
            "Minor concerns about defensive depth",
            "Arteta praised for tactical flexibility"
        ],
        "articles": [
            {"title": "Arsenal in Fine Form", "source": "Arsenal.com", "publish_date": "2026-01-18", "combined_score": 0.6},
            {"title": "Preview: Arsenal vs Chelsea", "source": "BBC Sport", "publish_date": "2026-01-17", "combined_score": 0.2},
        ]
    }

    sample_value = {
        "opportunities": [
            {
                "market": "home_win",
                "expected_value_percentage": 9.2,
                "edge_percentage": 4.5,
                "decimal_odds": 2.15,
                "bookmaker": "William Hill",
                "risk_level": "low",
                "is_high_value": True
            },
            {
                "market": "over_2_5",
                "expected_value_percentage": 6.3,
                "edge_percentage": 3.2,
                "decimal_odds": 1.90,
                "bookmaker": "Bet365",
                "risk_level": "medium",
                "is_high_value": True
            },
        ],
        "high_value_count": 2,
        "summary": {
            "recommendation": "Strong value identified in home win market."
        }
    }

    sample_news = {
        "articles": [
            {"title": "Arteta: We are ready for Chelsea", "source": "Arsenal.com", "url": "#"},
            {"title": "Arsenal injury update ahead of Chelsea clash", "source": "BBC Sport", "url": "#"},
        ],
        "quotes": [
            {"speaker": "Mikel Arteta (Arsenal Manager)", "quote": "We have prepared well and the team is ready."},
        ]
    }

    # Build report
    builder = ReportBuilder(include_charts=True)
    report = builder.build_report(
        match_id="20260120_ARS_CHE",
        home_team="Arsenal",
        away_team="Chelsea",
        match_date="2026-01-20T15:00:00Z",
        competition="Premier League",
        odds_data=sample_odds,
        ml_prediction=sample_prediction,
        sentiment_data=sample_sentiment,
        value_analysis=sample_value,
        news_data=sample_news,
    )

    print(f"\nReport built successfully!")
    print(f"Sections: {len(report.sections)}")
    print(f"Charts: {len(report.charts)}")
    print(f"Completeness: {report.data_completeness.completeness_score:.0f}%")

    # Generate outputs
    print("\n--- Generating Markdown Output ---")
    markdown = report.to_markdown()
    print(markdown[:1000] + "...\n")

    print("\n--- Generating HTML Output ---")
    html = report.to_html()
    print(f"HTML length: {len(html)} characters")

    # Save to files
    output_dir = Path(__file__).parent.parent / "data" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{report.match_id}_brief.md"
    html_path = output_dir / f"{report.match_id}_brief.html"

    with open(md_path, 'w') as f:
        f.write(markdown)
    print(f"\nMarkdown saved to: {md_path}")

    with open(html_path, 'w') as f:
        f.write(html)
    print(f"HTML saved to: {html_path}")

    print("\n" + "=" * 60)
    print("Demo complete.")
