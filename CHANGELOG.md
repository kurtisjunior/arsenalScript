# Changelog

All notable changes to Arsenal Pulse are documented in this file.

## [2.0.0] - 2026-01-15

### Added

**Data Collection Pipeline**
- Multi-bookmaker odds aggregation via The Odds API (`data_collection/odds_fetcher.py`)
- Odds format conversion: decimal, fractional, American (`data_collection/odds_data.py`)
- Historical match data collector with 5+ years of Arsenal data (`data_collection/historical_data.py`)
- News scraper for Arsenal.com and BBC Sport (`data_collection/news_scraper.py`)
- Social media lineup scraper for Reddit and Twitter (`data_collection/lineup_scraper.py`)
- Injury and availability parsing with NLP

**Machine Learning**
- Logistic regression match outcome predictor (`analysis/ml_predictor.py`)
- Feature engineering: form, H2H, home advantage, injuries
- Win/Draw/Loss probability with 95% confidence intervals
- Cross-validation with accuracy metrics
- Model serialization and versioning

**NLP & Sentiment Analysis**
- HuggingFace Transformers integration (`analysis/sentiment_analyzer.py`)
- News article sentiment classification
- Reddit r/Gunners sentiment aggregation
- Theme extraction (tactics, injuries, pressure, confidence)
- Source-weighted sentiment scoring

**Value Betting Analysis**
- Odds analyzer with overround calculation (`analysis/odds_analyzer.py`)
- Expected Value (EV) calculator (`analysis/value_calculator.py`)
- Kelly Criterion stake sizing
- Edge detection (model vs market)
- High-value opportunity flagging (EV > 5%)

**Reporting**
- Professional HTML intelligence brief generator (`reporting/report_builder.py`)
- Jinja2 templating with Arsenal-themed styling
- Chart generation: probability gauges, odds comparison (`reporting/chart_generator.py`)
- Markdown report output option
- Mobile-responsive email templates

**Infrastructure**
- Async pipeline orchestrator (`orchestrator.py`)
- Phase-based execution with error recovery
- GitHub Actions CI/CD workflow
- Comprehensive test suite (87 tests)
- JSON schemas for data validation

### Changed
- Renamed project from "Arsenal Match Notifier" to "Arsenal Pulse"
- Refactored from single-file script to modular architecture
- Updated scikit-learn compatibility (removed deprecated `multi_class` parameter)

### Fixed
- NoneType error when phase succeeds but returns no data
- String-to-float conversion in sentiment formatting
- OddsFetcher export in module __init__.py

---

## [1.0.0] - 2025-01-01

### Added
- Initial release as "Arsenal Match Notifier"
- Basic match notification via ntfy.sh
- Football-Data.org API integration for fixtures
- GitHub Actions scheduled workflow
- Timezone-aware notification timing

---

## Version History Summary

| Version | Name | Description |
|---------|------|-------------|
| 1.0.0 | Arsenal Match Notifier | Simple push notifications for upcoming matches |
| **2.0.0** | **Arsenal Pulse** | Full ML-powered intelligence platform |

## Upgrade Guide

### From 1.x to 2.x

1. **New Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **New Environment Variables**
   ```bash
   # Copy and update
   cp .env.example .env
   # Add: THE_ODDS_API_KEY, REDDIT_CLIENT_ID, SMTP_* (optional)
   ```

3. **New Entry Point**
   ```bash
   # Old (still works)
   python main.py

   # New
   python orchestrator.py --opponent "Chelsea" --match-date "2026-01-20"
   ```

4. **GitHub Secrets**
   Add new secrets for full functionality:
   - `THE_ODDS_API_KEY`
   - `REDDIT_CLIENT_ID` (optional)
   - `SMTP_USER` / `SMTP_PASSWORD` (optional)
