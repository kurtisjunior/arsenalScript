# Arsenal Pulse

> **ML-Powered Match Intelligence Platform for Arsenal FC**

An automated intelligence briefing system that combines real-time odds aggregation, machine learning predictions, NLP sentiment analysis, and value betting calculations to generate comprehensive pre-match reports for Arsenal FC.

[![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Bookmaker Odds** | Real-time aggregation from 15+ bookmakers via The Odds API |
| **ML Predictions** | Logistic regression model trained on 5+ years of historical data |
| **Sentiment Analysis** | NLP processing of news articles and social media (HuggingFace Transformers) |
| **Value Betting** | Expected Value calculations, Kelly Criterion stake sizing |
| **Automated Reports** | Professional HTML intelligence briefs via GitHub Actions |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR-USERNAME/arsenal-pulse.git
cd arsenal-pulse
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure (copy and edit with your API keys)
cp .env.example .env

# Generate intelligence brief
python orchestrator.py --opponent "Chelsea" --match-date "2026-01-20"
```

## Architecture

```
                              orchestrator.py
                           (Pipeline Controller)
                                    |
         +-------------+------------+------------+-------------+
         |             |            |            |             |
         v             v            v            v             v
   +-----------+ +-----------+ +-----------+ +-----------+ +-----------+
   |   Odds    | |Historical | |   News    | |  Lineup   | |    ML     |
   |  Fetcher  | |   Data    | | Scraper   | | Scraper   | | Predictor |
   +-----------+ +-----------+ +-----------+ +-----------+ +-----------+
         |             |            |            |             |
         +-------------+------------+------------+-------------+
                                    |
                                    v
                        +---------------------+
                        |  Sentiment Analyzer |
                        +---------------------+
                                    |
                                    v
                        +---------------------+
                        |  Value Calculator   |
                        +---------------------+
                                    |
                                    v
                        +---------------------+
                        |   Report Builder    |
                        |  (HTML/Markdown)    |
                        +---------------------+
```

## Project Structure

```
arsenal-pulse/
├── orchestrator.py                 # Main pipeline controller
├── main.py                         # Legacy notification script
├── requirements.txt                # Python dependencies
│
├── data_collection/                # Data gathering modules
│   ├── __init__.py                 # Module exports
│   ├── odds_data.py                # Odds data models and converters
│   ├── odds_fetcher.py             # The Odds API integration
│   ├── historical_data.py          # Historical match data collector
│   ├── news_scraper.py             # News scraping (Arsenal.com, BBC)
│   └── lineup_scraper.py           # Social media lineup intelligence
│
├── analysis/                       # Analysis and prediction modules
│   ├── __init__.py                 # Module exports
│   ├── ml_predictor.py             # ML match outcome predictions
│   ├── sentiment_analyzer.py       # NLP sentiment analysis
│   ├── value_calculator.py         # Value betting calculator
│   └── odds_analyzer.py            # Odds conversion and analysis
│
├── reporting/                      # Report generation
│   ├── __init__.py                 # Module exports
│   ├── report_builder.py           # Intelligence brief compiler
│   ├── chart_generator.py          # Visualization generator
│   └── templates/                  # Jinja2 templates
│       └── intelligence_brief.html # HTML report template
│
├── data/                           # Data storage
│   └── schemas/                    # JSON schema definitions
│       ├── odds.json
│       ├── lineups.json
│       └── news.json
│
├── tests/                          # Test suite
│   ├── test_integration.py         # End-to-end tests
│   └── test_odds_analyzer.py       # Unit tests
│
├── .github/workflows/              # CI/CD
│   ├── intelligence_brief.yml      # Intelligence brief workflow
│   └── notifier.yml                # Legacy notification workflow
│
├── .env.example                    # Environment template
└── README.md                       # This file
```

## Core Components

### 1. Odds Aggregation (`data_collection/odds_fetcher.py`)

Real-time odds collection from multiple bookmakers with rate limiting and caching.

```python
from data_collection import OddsFetcher

fetcher = OddsFetcher(api_key="your_key")
odds = await fetcher.get_match_odds("Arsenal", "Chelsea")
# Returns odds from Bet365, William Hill, Pinnacle, etc.
```

**Features:**
- 15+ bookmaker support
- Decimal/fractional/American format conversion
- Best odds identification per market
- Overround calculation and fair odds derivation

### 2. ML Predictions (`analysis/ml_predictor.py`)

Logistic regression model for match outcome prediction.

```python
from analysis import MatchPredictor

predictor = MatchPredictor()
prediction = predictor.predict(
    opponent="Chelsea",
    venue="home",
    arsenal_form="WWDWW",
    opponent_form="LDWDW"
)
# Returns: {'win': 0.65, 'draw': 0.22, 'loss': 0.13, 'confidence_interval': {...}}
```

**Features:**
- Win/Draw/Loss probability with confidence intervals
- Feature engineering: form, H2H, home advantage, injuries
- Cross-validation with accuracy metrics
- Model serialization and versioning

### 3. Sentiment Analysis (`analysis/sentiment_analyzer.py`)

NLP processing of news and social media content.

```python
from analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_articles(articles)
# Returns: {'overall_sentiment': 0.72, 'label': 'positive', 'themes': {...}}
```

**Features:**
- HuggingFace Transformers (DistilBERT)
- Source-weighted sentiment aggregation
- Theme extraction (tactics, injuries, confidence)
- Reddit/Twitter social sentiment

### 4. Value Calculator (`analysis/value_calculator.py`)

Identifies profitable betting opportunities.

```python
from analysis import ValueCalculator

calculator = ValueCalculator()
opportunities = calculator.find_value_bets(
    ml_predictions={'win': 0.65, 'draw': 0.22, 'loss': 0.13},
    bookmaker_odds={'win': 1.80, 'draw': 3.50, 'loss': 4.50}
)
# Returns value bets with EV, edge, and Kelly stake recommendations
```

**Features:**
- Expected Value (EV) calculation
- Kelly Criterion stake sizing
- Edge detection (model vs market)
- Risk assessment and bankroll management

### 5. Report Builder (`reporting/report_builder.py`)

Generates comprehensive HTML intelligence briefs.

```python
from reporting import ReportBuilder

builder = ReportBuilder()
report = builder.generate(
    fixture_info=fixture,
    odds_data=odds,
    predictions=ml_output,
    sentiment=sentiment_data,
    value_bets=opportunities
)
# Returns HTML report ready for email delivery
```

**Features:**
- Professional HTML templates (Jinja2)
- Executive summary with key insights
- Embedded charts (matplotlib)
- Mobile-responsive design

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

**Required:**
```bash
THE_ODDS_API_KEY=your_odds_api_key           # https://the-odds-api.com
FOOTBALL_DATA_API_KEY=your_football_data_key # https://football-data.org
```

**Optional - Social Sentiment:**
```bash
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=ArsenalPulse/2.0
```

**Optional - Email Delivery:**
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@example.com
```

**Optional - Push Notifications:**
```bash
NOTIFY_URL=ntfy://ntfy.sh/your-topic-name
USER_TIMEZONE=America/New_York
```

### GitHub Secrets (for CI/CD)

Add these in Settings > Secrets and variables > Actions:

| Secret | Required | Description |
|--------|----------|-------------|
| `THE_ODDS_API_KEY` | Yes | The Odds API key |
| `FOOTBALL_DATA_API_KEY` | Yes | Football-Data.org API key |
| `REDDIT_CLIENT_ID` | No | Reddit API client ID |
| `REDDIT_CLIENT_SECRET` | No | Reddit API secret |
| `SMTP_USER` | No | Email username |
| `SMTP_PASSWORD` | No | Email password/app password |
| `EMAIL_RECIPIENTS` | No | Comma-separated recipient list |

## Usage

### Command Line

```bash
# Basic usage
python orchestrator.py --opponent "Tottenham" --match-date "2026-01-20"

# Full options
python orchestrator.py \
    --opponent "Manchester City" \
    --match-date "2026-02-15" \
    --venue home \
    --competition "Premier League" \
    --output-dir ./reports \
    --send-email \
    --verbose
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--opponent` | Opposition team name | Required |
| `--match-date` | Match date (YYYY-MM-DD) | Required |
| `--venue` | `home` or `away` | Auto-detect |
| `--competition` | Competition name | Auto-detect |
| `--output-dir` | Report output directory | `./output` |
| `--send-email` | Email the report | `False` |
| `--skip-odds` | Skip odds collection | `False` |
| `--skip-news` | Skip news scraping | `False` |
| `--skip-sentiment` | Skip sentiment analysis | `False` |
| `--verbose` | Verbose logging | `False` |

### GitHub Actions

**Automated Schedule:** Runs daily at 8 AM UTC, checks for upcoming Arsenal matches.

**Manual Trigger:** Actions tab > "Arsenal Intelligence Brief" > Run workflow

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v
```

**Current Status:** 87 tests passing

## API Reference

### The Odds API
- **Purpose:** Real-time odds from 40+ bookmakers
- **Free Tier:** 500 requests/month
- **Docs:** https://the-odds-api.com/liveapi/guides/v4/

### Football-Data.org
- **Purpose:** Match fixtures, results, standings
- **Free Tier:** 10 req/min, unlimited monthly
- **Docs:** https://www.football-data.org/documentation/quickstart

### Reddit API
- **Purpose:** Social sentiment from r/Gunners
- **Free Tier:** 60 req/min (OAuth)
- **Docs:** https://www.reddit.com/dev/api/

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "API rate limit exceeded" | Too many requests | Wait for reset or upgrade plan |
| "No odds data available" | Match not listed yet | Odds available 7-14 days before match |
| "ML prediction failed" | Insufficient data | Retrain: `python -m analysis.ml_predictor --train` |
| "Email delivery failed" | SMTP auth error | Use Gmail App Password, not regular password |

Enable verbose logging:
```bash
python orchestrator.py --opponent "Chelsea" --match-date "2026-02-01" --verbose
```

## Cost

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Hosting | GitHub Actions | **$0** |
| Match Data | football-data.org | **$0** |
| Odds Data | The Odds API (500 req) | **$0** |
| NLP | HuggingFace | **$0** |
| Social Data | Reddit API | **$0** |
| Notifications | ntfy.sh | **$0** |
| **Total** | | **$0** |

## Tech Stack

- **Language:** Python 3.10+
- **ML:** scikit-learn, pandas, numpy
- **NLP:** HuggingFace Transformers
- **Web Scraping:** BeautifulSoup, aiohttp
- **Templating:** Jinja2
- **Visualization:** matplotlib
- **Testing:** pytest
- **CI/CD:** GitHub Actions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [The Odds API](https://the-odds-api.com/) - Betting odds data
- [Football-Data.org](https://www.football-data.org/) - Match fixtures
- [HuggingFace](https://huggingface.co/) - NLP models
- [ntfy.sh](https://ntfy.sh/) - Push notifications

---

**Come On You Gunners!**
