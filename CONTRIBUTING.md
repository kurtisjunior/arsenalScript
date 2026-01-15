# Contributing to Arsenal Pulse

Thank you for your interest in contributing to Arsenal Pulse! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- API keys (see [README.md](README.md#configuration))

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork via GitHub UI, then:
   git clone https://github.com/YOUR-USERNAME/arsenal-pulse.git
   cd arsenal-pulse
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Dev dependencies
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add WhatsApp notification support
fix: Handle empty odds response gracefully
docs: Update API configuration guide
refactor: Extract odds conversion to separate module
test: Add integration tests for value calculator
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests and linting**
   ```bash
   pytest tests/ -v
   flake8 .
   black --check .
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: Your descriptive message"
   git push origin feature/your-feature
   ```

5. **Open a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Wait for review

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting
- Use type hints where appropriate
- Document functions with docstrings

### Example

```python
def calculate_expected_value(
    probability: float,
    odds: float,
    stake: float = 1.0
) -> float:
    """
    Calculate expected value for a bet.

    Args:
        probability: Model's estimated win probability (0-1)
        odds: Decimal odds offered by bookmaker
        stake: Bet amount (default: 1.0)

    Returns:
        Expected value as a float

    Example:
        >>> calculate_expected_value(0.6, 1.8)
        0.08
    """
    return (probability * odds * stake) - stake
```

## Project Structure

```
arsenal-pulse/
├── data_collection/     # Data gathering modules
├── analysis/            # ML and analysis modules
├── reporting/           # Report generation
├── tests/               # Test suite
├── docs/                # Additional documentation
└── .github/workflows/   # CI/CD
```

### Module Guidelines

- **data_collection/**: External API integrations, web scraping
- **analysis/**: ML models, calculations, data processing
- **reporting/**: Report generation, templates, visualizations

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_integration.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Mock external APIs

```python
def test_odds_converter_handles_invalid_input():
    """Odds converter should return None for invalid odds."""
    converter = OddsConverter()
    result = converter.decimal_to_probability(-1.5)
    assert result is None
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Questions?

- Check existing [issues](https://github.com/YOUR-USERNAME/arsenal-pulse/issues)
- Open a new discussion
- Review the [README](README.md)

---

**Come On You Gunners!**
