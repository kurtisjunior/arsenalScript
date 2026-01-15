#!/usr/bin/env python3
"""
Odds Data Storage and Conversion Utilities

This module provides:
- OddsData: Class for storing, loading, and validating odds data
- OddsConverter: Utilities for converting between odds formats
- Helper functions for calculating implied probabilities and best odds

Schema Format (aligned with data/schemas/odds.json):
{
    "match_id": "YYYYMMDD_HOME_AWAY",
    "timestamp": "ISO8601 datetime",
    "bookmaker_odds": [...],
    "best_value": {...},
    "metadata": {...}
}
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ODDS_DIR = DATA_DIR / "odds"
SCHEMA_DIR = DATA_DIR / "schemas"


class OddsConverter:
    """Utility class for converting between different odds formats."""

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """
        Convert decimal odds to American odds format.

        Args:
            decimal_odds: Odds in decimal format (e.g., 1.85, 3.50)

        Returns:
            American odds as integer (e.g., -118, +250)
        """
        if decimal_odds < 1.0:
            raise ValueError("Decimal odds must be >= 1.0")

        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1) * 100))
        else:
            return int(round(-100 / (decimal_odds - 1)))

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to decimal format.

        Args:
            american_odds: Odds in American format (e.g., -118, +250)

        Returns:
            Decimal odds (e.g., 1.85, 3.50)
        """
        if american_odds > 0:
            return round((american_odds / 100) + 1, 2)
        else:
            return round((100 / abs(american_odds)) + 1, 2)

    @staticmethod
    def decimal_to_fractional(decimal_odds: float) -> str:
        """
        Convert decimal odds to fractional (UK) format.

        Args:
            decimal_odds: Odds in decimal format

        Returns:
            Fractional odds as string (e.g., "5/4", "2/1")
        """
        if decimal_odds < 1.0:
            raise ValueError("Decimal odds must be >= 1.0")

        # Common fractional odds lookup for precision
        common_fractions = {
            1.10: "1/10", 1.11: "1/9", 1.125: "1/8", 1.14: "1/7",
            1.17: "1/6", 1.20: "1/5", 1.25: "1/4", 1.33: "1/3",
            1.40: "2/5", 1.44: "4/9", 1.50: "1/2", 1.53: "8/15",
            1.57: "4/7", 1.62: "8/13", 1.67: "2/3", 1.73: "8/11",
            1.80: "4/5", 1.83: "5/6", 1.91: "10/11", 2.00: "1/1",
            2.10: "11/10", 2.20: "6/5", 2.25: "5/4", 2.38: "11/8",
            2.50: "6/4", 2.63: "13/8", 2.75: "7/4", 2.88: "15/8",
            3.00: "2/1", 3.25: "9/4", 3.50: "5/2", 3.75: "11/4",
            4.00: "3/1", 4.50: "7/2", 5.00: "4/1", 5.50: "9/2",
            6.00: "5/1", 7.00: "6/1", 8.00: "7/1", 9.00: "8/1",
            10.00: "9/1", 11.00: "10/1", 13.00: "12/1", 17.00: "16/1",
            21.00: "20/1", 26.00: "25/1", 34.00: "33/1", 51.00: "50/1",
            101.00: "100/1"
        }

        # Find closest common fraction
        closest = min(common_fractions.keys(), key=lambda x: abs(x - decimal_odds))
        if abs(closest - decimal_odds) < 0.05:
            return common_fractions[closest]

        # Calculate approximate fraction
        profit = decimal_odds - 1
        # Try to find nice denominator
        for denom in [1, 2, 4, 5, 8, 10, 20]:
            numer = round(profit * denom)
            if abs(numer / denom - profit) < 0.01:
                return f"{numer}/{denom}"

        # Fallback: approximate
        return f"{int(round(profit * 10))}/10"

    @staticmethod
    def implied_probability(decimal_odds: float) -> float:
        """
        Calculate implied probability from decimal odds.

        Args:
            decimal_odds: Odds in decimal format

        Returns:
            Implied probability as decimal (0-1)
        """
        if decimal_odds < 1.0:
            raise ValueError("Decimal odds must be >= 1.0")
        return round(1 / decimal_odds, 4)

    @staticmethod
    def remove_overround(probabilities: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Remove bookmaker overround (margin) from probabilities.

        Args:
            probabilities: Dict of outcome -> implied probability

        Returns:
            Tuple of (normalized probabilities, overround percentage)
        """
        total = sum(probabilities.values())
        overround = total - 1.0

        normalized = {
            outcome: round(prob / total, 4)
            for outcome, prob in probabilities.items()
        }

        return normalized, round(overround, 4)


@dataclass
class BookmakerOdds:
    """Represents odds from a single bookmaker."""
    bookmaker: str
    home_win: float
    draw: float
    away_win: float
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "bookmaker": self.bookmaker,
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win
        }
        if self.over_2_5 is not None:
            result["over_2_5"] = self.over_2_5
        if self.under_2_5 is not None:
            result["under_2_5"] = self.under_2_5
        return result


@dataclass
class BestValue:
    """Represents best value odds for a market."""
    bookmaker: str
    odds: float

    def to_dict(self) -> Dict[str, Any]:
        return {"bookmaker": self.bookmaker, "odds": self.odds}


@dataclass
class OddsData:
    """
    Main class for handling odds data storage and retrieval.

    Aligned with the simplified schema in data/schemas/odds.json.

    Provides methods for:
    - Loading/saving odds data to JSON files
    - Calculating best odds across bookmakers
    - Computing implied probabilities
    """

    match_id: str
    timestamp: str
    bookmaker_odds: List[Dict[str, Any]]
    best_value: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]

    @classmethod
    def generate_match_id(cls, match_date: str, home_team: str, away_team: str) -> str:
        """
        Generate a standardized match ID.

        Args:
            match_date: ISO format date string
            home_team: Home team name
            away_team: Away team name

        Returns:
            Match ID string (format: YYYYMMDD_HOME_AWAY)
        """
        # Parse date
        dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y%m%d')

        # Team abbreviation mapping
        team_abbrev = {
            "arsenal": "ARS", "chelsea": "CHE", "liverpool": "LIV",
            "manchester city": "MCI", "manchester united": "MUN",
            "tottenham": "TOT", "aston villa": "AVL", "newcastle": "NEW",
            "brighton": "BHA", "west ham": "WHU", "bournemouth": "BOU",
            "crystal palace": "CRY", "fulham": "FUL", "brentford": "BRE",
            "everton": "EVE", "nottingham forest": "NFO", "wolves": "WOL",
            "wolverhampton": "WOL", "ipswich": "IPS", "leicester": "LEI",
            "southampton": "SOU"
        }

        def get_abbrev(name: str) -> str:
            name_lower = name.lower()
            name_lower = re.sub(r'\s+(fc|afc|cf|town|city|united|hotspur)$', '', name_lower)
            name_lower = name_lower.strip()
            return team_abbrev.get(name_lower, name[:3].upper())

        home_abbrev = get_abbrev(home_team)
        away_abbrev = get_abbrev(away_team)

        return f"{date_str}_{home_abbrev}_{away_abbrev}"

    @classmethod
    def from_file(cls, match_id: str, odds_dir: Path = ODDS_DIR) -> Optional['OddsData']:
        """
        Load odds data from file.

        Args:
            match_id: The match ID to load
            odds_dir: Directory containing odds files

        Returns:
            OddsData instance or None if not found
        """
        file_path = odds_dir / f"{match_id}.json"

        if not file_path.exists():
            logger.warning(f"Odds file not found: {file_path}")
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            return cls(
                match_id=data['match_id'],
                timestamp=data['timestamp'],
                bookmaker_odds=data['bookmaker_odds'],
                best_value=data['best_value'],
                metadata=data['metadata']
            )
        except Exception as e:
            logger.error(f"Error loading odds file {file_path}: {e}")
            return None

    def to_file(self, odds_dir: Path = ODDS_DIR) -> bool:
        """
        Save odds data to file.

        Args:
            odds_dir: Directory to save to

        Returns:
            True if successful, False otherwise
        """
        # Ensure directory exists
        odds_dir.mkdir(parents=True, exist_ok=True)

        file_path = odds_dir / f"{self.match_id}.json"

        try:
            data = {
                'match_id': self.match_id,
                'timestamp': self.timestamp,
                'bookmaker_odds': self.bookmaker_odds,
                'best_value': self.best_value,
                'metadata': self.metadata
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved odds data to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving odds file {file_path}: {e}")
            return False

    @classmethod
    def calculate_best_value(cls, bookmaker_odds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate best odds across all bookmakers for each market.

        Args:
            bookmaker_odds: List of bookmaker odds dictionaries

        Returns:
            Dictionary of best value by market
        """
        best = {
            'home_win': {'bookmaker': '', 'odds': 0.0},
            'draw': {'bookmaker': '', 'odds': 0.0},
            'away_win': {'bookmaker': '', 'odds': 0.0},
        }

        # Check if over/under markets exist
        has_totals = any('over_2_5' in bm for bm in bookmaker_odds)
        if has_totals:
            best['over_2_5'] = {'bookmaker': '', 'odds': 0.0}
            best['under_2_5'] = {'bookmaker': '', 'odds': 0.0}

        for bm in bookmaker_odds:
            bookmaker = bm.get('bookmaker', '')

            for market in ['home_win', 'draw', 'away_win', 'over_2_5', 'under_2_5']:
                if market in bm and market in best:
                    odds = bm[market]
                    if odds > best[market]['odds']:
                        best[market] = {'bookmaker': bookmaker, 'odds': odds}

        # Remove empty entries
        return {k: v for k, v in best.items() if v['odds'] > 0}

    def get_implied_probabilities(self) -> Dict[str, float]:
        """
        Calculate implied probabilities from best value odds.

        Returns:
            Dictionary with normalized probabilities and overround
        """
        if not self.best_value:
            return {}

        required = ['home_win', 'draw', 'away_win']
        if not all(k in self.best_value for k in required):
            logger.warning("Missing required odds for probability calculation")
            return {}

        raw_probs = {
            'home_win': OddsConverter.implied_probability(self.best_value['home_win']['odds']),
            'draw': OddsConverter.implied_probability(self.best_value['draw']['odds']),
            'away_win': OddsConverter.implied_probability(self.best_value['away_win']['odds'])
        }

        normalized, overround = OddsConverter.remove_overround(raw_probs)
        normalized['overround'] = overround

        return normalized

    @classmethod
    def list_available_matches(cls, odds_dir: Path = ODDS_DIR) -> List[str]:
        """
        List all match IDs with saved odds data.

        Returns:
            List of match ID strings
        """
        if not odds_dir.exists():
            return []

        matches = []
        for f in odds_dir.glob("*.json"):
            if f.name != '.gitkeep' and not f.name.startswith('example_'):
                matches.append(f.stem)

        return sorted(matches)

    @classmethod
    def from_api_response(
        cls,
        match_id: str,
        timestamp: str,
        api_data: List[Dict],
        source: str = "the-odds-api"
    ) -> 'OddsData':
        """
        Create OddsData from API response.

        Args:
            match_id: The match identifier
            timestamp: Match kickoff time
            api_data: Raw API response data (list of bookmaker odds)
            source: API source name

        Returns:
            OddsData instance
        """
        bookmaker_odds = []

        for bm_data in api_data:
            bm_odds = {
                'bookmaker': bm_data.get('bookmaker', bm_data.get('key', 'unknown'))
            }

            # Extract h2h odds
            if 'home_win' in bm_data:
                bm_odds['home_win'] = bm_data['home_win']
                bm_odds['draw'] = bm_data['draw']
                bm_odds['away_win'] = bm_data['away_win']
            elif 'markets' in bm_data:
                # Handle The Odds API format
                for market in bm_data.get('markets', []):
                    if market.get('key') == 'h2h':
                        for outcome in market.get('outcomes', []):
                            name = outcome.get('name', '').lower()
                            price = outcome.get('price', 0)
                            if 'draw' in name:
                                bm_odds['draw'] = price
                            elif outcome.get('name') == bm_data.get('home_team'):
                                bm_odds['home_win'] = price
                            else:
                                bm_odds['away_win'] = price

            # Extract totals if available
            if 'over_2_5' in bm_data:
                bm_odds['over_2_5'] = bm_data['over_2_5']
            if 'under_2_5' in bm_data:
                bm_odds['under_2_5'] = bm_data['under_2_5']

            if 'home_win' in bm_odds:
                bookmaker_odds.append(bm_odds)

        best_value = cls.calculate_best_value(bookmaker_odds)

        return cls(
            match_id=match_id,
            timestamp=timestamp,
            bookmaker_odds=bookmaker_odds,
            best_value=best_value,
            metadata={
                'source': source,
                'fetch_timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        )


def example_usage():
    """Demonstrate usage of the odds data module."""

    print("=== OddsData Example Usage ===\n")

    # Generate a match ID
    match_id = OddsData.generate_match_id(
        match_date="2026-01-20T15:00:00Z",
        home_team="Arsenal FC",
        away_team="Aston Villa"
    )
    print(f"Generated match ID: {match_id}")

    # Odds conversion examples
    print("\n--- Odds Conversion ---")
    decimal_odds = 1.85
    print(f"Decimal: {decimal_odds}")
    print(f"American: {OddsConverter.decimal_to_american(decimal_odds)}")
    print(f"Fractional: {OddsConverter.decimal_to_fractional(decimal_odds)}")
    print(f"Implied Probability: {OddsConverter.implied_probability(decimal_odds):.2%}")

    # Load example data
    print("\n--- Loading Example Data ---")
    example_path = ODDS_DIR / "example_20260120_arsenal_vs_aston_villa.json"
    if example_path.exists():
        with open(example_path) as f:
            data = json.load(f)

        odds = OddsData(
            match_id=data['match_id'],
            timestamp=data['timestamp'],
            bookmaker_odds=data['bookmaker_odds'],
            best_value=data['best_value'],
            metadata=data['metadata']
        )

        print(f"Loaded match: {odds.match_id}")
        print(f"Bookmakers: {len(odds.bookmaker_odds)}")

        print("\nBest Value Odds:")
        for market, details in odds.best_value.items():
            print(f"  {market}: {details['odds']} ({details['bookmaker']})")

        # Calculate probabilities
        probs = odds.get_implied_probabilities()
        print("\nImplied Probabilities:")
        for outcome, prob in probs.items():
            if outcome != 'overround':
                print(f"  {outcome}: {prob:.1%}")
        print(f"  Overround: {probs.get('overround', 0):.1%}")

    # Create new odds data example
    print("\n--- Creating New Odds Data ---")
    new_odds = OddsData.from_api_response(
        match_id="20260125_ARS_MCI",
        timestamp="2026-01-25T17:30:00Z",
        api_data=[
            {"bookmaker": "draftkings", "home_win": 2.80, "draw": 3.40, "away_win": 2.50},
            {"bookmaker": "fanduel", "home_win": 2.75, "draw": 3.50, "away_win": 2.55},
        ],
        source="the-odds-api"
    )
    print(f"Created odds for: {new_odds.match_id}")
    print(f"Best home win: {new_odds.best_value['home_win']['odds']} ({new_odds.best_value['home_win']['bookmaker']})")


if __name__ == "__main__":
    example_usage()
