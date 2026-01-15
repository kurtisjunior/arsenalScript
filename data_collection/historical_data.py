#!/usr/bin/env python3
"""
Historical Match Data Collector for Arsenal FC

This module provides tools for collecting historical Arsenal match data
from football-data.org API for use in ML training pipelines.

Features:
- Fetch Arsenal matches by season
- Collect multi-season data for training datasets
- Calculate form statistics
- Export to CSV format for ML consumption

API: https://api.football-data.org/v4/
Env: FOOTBALL_DATA_API_KEY
"""

import csv
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"

# football-data.org API configuration
API_BASE_URL = "https://api.football-data.org/v4"
ARSENAL_TEAM_ID = 57  # Arsenal's team ID in the API
PREMIER_LEAGUE_CODE = "PL"

# Rate limiting: free tier allows 10 requests/minute
RATE_LIMIT_DELAY = 6.5  # seconds between requests (safe margin)


@dataclass
class MatchResult:
    """Represents a single match result with all relevant data."""
    date: str
    home_team: str
    away_team: str
    competition: str
    home_score: int
    away_score: int
    result: str  # W/D/L from Arsenal perspective
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    venue: str = ""  # "home" or "away" for Arsenal
    match_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON serialization."""
        return {
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "competition": self.competition,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "result": self.result,
            "home_xg": self.home_xg if self.home_xg is not None else "",
            "away_xg": self.away_xg if self.away_xg is not None else "",
            "venue": self.venue,
            "match_id": self.match_id if self.match_id else ""
        }


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    pass


class HistoricalDataCollector:
    """
    Collector for historical Arsenal match data from football-data.org.

    This class handles:
    - API authentication and rate limiting
    - Fetching match data by season
    - Multi-season data collection
    - Form calculation
    - CSV export for ML training

    Usage:
        collector = HistoricalDataCollector()
        matches = collector.fetch_season_matches("2024")
        collector.save_to_csv("data/historical/arsenal_matches.csv")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the collector with API credentials.

        Args:
            api_key: football-data.org API key. If not provided,
                     reads from FOOTBALL_DATA_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            logger.warning(
                "No API key provided. Set FOOTBALL_DATA_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": self.api_key or "",
            "Content-Type": "application/json"
        })

        self._last_request_time: float = 0
        self._matches: List[MatchResult] = []

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a rate-limited API request.

        Args:
            endpoint: API endpoint path (without base URL)
            params: Optional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            RateLimitError: If rate limit is exceeded
            APIError: For other API errors
        """
        self._rate_limit()

        url = f"{API_BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 429:
                retry_after = response.headers.get("X-RequestCounter-Reset", 60)
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds."
                )

            if response.status_code == 403:
                raise APIError(
                    "API authentication failed. Check your FOOTBALL_DATA_API_KEY."
                )

            if response.status_code == 404:
                raise APIError(f"Resource not found: {endpoint}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise APIError(f"Request timeout for {endpoint}")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Connection error for {endpoint}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}")

    def _determine_result(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int
    ) -> str:
        """
        Determine match result from Arsenal's perspective.

        Args:
            home_team: Name of home team
            away_team: Name of away team
            home_score: Home team goals
            away_score: Away team goals

        Returns:
            'W' for win, 'D' for draw, 'L' for loss (Arsenal perspective)
        """
        arsenal_home = "arsenal" in home_team.lower()

        if home_score == away_score:
            return "D"
        elif home_score > away_score:
            return "W" if arsenal_home else "L"
        else:
            return "L" if arsenal_home else "W"

    def _determine_venue(self, home_team: str) -> str:
        """Determine if Arsenal played home or away."""
        return "home" if "arsenal" in home_team.lower() else "away"

    def _parse_match(self, match_data: Dict) -> Optional[MatchResult]:
        """
        Parse API match data into MatchResult object.

        Args:
            match_data: Raw match data from API

        Returns:
            MatchResult object or None if parsing fails
        """
        try:
            # Skip matches that haven't been played
            status = match_data.get("status", "")
            if status not in ["FINISHED", "AWARDED"]:
                return None

            home_team = match_data.get("homeTeam", {}).get("name", "")
            away_team = match_data.get("awayTeam", {}).get("name", "")

            score = match_data.get("score", {})
            full_time = score.get("fullTime", {})
            home_score = full_time.get("home")
            away_score = full_time.get("away")

            if home_score is None or away_score is None:
                return None

            # Parse date
            utc_date = match_data.get("utcDate", "")
            if utc_date:
                dt = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = ""

            # Get competition name
            competition = match_data.get("competition", {}).get("name", "Premier League")

            # Determine result and venue from Arsenal's perspective
            result = self._determine_result(home_team, away_team, home_score, away_score)
            venue = self._determine_venue(home_team)

            return MatchResult(
                date=date_str,
                home_team=home_team,
                away_team=away_team,
                competition=competition,
                home_score=home_score,
                away_score=away_score,
                result=result,
                home_xg=None,  # xG not available in free tier
                away_xg=None,
                venue=venue,
                match_id=match_data.get("id")
            )

        except Exception as e:
            logger.error(f"Error parsing match data: {e}")
            return None

    def fetch_season_matches(self, season: str) -> List[MatchResult]:
        """
        Fetch all Arsenal matches for a specific season.

        Args:
            season: Season year (e.g., "2024" for 2024-25 season,
                    "2023" for 2023-24 season)

        Returns:
            List of MatchResult objects for the season

        Example:
            matches = collector.fetch_season_matches("2024")
        """
        logger.info(f"Fetching Arsenal matches for {season}-{int(season)+1} season")

        endpoint = f"/teams/{ARSENAL_TEAM_ID}/matches"
        params = {
            "season": season,
            "status": "FINISHED"
        }

        try:
            data = self._make_request(endpoint, params)
            matches_data = data.get("matches", [])

            matches = []
            for match_data in matches_data:
                match = self._parse_match(match_data)
                if match:
                    matches.append(match)

            # Sort by date
            matches.sort(key=lambda x: x.date)

            logger.info(f"Found {len(matches)} completed matches for {season} season")

            # Add to internal storage
            self._matches.extend(matches)

            return matches

        except APIError as e:
            logger.error(f"Failed to fetch {season} season: {e}")
            return []

    def fetch_multiple_seasons(
        self,
        seasons: Optional[List[str]] = None
    ) -> List[MatchResult]:
        """
        Fetch Arsenal matches for multiple seasons.

        Args:
            seasons: List of season years. Defaults to last 3 seasons.
                     E.g., ["2022", "2023", "2024"] for 2022-23, 2023-24, 2024-25

        Returns:
            List of all MatchResult objects across seasons

        Example:
            # Fetch last 3 seasons (default)
            matches = collector.fetch_multiple_seasons()

            # Or specify seasons explicitly
            matches = collector.fetch_multiple_seasons(["2022", "2023", "2024"])
        """
        if seasons is None:
            # Default to last 3 seasons
            current_year = datetime.now().year
            # If we're in first half of year, current season started last year
            if datetime.now().month < 8:
                current_season = current_year - 1
            else:
                current_season = current_year

            seasons = [str(current_season - i) for i in range(3)]
            seasons.reverse()  # Oldest first

        logger.info(f"Fetching data for seasons: {seasons}")

        # Clear existing matches
        self._matches = []

        all_matches = []
        for season in seasons:
            matches = self.fetch_season_matches(season)
            all_matches.extend(matches)

        # Sort all matches by date
        all_matches.sort(key=lambda x: x.date)
        self._matches = all_matches

        logger.info(f"Total matches collected: {len(all_matches)}")
        return all_matches

    def save_to_csv(
        self,
        filepath: Optional[str] = None,
        matches: Optional[List[MatchResult]] = None
    ) -> str:
        """
        Export match data to CSV file for ML training.

        Args:
            filepath: Output file path. Defaults to data/historical/arsenal_matches.csv
            matches: List of matches to export. Uses internal storage if not provided.

        Returns:
            Path to the created CSV file

        Example:
            collector.save_to_csv()  # Uses default path
            collector.save_to_csv("custom/path/matches.csv")
        """
        if filepath is None:
            filepath = str(HISTORICAL_DIR / "arsenal_matches.csv")

        if matches is None:
            matches = self._matches

        if not matches:
            logger.warning("No matches to export")
            return filepath

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # CSV field names
        fieldnames = [
            "date", "home_team", "away_team", "competition",
            "home_score", "away_score", "result",
            "home_xg", "away_xg", "venue", "match_id"
        ]

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for match in matches:
                    writer.writerow(match.to_dict())

            logger.info(f"Exported {len(matches)} matches to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise

    def get_form_stats(
        self,
        n_games: int = 5,
        matches: Optional[List[MatchResult]] = None
    ) -> Dict[str, Any]:
        """
        Calculate recent form statistics for Arsenal.

        Args:
            n_games: Number of recent games to consider (default: 5)
            matches: List of matches to analyze. Uses internal storage if not provided.

        Returns:
            Dictionary with form statistics:
            - wins: Number of wins
            - draws: Number of draws
            - losses: Number of losses
            - points: Total points (W=3, D=1, L=0)
            - goals_scored: Total goals scored
            - goals_conceded: Total goals conceded
            - goal_difference: Goals scored - conceded
            - form_string: Recent results as string (e.g., "WWDLW")
            - clean_sheets: Number of clean sheets
            - failed_to_score: Number of games without scoring

        Example:
            form = collector.get_form_stats(n_games=5)
            print(f"Last 5 games: {form['form_string']}")
            print(f"Points: {form['points']}/15")
        """
        if matches is None:
            matches = self._matches

        if not matches:
            logger.warning("No matches available for form calculation")
            return {}

        # Get last n completed matches, sorted by date
        sorted_matches = sorted(matches, key=lambda x: x.date, reverse=True)
        recent = sorted_matches[:n_games]

        wins = sum(1 for m in recent if m.result == "W")
        draws = sum(1 for m in recent if m.result == "D")
        losses = sum(1 for m in recent if m.result == "L")

        # Calculate goals from Arsenal's perspective
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0
        failed_to_score = 0

        for match in recent:
            if match.venue == "home":
                goals_scored += match.home_score
                goals_conceded += match.away_score
                if match.away_score == 0:
                    clean_sheets += 1
                if match.home_score == 0:
                    failed_to_score += 1
            else:
                goals_scored += match.away_score
                goals_conceded += match.home_score
                if match.home_score == 0:
                    clean_sheets += 1
                if match.away_score == 0:
                    failed_to_score += 1

        # Form string (most recent first)
        form_string = "".join(m.result for m in recent)

        return {
            "n_games": len(recent),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": (wins * 3) + draws,
            "max_points": len(recent) * 3,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "form_string": form_string,
            "clean_sheets": clean_sheets,
            "failed_to_score": failed_to_score
        }

    def get_head_to_head(
        self,
        opponent: str,
        matches: Optional[List[MatchResult]] = None
    ) -> Dict[str, Any]:
        """
        Get head-to-head record against a specific opponent.

        Args:
            opponent: Team name to check (partial match supported)
            matches: List of matches to analyze

        Returns:
            Dictionary with H2H statistics
        """
        if matches is None:
            matches = self._matches

        opponent_lower = opponent.lower()
        h2h_matches = [
            m for m in matches
            if opponent_lower in m.home_team.lower() or
               opponent_lower in m.away_team.lower()
        ]

        if not h2h_matches:
            return {"matches_played": 0, "opponent": opponent}

        wins = sum(1 for m in h2h_matches if m.result == "W")
        draws = sum(1 for m in h2h_matches if m.result == "D")
        losses = sum(1 for m in h2h_matches if m.result == "L")

        return {
            "opponent": opponent,
            "matches_played": len(h2h_matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / len(h2h_matches) if h2h_matches else 0
        }


def example_usage():
    """Demonstrate usage of the HistoricalDataCollector."""

    print("=== Historical Data Collector Example ===\n")

    # Check for API key
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("Note: FOOTBALL_DATA_API_KEY not set.")
        print("Set it to fetch live data from football-data.org\n")
        print("Example:")
        print("  export FOOTBALL_DATA_API_KEY='your-api-key'\n")

        # Demo with mock data
        print("--- Demo with sample data ---")
        collector = HistoricalDataCollector()

        # Create sample matches for demonstration
        sample_matches = [
            MatchResult(
                date="2024-08-17",
                home_team="Arsenal FC",
                away_team="Wolverhampton Wanderers FC",
                competition="Premier League",
                home_score=2,
                away_score=0,
                result="W",
                venue="home",
                match_id=1
            ),
            MatchResult(
                date="2024-08-24",
                home_team="Aston Villa FC",
                away_team="Arsenal FC",
                competition="Premier League",
                home_score=0,
                away_score=2,
                result="W",
                venue="away",
                match_id=2
            ),
            MatchResult(
                date="2024-08-31",
                home_team="Arsenal FC",
                away_team="Brighton & Hove Albion FC",
                competition="Premier League",
                home_score=1,
                away_score=1,
                result="D",
                venue="home",
                match_id=3
            ),
            MatchResult(
                date="2024-09-14",
                home_team="Tottenham Hotspur FC",
                away_team="Arsenal FC",
                competition="Premier League",
                home_score=0,
                away_score=1,
                result="W",
                venue="away",
                match_id=4
            ),
            MatchResult(
                date="2024-09-21",
                home_team="Arsenal FC",
                away_team="Manchester City FC",
                competition="Premier League",
                home_score=2,
                away_score=2,
                result="D",
                venue="home",
                match_id=5
            ),
        ]

        collector._matches = sample_matches

        # Calculate form
        form = collector.get_form_stats(n_games=5)
        print(f"Last 5 games form: {form['form_string']}")
        print(f"Points: {form['points']}/{form['max_points']}")
        print(f"Goals: {form['goals_scored']} scored, {form['goals_conceded']} conceded")
        print(f"Record: {form['wins']}W-{form['draws']}D-{form['losses']}L")

        # Save to CSV
        csv_path = collector.save_to_csv()
        print(f"\nSample data saved to: {csv_path}")

        return

    # With API key - fetch real data
    print("--- Fetching live data ---")
    collector = HistoricalDataCollector(api_key)

    # Fetch last 3 seasons
    print("\nFetching last 3 seasons of Arsenal matches...")
    matches = collector.fetch_multiple_seasons()

    print(f"\nTotal matches: {len(matches)}")

    # Show recent form
    form = collector.get_form_stats(n_games=5)
    print(f"\nLast 5 games: {form['form_string']}")
    print(f"Points: {form['points']}/{form['max_points']}")

    # Save to CSV
    csv_path = collector.save_to_csv()
    print(f"\nData exported to: {csv_path}")

    # Show head-to-head example
    h2h = collector.get_head_to_head("Tottenham")
    print(f"\nH2H vs Tottenham: {h2h['wins']}W-{h2h['draws']}D-{h2h['losses']}L")


if __name__ == "__main__":
    example_usage()
