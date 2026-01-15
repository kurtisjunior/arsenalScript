#!/usr/bin/env python3
"""
Odds Fetcher - API Client for The Odds API

This module provides the OddsFetcher class for fetching betting odds
from The Odds API (https://the-odds-api.com/).

Features:
- Rate limiting to stay within free tier (500 requests/month)
- Response caching to minimize API calls
- Automatic retry with exponential backoff
- Conversion to OddsData format for storage

Usage:
    from odds_fetcher import OddsFetcher

    fetcher = OddsFetcher()  # Uses THE_ODDS_API_KEY env var
    matches = fetcher.fetch_upcoming_matches()
    arsenal_matches = fetcher.fetch_arsenal_matches()
"""

import json
import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import wraps
import urllib.request
import urllib.error
import urllib.parse

from .odds_data import OddsData

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache" / "odds_api"

# API Configuration
BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "soccer_epl"
DEFAULT_REGIONS = "us,uk"
DEFAULT_MARKETS = "h2h,totals"

# Rate limiting configuration (free tier: 500 requests/month)
MAX_MONTHLY_REQUESTS = 500
MAX_DAILY_REQUESTS = 15  # ~450/month, leaves buffer
REQUEST_INTERVAL_SECONDS = 5  # Minimum time between consecutive calls
CACHE_DURATION_HOURS = 6  # Re-use cached data within window

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class APIError(Exception):
    """Raised when API returns an error."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class OddsFetcher:
    """
    API client for The Odds API.

    Provides methods for fetching betting odds for football matches,
    with built-in rate limiting, caching, and error handling.

    Attributes:
        api_key: The Odds API key
        cache_dir: Directory for cached responses
        last_request_time: Timestamp of last API request
        daily_request_count: Number of requests made today

    Example:
        >>> fetcher = OddsFetcher()
        >>> matches = fetcher.fetch_upcoming_matches()
        >>> for match in matches:
        ...     print(f"{match['home_team']} vs {match['away_team']}")
    """

    def __init__(self, api_key: str = None, cache_dir: Path = None):
        """
        Initialize the OddsFetcher with API key and cache directory.

        Args:
            api_key: The Odds API key. If not provided, reads from
                     THE_ODDS_API_KEY environment variable.
            cache_dir: Directory for cached responses. Defaults to
                       data/cache/odds_api.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set THE_ODDS_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.last_request_time: Optional[float] = None
        self._load_request_count()

        logger.info(f"OddsFetcher initialized. Daily requests: {self.daily_request_count}/{MAX_DAILY_REQUESTS}")

    def _load_request_count(self) -> None:
        """Load the daily request count from cache file."""
        count_file = self.cache_dir / "request_count.json"
        today = datetime.utcnow().strftime("%Y-%m-%d")

        if count_file.exists():
            try:
                with open(count_file, 'r') as f:
                    data = json.load(f)
                if data.get("date") == today:
                    self.daily_request_count = data.get("count", 0)
                else:
                    # New day, reset count
                    self.daily_request_count = 0
            except Exception as e:
                logger.warning(f"Error loading request count: {e}")
                self.daily_request_count = 0
        else:
            self.daily_request_count = 0

    def _save_request_count(self) -> None:
        """Save the daily request count to cache file."""
        count_file = self.cache_dir / "request_count.json"
        today = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            with open(count_file, 'w') as f:
                json.dump({"date": today, "count": self.daily_request_count}, f)
        except Exception as e:
            logger.warning(f"Error saving request count: {e}")

    def _get_cache_key(self, endpoint: str, params: Dict[str, str]) -> str:
        """Generate a cache key for the given request."""
        # Remove API key from params for cache key
        cache_params = {k: v for k, v in params.items() if k != "apiKey"}
        key_string = f"{endpoint}:{json.dumps(cache_params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if it exists and is still valid.

        Args:
            cache_key: The cache key for the request

        Returns:
            Cached response data if valid, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            cache_age = datetime.utcnow() - cache_time

            if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
                logger.debug(f"Cache hit for {cache_key}")
                return cached.get("data")
            else:
                logger.debug(f"Cache expired for {cache_key}")
                return None

        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save response data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "cached_at": datetime.utcnow().isoformat(),
                    "data": data
                }, f)
            logger.debug(f"Cached response for {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting before making a request.

        Raises:
            RateLimitExceeded: If daily request limit is reached
        """
        # Check daily limit
        if self.daily_request_count >= MAX_DAILY_REQUESTS:
            raise RateLimitExceeded(
                f"Daily request limit reached ({MAX_DAILY_REQUESTS}). "
                "Try again tomorrow or use cached data."
            )

        # Enforce minimum interval between requests
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < REQUEST_INTERVAL_SECONDS:
                sleep_time = REQUEST_INTERVAL_SECONDS - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to The Odds API with caching and retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/sports/soccer_epl/odds")
            params: Query parameters (api key added automatically)
            use_cache: Whether to use cached responses

        Returns:
            API response data as dictionary

        Raises:
            APIError: If API returns an error
            RateLimitExceeded: If rate limit is exceeded
        """
        params = params or {}
        params["apiKey"] = self.api_key

        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        if use_cache:
            cached = self._get_cached_response(cache_key)
            if cached is not None:
                return cached

        # Enforce rate limiting
        self._enforce_rate_limit()

        # Build URL
        url = f"{BASE_URL}{endpoint}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        # Make request with retry logic
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"API request: {endpoint} (attempt {attempt + 1}/{MAX_RETRIES})")

                request = urllib.request.Request(
                    url,
                    headers={"Accept": "application/json"}
                )

                with urllib.request.urlopen(request, timeout=30) as response:
                    self.last_request_time = time.time()
                    self.daily_request_count += 1
                    self._save_request_count()

                    data = json.loads(response.read().decode())

                    # Log remaining requests from headers
                    remaining = response.headers.get("x-requests-remaining")
                    if remaining:
                        logger.info(f"API requests remaining this month: {remaining}")

                    # Cache successful response
                    self._save_to_cache(cache_key, data)

                    return data

            except urllib.error.HTTPError as e:
                last_error = e
                status_code = e.code

                if status_code == 401:
                    raise APIError("Invalid API key", status_code=401)
                elif status_code == 429:
                    raise RateLimitExceeded("API rate limit exceeded")
                elif status_code == 404:
                    raise APIError(f"Resource not found: {endpoint}", status_code=404)
                elif status_code >= 500:
                    # Server error, retry with backoff
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(f"Server error {status_code}, retrying in {backoff}s")
                    time.sleep(backoff)
                else:
                    raise APIError(f"HTTP error {status_code}", status_code=status_code)

            except urllib.error.URLError as e:
                last_error = e
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(f"Network error: {e}, retrying in {backoff}s")
                time.sleep(backoff)

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                raise APIError(f"Unexpected error: {e}")

        # All retries failed
        raise APIError(f"Request failed after {MAX_RETRIES} attempts: {last_error}")

    def fetch_upcoming_matches(
        self,
        sport: str = DEFAULT_SPORT,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch upcoming matches with odds for a given sport.

        Args:
            sport: Sport key (default: "soccer_epl" for Premier League)
            regions: Comma-separated region codes for bookmakers (default: "us,uk")
            markets: Comma-separated market types (default: "h2h,totals")
            use_cache: Whether to use cached responses (default: True)

        Returns:
            List of match dictionaries with odds from multiple bookmakers.
            Each match contains:
            - id: Unique match ID from the API
            - sport_key: Sport identifier
            - sport_title: Human-readable sport name
            - commence_time: Match start time (ISO 8601)
            - home_team: Home team name
            - away_team: Away team name
            - bookmakers: List of bookmaker odds

        Example:
            >>> matches = fetcher.fetch_upcoming_matches()
            >>> for match in matches:
            ...     print(f"{match['home_team']} vs {match['away_team']}")
            ...     print(f"  Kickoff: {match['commence_time']}")
        """
        endpoint = f"/sports/{sport}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }

        try:
            data = self._make_request(endpoint, params, use_cache=use_cache)
            logger.info(f"Fetched {len(data)} upcoming matches for {sport}")
            return data
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            raise

    def fetch_match_odds(
        self,
        match_id: str,
        sport: str = DEFAULT_SPORT,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch odds for a specific match by ID.

        Args:
            match_id: The Odds API event ID for the match
            sport: Sport key (default: "soccer_epl")
            regions: Comma-separated region codes for bookmakers
            markets: Comma-separated market types
            use_cache: Whether to use cached responses

        Returns:
            Match dictionary with odds, or None if match not found.

        Note:
            The Odds API does not have a direct endpoint for single matches.
            This method fetches all upcoming matches and filters by ID.
            The response is cached to minimize API calls.

        Example:
            >>> odds = fetcher.fetch_match_odds("abc123def456")
            >>> if odds:
            ...     print(f"Found: {odds['home_team']} vs {odds['away_team']}")
        """
        try:
            matches = self.fetch_upcoming_matches(
                sport=sport,
                regions=regions,
                markets=markets,
                use_cache=use_cache
            )

            for match in matches:
                if match.get("id") == match_id:
                    logger.info(f"Found match: {match['home_team']} vs {match['away_team']}")
                    return match

            logger.warning(f"Match not found: {match_id}")
            return None

        except Exception as e:
            logger.error(f"Error fetching match odds: {e}")
            raise

    def fetch_arsenal_matches(
        self,
        sport: str = DEFAULT_SPORT,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch upcoming matches involving Arsenal.

        Args:
            sport: Sport key (default: "soccer_epl")
            use_cache: Whether to use cached responses

        Returns:
            List of match dictionaries where Arsenal is playing.

        Example:
            >>> arsenal_matches = fetcher.fetch_arsenal_matches()
            >>> for match in arsenal_matches:
            ...     is_home = "Arsenal" in match['home_team']
            ...     opponent = match['away_team'] if is_home else match['home_team']
            ...     venue = "home" if is_home else "away"
            ...     print(f"Arsenal ({venue}) vs {opponent}")
        """
        arsenal_keywords = ["arsenal", "ars"]

        try:
            all_matches = self.fetch_upcoming_matches(sport=sport, use_cache=use_cache)

            arsenal_matches = []
            for match in all_matches:
                home = match.get("home_team", "").lower()
                away = match.get("away_team", "").lower()

                if any(kw in home or kw in away for kw in arsenal_keywords):
                    arsenal_matches.append(match)
                    logger.debug(
                        f"Arsenal match found: {match['home_team']} vs {match['away_team']}"
                    )

            logger.info(f"Found {len(arsenal_matches)} Arsenal matches")
            return arsenal_matches

        except Exception as e:
            logger.error(f"Error fetching Arsenal matches: {e}")
            raise

    def convert_to_odds_data(self, match: Dict[str, Any]) -> OddsData:
        """
        Convert API response match to OddsData format.

        Args:
            match: Match dictionary from The Odds API

        Returns:
            OddsData instance ready for storage

        Example:
            >>> match = fetcher.fetch_match_odds("abc123")
            >>> odds_data = fetcher.convert_to_odds_data(match)
            >>> odds_data.to_file()  # Save to data/odds/
        """
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        commence_time = match.get("commence_time", datetime.utcnow().isoformat())

        # Generate match ID in our format
        match_id = OddsData.generate_match_id(
            match_date=commence_time,
            home_team=home_team,
            away_team=away_team
        )

        # Extract bookmaker odds
        bookmaker_odds = []
        for bookmaker in match.get("bookmakers", []):
            bm_name = bookmaker.get("key", bookmaker.get("title", "unknown"))
            bm_odds = {"bookmaker": bm_name}

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h":
                    # 1X2 market
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)

                        if name.lower() == "draw":
                            bm_odds["draw"] = price
                        elif name == home_team:
                            bm_odds["home_win"] = price
                        else:
                            bm_odds["away_win"] = price

                elif market_key == "totals":
                    # Over/under market
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        price = outcome.get("price", 0)

                        # Only track 2.5 goals line
                        if point == 2.5:
                            if name.lower() == "over":
                                bm_odds["over_2_5"] = price
                            elif name.lower() == "under":
                                bm_odds["under_2_5"] = price

            # Only add if we have the core h2h odds
            if all(k in bm_odds for k in ["home_win", "draw", "away_win"]):
                bookmaker_odds.append(bm_odds)

        # Calculate best value
        best_value = OddsData.calculate_best_value(bookmaker_odds)

        return OddsData(
            match_id=match_id,
            timestamp=commence_time,
            bookmaker_odds=bookmaker_odds,
            best_value=best_value,
            metadata={
                "source": "the-odds-api",
                "fetch_timestamp": datetime.utcnow().isoformat() + "Z",
                "api_match_id": match.get("id", "")
            }
        )

    def fetch_and_store_arsenal_odds(self) -> List[OddsData]:
        """
        Fetch Arsenal match odds and store them to disk.

        This is a convenience method that:
        1. Fetches all upcoming Arsenal matches
        2. Converts each to OddsData format
        3. Saves to data/odds/ directory

        Returns:
            List of OddsData instances that were saved

        Example:
            >>> fetcher = OddsFetcher()
            >>> saved = fetcher.fetch_and_store_arsenal_odds()
            >>> print(f"Saved odds for {len(saved)} matches")
        """
        saved_odds = []

        try:
            arsenal_matches = self.fetch_arsenal_matches()

            for match in arsenal_matches:
                try:
                    odds_data = self.convert_to_odds_data(match)
                    if odds_data.to_file():
                        saved_odds.append(odds_data)
                        logger.info(f"Saved odds for {odds_data.match_id}")
                except Exception as e:
                    logger.error(f"Error processing match: {e}")
                    continue

            logger.info(f"Successfully saved odds for {len(saved_odds)} Arsenal matches")
            return saved_odds

        except Exception as e:
            logger.error(f"Error in fetch_and_store_arsenal_odds: {e}")
            raise

    def get_remaining_requests(self) -> Dict[str, int]:
        """
        Get information about remaining API requests.

        Returns:
            Dictionary with request quota information:
            - daily_used: Requests made today
            - daily_remaining: Requests remaining today
            - daily_limit: Maximum daily requests

        Example:
            >>> quota = fetcher.get_remaining_requests()
            >>> print(f"Used {quota['daily_used']} of {quota['daily_limit']} daily requests")
        """
        return {
            "daily_used": self.daily_request_count,
            "daily_remaining": max(0, MAX_DAILY_REQUESTS - self.daily_request_count),
            "daily_limit": MAX_DAILY_REQUESTS,
            "monthly_limit": MAX_MONTHLY_REQUESTS
        }

    def clear_cache(self) -> int:
        """
        Clear all cached API responses.

        Returns:
            Number of cache files deleted
        """
        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name != "request_count.json":
                try:
                    cache_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted} cached responses")
        return deleted


def main():
    """Example usage of the OddsFetcher class."""
    print("=" * 60)
    print("OddsFetcher - The Odds API Client")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("\nNote: THE_ODDS_API_KEY environment variable not set.")
        print("To use this client, set your API key:")
        print("  export THE_ODDS_API_KEY='your_api_key_here'")
        print("\nDemonstrating with mock data instead...\n")

        # Demo with mock data
        mock_match = {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-01-20T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Aston Villa",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.65},
                                {"name": "Draw", "price": 3.80},
                                {"name": "Aston Villa", "price": 5.50}
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
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 1.70},
                                {"name": "Draw", "price": 3.75},
                                {"name": "Aston Villa", "price": 5.25}
                            ]
                        }
                    ]
                }
            ]
        }

        print("Mock API Response:")
        print(f"  Match: {mock_match['home_team']} vs {mock_match['away_team']}")
        print(f"  Kickoff: {mock_match['commence_time']}")
        print(f"  Bookmakers: {len(mock_match['bookmakers'])}")

        # Create a temporary fetcher just for conversion (will fail on API calls)
        # We can still demonstrate the conversion logic
        print("\nConverting to OddsData format...")

        home_team = mock_match["home_team"]
        away_team = mock_match["away_team"]

        # Manual conversion for demo
        bookmaker_odds = []
        for bm in mock_match["bookmakers"]:
            bm_odds = {"bookmaker": bm["key"]}
            for market in bm["markets"]:
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Draw":
                            bm_odds["draw"] = outcome["price"]
                        elif outcome["name"] == home_team:
                            bm_odds["home_win"] = outcome["price"]
                        else:
                            bm_odds["away_win"] = outcome["price"]
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome.get("point") == 2.5:
                            if outcome["name"] == "Over":
                                bm_odds["over_2_5"] = outcome["price"]
                            else:
                                bm_odds["under_2_5"] = outcome["price"]
            bookmaker_odds.append(bm_odds)

        best_value = OddsData.calculate_best_value(bookmaker_odds)
        match_id = OddsData.generate_match_id(
            mock_match["commence_time"], home_team, away_team
        )

        print(f"\nGenerated Match ID: {match_id}")
        print("\nBookmaker Odds:")
        for bm in bookmaker_odds:
            print(f"  {bm['bookmaker']}:")
            print(f"    Home Win: {bm.get('home_win', 'N/A')}")
            print(f"    Draw: {bm.get('draw', 'N/A')}")
            print(f"    Away Win: {bm.get('away_win', 'N/A')}")
            if "over_2_5" in bm:
                print(f"    Over 2.5: {bm['over_2_5']}")
            if "under_2_5" in bm:
                print(f"    Under 2.5: {bm['under_2_5']}")

        print("\nBest Value:")
        for market, value in best_value.items():
            print(f"  {market}: {value['odds']} ({value['bookmaker']})")

        return

    # Real API usage
    try:
        fetcher = OddsFetcher()

        # Show quota
        quota = fetcher.get_remaining_requests()
        print(f"\nAPI Quota:")
        print(f"  Daily: {quota['daily_used']}/{quota['daily_limit']} used")
        print(f"  Monthly limit: {quota['monthly_limit']}")

        # Fetch Arsenal matches
        print("\nFetching Arsenal matches...")
        arsenal_matches = fetcher.fetch_arsenal_matches()

        if not arsenal_matches:
            print("No upcoming Arsenal matches found.")
        else:
            print(f"\nFound {len(arsenal_matches)} upcoming Arsenal match(es):")
            for match in arsenal_matches:
                print(f"\n  {match['home_team']} vs {match['away_team']}")
                print(f"  Kickoff: {match['commence_time']}")

                # Convert and display odds
                odds_data = fetcher.convert_to_odds_data(match)
                print(f"  Match ID: {odds_data.match_id}")
                print("  Best Odds:")
                for market, value in odds_data.best_value.items():
                    print(f"    {market}: {value['odds']} ({value['bookmaker']})")

        # Updated quota
        quota = fetcher.get_remaining_requests()
        print(f"\nRemaining daily requests: {quota['daily_remaining']}")

    except RateLimitExceeded as e:
        print(f"\nRate limit exceeded: {e}")
    except APIError as e:
        print(f"\nAPI error: {e}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
