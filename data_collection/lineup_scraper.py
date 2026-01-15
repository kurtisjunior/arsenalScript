#!/usr/bin/env python3
"""
Lineup Scraper - Social Media Lineup/Injury Intelligence Scraping

This module provides scraping capabilities for collecting lineup predictions,
injury updates, and team news from social media sources including Reddit and
Twitter/X journalists.

Features:
- Reddit API (praw) integration for r/Gunners Pre-Match Threads
- Nitter-based Twitter scraping for journalist accounts (avoiding expensive API)
- NLP-based injury keyword extraction
- Formation parsing (4-3-3, 4-2-3-1, etc.)
- Source reliability scoring system
- Schema-compliant output (data/schemas/lineups.json)

Sources:
- Reddit r/Gunners: Community discussions, pre-match threads
- Twitter/X: @FabrizioRomano, @David_Ornstein, @charles_watts

Usage:
    from lineup_scraper import LineupScraper

    scraper = LineupScraper()
    lineup_data = await scraper.scrape_lineup_intelligence(
        match_id="20260118_ARS_CHE",
        opponent="Chelsea"
    )

Environment Variables:
    REDDIT_CLIENT_ID: Reddit API client ID
    REDDIT_CLIENT_SECRET: Reddit API client secret
    REDDIT_USER_AGENT: Reddit API user agent string
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, quote as url_quote
from enum import Enum

# Third-party imports
try:
    import aiohttp
    from bs4 import BeautifulSoup
except ImportError as e:
    raise ImportError(
        f"Required dependency not installed: {e}. "
        "Install with: pip install aiohttp beautifulsoup4"
    )

# Optional praw import for Reddit
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache" / "lineups"
LINEUPS_DIR = DATA_DIR / "lineups"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 5
REQUEST_INTERVAL_SECONDS = 60 / MAX_REQUESTS_PER_MINUTE
CACHE_DURATION_HOURS = 0.5  # 30 minutes for lineup data (more time-sensitive)

# Nitter instances (fallback list if one is down)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.cz",
]

# Target journalists for Twitter scraping
TARGET_JOURNALISTS = {
    "FabrizioRomano": {
        "name": "Fabrizio Romano",
        "type": "journalist",
        "reliability_score": 0.88,
        "tier": 1,
    },
    "David_Ornstein": {
        "name": "David Ornstein",
        "type": "journalist",
        "reliability_score": 0.95,
        "tier": 1,
    },
    "charles_watts": {
        "name": "Charles Watts",
        "type": "journalist",
        "reliability_score": 0.85,
        "tier": 2,
    },
}

# User agent for scraping
USER_AGENT = (
    "Mozilla/5.0 (compatible; ArsenalScriptBot/1.0; "
    "+https://github.com/arsenalscript) AppleWebKit/537.36"
)


class InjuryStatus(Enum):
    """Injury status enum matching schema."""
    OUT = "out"
    DOUBTFUL = "doubtful"
    QUESTIONABLE = "questionable"
    PROBABLE = "probable"


class SourceType(Enum):
    """Source type enum matching schema."""
    OFFICIAL = "official"
    JOURNALIST = "journalist"
    AGGREGATOR = "aggregator"
    FAN_ACCOUNT = "fan_account"
    UNKNOWN = "unknown"


# Arsenal player roster for entity recognition
ARSENAL_ROSTER = {
    # Goalkeepers
    "David Raya": {"position": "GK", "aliases": ["Raya"]},
    "Aaron Ramsdale": {"position": "GK", "aliases": ["Ramsdale"]},
    "Karl Hein": {"position": "GK", "aliases": ["Hein"]},

    # Defenders
    "Ben White": {"position": "RB", "aliases": ["White", "B. White"]},
    "William Saliba": {"position": "CB", "aliases": ["Saliba"]},
    "Gabriel Magalhaes": {"position": "CB", "aliases": ["Gabriel", "Gabriel M", "Magalhaes"]},
    "Jurrien Timber": {"position": "RB", "aliases": ["Timber"]},
    "Oleksandr Zinchenko": {"position": "LB", "aliases": ["Zinchenko", "Zinny"]},
    "Takehiro Tomiyasu": {"position": "RB", "aliases": ["Tomiyasu", "Tomi"]},
    "Jakub Kiwior": {"position": "CB", "aliases": ["Kiwior"]},
    "Kieran Tierney": {"position": "LB", "aliases": ["Tierney", "KT"]},
    "Riccardo Calafiori": {"position": "LB", "aliases": ["Calafiori"]},

    # Midfielders
    "Martin Odegaard": {"position": "CAM", "aliases": ["Odegaard", "Ode", "MO8"]},
    "Declan Rice": {"position": "CM", "aliases": ["Rice", "Dec"]},
    "Thomas Partey": {"position": "CM", "aliases": ["Partey", "TP"]},
    "Jorginho": {"position": "CM", "aliases": ["Jorgi"]},
    "Fabio Vieira": {"position": "CAM", "aliases": ["Vieira", "Fabio V"]},
    "Emile Smith Rowe": {"position": "CAM", "aliases": ["ESR", "Smith Rowe"]},
    "Ethan Nwaneri": {"position": "CAM", "aliases": ["Nwaneri"]},

    # Forwards
    "Bukayo Saka": {"position": "RW", "aliases": ["Saka", "Starboy"]},
    "Gabriel Martinelli": {"position": "LW", "aliases": ["Martinelli", "Gabi M"]},
    "Kai Havertz": {"position": "ST", "aliases": ["Havertz", "Kai"]},
    "Leandro Trossard": {"position": "LW", "aliases": ["Trossard"]},
    "Gabriel Jesus": {"position": "ST", "aliases": ["Jesus", "GJ", "G. Jesus"]},
    "Eddie Nketiah": {"position": "ST", "aliases": ["Nketiah", "Eddie"]},
    "Reiss Nelson": {"position": "RW", "aliases": ["Nelson", "Reiss"]},
    "Raheem Sterling": {"position": "LW", "aliases": ["Sterling"]},
}

# Build reverse lookup for aliases
PLAYER_ALIAS_MAP: Dict[str, str] = {}
for player_name, info in ARSENAL_ROSTER.items():
    PLAYER_ALIAS_MAP[player_name.lower()] = player_name
    for alias in info.get("aliases", []):
        PLAYER_ALIAS_MAP[alias.lower()] = player_name

# Common formations
FORMATIONS = [
    "4-3-3", "4-2-3-1", "4-4-2", "3-4-3", "3-5-2", "4-1-4-1",
    "4-4-1-1", "3-4-2-1", "5-3-2", "5-4-1", "4-3-2-1", "4-5-1",
]

# Formation regex pattern
FORMATION_PATTERN = re.compile(r'\b(\d[-–]\d[-–]\d(?:[-–]\d)?)\b')

# Injury keywords for NLP extraction
INJURY_KEYWORDS = {
    "out": ["out", "ruled out", "sidelined", "miss", "missing", "unavailable", "not available"],
    "doubt": ["doubt", "doubtful", "uncertain", "touch and go", "50/50", "concern"],
    "available": ["available", "fit", "ready", "cleared", "back in training", "in contention"],
    "injured": ["injured", "injury", "hamstring", "muscle", "knock", "strain", "sprain",
                "fracture", "broken", "torn", "ligament", "ACL", "MCL", "ankle", "knee",
                "thigh", "calf", "groin", "back", "shoulder"],
    "recovering": ["recovering", "recovery", "rehab", "rehabilitation", "return", "returning",
                   "comeback", "back soon", "weeks away", "months away"],
}


@dataclass
class InjuryInfo:
    """Represents injury information for a player."""
    player: str
    status: str  # "out", "doubtful", "questionable", "probable"
    return_date: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "player": self.player,
            "status": self.status,
            "return_date": self.return_date,
            "source": self.source,
        }


@dataclass
class PlayerPosition:
    """Represents a player's position in a lineup."""
    position: str
    name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "position": self.position,
            "name": self.name,
            "confidence": self.confidence,
        }


@dataclass
class LineupPrediction:
    """Represents a predicted or confirmed lineup."""
    formation: str
    players: List[PlayerPosition]
    source: str
    confidence: float
    is_confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "formation": self.formation,
            "players": [p.to_dict() for p in self.players],
        }


@dataclass
class SourceReliability:
    """Tracks source reliability metrics."""
    name: str
    type: str  # "official", "journalist", "aggregator", "fan_account", "unknown"
    reliability_score: float
    historical_accuracy: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        result = {
            "name": self.name,
            "type": self.type,
            "reliability_score": self.reliability_score,
        }
        if self.historical_accuracy:
            result["historical_accuracy"] = self.historical_accuracy
        return result


@dataclass
class LineupData:
    """
    Complete lineup data for a match, matching data/schemas/lineups.json schema.
    """
    match_id: str
    timestamp: str
    injuries: List[InjuryInfo] = field(default_factory=list)
    rumored_lineup: Optional[LineupPrediction] = None
    confirmed_lineup: Optional[LineupPrediction] = None
    source_reliability: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values."""
        if not self.source_reliability:
            self.source_reliability = {
                "sources": [],
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "match_id": self.match_id,
            "timestamp": self.timestamp,
            "injuries": [i.to_dict() for i in self.injuries],
            "rumored_lineup": self.rumored_lineup.to_dict() if self.rumored_lineup else None,
            "confirmed_lineup": self.confirmed_lineup.to_dict() if self.confirmed_lineup else None,
            "source_reliability": self.source_reliability,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, filepath: Optional[Path] = None) -> Path:
        """
        Save lineup data to JSON file.

        Args:
            filepath: Optional custom path. Defaults to data/lineups/{match_id}.json

        Returns:
            Path to saved file
        """
        if filepath is None:
            LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = LINEUPS_DIR / f"{self.match_id}_lineup.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        logger.info(f"Saved lineup data to {filepath}")
        return filepath


class RateLimiter:
    """
    Rate limiter for managing request frequency per domain.
    """

    def __init__(self, requests_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self._domain_timestamps: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc

    async def acquire(self, url: str) -> None:
        """Wait until a request to the given URL is allowed."""
        domain = self._get_domain(url)

        async with self._lock:
            now = time.time()

            if domain not in self._domain_timestamps:
                self._domain_timestamps[domain] = []

            # Remove timestamps older than 1 minute
            self._domain_timestamps[domain] = [
                ts for ts in self._domain_timestamps[domain]
                if now - ts < 60.0
            ]

            # Check if we need to wait
            if len(self._domain_timestamps[domain]) >= self.requests_per_minute:
                oldest = self._domain_timestamps[domain][0]
                wait_time = 60.0 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limiting {domain}: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    now = time.time()
                    self._domain_timestamps[domain] = [
                        ts for ts in self._domain_timestamps[domain]
                        if now - ts < 60.0
                    ]

            self._domain_timestamps[domain].append(time.time())


class CacheManager:
    """Manages caching of scraped content with configurable expiration."""

    def __init__(self, cache_dir: Path = CACHE_DIR, duration_hours: float = CACHE_DURATION_HOURS):
        self.cache_dir = cache_dir
        self.duration = timedelta(hours=duration_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{self._get_cache_key(key)}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached content if valid."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)

            cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            if datetime.utcnow() - cached_at > self.duration:
                logger.debug(f"Cache expired for {key}")
                return None

            logger.debug(f"Cache hit for {key}")
            return cached.get("data")

        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {e}")
            return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Cache content."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "cached_at": datetime.utcnow().isoformat(),
                    "key": key,
                    "data": data
                }, f, ensure_ascii=False)
            logger.debug(f"Cached content for {key}")
        except Exception as e:
            logger.warning(f"Error caching {key}: {e}")


class InjuryParser:
    """
    NLP-based parser for extracting injury information from text.

    Identifies player names and associated injury status using keyword matching
    and context analysis.
    """

    def __init__(self, roster: Dict[str, Dict] = None, alias_map: Dict[str, str] = None):
        """
        Initialize injury parser.

        Args:
            roster: Player roster dictionary
            alias_map: Mapping of aliases to canonical player names
        """
        self.roster = roster or ARSENAL_ROSTER
        self.alias_map = alias_map or PLAYER_ALIAS_MAP

        # Build regex patterns for player names
        all_names = list(self.alias_map.keys())
        # Sort by length (longest first) to match "Gabriel Magalhaes" before "Gabriel"
        all_names.sort(key=len, reverse=True)
        self.player_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(name) for name in all_names) + r')\b',
            re.IGNORECASE
        )

    def _get_canonical_name(self, name: str) -> Optional[str]:
        """Get canonical player name from alias."""
        return self.alias_map.get(name.lower())

    def _determine_status(self, context: str) -> Tuple[str, float]:
        """
        Determine injury status from context text.

        Args:
            context: Text surrounding the player mention

        Returns:
            Tuple of (status, confidence)
        """
        context_lower = context.lower()

        # Check for "out" keywords - most severe
        for keyword in INJURY_KEYWORDS["out"]:
            if keyword in context_lower:
                return "out", 0.85

        # Check for "doubt" keywords
        for keyword in INJURY_KEYWORDS["doubt"]:
            if keyword in context_lower:
                return "doubtful", 0.75

        # Check for injury keywords (indicates some issue)
        for keyword in INJURY_KEYWORDS["injured"]:
            if keyword in context_lower:
                # Check if also mentioned as recovering
                for recover_kw in INJURY_KEYWORDS["recovering"]:
                    if recover_kw in context_lower:
                        return "questionable", 0.65
                return "doubtful", 0.70

        # Check for recovering - close to return
        for keyword in INJURY_KEYWORDS["recovering"]:
            if keyword in context_lower:
                return "questionable", 0.60

        # Check for available/fit keywords - positive
        for keyword in INJURY_KEYWORDS["available"]:
            if keyword in context_lower:
                return "probable", 0.80

        # Default - not enough info
        return "questionable", 0.40

    def _extract_return_date(self, context: str) -> Optional[str]:
        """
        Extract expected return date from context.

        Args:
            context: Text surrounding the player mention

        Returns:
            ISO date string or None
        """
        # Patterns for date extraction
        date_patterns = [
            # "back by January 20" or "return January 20"
            r'(?:back|return|available|fit).*?(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))',
            # "out until January 20"
            r'(?:out|sidelined|miss).*?until\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))',
            # "2-3 weeks"
            r'(\d+[-–]\d+\s+weeks?)',
            # "a few weeks"
            r'((?:few|couple|several)\s+weeks?)',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                date_str = match.group(1)

                # Try to parse actual date
                try:
                    # Handle "20th January" style
                    clean_date = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', date_str)
                    current_year = datetime.now().year

                    for fmt in ['%d %B', '%B %d']:
                        try:
                            dt = datetime.strptime(clean_date, fmt)
                            dt = dt.replace(year=current_year)
                            # If date is in past, assume next year
                            if dt < datetime.now():
                                dt = dt.replace(year=current_year + 1)
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                except Exception:
                    pass

                # For relative dates like "2-3 weeks", calculate estimate
                weeks_match = re.search(r'(\d+)[-–](\d+)\s+weeks?', date_str)
                if weeks_match:
                    avg_weeks = (int(weeks_match.group(1)) + int(weeks_match.group(2))) / 2
                    return_date = datetime.now() + timedelta(weeks=avg_weeks)
                    return return_date.strftime('%Y-%m-%d')

        return None

    def parse(self, text: str, source: str = "unknown") -> List[InjuryInfo]:
        """
        Parse text for injury information.

        Args:
            text: Text to parse for injury information
            source: Source of the text

        Returns:
            List of InjuryInfo objects
        """
        injuries = []
        seen_players: Set[str] = set()

        # Find all player mentions
        for match in self.player_pattern.finditer(text):
            player_mention = match.group(1)
            canonical_name = self._get_canonical_name(player_mention)

            if not canonical_name or canonical_name in seen_players:
                continue

            seen_players.add(canonical_name)

            # Get context around the mention (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]

            # Determine status and confidence
            status, confidence = self._determine_status(context)

            # Extract return date if possible
            return_date = self._extract_return_date(context)

            injuries.append(InjuryInfo(
                player=canonical_name,
                status=status,
                return_date=return_date,
                source=source,
                confidence=confidence
            ))

        return injuries

    def parse_multiple(self, texts: List[Tuple[str, str]]) -> List[InjuryInfo]:
        """
        Parse multiple texts and merge injury information.

        Args:
            texts: List of (text, source) tuples

        Returns:
            Merged list of InjuryInfo objects
        """
        all_injuries: Dict[str, List[InjuryInfo]] = {}

        for text, source in texts:
            injuries = self.parse(text, source)
            for injury in injuries:
                if injury.player not in all_injuries:
                    all_injuries[injury.player] = []
                all_injuries[injury.player].append(injury)

        # Merge injuries for same player - prefer higher confidence
        merged = []
        for player, player_injuries in all_injuries.items():
            # Sort by confidence descending
            player_injuries.sort(key=lambda x: x.confidence, reverse=True)
            best = player_injuries[0]

            # If multiple high-confidence sources agree, boost confidence
            if len(player_injuries) > 1:
                agreeing = sum(1 for i in player_injuries if i.status == best.status)
                if agreeing > 1:
                    best.confidence = min(1.0, best.confidence + 0.1 * (agreeing - 1))

            merged.append(best)

        return merged


class LineupExtractor:
    """
    Extracts lineup predictions and formations from text.

    Parses formation patterns (4-3-3, 4-2-3-1) and player positions
    to construct predicted lineups.
    """

    # Position mappings for different formations
    FORMATION_POSITIONS = {
        "4-3-3": ["GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "RW", "ST", "LW"],
        "4-2-3-1": ["GK", "RB", "CB", "CB", "LB", "CDM", "CDM", "RAM", "CAM", "LAM", "ST"],
        "4-4-2": ["GK", "RB", "CB", "CB", "LB", "RM", "CM", "CM", "LM", "ST", "ST"],
        "3-4-3": ["GK", "CB", "CB", "CB", "RM", "CM", "CM", "LM", "RW", "ST", "LW"],
        "3-5-2": ["GK", "CB", "CB", "CB", "RWB", "CM", "CM", "CM", "LWB", "ST", "ST"],
        "4-1-4-1": ["GK", "RB", "CB", "CB", "LB", "CDM", "RM", "CM", "CM", "LM", "ST"],
    }

    def __init__(self, roster: Dict[str, Dict] = None, alias_map: Dict[str, str] = None):
        """
        Initialize lineup extractor.

        Args:
            roster: Player roster dictionary
            alias_map: Mapping of aliases to canonical player names
        """
        self.roster = roster or ARSENAL_ROSTER
        self.alias_map = alias_map or PLAYER_ALIAS_MAP

        # Build player pattern
        all_names = list(self.alias_map.keys())
        all_names.sort(key=len, reverse=True)
        self.player_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(name) for name in all_names) + r')\b',
            re.IGNORECASE
        )

    def _get_canonical_name(self, name: str) -> Optional[str]:
        """Get canonical player name from alias."""
        return self.alias_map.get(name.lower())

    def extract_formation(self, text: str) -> Optional[str]:
        """
        Extract formation from text.

        Args:
            text: Text to search for formation

        Returns:
            Formation string (e.g., "4-3-3") or None
        """
        # Normalize dashes
        normalized = text.replace('–', '-').replace('—', '-')

        match = FORMATION_PATTERN.search(normalized)
        if match:
            formation = match.group(1)
            # Validate it looks like a real formation
            parts = formation.split('-')
            if len(parts) >= 3:
                total = sum(int(p) for p in parts)
                if 9 <= total <= 11:  # Valid formation sums to 10 (outfield) or 11 (with GK)
                    return formation

        return None

    def extract_players(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract player names from text with confidence scores.

        Args:
            text: Text to search for players

        Returns:
            List of (player_name, confidence) tuples
        """
        players = []
        seen = set()

        for match in self.player_pattern.finditer(text):
            player_mention = match.group(1)
            canonical_name = self._get_canonical_name(player_mention)

            if canonical_name and canonical_name not in seen:
                seen.add(canonical_name)

                # Base confidence
                confidence = 0.7

                # Boost if full name used
                if player_mention.lower() == canonical_name.lower():
                    confidence += 0.1

                # Check context for confidence indicators
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].lower()

                if any(word in context for word in ["start", "starting", "lineup", "xi"]):
                    confidence += 0.1
                if any(word in context for word in ["likely", "expected", "predicted"]):
                    confidence += 0.05
                if any(word in context for word in ["could", "might", "possibly"]):
                    confidence -= 0.1
                if any(word in context for word in ["won't", "not", "doubt"]):
                    confidence -= 0.2

                confidence = max(0.1, min(1.0, confidence))
                players.append((canonical_name, confidence))

        return players

    def build_lineup(
        self,
        formation: str,
        players: List[Tuple[str, float]],
        source: str = "unknown"
    ) -> Optional[LineupPrediction]:
        """
        Build a lineup prediction from formation and player list.

        Args:
            formation: Formation string
            players: List of (player_name, confidence) tuples
            source: Source of the prediction

        Returns:
            LineupPrediction or None if insufficient data
        """
        if not formation or len(players) < 11:
            return None

        # Get position template for formation
        positions = self.FORMATION_POSITIONS.get(formation)
        if not positions:
            # Use 4-3-3 as default
            positions = self.FORMATION_POSITIONS["4-3-3"]

        # Sort players by their natural position
        def position_priority(player_tuple):
            player_name, _ = player_tuple
            player_info = self.roster.get(player_name, {})
            pos = player_info.get("position", "CM")
            # GK first, then defenders, then midfielders, then forwards
            order = {"GK": 0, "RB": 1, "CB": 2, "LB": 3,
                     "CDM": 4, "CM": 5, "CAM": 6,
                     "RW": 7, "LW": 8, "ST": 9, "RM": 5, "LM": 5}
            return order.get(pos, 5)

        players_sorted = sorted(players[:11], key=position_priority)

        # Build lineup
        lineup_players = []
        for i, (player_name, confidence) in enumerate(players_sorted):
            position = positions[i] if i < len(positions) else "SUB"
            lineup_players.append(PlayerPosition(
                position=position,
                name=player_name,
                confidence=round(confidence, 2)
            ))

        # Calculate overall confidence
        avg_confidence = sum(p.confidence for p in lineup_players) / len(lineup_players)

        return LineupPrediction(
            formation=formation,
            players=lineup_players,
            source=source,
            confidence=round(avg_confidence, 2)
        )

    def extract_from_text(self, text: str, source: str = "unknown") -> Optional[LineupPrediction]:
        """
        Extract a complete lineup prediction from text.

        Args:
            text: Text containing lineup information
            source: Source of the text

        Returns:
            LineupPrediction or None
        """
        formation = self.extract_formation(text)
        players = self.extract_players(text)

        if not formation:
            formation = "4-3-3"  # Default Arsenal formation

        return self.build_lineup(formation, players, source)


class RedditScraper:
    """
    Scrapes r/Gunners for Pre-Match Threads and injury discussions.

    Uses praw (Python Reddit API Wrapper) for authenticated access.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize Reddit scraper.

        Args:
            client_id: Reddit API client ID (or REDDIT_CLIENT_ID env var)
            client_secret: Reddit API client secret (or REDDIT_CLIENT_SECRET env var)
            user_agent: User agent string (or REDDIT_USER_AGENT env var)
        """
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", USER_AGENT)

        self._reddit: Optional[Any] = None
        self._initialized = False

    def _init_reddit(self) -> bool:
        """Initialize Reddit API connection."""
        if self._initialized:
            return self._reddit is not None

        self._initialized = True

        if not PRAW_AVAILABLE:
            logger.warning("praw not installed. Reddit scraping disabled.")
            return False

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit credentials not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
            )
            return False

        try:
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
            logger.info("Reddit API initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            return False

    def scrape_prematch_threads(
        self,
        opponent: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scrape Pre-Match Thread posts from r/Gunners.

        Args:
            opponent: Optional opponent name to filter threads
            limit: Maximum number of threads to fetch

        Returns:
            List of thread data dictionaries
        """
        if not self._init_reddit():
            return []

        threads = []

        try:
            subreddit = self._reddit.subreddit("Gunners")

            # Search for pre-match threads
            search_query = "Pre-Match Thread"
            if opponent:
                search_query += f" {opponent}"

            for submission in subreddit.search(search_query, sort="new", limit=limit):
                thread_data = {
                    "title": submission.title,
                    "url": f"https://reddit.com{submission.permalink}",
                    "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat() + "Z",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "comments": [],
                }

                # Get top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments[:20]:
                    if hasattr(comment, 'body'):
                        thread_data["comments"].append({
                            "body": comment.body,
                            "score": comment.score,
                        })

                threads.append(thread_data)
                logger.info(f"Scraped Pre-Match Thread: {submission.title}")

        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")

        return threads

    def scrape_injury_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape posts related to injuries from r/Gunners.

        Args:
            limit: Maximum number of posts to fetch

        Returns:
            List of post data dictionaries
        """
        if not self._init_reddit():
            return []

        posts = []

        try:
            subreddit = self._reddit.subreddit("Gunners")

            # Search for injury-related posts
            for query in ["injury", "injured", "fitness", "team news"]:
                for submission in subreddit.search(query, sort="new", limit=limit // 4):
                    post_data = {
                        "title": submission.title,
                        "url": f"https://reddit.com{submission.permalink}",
                        "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat() + "Z",
                        "selftext": submission.selftext,
                        "flair": submission.link_flair_text,
                    }

                    # Avoid duplicates
                    if post_data["url"] not in [p["url"] for p in posts]:
                        posts.append(post_data)

        except Exception as e:
            logger.error(f"Error scraping Reddit injury posts: {e}")

        return posts


class TwitterScraper:
    """
    Scrapes Twitter/X content via Nitter instances.

    Targets journalist accounts:
    - @FabrizioRomano
    - @David_Ornstein
    - @charles_watts

    Uses Nitter (open-source Twitter frontend) to avoid expensive Twitter API.
    """

    def __init__(
        self,
        nitter_instance: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[CacheManager] = None
    ):
        """
        Initialize Twitter scraper.

        Args:
            nitter_instance: Specific Nitter instance URL to use
            rate_limiter: Rate limiter instance
            cache: Cache manager instance
        """
        self.nitter_instances = NITTER_INSTANCES.copy()
        if nitter_instance:
            self.nitter_instances.insert(0, nitter_instance)

        self.current_instance: Optional[str] = None
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _find_working_instance(self) -> Optional[str]:
        """Find a working Nitter instance."""
        session = await self._get_session()

        for instance in self.nitter_instances:
            try:
                async with session.get(instance, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        self.current_instance = instance
                        logger.info(f"Using Nitter instance: {instance}")
                        return instance
            except Exception:
                continue

        logger.warning("No working Nitter instances found")
        return None

    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with rate limiting."""
        # Check cache
        if self.cache:
            cached = self.cache.get(url)
            if cached:
                return cached.get("html")

        await self.rate_limiter.acquire(url)

        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    if self.cache:
                        self.cache.set(url, {"html": html})
                    return html
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")

        return None

    def _parse_nitter_tweets(self, html: str, username: str) -> List[Dict[str, Any]]:
        """
        Parse tweets from Nitter HTML.

        Args:
            html: HTML content from Nitter
            username: Twitter username

        Returns:
            List of tweet dictionaries
        """
        soup = BeautifulSoup(html, 'html.parser')
        tweets = []

        # Nitter tweet structure
        tweet_elements = soup.select('.timeline-item, .tweet-body, .tweet')

        for elem in tweet_elements:
            try:
                # Get tweet text
                content_elem = elem.select_one('.tweet-content, .tweet-text')
                if not content_elem:
                    continue

                content = content_elem.get_text(strip=True)
                if not content:
                    continue

                # Get timestamp
                time_elem = elem.select_one('.tweet-date a, time')
                timestamp = None
                if time_elem:
                    timestamp = time_elem.get('title') or time_elem.get_text(strip=True)

                # Get link
                link_elem = elem.select_one('.tweet-link, a[href*="/status/"]')
                link = None
                if link_elem:
                    href = link_elem.get('href', '')
                    if self.current_instance and href.startswith('/'):
                        # Convert Nitter link to Twitter link
                        link = f"https://twitter.com{href}"
                    elif href:
                        link = href

                tweets.append({
                    "username": username,
                    "content": content,
                    "timestamp": timestamp,
                    "link": link,
                })

            except Exception as e:
                logger.debug(f"Error parsing tweet: {e}")
                continue

        return tweets

    async def scrape_journalist(
        self,
        username: str,
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape tweets from a journalist account.

        Args:
            username: Twitter username (without @)
            keywords: Optional keywords to filter tweets

        Returns:
            List of relevant tweet dictionaries
        """
        if not self.current_instance:
            await self._find_working_instance()

        if not self.current_instance:
            return []

        tweets = []

        # Build URL
        url = f"{self.current_instance}/{username}"

        html = await self._fetch_page(url)
        if not html:
            return []

        all_tweets = self._parse_nitter_tweets(html, username)

        # Filter by keywords if provided
        if keywords:
            keywords_lower = [kw.lower() for kw in keywords]
            for tweet in all_tweets:
                content_lower = tweet["content"].lower()
                if any(kw in content_lower for kw in keywords_lower):
                    tweets.append(tweet)
        else:
            tweets = all_tweets

        logger.info(f"Scraped {len(tweets)} tweets from @{username}")
        return tweets

    async def scrape_all_journalists(
        self,
        keywords: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape tweets from all target journalists.

        Args:
            keywords: Optional keywords to filter tweets

        Returns:
            Dictionary mapping username to list of tweets
        """
        default_keywords = ["Arsenal", "Gunners", "injury", "lineup", "team news"]
        keywords = keywords or default_keywords

        results = {}

        for username in TARGET_JOURNALISTS:
            tweets = await self.scrape_journalist(username, keywords)
            results[username] = tweets

        return results


class LineupScraper:
    """
    Main orchestrator class for lineup and injury intelligence scraping.

    Coordinates scraping from multiple sources:
    - Reddit r/Gunners
    - Twitter/X journalists via Nitter

    Provides unified interface for collecting lineup predictions,
    injury updates, and source reliability tracking.

    Usage:
        scraper = LineupScraper()

        lineup_data = await scraper.scrape_lineup_intelligence(
            match_id="20260118_ARS_CHE",
            opponent="Chelsea"
        )

        lineup_data.save_to_file()
    """

    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        nitter_instance: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize lineup scraper.

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            nitter_instance: Specific Nitter instance URL
            cache_enabled: Whether to enable caching
        """
        self.rate_limiter = RateLimiter()
        self.cache = CacheManager() if cache_enabled else None

        # Initialize sub-scrapers
        self.reddit_scraper = RedditScraper(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret
        )
        self.twitter_scraper = TwitterScraper(
            nitter_instance=nitter_instance,
            rate_limiter=self.rate_limiter,
            cache=self.cache
        )

        # Initialize parsers
        self.injury_parser = InjuryParser()
        self.lineup_extractor = LineupExtractor()

        # Source reliability tracking
        self.source_reliability: Dict[str, SourceReliability] = {}
        self._init_source_reliability()

    def _init_source_reliability(self):
        """Initialize source reliability scores."""
        # Official sources
        self.source_reliability["Arsenal Official"] = SourceReliability(
            name="Arsenal Official",
            type="official",
            reliability_score=1.0,
            historical_accuracy={"correct_predictions": 100, "total_predictions": 100}
        )

        # Add journalists
        for username, info in TARGET_JOURNALISTS.items():
            self.source_reliability[info["name"]] = SourceReliability(
                name=info["name"],
                type=info["type"],
                reliability_score=info["reliability_score"],
                historical_accuracy={"correct_predictions": 85, "total_predictions": 100}
            )

        # Reddit sources
        self.source_reliability["r/Gunners"] = SourceReliability(
            name="r/Gunners",
            type="fan_account",
            reliability_score=0.5,
        )

    async def close(self) -> None:
        """Close all scraper sessions."""
        await self.twitter_scraper.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _aggregate_injuries(
        self,
        reddit_injuries: List[InjuryInfo],
        twitter_injuries: List[InjuryInfo]
    ) -> List[InjuryInfo]:
        """
        Aggregate injuries from multiple sources, resolving conflicts.

        Args:
            reddit_injuries: Injuries from Reddit
            twitter_injuries: Injuries from Twitter

        Returns:
            Merged and deduplicated injury list
        """
        all_injuries: Dict[str, List[InjuryInfo]] = {}

        # Twitter journalists are more reliable, add them first
        for injury in twitter_injuries:
            if injury.player not in all_injuries:
                all_injuries[injury.player] = []
            # Boost confidence for journalist sources
            source_info = self.source_reliability.get(injury.source)
            if source_info:
                injury.confidence = min(1.0, injury.confidence * source_info.reliability_score * 1.2)
            all_injuries[injury.player].append(injury)

        # Add Reddit injuries
        for injury in reddit_injuries:
            if injury.player not in all_injuries:
                all_injuries[injury.player] = []
            # Lower weight for fan sources
            injury.confidence *= 0.7
            all_injuries[injury.player].append(injury)

        # Select best injury info for each player
        merged = []
        for player, injuries in all_injuries.items():
            # Sort by confidence
            injuries.sort(key=lambda x: x.confidence, reverse=True)
            best = injuries[0]
            merged.append(best)

        return merged

    def _select_best_lineup(
        self,
        lineups: List[LineupPrediction]
    ) -> Optional[LineupPrediction]:
        """
        Select the best lineup prediction from multiple sources.

        Args:
            lineups: List of lineup predictions

        Returns:
            Best lineup prediction or None
        """
        if not lineups:
            return None

        # Weight by source reliability
        for lineup in lineups:
            source_info = self.source_reliability.get(lineup.source)
            if source_info:
                lineup.confidence *= source_info.reliability_score

        # Sort by confidence
        lineups.sort(key=lambda x: x.confidence, reverse=True)

        return lineups[0]

    async def scrape_reddit(self, opponent: Optional[str] = None) -> Tuple[List[InjuryInfo], List[LineupPrediction]]:
        """
        Scrape Reddit for injury and lineup information.

        Args:
            opponent: Optional opponent name

        Returns:
            Tuple of (injuries, lineup_predictions)
        """
        injuries = []
        lineups = []

        # Scrape pre-match threads
        threads = self.reddit_scraper.scrape_prematch_threads(opponent=opponent, limit=3)

        for thread in threads:
            # Parse main post
            text = f"{thread['title']} {thread['selftext']}"
            thread_injuries = self.injury_parser.parse(text, "r/Gunners")
            injuries.extend(thread_injuries)

            # Parse comments
            for comment in thread.get("comments", []):
                if comment.get("score", 0) > 5:  # Only consider upvoted comments
                    comment_injuries = self.injury_parser.parse(comment["body"], "r/Gunners")
                    injuries.extend(comment_injuries)

                    # Try to extract lineup
                    lineup = self.lineup_extractor.extract_from_text(comment["body"], "r/Gunners")
                    if lineup and len(lineup.players) >= 11:
                        lineups.append(lineup)

        # Scrape injury posts
        injury_posts = self.reddit_scraper.scrape_injury_posts(limit=10)
        for post in injury_posts:
            text = f"{post['title']} {post['selftext']}"
            post_injuries = self.injury_parser.parse(text, "r/Gunners")
            injuries.extend(post_injuries)

        return injuries, lineups

    async def scrape_twitter(self, opponent: Optional[str] = None) -> Tuple[List[InjuryInfo], List[LineupPrediction]]:
        """
        Scrape Twitter for injury and lineup information.

        Args:
            opponent: Optional opponent name

        Returns:
            Tuple of (injuries, lineup_predictions)
        """
        injuries = []
        lineups = []

        # Build keywords
        keywords = ["Arsenal", "injury", "team news", "lineup"]
        if opponent:
            keywords.append(opponent)

        # Scrape all journalists
        all_tweets = await self.twitter_scraper.scrape_all_journalists(keywords)

        for username, tweets in all_tweets.items():
            journalist_info = TARGET_JOURNALISTS.get(username, {})
            source_name = journalist_info.get("name", username)

            for tweet in tweets:
                content = tweet.get("content", "")

                # Parse injuries
                tweet_injuries = self.injury_parser.parse(content, source_name)
                injuries.extend(tweet_injuries)

                # Try to extract lineup
                lineup = self.lineup_extractor.extract_from_text(content, source_name)
                if lineup and len(lineup.players) >= 7:  # Lower threshold for tweets
                    lineups.append(lineup)

        return injuries, lineups

    async def scrape_lineup_intelligence(
        self,
        match_id: str,
        opponent: str,
        include_reddit: bool = True,
        include_twitter: bool = True
    ) -> LineupData:
        """
        Scrape all sources for lineup and injury intelligence.

        This is the main entry point for lineup data collection.

        Args:
            match_id: Unique match identifier
            opponent: Opponent team name
            include_reddit: Whether to scrape Reddit
            include_twitter: Whether to scrape Twitter

        Returns:
            LineupData object with injuries, predictions, and sources
        """
        logger.info(f"Scraping lineup intelligence for {match_id} vs {opponent}")

        all_injuries = []
        all_lineups = []
        sources_used = []

        # Scrape sources concurrently
        tasks = []

        if include_reddit:
            tasks.append(("reddit", self.scrape_reddit(opponent)))

        if include_twitter:
            tasks.append(("twitter", self.scrape_twitter(opponent)))

        # Run tasks
        if tasks:
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for (source_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Error scraping {source_name}: {result}")
                    continue

                injuries, lineups = result

                if source_name == "reddit" and (injuries or lineups):
                    sources_used.append("r/Gunners")
                    all_injuries.extend(injuries)
                    all_lineups.extend(lineups)

                elif source_name == "twitter" and (injuries or lineups):
                    for username, info in TARGET_JOURNALISTS.items():
                        sources_used.append(info["name"])
                    all_injuries.extend(injuries)
                    all_lineups.extend(lineups)

        # Aggregate and deduplicate
        merged_injuries = self._aggregate_injuries(
            [i for i in all_injuries if i.source == "r/Gunners"],
            [i for i in all_injuries if i.source != "r/Gunners"]
        )

        # Select best lineup
        best_lineup = self._select_best_lineup(all_lineups)

        # Build source reliability list
        source_reliability_list = []
        for source_name in set(sources_used):
            if source_name in self.source_reliability:
                source_reliability_list.append(
                    self.source_reliability[source_name].to_dict()
                )

        # Create LineupData
        lineup_data = LineupData(
            match_id=match_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            injuries=merged_injuries,
            rumored_lineup=best_lineup,
            confirmed_lineup=None,  # Would need official source
            source_reliability={
                "sources": source_reliability_list,
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
        )

        logger.info(
            f"Completed scraping for {match_id}: "
            f"{len(merged_injuries)} injuries, "
            f"{'lineup predicted' if best_lineup else 'no lineup'}"
        )

        return lineup_data


# ============================================================================
# Example Usage and Mock Data Demonstration
# ============================================================================

def create_mock_lineup_data() -> LineupData:
    """Create mock lineup data for demonstration."""

    # Mock injuries
    injuries = [
        InjuryInfo(
            player="Bukayo Saka",
            status="doubtful",
            return_date="2026-01-20",
            source="David Ornstein",
            confidence=0.85
        ),
        InjuryInfo(
            player="Takehiro Tomiyasu",
            status="out",
            return_date=None,
            source="Arsenal Official",
            confidence=1.0
        ),
        InjuryInfo(
            player="Gabriel Jesus",
            status="out",
            return_date="2026-03-15",
            source="Charles Watts",
            confidence=0.80
        ),
        InjuryInfo(
            player="Jurrien Timber",
            status="probable",
            return_date=None,
            source="r/Gunners",
            confidence=0.60
        ),
    ]

    # Mock rumored lineup
    rumored_players = [
        PlayerPosition("GK", "David Raya", 0.95),
        PlayerPosition("RB", "Ben White", 0.90),
        PlayerPosition("CB", "William Saliba", 0.98),
        PlayerPosition("CB", "Gabriel Magalhaes", 0.98),
        PlayerPosition("LB", "Oleksandr Zinchenko", 0.75),
        PlayerPosition("CM", "Declan Rice", 0.95),
        PlayerPosition("CM", "Martin Odegaard", 0.92),
        PlayerPosition("CM", "Thomas Partey", 0.80),
        PlayerPosition("RW", "Gabriel Martinelli", 0.70),
        PlayerPosition("ST", "Kai Havertz", 0.88),
        PlayerPosition("LW", "Leandro Trossard", 0.65),
    ]

    rumored_lineup = LineupPrediction(
        formation="4-3-3",
        players=rumored_players,
        source="David Ornstein",
        confidence=0.85
    )

    # Source reliability
    source_reliability = {
        "sources": [
            {
                "name": "David Ornstein",
                "type": "journalist",
                "reliability_score": 0.95,
                "historical_accuracy": {
                    "correct_predictions": 47,
                    "total_predictions": 50
                }
            },
            {
                "name": "Charles Watts",
                "type": "journalist",
                "reliability_score": 0.85,
                "historical_accuracy": {
                    "correct_predictions": 42,
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
            },
            {
                "name": "r/Gunners",
                "type": "fan_account",
                "reliability_score": 0.5
            }
        ],
        "last_updated": datetime.utcnow().isoformat() + "Z"
    }

    return LineupData(
        match_id="20260118_ARS_CHE",
        timestamp=datetime.utcnow().isoformat() + "Z",
        injuries=injuries,
        rumored_lineup=rumored_lineup,
        confirmed_lineup=None,
        source_reliability=source_reliability
    )


async def example_usage():
    """Demonstrate usage of the LineupScraper."""
    print("=" * 70)
    print("Lineup Scraper - Social Media Intelligence Collection")
    print("=" * 70)

    # Create mock data for demonstration
    print("\n[Demo Mode] Creating mock lineup data...")
    mock_data = create_mock_lineup_data()

    print(f"\nMatch: {mock_data.match_id}")
    print(f"Timestamp: {mock_data.timestamp}")

    print("\n--- Injuries ---")
    for injury in mock_data.injuries:
        status_icon = {
            "out": "[X]",
            "doubtful": "[?]",
            "questionable": "[~]",
            "probable": "[+]"
        }.get(injury.status, "[?]")
        return_info = f" (return: {injury.return_date})" if injury.return_date else ""
        print(f"  {status_icon} {injury.player}: {injury.status}{return_info}")
        print(f"      Source: {injury.source} (confidence: {injury.confidence:.0%})")

    if mock_data.rumored_lineup:
        print(f"\n--- Rumored Lineup ({mock_data.rumored_lineup.formation}) ---")
        print(f"  Source: {mock_data.rumored_lineup.source}")
        print(f"  Confidence: {mock_data.rumored_lineup.confidence:.0%}")
        print("\n  Players:")
        for player in mock_data.rumored_lineup.players:
            print(f"    {player.position:5} - {player.name} ({player.confidence:.0%})")

    print("\n--- Source Reliability ---")
    for source in mock_data.source_reliability.get("sources", []):
        tier = "Official" if source["type"] == "official" else f"Tier {1 if source['reliability_score'] > 0.9 else 2}"
        print(f"  {source['name']} ({tier}): {source['reliability_score']:.0%}")

    # Save to file
    print("\n--- Output ---")
    output_path = mock_data.save_to_file()
    print(f"Saved to: {output_path}")

    # Show JSON preview
    print("\nJSON Preview:")
    print("-" * 40)
    json_data = json.loads(mock_data.to_json())
    # Truncate for display
    json_data["injuries"] = f"[{len(json_data['injuries'])} injuries]"
    if json_data.get("rumored_lineup"):
        json_data["rumored_lineup"]["players"] = f"[11 players]"
    print(json.dumps(json_data, indent=2))

    # Demonstrate parser capabilities
    print("\n" + "=" * 70)
    print("Parser Demonstrations")
    print("=" * 70)

    # Injury parser demo
    print("\n--- Injury Parser Demo ---")
    parser = InjuryParser()

    sample_texts = [
        "Breaking: Saka is ruled out of the Chelsea match with a hamstring injury. Expected return in 2-3 weeks.",
        "Odegaard back in training and available for selection. Rice also fit to start.",
        "Doubt over Martinelli's fitness - he picked up a knock in training yesterday.",
    ]

    for text in sample_texts:
        print(f"\nText: \"{text}\"")
        injuries = parser.parse(text, "test")
        for injury in injuries:
            print(f"  -> {injury.player}: {injury.status} (confidence: {injury.confidence:.0%})")

    # Formation extractor demo
    print("\n--- Formation Extractor Demo ---")
    extractor = LineupExtractor()

    formation_texts = [
        "Arsenal expected to line up in a 4-3-3 with Havertz leading the line",
        "Arteta could switch to a 4-2-3-1 formation against Chelsea",
        "The team will likely play 3-4-3 with White at right center-back",
    ]

    for text in formation_texts:
        print(f"\nText: \"{text}\"")
        formation = extractor.extract_formation(text)
        print(f"  -> Formation: {formation or 'Not detected'}")

    # Live scraping demo (if credentials available)
    print("\n" + "=" * 70)
    print("Live Scraping (requires credentials)")
    print("=" * 70)

    async with LineupScraper() as scraper:
        print("\nReddit API Status:", "Available" if PRAW_AVAILABLE else "Not installed (pip install praw)")
        print("Reddit Credentials:", "Configured" if os.environ.get("REDDIT_CLIENT_ID") else "Not configured")

        print("\nTo enable live scraping, set environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("  export REDDIT_USER_AGENT='your_user_agent'")

        # Test Twitter scraper (Nitter doesn't require auth)
        print("\nTesting Nitter availability...")
        instance = await scraper.twitter_scraper._find_working_instance()
        if instance:
            print(f"  Nitter instance available: {instance}")
        else:
            print("  No Nitter instances available")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        # Live mode - attempt real scraping
        async def run_live():
            match_id = sys.argv[2] if len(sys.argv) > 2 else f"{datetime.now().strftime('%Y%m%d')}_ARS_OPP"
            opponent = sys.argv[3] if len(sys.argv) > 3 else "Opponent"

            print(f"Running live scrape for {match_id} vs {opponent}")

            async with LineupScraper() as scraper:
                data = await scraper.scrape_lineup_intelligence(
                    match_id=match_id,
                    opponent=opponent
                )
                data.save_to_file()
                print(f"Scraped {len(data.injuries)} injuries")
                if data.rumored_lineup:
                    print(f"Lineup: {data.rumored_lineup.formation}")

        asyncio.run(run_live())
    else:
        # Demo mode
        asyncio.run(example_usage())


if __name__ == "__main__":
    main()
