#!/usr/bin/env python3
"""
News Scraper - Web Scraping for Arsenal Match News and Quotes

This module provides async web scraping capabilities for collecting news articles
and manager/player quotes from Arsenal.com and BBC Sport.

Features:
- Async scraping with aiohttp for performance
- Rate limiting (max 5 req/min per site)
- 1-hour caching to minimize requests
- Quote extraction from press conferences
- Full text extraction for NLP analysis
- Schema-compliant output (data/schemas/news.json)

Sources:
- Arsenal.com: Official team news, press conferences
- BBC Sport: Match previews, analysis

Usage:
    from news_scraper import NewsScraper

    scraper = NewsScraper()
    news_data = await scraper.scrape_match_news(
        match_id="20260118_ARS_CHE",
        opponent="Chelsea"
    )
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin, urlparse, quote

# Third-party imports
try:
    import aiohttp
    from bs4 import BeautifulSoup
except ImportError as e:
    raise ImportError(
        f"Required dependency not installed: {e}. "
        "Install with: pip install aiohttp beautifulsoup4"
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache" / "news"
NEWS_DIR = DATA_DIR / "news"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 5
REQUEST_INTERVAL_SECONDS = 60 / MAX_REQUESTS_PER_MINUTE  # 12 seconds between requests
CACHE_DURATION_HOURS = 1

# Source URLs
ARSENAL_BASE_URL = "https://www.arsenal.com"
ARSENAL_NEWS_URL = f"{ARSENAL_BASE_URL}/news"
BBC_SPORT_BASE_URL = "https://www.bbc.co.uk/sport/football"

# User agent to identify our scraper
USER_AGENT = (
    "Mozilla/5.0 (compatible; ArsenalScriptBot/1.0; "
    "+https://github.com/arsenalscript) AppleWebKit/537.36"
)

# Known managers and players for quote extraction
ARSENAL_MANAGERS = ["Mikel Arteta", "Arteta"]
ARSENAL_PLAYERS = [
    "Bukayo Saka", "Martin Odegaard", "Gabriel Jesus", "Declan Rice",
    "William Saliba", "Gabriel Magalhaes", "Aaron Ramsdale", "David Raya",
    "Kai Havertz", "Leandro Trossard", "Gabriel Martinelli", "Ben White",
    "Jurrien Timber", "Oleksandr Zinchenko", "Thomas Partey", "Jorginho",
    "Eddie Nketiah", "Reiss Nelson", "Fabio Vieira", "Takehiro Tomiyasu"
]


@dataclass
class Article:
    """Represents a news article."""
    title: str
    url: str
    source: str
    publish_date: str
    author: Optional[str] = None
    full_text: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "author": self.author,
            "publish_date": self.publish_date,
            "full_text": self.full_text,
            "summary": self.summary
        }


@dataclass
class Quote:
    """Represents an extracted quote."""
    speaker: str
    quote: str
    source_url: str
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "speaker": self.speaker,
            "quote": self.quote,
            "context": self.context,
            "source_url": self.source_url
        }


@dataclass
class NewsData:
    """
    Complete news data for a match, matching data/schemas/news.json schema.
    """
    match_id: str
    timestamp: str
    articles: List[Article] = field(default_factory=list)
    quotes: List[Quote] = field(default_factory=list)
    sentiment_scores: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values for sentiment and metadata."""
        if not self.sentiment_scores:
            self.sentiment_scores = {
                "overall": 0.0,
                "by_source": []
            }
        if not self.metadata:
            self.metadata = {
                "sources_scraped": [],
                "fetch_timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to schema-compliant dictionary."""
        return {
            "match_id": self.match_id,
            "timestamp": self.timestamp,
            "articles": [a.to_dict() for a in self.articles],
            "quotes": [q.to_dict() for q in self.quotes],
            "sentiment_scores": self.sentiment_scores,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, filepath: Optional[Path] = None) -> Path:
        """
        Save news data to JSON file.

        Args:
            filepath: Optional custom path. Defaults to data/news/{match_id}.json

        Returns:
            Path to saved file
        """
        if filepath is None:
            NEWS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = NEWS_DIR / f"{self.match_id}_news.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        logger.info(f"Saved news data to {filepath}")
        return filepath


class RateLimiter:
    """
    Rate limiter for managing request frequency per domain.

    Enforces max 5 requests per minute per domain to be respectful
    to source websites.
    """

    def __init__(self, requests_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per domain
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self._domain_timestamps: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc

    async def acquire(self, url: str) -> None:
        """
        Wait until a request to the given URL is allowed.

        Args:
            url: The URL to be requested
        """
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
                wait_time = 60.0 - (now - oldest) + 0.1  # Small buffer
                if wait_time > 0:
                    logger.debug(f"Rate limiting {domain}: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Refresh timestamp after waiting
                    now = time.time()
                    self._domain_timestamps[domain] = [
                        ts for ts in self._domain_timestamps[domain]
                        if now - ts < 60.0
                    ]

            # Record this request
            self._domain_timestamps[domain].append(time.time())


class CacheManager:
    """
    Manages caching of scraped content with 1-hour expiration.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, duration_hours: float = CACHE_DURATION_HOURS):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            duration_hours: How long cached items remain valid
        """
        self.cache_dir = cache_dir
        self.duration = timedelta(hours=duration_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL."""
        return self.cache_dir / f"{self._get_cache_key(url)}.json"

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached content for URL if valid.

        Args:
            url: The URL to look up

        Returns:
            Cached data if valid, None otherwise
        """
        cache_path = self._get_cache_path(url)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)

            cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            if datetime.utcnow() - cached_at > self.duration:
                logger.debug(f"Cache expired for {url}")
                return None

            logger.debug(f"Cache hit for {url}")
            return cached.get("data")

        except Exception as e:
            logger.warning(f"Error reading cache for {url}: {e}")
            return None

    def set(self, url: str, data: Dict[str, Any]) -> None:
        """
        Cache content for URL.

        Args:
            url: The URL to cache
            data: The data to cache
        """
        cache_path = self._get_cache_path(url)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "cached_at": datetime.utcnow().isoformat(),
                    "url": url,
                    "data": data
                }, f, ensure_ascii=False)
            logger.debug(f"Cached content for {url}")
        except Exception as e:
            logger.warning(f"Error caching {url}: {e}")

    def clear(self) -> int:
        """
        Clear all cached content.

        Returns:
            Number of cache files deleted
        """
        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted} cache files")
        return deleted


class QuoteExtractor:
    """
    Extracts quotes from article text, identifying speakers.
    """

    # Patterns for identifying quotes
    QUOTE_PATTERNS = [
        # "Quote text," said Speaker Name
        r'"([^"]+)"\s*,?\s*(?:said|says|explained|added|revealed|admitted|insisted|claimed|stated)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        # Speaker Name said: "Quote text"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|says|explained|added|revealed|admitted|insisted|claimed|stated)[:\s]+["\']([^"\']+)["\']',
        # "Quote text," according to Speaker Name
        r'"([^"]+)"\s*,?\s*according to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        # Direct attribution: Speaker: "Quote"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*["\']([^"\']+)["\']',
    ]

    def __init__(
        self,
        managers: List[str] = None,
        players: List[str] = None,
        opposing_managers: List[str] = None
    ):
        """
        Initialize quote extractor with known speakers.

        Args:
            managers: List of manager names to look for
            players: List of player names to look for
            opposing_managers: List of opposing manager names
        """
        self.managers = managers or ARSENAL_MANAGERS.copy()
        self.players = players or ARSENAL_PLAYERS.copy()
        self.opposing_managers = opposing_managers or []

        # Build lookup for speaker roles
        self._speaker_roles: Dict[str, str] = {}
        for name in self.managers:
            self._speaker_roles[name.lower()] = "Arsenal Manager"
        for name in self.players:
            self._speaker_roles[name.lower()] = "Arsenal Player"
        for name in self.opposing_managers:
            self._speaker_roles[name.lower()] = "Manager"

    def add_opposing_manager(self, name: str, team: str) -> None:
        """Add an opposing manager to track."""
        self.opposing_managers.append(name)
        self._speaker_roles[name.lower()] = f"{team} Manager"

    def _normalize_speaker(self, name: str) -> Tuple[str, str]:
        """
        Normalize speaker name and determine role.

        Args:
            name: Raw speaker name from text

        Returns:
            Tuple of (formatted_name, role_description)
        """
        name_clean = name.strip()
        name_lower = name_clean.lower()

        # Check exact matches first
        if name_lower in self._speaker_roles:
            role = self._speaker_roles[name_lower]
            return f"{name_clean} ({role})", role

        # Check partial matches (e.g., "Arteta" matching "Mikel Arteta")
        for known_name, role in self._speaker_roles.items():
            if name_lower in known_name or known_name in name_lower:
                return f"{name_clean} ({role})", role

        # Unknown speaker
        return name_clean, "Unknown"

    def extract_quotes(
        self,
        text: str,
        source_url: str,
        context: Optional[str] = None
    ) -> List[Quote]:
        """
        Extract quotes from article text.

        Args:
            text: The article text to analyze
            source_url: URL of the source article
            context: Optional context (e.g., "pre-match press conference")

        Returns:
            List of Quote objects found in the text
        """
        quotes = []
        seen_quotes = set()  # Avoid duplicates

        for pattern in self.QUOTE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                groups = match.groups()

                # Determine which group is quote vs speaker based on pattern
                if pattern.startswith('"') or pattern.startswith("'"):
                    quote_text, speaker_name = groups[0], groups[1]
                else:
                    speaker_name, quote_text = groups[0], groups[1]

                # Skip if we've seen this quote
                quote_key = quote_text.strip().lower()[:50]
                if quote_key in seen_quotes:
                    continue
                seen_quotes.add(quote_key)

                # Normalize speaker
                speaker_formatted, role = self._normalize_speaker(speaker_name)

                # Only include quotes from known/relevant people or if they mention Arsenal/football
                is_relevant = (
                    role != "Unknown" or
                    "arsenal" in quote_text.lower() or
                    any(name.lower() in quote_text.lower() for name in self.managers)
                )

                if is_relevant and len(quote_text) > 20:  # Skip very short quotes
                    quotes.append(Quote(
                        speaker=speaker_formatted,
                        quote=quote_text.strip(),
                        source_url=source_url,
                        context=context
                    ))

        return quotes


class NewsScraper:
    """
    Async web scraper for Arsenal news and quotes.

    Scrapes Arsenal.com and BBC Sport for:
    - Official team news and press conferences
    - Match previews and analysis
    - Manager and player quotes

    Features:
    - Async operation with aiohttp for performance
    - Rate limiting (5 req/min per site)
    - 1-hour result caching
    - Quote extraction with speaker identification

    Usage:
        scraper = NewsScraper()

        # Scrape news for a specific match
        news = await scraper.scrape_match_news(
            match_id="20260118_ARS_CHE",
            opponent="Chelsea"
        )

        # Save to file
        news.save_to_file()
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        rate_limit_rpm: int = MAX_REQUESTS_PER_MINUTE
    ):
        """
        Initialize the news scraper.

        Args:
            cache_enabled: Whether to use caching (default: True)
            rate_limit_rpm: Max requests per minute per domain (default: 5)
        """
        self.rate_limiter = RateLimiter(rate_limit_rpm)
        self.cache = CacheManager() if cache_enabled else None
        self.quote_extractor = QuoteExtractor()

        # Session will be created in async context
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page with rate limiting and caching.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if fetch failed
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(url)
            if cached:
                return cached.get("html")

        # Wait for rate limit
        await self.rate_limiter.acquire(url)

        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()

                    # Cache the result
                    if self.cache:
                        self.cache.set(url, {"html": html})

                    return html
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def _parse_arsenal_article_list(self, html: str) -> List[Dict[str, str]]:
        """
        Parse Arsenal.com news listing page for article links.

        Args:
            html: HTML content of news listing page

        Returns:
            List of dicts with 'title' and 'url' keys
        """
        soup = BeautifulSoup(html, 'html.parser')
        articles = []

        # Arsenal.com article card structure
        # Look for article cards/links
        selectors = [
            'article a[href*="/news/"]',
            '.news-article a',
            'a.article-card',
            '.card a[href*="/news/"]',
            'a[href*="/news/"][class*="card"]',
            'a[href^="/news/"]',
        ]

        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href', '')
                if '/news/' in href:
                    # Get absolute URL
                    url = urljoin(ARSENAL_BASE_URL, href)

                    # Try to get title
                    title = None
                    title_elem = elem.find(['h2', 'h3', 'h4', '.title', '.headline'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    elif elem.get('title'):
                        title = elem.get('title')
                    elif elem.get_text(strip=True):
                        title = elem.get_text(strip=True)[:100]

                    if title and url and url not in [a['url'] for a in articles]:
                        articles.append({
                            'title': title,
                            'url': url
                        })

        return articles

    def _parse_arsenal_article(self, html: str, url: str) -> Optional[Article]:
        """
        Parse a single Arsenal.com article page.

        Args:
            html: HTML content of article page
            url: URL of the article

        Returns:
            Article object or None if parsing failed
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title = None
        title_selectors = ['h1', 'article h1', '.article-header h1', '[class*="title"] h1']
        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                break

        if not title:
            # Fallback to meta title
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                title = meta_title.get('content', '')

        if not title:
            logger.warning(f"Could not extract title from {url}")
            return None

        # Extract publish date
        publish_date = None
        date_selectors = [
            'time[datetime]',
            'meta[property="article:published_time"]',
            '.date',
            '.published',
            '[class*="date"]',
        ]

        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.name == 'meta':
                    publish_date = elem.get('content')
                elif elem.get('datetime'):
                    publish_date = elem.get('datetime')
                else:
                    # Try to parse text date
                    date_text = elem.get_text(strip=True)
                    try:
                        # Try common formats
                        for fmt in ['%d %B %Y', '%B %d, %Y', '%Y-%m-%d', '%d/%m/%Y']:
                            try:
                                dt = datetime.strptime(date_text, fmt)
                                publish_date = dt.isoformat() + "Z"
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                if publish_date:
                    break

        if not publish_date:
            publish_date = datetime.utcnow().isoformat() + "Z"

        # Extract author
        author = None
        author_selectors = [
            'meta[name="author"]',
            '.author',
            '[class*="author"]',
            '.byline',
        ]
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.name == 'meta':
                    author = elem.get('content')
                else:
                    author = elem.get_text(strip=True)
                if author:
                    break

        # Extract full text
        full_text = None
        content_selectors = [
            'article .content',
            '.article-body',
            '.article-content',
            'article [class*="body"]',
            '.story-body',
            'article p',
        ]

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get text from all matching elements
                paragraphs = []
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:  # Skip very short paragraphs
                        paragraphs.append(text)

                if paragraphs:
                    full_text = '\n\n'.join(paragraphs)
                    break

        # Generate summary (first 2-3 sentences)
        summary = None
        if full_text:
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            if sentences:
                summary = ' '.join(sentences[:3])
                if len(summary) > 500:
                    summary = summary[:497] + '...'

        return Article(
            title=title,
            url=url,
            source="Arsenal.com",
            publish_date=publish_date,
            author=author,
            full_text=full_text,
            summary=summary
        )

    def _parse_bbc_article_list(self, html: str, team: str = "arsenal") -> List[Dict[str, str]]:
        """
        Parse BBC Sport page for article links.

        Args:
            html: HTML content of BBC Sport page
            team: Team name to filter for

        Returns:
            List of dicts with 'title' and 'url' keys
        """
        soup = BeautifulSoup(html, 'html.parser')
        articles = []

        # BBC Sport article selectors
        selectors = [
            'a[href*="/sport/football/"]',
            '.gs-c-promo a',
            'article a',
            'a[data-bbc-title]',
        ]

        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href', '')
                if '/sport/football/' in href or '/sport/av/' in href:
                    # Get absolute URL
                    if href.startswith('/'):
                        url = f"https://www.bbc.co.uk{href}"
                    else:
                        url = href

                    # Get title
                    title = None
                    title_elem = elem.find(['h3', 'h2', 'span', 'p'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    elif elem.get('data-bbc-title'):
                        title = elem.get('data-bbc-title')
                    elif elem.get_text(strip=True):
                        title = elem.get_text(strip=True)[:100]

                    # Filter for relevance
                    is_relevant = (
                        title and
                        url and
                        url not in [a['url'] for a in articles] and
                        (team.lower() in title.lower() or team.lower() in url.lower())
                    )

                    if is_relevant:
                        articles.append({
                            'title': title,
                            'url': url
                        })

        return articles

    def _parse_bbc_article(self, html: str, url: str) -> Optional[Article]:
        """
        Parse a single BBC Sport article page.

        Args:
            html: HTML content of article page
            url: URL of the article

        Returns:
            Article object or None if parsing failed
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Extract title
        title = None
        title_selectors = ['h1', 'article h1', '#main-heading', '[class*="Headline"]']
        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                break

        if not title:
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                title = meta_title.get('content', '')

        if not title:
            logger.warning(f"Could not extract title from {url}")
            return None

        # Extract publish date
        publish_date = None
        date_selectors = [
            'time[datetime]',
            'meta[property="article:published_time"]',
            '[data-testid="timestamp"]',
        ]

        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.name == 'meta':
                    publish_date = elem.get('content')
                elif elem.get('datetime'):
                    publish_date = elem.get('datetime')
                if publish_date:
                    break

        if not publish_date:
            publish_date = datetime.utcnow().isoformat() + "Z"

        # Extract author
        author = None
        author_selectors = [
            'meta[name="author"]',
            '[class*="Contributor"]',
            '.ssrcss-68pt20-Text-TextContributorName',
            '.byline',
        ]
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.name == 'meta':
                    author = elem.get('content')
                else:
                    author = elem.get_text(strip=True)
                if author:
                    break

        # Extract full text
        full_text = None
        content_selectors = [
            'article [data-component="text-block"]',
            'article p',
            '.ssrcss-uf6wea-RichTextContainer',
            '#main-content p',
            '[class*="TextContainerBlock"] p',
        ]

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                paragraphs = []
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 20:
                        paragraphs.append(text)

                if paragraphs:
                    full_text = '\n\n'.join(paragraphs)
                    break

        # Generate summary
        summary = None
        if full_text:
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            if sentences:
                summary = ' '.join(sentences[:3])
                if len(summary) > 500:
                    summary = summary[:497] + '...'

        return Article(
            title=title,
            url=url,
            source="BBC Sport",
            publish_date=publish_date,
            author=author,
            full_text=full_text,
            summary=summary
        )

    async def scrape_arsenal_news(
        self,
        search_terms: List[str] = None,
        max_articles: int = 10
    ) -> Tuple[List[Article], List[Quote]]:
        """
        Scrape news from Arsenal.com.

        Args:
            search_terms: Optional terms to filter articles
            max_articles: Maximum number of articles to fetch

        Returns:
            Tuple of (articles, quotes)
        """
        articles = []
        quotes = []

        logger.info("Scraping Arsenal.com news...")

        # Fetch news listing page
        html = await self._fetch_page(ARSENAL_NEWS_URL)
        if not html:
            logger.warning("Failed to fetch Arsenal.com news listing")
            return articles, quotes

        # Parse article links
        article_links = self._parse_arsenal_article_list(html)
        logger.info(f"Found {len(article_links)} article links on Arsenal.com")

        # Filter by search terms if provided
        if search_terms:
            filtered = []
            for link in article_links:
                title_lower = link['title'].lower()
                if any(term.lower() in title_lower for term in search_terms):
                    filtered.append(link)
            article_links = filtered
            logger.info(f"Filtered to {len(article_links)} relevant articles")

        # Fetch individual articles
        for link in article_links[:max_articles]:
            html = await self._fetch_page(link['url'])
            if html:
                article = self._parse_arsenal_article(html, link['url'])
                if article:
                    articles.append(article)

                    # Extract quotes from full text
                    if article.full_text:
                        context = "press conference" if "press" in link['title'].lower() else None
                        extracted_quotes = self.quote_extractor.extract_quotes(
                            article.full_text,
                            article.url,
                            context=context
                        )
                        quotes.extend(extracted_quotes)

        logger.info(f"Scraped {len(articles)} articles and {len(quotes)} quotes from Arsenal.com")
        return articles, quotes

    async def scrape_bbc_sport(
        self,
        opponent: str = None,
        max_articles: int = 10
    ) -> Tuple[List[Article], List[Quote]]:
        """
        Scrape news from BBC Sport.

        Args:
            opponent: Opponent team name to search for
            max_articles: Maximum number of articles to fetch

        Returns:
            Tuple of (articles, quotes)
        """
        articles = []
        quotes = []

        logger.info("Scraping BBC Sport...")

        # Build search URL
        search_terms = ["Arsenal"]
        if opponent:
            search_terms.append(opponent)

        # Try team-specific page
        team_url = f"{BBC_SPORT_BASE_URL}/teams/arsenal"
        html = await self._fetch_page(team_url)

        if not html:
            # Fallback to search
            search_query = quote(" ".join(search_terms))
            search_url = f"https://www.bbc.co.uk/search?q={search_query}&filter=sport"
            html = await self._fetch_page(search_url)

        if not html:
            logger.warning("Failed to fetch BBC Sport content")
            return articles, quotes

        # Parse article links
        article_links = self._parse_bbc_article_list(html, team="arsenal")

        # Also search for opponent if provided
        if opponent:
            opponent_links = self._parse_bbc_article_list(html, team=opponent)
            # Merge without duplicates
            existing_urls = {link['url'] for link in article_links}
            for link in opponent_links:
                if link['url'] not in existing_urls:
                    article_links.append(link)

        logger.info(f"Found {len(article_links)} article links on BBC Sport")

        # Fetch individual articles
        for link in article_links[:max_articles]:
            html = await self._fetch_page(link['url'])
            if html:
                article = self._parse_bbc_article(html, link['url'])
                if article:
                    articles.append(article)

                    # Extract quotes
                    if article.full_text:
                        context = "match preview" if "preview" in link['title'].lower() else None
                        extracted_quotes = self.quote_extractor.extract_quotes(
                            article.full_text,
                            article.url,
                            context=context
                        )
                        quotes.extend(extracted_quotes)

        logger.info(f"Scraped {len(articles)} articles and {len(quotes)} quotes from BBC Sport")
        return articles, quotes

    async def scrape_match_news(
        self,
        match_id: str,
        opponent: str,
        opposing_manager: Optional[str] = None,
        max_articles_per_source: int = 5
    ) -> NewsData:
        """
        Scrape all news related to a specific match.

        This is the main entry point for match-specific news collection.
        Scrapes Arsenal.com and BBC Sport, extracts quotes, and returns
        data matching the news.json schema.

        Args:
            match_id: Unique match identifier (e.g., "20260118_ARS_CHE")
            opponent: Name of the opposing team
            opposing_manager: Name of the opposing manager (for quote extraction)
            max_articles_per_source: Max articles to fetch from each source

        Returns:
            NewsData object with articles, quotes, and metadata

        Example:
            async with NewsScraper() as scraper:
                news = await scraper.scrape_match_news(
                    match_id="20260118_ARS_CHE",
                    opponent="Chelsea",
                    opposing_manager="Enzo Maresca"
                )
                news.save_to_file()
        """
        logger.info(f"Scraping news for match {match_id} vs {opponent}")

        # Add opposing manager to quote extractor if provided
        if opposing_manager:
            self.quote_extractor.add_opposing_manager(opposing_manager, opponent)

        search_terms = [opponent, "preview", "press conference", "team news"]

        # Scrape both sources concurrently
        arsenal_task = self.scrape_arsenal_news(
            search_terms=search_terms,
            max_articles=max_articles_per_source
        )
        bbc_task = self.scrape_bbc_sport(
            opponent=opponent,
            max_articles=max_articles_per_source
        )

        (arsenal_articles, arsenal_quotes), (bbc_articles, bbc_quotes) = await asyncio.gather(
            arsenal_task, bbc_task
        )

        # Combine results
        all_articles = arsenal_articles + bbc_articles
        all_quotes = arsenal_quotes + bbc_quotes

        # De-duplicate quotes by content
        seen_quotes = set()
        unique_quotes = []
        for q in all_quotes:
            quote_key = q.quote.lower()[:50]
            if quote_key not in seen_quotes:
                seen_quotes.add(quote_key)
                unique_quotes.append(q)

        # Calculate basic sentiment scores (placeholder - can be enhanced with NLP)
        sentiment_by_source = []
        sources_scraped = []

        if arsenal_articles:
            sources_scraped.append("Arsenal.com")
            sentiment_by_source.append({
                "source": "Arsenal.com",
                "score": 0.5  # Official source tends to be neutral-positive
            })

        if bbc_articles:
            sources_scraped.append("BBC Sport")
            sentiment_by_source.append({
                "source": "BBC Sport",
                "score": 0.0  # BBC tends to be neutral
            })

        overall_sentiment = sum(s["score"] for s in sentiment_by_source) / len(sentiment_by_source) if sentiment_by_source else 0.0

        # Build NewsData object
        news_data = NewsData(
            match_id=match_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            articles=all_articles,
            quotes=unique_quotes,
            sentiment_scores={
                "overall": round(overall_sentiment, 2),
                "by_source": sentiment_by_source
            },
            metadata={
                "sources_scraped": sources_scraped,
                "fetch_timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

        logger.info(
            f"Completed scraping for {match_id}: "
            f"{len(all_articles)} articles, {len(unique_quotes)} quotes"
        )

        return news_data

    def clear_cache(self) -> int:
        """
        Clear the scraper cache.

        Returns:
            Number of cache entries cleared
        """
        if self.cache:
            return self.cache.clear()
        return 0


async def example_usage():
    """Demonstrate usage of the NewsScraper."""
    print("=" * 60)
    print("News Scraper - Arsenal Match News Collection")
    print("=" * 60)

    async with NewsScraper() as scraper:
        # Example: Scrape news for Arsenal vs Chelsea
        match_id = "20260118_ARS_CHE"
        opponent = "Chelsea"
        opposing_manager = "Enzo Maresca"

        print(f"\nScraping news for {match_id}...")
        print(f"Opponent: {opponent}")
        print(f"Opposing Manager: {opposing_manager}")
        print("-" * 40)

        try:
            news = await scraper.scrape_match_news(
                match_id=match_id,
                opponent=opponent,
                opposing_manager=opposing_manager,
                max_articles_per_source=3
            )

            print(f"\nResults:")
            print(f"  Articles: {len(news.articles)}")
            print(f"  Quotes: {len(news.quotes)}")
            print(f"  Sources: {news.metadata['sources_scraped']}")

            if news.articles:
                print("\nSample Articles:")
                for article in news.articles[:3]:
                    print(f"  - {article.title[:60]}...")
                    print(f"    Source: {article.source}")

            if news.quotes:
                print("\nSample Quotes:")
                for quote in news.quotes[:3]:
                    print(f"  - \"{quote.quote[:80]}...\"")
                    print(f"    - {quote.speaker}")

            # Save to file
            output_path = news.save_to_file()
            print(f"\nSaved to: {output_path}")

            # Show JSON structure
            print("\nJSON structure preview:")
            preview = json.loads(news.to_json())
            preview['articles'] = f"[{len(preview['articles'])} articles]"
            preview['quotes'] = f"[{len(preview['quotes'])} quotes]"
            print(json.dumps(preview, indent=2))

        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            print(f"\nError: {e}")
            print("Note: This scraper requires internet access to Arsenal.com and BBC Sport.")


def main():
    """Main entry point for command-line usage."""
    import sys

    if len(sys.argv) > 1:
        match_id = sys.argv[1]
        opponent = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
        opposing_manager = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        match_id = f"{datetime.now().strftime('%Y%m%d')}_ARS_OPP"
        opponent = "Opponent"
        opposing_manager = None

    print(f"Scraping news for match: {match_id}")
    print(f"Opponent: {opponent}")

    async def run():
        async with NewsScraper() as scraper:
            news = await scraper.scrape_match_news(
                match_id=match_id,
                opponent=opponent,
                opposing_manager=opposing_manager
            )
            news.save_to_file()
            print(f"Scraped {len(news.articles)} articles and {len(news.quotes)} quotes")
            return news

    asyncio.run(run())


if __name__ == "__main__":
    main()
