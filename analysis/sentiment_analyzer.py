#!/usr/bin/env python3
"""
Sentiment Analyzer Module for Arsenal Intelligence Brief.

This module provides functionality to:
- Analyze news article sentiment using HuggingFace transformers
- Analyze Reddit comment sentiment
- Extract key themes and keywords from text
- Generate sentiment summary reports for intelligence briefs

Task: arsenalScript-vqp.28-32 - Sentiment analysis for news and social media

Uses distilbert-base-uncased-finetuned-sst-2-english for sentiment classification.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ==========================================================================
# DATA CLASSES
# ==========================================================================

@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single text."""
    text: str
    label: str  # "positive", "negative", or "neutral"
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 to 1.0 (negative to positive scale)
    raw_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArticleSentiment:
    """Sentiment analysis result for a news article."""
    title: str
    url: str
    source: str
    publish_date: Optional[str]
    title_sentiment: SentimentResult
    content_sentiment: Optional[SentimentResult]
    combined_score: float  # Weighted combination of title and content


@dataclass
class CommentSentiment:
    """Sentiment analysis result for a Reddit comment."""
    comment_id: str
    author: str
    text: str
    subreddit: str
    sentiment: SentimentResult
    upvotes: int = 0
    created_utc: Optional[str] = None


@dataclass
class ThemeExtraction:
    """Extracted themes and keywords from text corpus."""
    keywords: List[Tuple[str, int]]  # (keyword, frequency)
    bigrams: List[Tuple[str, int]]  # (bigram phrase, frequency)
    entities: List[Tuple[str, str, int]]  # (entity, type, frequency)
    trending_concerns: List[str]
    trending_optimism: List[str]


@dataclass
class SentimentSummaryReport:
    """Complete sentiment summary report for intelligence brief."""
    match_id: str
    generated_at: str
    overall_sentiment: float  # -1.0 to 1.0
    overall_label: str  # "positive", "negative", "neutral"
    sentiment_distribution: Dict[str, int]  # count by label
    articles_analyzed: int
    comments_analyzed: int
    sentiment_by_source: Dict[str, float]
    themes: ThemeExtraction
    key_insights: List[str]
    articles: List[ArticleSentiment]
    comments: List[CommentSentiment]


# ==========================================================================
# SENTIMENT ANALYZER CLASS
# ==========================================================================

class SentimentAnalyzer:
    """
    Analyzer for sentiment analysis of news articles and social media.

    Uses HuggingFace transformers with distilbert-base-uncased-finetuned-sst-2-english
    for sentiment classification. Supports batch processing and handles long text
    through chunking/truncation.
    """

    # Model configuration
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_LENGTH = 512  # DistilBERT max token length
    CHUNK_OVERLAP = 50  # Token overlap for chunking long texts

    # Neutral threshold - scores between these are considered neutral
    NEUTRAL_THRESHOLD_LOW = 0.4
    NEUTRAL_THRESHOLD_HIGH = 0.6

    # Stopwords for keyword extraction (football-specific additions)
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'then', 'once', 'if', 'as', 'while', 'because', 'although', 'though',
        'until', 'unless', 'since', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'says', 'said', 'say',
        'get', 'got', 'gets', 'getting', 'going', 'go', 'goes', 'went',
        'come', 'came', 'comes', 'coming', 'make', 'made', 'makes', 'making',
        'take', 'took', 'takes', 'taking', 'know', 'knew', 'knows', 'knowing',
        'think', 'thought', 'thinks', 'thinking', 'see', 'saw', 'sees', 'seeing',
        'want', 'wanted', 'wants', 'wanting', 'look', 'looked', 'looks', 'looking',
        'use', 'used', 'uses', 'using', 'find', 'found', 'finds', 'finding',
        'give', 'gave', 'gives', 'giving', 'tell', 'told', 'tells', 'telling',
        'become', 'became', 'becomes', 'becoming', 'leave', 'left', 'leaves',
        'put', 'puts', 'putting', 'keep', 'kept', 'keeps', 'keeping',
        'let', 'lets', 'letting', 'begin', 'began', 'begins', 'beginning',
        'seem', 'seemed', 'seems', 'seeming', 'help', 'helped', 'helps',
        'show', 'showed', 'shows', 'showing', 'hear', 'heard', 'hears',
        'play', 'played', 'plays', 'playing', 'run', 'ran', 'runs', 'running',
        'move', 'moved', 'moves', 'moving', 'live', 'lived', 'lives', 'living',
        'believe', 'believed', 'believes', 'work', 'worked', 'works', 'working',
        'last', 'first', 'next', 'new', 'old', 'high', 'low', 'long', 'little',
        'big', 'great', 'good', 'bad', 'right', 'left', 'still', 'well',
        'back', 'even', 'much', 'any', 'his', 'her', 'its', 'their', 'our',
        'my', 'your', 'me', 'him', 'them', 'us', 'been', 'being', 'have',
        # Football common words (too generic for keyword extraction)
        'match', 'game', 'team', 'football', 'soccer', 'league', 'season',
        'club', 'player', 'players', 'manager', 'coach', 'vs', 'versus',
    }

    # Concern-related keywords for trending concerns detection
    CONCERN_KEYWORDS = {
        'injury', 'injured', 'doubt', 'doubtful', 'concern', 'worried', 'worry',
        'problem', 'issue', 'miss', 'missing', 'absent', 'unavailable', 'out',
        'struggle', 'struggling', 'poor', 'disappointing', 'disappoint', 'fail',
        'failing', 'failure', 'loss', 'lose', 'losing', 'defeat', 'defeated',
        'crisis', 'pressure', 'under fire', 'sack', 'sacked', 'dismissed',
        'suspension', 'suspended', 'ban', 'banned', 'red card', 'controversial',
        'controversy', 'uncertainty', 'uncertain', 'risk', 'risky', 'threat',
        'setback', 'blow', 'bad news', 'worse', 'worst', 'decline', 'slump',
        'form', 'fitness', 'fatigue', 'tired', 'exhausted', 'rotation',
    }

    # Optimism-related keywords
    OPTIMISM_KEYWORDS = {
        'win', 'winning', 'victory', 'victorious', 'triumph', 'success',
        'successful', 'confident', 'confidence', 'optimistic', 'optimism',
        'positive', 'good', 'great', 'excellent', 'outstanding', 'brilliant',
        'fantastic', 'amazing', 'incredible', 'impressive', 'strong', 'strength',
        'improve', 'improving', 'improvement', 'better', 'best', 'top', 'title',
        'champion', 'championship', 'hope', 'hopeful', 'promising', 'potential',
        'boost', 'return', 'returning', 'back', 'fit', 'fitness', 'ready',
        'form', 'in-form', 'momentum', 'unbeaten', 'streak', 'run', 'dominant',
        'dominate', 'dominating', 'control', 'clinical', 'sharp', 'scoring',
        'goals', 'clean sheet', 'solid', 'defensively', 'attacking',
    }

    def __init__(self, device: Optional[str] = None, batch_size: int = 8):
        """
        Initialize the SentimentAnalyzer.

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for processing multiple texts
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.batch_size = batch_size
        self._pipeline = None
        self._tokenizer = None
        self._device = device

    def _load_pipeline(self):
        """Lazy load the sentiment analysis pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer

                self.logger.info(f"Loading sentiment model: {self.MODEL_NAME}")

                # Load tokenizer for text chunking
                self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

                # Load sentiment pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.MODEL_NAME,
                    tokenizer=self._tokenizer,
                    device=self._device,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                )

                self.logger.info("Sentiment model loaded successfully")

            except ImportError as e:
                self.logger.error(
                    "transformers library not installed. "
                    "Install with: pip install transformers torch"
                )
                raise ImportError(
                    "transformers library required for sentiment analysis. "
                    "Install with: pip install transformers torch"
                ) from e
            except Exception as e:
                self.logger.error(f"Failed to load sentiment model: {e}")
                raise

        return self._pipeline

    # ==========================================================================
    # TEXT PREPROCESSING
    # ==========================================================================

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation important for sentiment
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)

        return text.strip()

    def _chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Split long text into chunks that fit within model's max length.

        Uses sentence-aware splitting to maintain context.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (default: MODEL max - overlap)

        Returns:
            List of text chunks
        """
        if max_tokens is None:
            max_tokens = self.MAX_LENGTH - self.CHUNK_OVERLAP

        # If tokenizer not loaded, use character-based chunking
        if self._tokenizer is None:
            # Approximate: ~4 characters per token
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return [text]

            # Split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) > max_chars and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence)

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks if chunks else [text]

        # Token-based chunking
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            return [text]

        # Split by sentences first for cleaner chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk_tokens = []
        current_chunk_text = []

        for sentence in sentences:
            sentence_tokens = self._tokenizer.encode(sentence, add_special_tokens=False)

            if len(current_chunk_tokens) + len(sentence_tokens) > max_tokens:
                if current_chunk_text:
                    chunks.append(' '.join(current_chunk_text))
                current_chunk_tokens = sentence_tokens
                current_chunk_text = [sentence]
            else:
                current_chunk_tokens.extend(sentence_tokens)
                current_chunk_text.append(sentence)

        if current_chunk_text:
            chunks.append(' '.join(current_chunk_text))

        return chunks if chunks else [text]

    # ==========================================================================
    # CORE SENTIMENT ANALYSIS
    # ==========================================================================

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Handles long text by chunking and averaging scores.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with label, confidence, and score
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text or "",
                label="neutral",
                confidence=0.0,
                score=0.0,
                raw_scores={"POSITIVE": 0.5, "NEGATIVE": 0.5}
            )

        pipeline = self._load_pipeline()
        cleaned_text = self._preprocess_text(text)

        # Chunk text if too long
        chunks = self._chunk_text(cleaned_text)

        # Analyze each chunk
        chunk_results = []
        for chunk in chunks:
            if chunk.strip():
                try:
                    result = pipeline(chunk)[0]
                    chunk_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze chunk: {e}")

        if not chunk_results:
            return SentimentResult(
                text=text,
                label="neutral",
                confidence=0.0,
                score=0.0,
                raw_scores={"POSITIVE": 0.5, "NEGATIVE": 0.5}
            )

        # Aggregate results from chunks
        pos_scores = []
        neg_scores = []

        for result in chunk_results:
            if result['label'] == 'POSITIVE':
                pos_scores.append(result['score'])
                neg_scores.append(1 - result['score'])
            else:
                neg_scores.append(result['score'])
                pos_scores.append(1 - result['score'])

        avg_pos = sum(pos_scores) / len(pos_scores)
        avg_neg = sum(neg_scores) / len(neg_scores)

        # Determine label and confidence
        raw_scores = {"POSITIVE": avg_pos, "NEGATIVE": avg_neg}

        # Convert to -1 to 1 scale (negative to positive)
        score = avg_pos - avg_neg

        # Determine label with neutral threshold
        if avg_pos > self.NEUTRAL_THRESHOLD_HIGH:
            label = "positive"
            confidence = avg_pos
        elif avg_neg > self.NEUTRAL_THRESHOLD_HIGH:
            label = "negative"
            confidence = avg_neg
        else:
            label = "neutral"
            confidence = 1 - abs(avg_pos - avg_neg)

        return SentimentResult(
            text=text[:500] + "..." if len(text) > 500 else text,
            label=label,
            confidence=round(confidence, 4),
            score=round(score, 4),
            raw_scores=raw_scores
        )

    def analyze_texts_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts in batch for efficiency.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                result = self.analyze_text(text)
                results.append(result)

        return results

    # ==========================================================================
    # NEWS ARTICLE ANALYSIS
    # ==========================================================================

    def analyze_article(
        self,
        title: str,
        url: str,
        source: str,
        publish_date: Optional[str] = None,
        full_text: Optional[str] = None,
        title_weight: float = 0.4
    ) -> ArticleSentiment:
        """
        Analyze sentiment of a news article.

        Combines title and content sentiment with configurable weighting.

        Args:
            title: Article headline
            url: Article URL
            source: Publication source name
            publish_date: ISO format publication date
            full_text: Full article content (optional)
            title_weight: Weight given to title sentiment (0-1)

        Returns:
            ArticleSentiment with detailed analysis
        """
        # Analyze title
        title_sentiment = self.analyze_text(title)

        # Analyze content if available
        content_sentiment = None
        if full_text and full_text.strip():
            content_sentiment = self.analyze_text(full_text)

        # Calculate combined score
        if content_sentiment:
            combined_score = (
                title_weight * title_sentiment.score +
                (1 - title_weight) * content_sentiment.score
            )
        else:
            combined_score = title_sentiment.score

        return ArticleSentiment(
            title=title,
            url=url,
            source=source,
            publish_date=publish_date,
            title_sentiment=title_sentiment,
            content_sentiment=content_sentiment,
            combined_score=round(combined_score, 4)
        )

    def analyze_articles_batch(
        self,
        articles: List[Dict[str, Any]],
        title_weight: float = 0.4
    ) -> List[ArticleSentiment]:
        """
        Analyze sentiment of multiple news articles in batch.

        Args:
            articles: List of article dicts with keys: title, url, source,
                     publish_date, full_text (optional)
            title_weight: Weight given to title sentiment

        Returns:
            List of ArticleSentiment objects
        """
        results = []

        for article in articles:
            result = self.analyze_article(
                title=article.get('title', ''),
                url=article.get('url', ''),
                source=article.get('source', ''),
                publish_date=article.get('publish_date'),
                full_text=article.get('full_text'),
                title_weight=title_weight
            )
            results.append(result)

        return results

    # ==========================================================================
    # REDDIT COMMENT ANALYSIS
    # ==========================================================================

    def analyze_comment(
        self,
        comment_id: str,
        author: str,
        text: str,
        subreddit: str,
        upvotes: int = 0,
        created_utc: Optional[str] = None
    ) -> CommentSentiment:
        """
        Analyze sentiment of a Reddit comment.

        Args:
            comment_id: Unique comment identifier
            author: Reddit username
            text: Comment text
            subreddit: Subreddit name (without r/)
            upvotes: Comment upvote count
            created_utc: UTC timestamp

        Returns:
            CommentSentiment with analysis
        """
        sentiment = self.analyze_text(text)

        return CommentSentiment(
            comment_id=comment_id,
            author=author,
            text=text[:500] + "..." if len(text) > 500 else text,
            subreddit=subreddit,
            sentiment=sentiment,
            upvotes=upvotes,
            created_utc=created_utc
        )

    def analyze_comments_batch(
        self,
        comments: List[Dict[str, Any]]
    ) -> List[CommentSentiment]:
        """
        Analyze sentiment of multiple Reddit comments in batch.

        Args:
            comments: List of comment dicts with keys: comment_id, author,
                     text, subreddit, upvotes (optional), created_utc (optional)

        Returns:
            List of CommentSentiment objects
        """
        results = []

        for comment in comments:
            result = self.analyze_comment(
                comment_id=comment.get('comment_id', comment.get('id', '')),
                author=comment.get('author', ''),
                text=comment.get('text', comment.get('body', '')),
                subreddit=comment.get('subreddit', ''),
                upvotes=comment.get('upvotes', comment.get('score', 0)),
                created_utc=comment.get('created_utc')
            )
            results.append(result)

        return results

    # ==========================================================================
    # KEYWORD AND THEME EXTRACTION
    # ==========================================================================

    def extract_keywords(
        self,
        texts: List[str],
        top_n: int = 20,
        min_word_length: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Extract most frequent keywords from a corpus of texts.

        Args:
            texts: List of texts to analyze
            top_n: Number of top keywords to return
            min_word_length: Minimum word length to consider

        Returns:
            List of (keyword, frequency) tuples sorted by frequency
        """
        word_counts = Counter()

        for text in texts:
            if not text:
                continue

            # Clean and tokenize
            cleaned = self._preprocess_text(text.lower())
            words = re.findall(r'\b[a-z]+\b', cleaned)

            # Filter stopwords and short words
            filtered_words = [
                w for w in words
                if w not in self.STOPWORDS and len(w) >= min_word_length
            ]

            word_counts.update(filtered_words)

        return word_counts.most_common(top_n)

    def extract_bigrams(
        self,
        texts: List[str],
        top_n: int = 15,
        min_word_length: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Extract most frequent bigrams (two-word phrases) from texts.

        Args:
            texts: List of texts to analyze
            top_n: Number of top bigrams to return
            min_word_length: Minimum word length for each word in bigram

        Returns:
            List of (bigram, frequency) tuples sorted by frequency
        """
        bigram_counts = Counter()

        for text in texts:
            if not text:
                continue

            # Clean and tokenize
            cleaned = self._preprocess_text(text.lower())
            words = re.findall(r'\b[a-z]+\b', cleaned)

            # Generate bigrams, filtering stopwords
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if (w1 not in self.STOPWORDS and w2 not in self.STOPWORDS and
                    len(w1) >= min_word_length and len(w2) >= min_word_length):
                    bigram_counts[f"{w1} {w2}"] += 1

        return bigram_counts.most_common(top_n)

    def extract_entities(
        self,
        texts: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, str, int]]:
        """
        Extract named entities (players, teams, etc.) from texts.

        Uses pattern matching for common football entity types.

        Args:
            texts: List of texts to analyze
            top_n: Number of top entities to return

        Returns:
            List of (entity, type, frequency) tuples
        """
        entity_counts = Counter()

        # Patterns for entity extraction
        patterns = {
            'PERSON': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # "Bukayo Saka"
            'TEAM': r'\b(Arsenal|Chelsea|Liverpool|Manchester (?:United|City)|Tottenham|Newcastle|Brighton|Aston Villa|West Ham|Crystal Palace|Fulham|Brentford|Nottingham Forest|Bournemouth|Wolves|Everton|Leicester|Southampton|Ipswich|Real Madrid|Barcelona|Bayern|PSG|Juventus|Inter|Milan)\b',
        }

        for text in texts:
            if not text:
                continue

            for entity_type, pattern in patterns.items():
                matches = re.findall(pattern, text)
                for match in matches:
                    entity_counts[(match, entity_type)] += 1

        # Sort by frequency and return top N
        sorted_entities = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return [(e[0], e[1], count) for (e, count) in sorted_entities]

    def identify_trending_concerns(
        self,
        texts: List[str],
        top_n: int = 5
    ) -> List[str]:
        """
        Identify trending concerns from text corpus.

        Looks for negative sentiment indicators and concern-related keywords.

        Args:
            texts: List of texts to analyze
            top_n: Number of top concerns to return

        Returns:
            List of concern phrases/topics
        """
        concern_phrases = Counter()

        for text in texts:
            if not text:
                continue

            text_lower = text.lower()

            # Check for concern keywords in context
            sentences = re.split(r'[.!?]', text_lower)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Count concern keywords found
                concerns_found = [
                    kw for kw in self.CONCERN_KEYWORDS
                    if kw in sentence
                ]

                if concerns_found:
                    # Extract the key phrase around the concern
                    for kw in concerns_found:
                        # Get surrounding context
                        idx = sentence.find(kw)
                        start = max(0, idx - 30)
                        end = min(len(sentence), idx + len(kw) + 30)
                        context = sentence[start:end].strip()

                        # Clean up and add
                        if len(context) > 10:
                            concern_phrases[context] += 1

        # Get top concerns and clean them up
        top_concerns = []
        for phrase, count in concern_phrases.most_common(top_n * 2):
            # Clean up the phrase
            cleaned = phrase.strip('.,!? ')
            if cleaned and cleaned not in top_concerns:
                top_concerns.append(cleaned)
            if len(top_concerns) >= top_n:
                break

        return top_concerns

    def identify_trending_optimism(
        self,
        texts: List[str],
        top_n: int = 5
    ) -> List[str]:
        """
        Identify trending positive/optimistic themes from text corpus.

        Args:
            texts: List of texts to analyze
            top_n: Number of top optimistic themes to return

        Returns:
            List of positive phrases/topics
        """
        optimism_phrases = Counter()

        for text in texts:
            if not text:
                continue

            text_lower = text.lower()
            sentences = re.split(r'[.!?]', text_lower)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                optimism_found = [
                    kw for kw in self.OPTIMISM_KEYWORDS
                    if kw in sentence
                ]

                if optimism_found:
                    for kw in optimism_found:
                        idx = sentence.find(kw)
                        start = max(0, idx - 30)
                        end = min(len(sentence), idx + len(kw) + 30)
                        context = sentence[start:end].strip()

                        if len(context) > 10:
                            optimism_phrases[context] += 1

        top_optimism = []
        for phrase, count in optimism_phrases.most_common(top_n * 2):
            cleaned = phrase.strip('.,!? ')
            if cleaned and cleaned not in top_optimism:
                top_optimism.append(cleaned)
            if len(top_optimism) >= top_n:
                break

        return top_optimism

    def extract_themes(
        self,
        texts: List[str],
        top_keywords: int = 20,
        top_bigrams: int = 15,
        top_entities: int = 20,
        top_concerns: int = 5,
        top_optimism: int = 5
    ) -> ThemeExtraction:
        """
        Extract all themes and keywords from a corpus of texts.

        Args:
            texts: List of texts to analyze
            top_keywords: Number of keywords to extract
            top_bigrams: Number of bigrams to extract
            top_entities: Number of entities to extract
            top_concerns: Number of concerns to identify
            top_optimism: Number of optimistic themes to identify

        Returns:
            ThemeExtraction with all extracted themes
        """
        return ThemeExtraction(
            keywords=self.extract_keywords(texts, top_keywords),
            bigrams=self.extract_bigrams(texts, top_bigrams),
            entities=self.extract_entities(texts, top_entities),
            trending_concerns=self.identify_trending_concerns(texts, top_concerns),
            trending_optimism=self.identify_trending_optimism(texts, top_optimism)
        )

    # ==========================================================================
    # REPORT GENERATION
    # ==========================================================================

    def generate_sentiment_report(
        self,
        match_id: str,
        articles: List[Dict[str, Any]],
        comments: Optional[List[Dict[str, Any]]] = None,
        title_weight: float = 0.4
    ) -> SentimentSummaryReport:
        """
        Generate a comprehensive sentiment summary report for intelligence brief.

        Args:
            match_id: Match identifier (e.g., "20260118_ARS_CHE")
            articles: List of article dicts from news data
            comments: Optional list of Reddit comment dicts
            title_weight: Weight for title vs content sentiment

        Returns:
            SentimentSummaryReport with complete analysis
        """
        # Analyze articles
        article_results = self.analyze_articles_batch(articles, title_weight)

        # Analyze comments if provided
        comment_results = []
        if comments:
            comment_results = self.analyze_comments_batch(comments)

        # Collect all scores and texts for aggregation
        all_scores = []
        all_texts = []
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        sentiment_by_source = {}
        source_counts = {}

        # Process article results
        for article in article_results:
            all_scores.append(article.combined_score)
            all_texts.append(article.title)
            if article.content_sentiment and article.content_sentiment.text:
                all_texts.append(article.content_sentiment.text)

            # Track sentiment distribution
            sentiment_distribution[article.title_sentiment.label] += 1

            # Track by source
            source = article.source
            if source not in sentiment_by_source:
                sentiment_by_source[source] = 0.0
                source_counts[source] = 0
            sentiment_by_source[source] += article.combined_score
            source_counts[source] += 1

        # Process comment results
        for comment in comment_results:
            all_scores.append(comment.sentiment.score)
            all_texts.append(comment.text)
            sentiment_distribution[comment.sentiment.label] += 1

        # Calculate overall sentiment
        if all_scores:
            overall_sentiment = sum(all_scores) / len(all_scores)
        else:
            overall_sentiment = 0.0

        # Determine overall label
        if overall_sentiment > 0.2:
            overall_label = "positive"
        elif overall_sentiment < -0.2:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        # Average sentiment by source
        for source in sentiment_by_source:
            if source_counts[source] > 0:
                sentiment_by_source[source] = round(
                    sentiment_by_source[source] / source_counts[source],
                    4
                )

        # Extract themes from all text
        themes = self.extract_themes(all_texts)

        # Generate key insights
        key_insights = self._generate_insights(
            overall_sentiment,
            overall_label,
            sentiment_distribution,
            sentiment_by_source,
            themes,
            len(article_results),
            len(comment_results)
        )

        return SentimentSummaryReport(
            match_id=match_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            overall_sentiment=round(overall_sentiment, 4),
            overall_label=overall_label,
            sentiment_distribution=sentiment_distribution,
            articles_analyzed=len(article_results),
            comments_analyzed=len(comment_results),
            sentiment_by_source=sentiment_by_source,
            themes=themes,
            key_insights=key_insights,
            articles=article_results,
            comments=comment_results
        )

    def _generate_insights(
        self,
        overall_sentiment: float,
        overall_label: str,
        distribution: Dict[str, int],
        by_source: Dict[str, float],
        themes: ThemeExtraction,
        article_count: int,
        comment_count: int
    ) -> List[str]:
        """
        Generate key insights from sentiment analysis.

        Args:
            overall_sentiment: Overall sentiment score
            overall_label: Overall sentiment label
            distribution: Sentiment distribution counts
            by_source: Sentiment by source
            themes: Extracted themes
            article_count: Number of articles analyzed
            comment_count: Number of comments analyzed

        Returns:
            List of insight strings
        """
        insights = []

        # Overall sentiment insight
        total = sum(distribution.values())
        if total > 0:
            pos_pct = distribution['positive'] / total * 100
            neg_pct = distribution['negative'] / total * 100

            if overall_label == "positive":
                insights.append(
                    f"Overall media sentiment is positive ({pos_pct:.0f}% positive coverage), "
                    f"suggesting favorable pre-match atmosphere."
                )
            elif overall_label == "negative":
                insights.append(
                    f"Overall media sentiment is negative ({neg_pct:.0f}% negative coverage), "
                    f"indicating concerns or skepticism in coverage."
                )
            else:
                insights.append(
                    f"Overall media sentiment is neutral/mixed with {pos_pct:.0f}% positive "
                    f"and {neg_pct:.0f}% negative coverage."
                )

        # Source variation insight
        if by_source:
            most_positive = max(by_source.items(), key=lambda x: x[1])
            most_negative = min(by_source.items(), key=lambda x: x[1])

            if most_positive[1] > 0.3:
                insights.append(
                    f"Most positive coverage from {most_positive[0]} "
                    f"(score: {most_positive[1]:.2f})."
                )
            if most_negative[1] < -0.3:
                insights.append(
                    f"Most negative coverage from {most_negative[0]} "
                    f"(score: {most_negative[1]:.2f})."
                )

        # Trending concerns insight
        if themes.trending_concerns:
            concerns_summary = "; ".join(themes.trending_concerns[:3])
            insights.append(f"Key concerns identified: {concerns_summary}")

        # Trending optimism insight
        if themes.trending_optimism:
            optimism_summary = "; ".join(themes.trending_optimism[:3])
            insights.append(f"Positive narratives: {optimism_summary}")

        # Top keywords insight
        if themes.keywords:
            top_kws = [kw for kw, _ in themes.keywords[:5]]
            insights.append(f"Dominant topics: {', '.join(top_kws)}")

        # Entity insight
        if themes.entities:
            persons = [e[0] for e in themes.entities if e[1] == 'PERSON'][:3]
            if persons:
                insights.append(f"Most mentioned individuals: {', '.join(persons)}")

        return insights

    def format_report_for_brief(
        self,
        report: SentimentSummaryReport
    ) -> str:
        """
        Format a sentiment report as text for inclusion in intelligence brief.

        Args:
            report: SentimentSummaryReport to format

        Returns:
            Formatted text string for intelligence brief
        """
        lines = [
            f"SENTIMENT ANALYSIS REPORT",
            f"Match: {report.match_id}",
            f"Generated: {report.generated_at}",
            f"",
            f"OVERALL ASSESSMENT",
            f"  Sentiment: {report.overall_label.upper()} (score: {report.overall_sentiment:+.2f})",
            f"  Distribution: {report.sentiment_distribution['positive']} positive, "
            f"{report.sentiment_distribution['neutral']} neutral, "
            f"{report.sentiment_distribution['negative']} negative",
            f"  Sources analyzed: {report.articles_analyzed} articles, "
            f"{report.comments_analyzed} comments",
            f"",
            f"SENTIMENT BY SOURCE",
        ]

        for source, score in sorted(
            report.sentiment_by_source.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {source}: {score:+.2f}")

        lines.extend([
            f"",
            f"KEY THEMES",
            f"  Keywords: {', '.join([kw for kw, _ in report.themes.keywords[:10]])}",
            f"  Entities: {', '.join([e[0] for e in report.themes.entities[:5]])}",
            f"",
            f"TRENDING CONCERNS",
        ])

        for concern in report.themes.trending_concerns:
            lines.append(f"  - {concern}")

        lines.extend([
            f"",
            f"TRENDING OPTIMISM",
        ])

        for optimism in report.themes.trending_optimism:
            lines.append(f"  - {optimism}")

        lines.extend([
            f"",
            f"KEY INSIGHTS",
        ])

        for insight in report.key_insights:
            lines.append(f"  - {insight}")

        return "\n".join(lines)


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def analyze_news_data(news_data: Dict[str, Any]) -> SentimentSummaryReport:
    """
    Convenience function to analyze news data conforming to news.json schema.

    Args:
        news_data: News data dict matching the news.json schema

    Returns:
        SentimentSummaryReport with analysis
    """
    analyzer = SentimentAnalyzer()

    return analyzer.generate_sentiment_report(
        match_id=news_data.get('match_id', 'UNKNOWN'),
        articles=news_data.get('articles', []),
        comments=None  # Reddit comments would come from a separate source
    )


def calculate_sentiment_score(text: str) -> float:
    """
    Quick sentiment score calculation for a single text.

    Args:
        text: Text to analyze

    Returns:
        Sentiment score from -1.0 (negative) to 1.0 (positive)
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text(text)
    return result.score


def batch_sentiment_scores(texts: List[str]) -> List[float]:
    """
    Calculate sentiment scores for multiple texts efficiently.

    Args:
        texts: List of texts to analyze

    Returns:
        List of sentiment scores
    """
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_texts_batch(texts)
    return [r.score for r in results]


# ==========================================================================
# MAIN / DEMO
# ==========================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("Arsenal Intelligence Brief - Sentiment Analyzer Demo")
    print("=" * 70)

    # Create analyzer instance
    analyzer = SentimentAnalyzer()

    # Demo 1: Single text analysis
    print("\n--- Single Text Analysis ---")
    sample_texts = [
        "Arsenal showed brilliant form today, dominating Chelsea from start to finish.",
        "Disappointing performance from the team. Serious concerns about defensive issues.",
        "The match ended in a draw with both teams having chances.",
    ]

    for text in sample_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Score: {result.score:+.2f}")

    # Demo 2: Article analysis
    print("\n--- Article Sentiment Analysis ---")
    demo_articles = [
        {
            "title": "Arsenal vs Chelsea: Key Battles to Watch in London Derby",
            "url": "https://example.com/article1",
            "source": "BBC Sport",
            "publish_date": "2026-01-17T08:30:00Z",
            "full_text": "The North London club faces their West London rivals in what promises to be a thrilling encounter. Arsenal's recent form has been impressive, with Arteta's side winning their last five matches."
        },
        {
            "title": "Injury concerns mount for Arsenal ahead of Chelsea clash",
            "url": "https://example.com/article2",
            "source": "The Guardian",
            "publish_date": "2026-01-16T16:45:00Z",
            "full_text": "Arsenal face an injury crisis with multiple key players doubtful for the crucial London derby. Manager Mikel Arteta expressed concern in his press conference."
        }
    ]

    article_results = analyzer.analyze_articles_batch(demo_articles)

    for result in article_results:
        print(f"\nArticle: {result.title[:50]}...")
        print(f"  Source: {result.source}")
        print(f"  Title Sentiment: {result.title_sentiment.label} ({result.title_sentiment.score:+.2f})")
        if result.content_sentiment:
            print(f"  Content Sentiment: {result.content_sentiment.label} ({result.content_sentiment.score:+.2f})")
        print(f"  Combined Score: {result.combined_score:+.2f}")

    # Demo 3: Reddit comment analysis
    print("\n--- Reddit Comment Analysis ---")
    demo_comments = [
        {
            "comment_id": "abc123",
            "author": "gunner_fan",
            "text": "COYG! We're going to smash Chelsea! Saka is in incredible form!",
            "subreddit": "Gunners",
            "upvotes": 156
        },
        {
            "comment_id": "def456",
            "author": "worried_supporter",
            "text": "I'm nervous about this one. Our defense has been shaky lately.",
            "subreddit": "Gunners",
            "upvotes": 42
        }
    ]

    comment_results = analyzer.analyze_comments_batch(demo_comments)

    for result in comment_results:
        print(f"\nComment by u/{result.author} (r/{result.subreddit}):")
        print(f"  Text: {result.text[:50]}...")
        print(f"  Sentiment: {result.sentiment.label} ({result.sentiment.score:+.2f})")
        print(f"  Upvotes: {result.upvotes}")

    # Demo 4: Theme extraction
    print("\n--- Theme Extraction ---")
    all_texts = [a['title'] for a in demo_articles] + [a.get('full_text', '') for a in demo_articles]
    all_texts += [c['text'] for c in demo_comments]

    themes = analyzer.extract_themes(all_texts)

    print(f"\nTop Keywords: {[kw for kw, _ in themes.keywords[:10]]}")
    print(f"Top Bigrams: {[bg for bg, _ in themes.bigrams[:5]]}")
    print(f"Entities: {[(e[0], e[1]) for e in themes.entities[:5]]}")
    print(f"Trending Concerns: {themes.trending_concerns}")
    print(f"Trending Optimism: {themes.trending_optimism}")

    # Demo 5: Full report generation
    print("\n--- Full Sentiment Report ---")
    report = analyzer.generate_sentiment_report(
        match_id="20260118_ARS_CHE",
        articles=demo_articles,
        comments=demo_comments
    )

    print(f"\nMatch: {report.match_id}")
    print(f"Overall Sentiment: {report.overall_label} ({report.overall_sentiment:+.2f})")
    print(f"Distribution: {report.sentiment_distribution}")
    print(f"Articles Analyzed: {report.articles_analyzed}")
    print(f"Comments Analyzed: {report.comments_analyzed}")

    print("\nKey Insights:")
    for insight in report.key_insights:
        print(f"  - {insight}")

    # Demo 6: Formatted report for brief
    print("\n" + "=" * 70)
    print("FORMATTED REPORT FOR INTELLIGENCE BRIEF")
    print("=" * 70)
    formatted = analyzer.format_report_for_brief(report)
    print(formatted)
