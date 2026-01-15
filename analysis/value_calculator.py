#!/usr/bin/env python3
"""
Value Calculator Module for Arsenal Intelligence Brief.

This module provides functionality to:
- Compare ML model predictions with bookmaker implied odds probabilities
- Calculate expected value (EV = ML probability x odds - 1)
- Flag high-value betting opportunities (EV > 5%)
- Rank opportunities by expected value
- Include confidence level and risk assessment

Task: arsenalScript-vqp.34-36 - Value betting calculations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging

from .odds_analyzer import OddsAnalyzer, calculate_expected_value

logger = logging.getLogger(__name__)


# ==========================================================================
# ENUMS AND CONSTANTS
# ==========================================================================

class RiskLevel(Enum):
    """Risk level classification for betting opportunities."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ConfidenceLevel(Enum):
    """Confidence level for ML model predictions."""
    VERY_HIGH = "very_high"     # >= 90% model confidence
    HIGH = "high"               # 75-90%
    MEDIUM = "medium"           # 60-75%
    LOW = "low"                 # 45-60%
    VERY_LOW = "very_low"       # < 45%


class MarketType(Enum):
    """Supported betting market types."""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    OVER_2_5 = "over_2_5"
    UNDER_2_5 = "under_2_5"
    OVER_1_5 = "over_1_5"
    UNDER_1_5 = "under_1_5"
    BTTS_YES = "btts_yes"       # Both teams to score
    BTTS_NO = "btts_no"


# EV thresholds for flagging opportunities
EV_THRESHOLD_HIGH_VALUE = 0.05      # 5% EV - high value flag
EV_THRESHOLD_MEDIUM_VALUE = 0.03    # 3% EV - medium value
EV_THRESHOLD_MIN_VALUE = 0.01       # 1% EV - minimum for consideration


# ==========================================================================
# DATA CLASSES
# ==========================================================================

@dataclass
class MLPrediction:
    """
    Represents an ML model's prediction for a match outcome.

    Attributes:
        probability: The predicted probability (0-1)
        confidence: Model confidence in the prediction
        model_name: Name of the model used
        features_used: Optional list of features used in prediction
        prediction_timestamp: When the prediction was made
    """
    probability: float
    confidence: float
    model_name: str = "default_model"
    features_used: Optional[List[str]] = None
    prediction_timestamp: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    def get_confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if self.confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.45:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class BookmakerOddsInput:
    """
    Input structure for bookmaker odds across markets.

    Attributes:
        bookmaker: Name of the bookmaker
        home_win: Decimal odds for home win
        draw: Decimal odds for draw
        away_win: Decimal odds for away win
        over_2_5: Optional odds for over 2.5 goals
        under_2_5: Optional odds for under 2.5 goals
        over_1_5: Optional odds for over 1.5 goals
        under_1_5: Optional odds for under 1.5 goals
        btts_yes: Optional odds for both teams to score
        btts_no: Optional odds for both teams not to score
    """
    bookmaker: str
    home_win: float
    draw: float
    away_win: float
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None
    over_1_5: Optional[float] = None
    under_1_5: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None

    def get_odds_for_market(self, market: MarketType) -> Optional[float]:
        """Get odds for a specific market type."""
        market_map = {
            MarketType.HOME_WIN: self.home_win,
            MarketType.DRAW: self.draw,
            MarketType.AWAY_WIN: self.away_win,
            MarketType.OVER_2_5: self.over_2_5,
            MarketType.UNDER_2_5: self.under_2_5,
            MarketType.OVER_1_5: self.over_1_5,
            MarketType.UNDER_1_5: self.under_1_5,
            MarketType.BTTS_YES: self.btts_yes,
            MarketType.BTTS_NO: self.btts_no,
        }
        return market_map.get(market)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "bookmaker": self.bookmaker,
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
        }
        for attr, key in [
            (self.over_2_5, "over_2_5"),
            (self.under_2_5, "under_2_5"),
            (self.over_1_5, "over_1_5"),
            (self.under_1_5, "under_1_5"),
            (self.btts_yes, "btts_yes"),
            (self.btts_no, "btts_no"),
        ]:
            if attr is not None:
                result[key] = attr
        return result


@dataclass
class ValueOpportunity:
    """
    Represents a single value betting opportunity.

    Contains all relevant information for making an informed betting decision.
    """
    market: MarketType
    expected_value: float               # EV as decimal (0.05 = 5%)
    expected_value_percentage: float    # EV as percentage (5.0 = 5%)
    ml_probability: float              # ML model's predicted probability
    implied_probability: float         # Bookmaker's implied probability
    edge: float                        # ML prob - implied prob
    edge_percentage: float             # Edge as percentage
    decimal_odds: float                # Best available decimal odds
    bookmaker: str                     # Best bookmaker for this market
    confidence_level: ConfidenceLevel  # ML model confidence
    risk_level: RiskLevel              # Calculated risk level
    kelly_fraction: float              # Optimal Kelly criterion stake
    is_high_value: bool                # True if EV > 5%
    match_id: Optional[str] = None     # Reference to match
    analysis_timestamp: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "market": self.market.value,
            "expected_value": round(self.expected_value, 4),
            "expected_value_percentage": round(self.expected_value_percentage, 2),
            "ml_probability": round(self.ml_probability, 4),
            "implied_probability": round(self.implied_probability, 4),
            "edge": round(self.edge, 4),
            "edge_percentage": round(self.edge_percentage, 2),
            "decimal_odds": round(self.decimal_odds, 2),
            "bookmaker": self.bookmaker,
            "confidence_level": self.confidence_level.value,
            "risk_level": self.risk_level.value,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "is_high_value": self.is_high_value,
            "match_id": self.match_id,
            "analysis_timestamp": self.analysis_timestamp,
            "notes": self.notes,
        }

    def to_brief_summary(self) -> str:
        """Generate a brief summary for intelligence report."""
        market_display = self.market.value.replace("_", " ").title()
        return (
            f"{market_display}: EV {self.expected_value_percentage:+.1f}% | "
            f"Edge {self.edge_percentage:.1f}% | "
            f"Odds {self.decimal_odds:.2f} ({self.bookmaker}) | "
            f"Risk: {self.risk_level.value.replace('_', ' ').title()}"
        )


@dataclass
class ValueAnalysisResult:
    """
    Complete result of value analysis for a match.

    Contains all opportunities found, ranked by expected value.
    """
    match_id: str
    analysis_timestamp: str
    opportunities: List[ValueOpportunity]
    high_value_count: int
    total_markets_analyzed: int
    best_opportunity: Optional[ValueOpportunity]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match_id": self.match_id,
            "analysis_timestamp": self.analysis_timestamp,
            "opportunities": [opp.to_dict() for opp in self.opportunities],
            "high_value_count": self.high_value_count,
            "total_markets_analyzed": self.total_markets_analyzed,
            "best_opportunity": self.best_opportunity.to_dict() if self.best_opportunity else None,
            "summary": self.summary,
        }

    def to_intelligence_brief(self) -> str:
        """
        Generate formatted output suitable for intelligence brief.

        Returns a human-readable summary of value opportunities.
        """
        lines = [
            "=" * 60,
            f"VALUE ANALYSIS: {self.match_id}",
            f"Timestamp: {self.analysis_timestamp}",
            "=" * 60,
            "",
        ]

        if not self.opportunities:
            lines.append("No value opportunities identified.")
            return "\n".join(lines)

        # Summary section
        lines.append(f"SUMMARY:")
        lines.append(f"  Markets Analyzed: {self.total_markets_analyzed}")
        lines.append(f"  Value Opportunities Found: {len(self.opportunities)}")
        lines.append(f"  High Value Opportunities (EV > 5%): {self.high_value_count}")
        lines.append("")

        # Best opportunity highlight
        if self.best_opportunity:
            lines.append("BEST OPPORTUNITY:")
            lines.append(f"  {self.best_opportunity.to_brief_summary()}")
            lines.append(f"  ML Probability: {self.best_opportunity.ml_probability:.1%}")
            lines.append(f"  Implied Probability: {self.best_opportunity.implied_probability:.1%}")
            lines.append(f"  Confidence: {self.best_opportunity.confidence_level.value.replace('_', ' ').title()}")
            lines.append(f"  Kelly Stake: {self.best_opportunity.kelly_fraction:.1%} of bankroll")
            lines.append("")

        # All opportunities ranked
        if len(self.opportunities) > 1:
            lines.append("ALL OPPORTUNITIES (Ranked by EV):")
            lines.append("-" * 50)
            for i, opp in enumerate(self.opportunities, 1):
                flag = " [HIGH VALUE]" if opp.is_high_value else ""
                lines.append(f"  {i}. {opp.to_brief_summary()}{flag}")
            lines.append("")

        # Risk warning
        high_risk_count = sum(1 for o in self.opportunities
                             if o.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH])
        if high_risk_count > 0:
            lines.append(f"WARNING: {high_risk_count} opportunity/ies flagged as HIGH RISK")

        return "\n".join(lines)


# ==========================================================================
# VALUE CALCULATOR CLASS
# ==========================================================================

class ValueCalculator:
    """
    Calculator for identifying and ranking value betting opportunities.

    Compares ML model predictions with bookmaker odds to find edges and
    calculate expected value. Supports multiple betting markets.

    Usage:
        calculator = ValueCalculator()

        # Set up ML predictions
        predictions = {
            MarketType.HOME_WIN: MLPrediction(probability=0.52, confidence=0.85),
            MarketType.DRAW: MLPrediction(probability=0.26, confidence=0.80),
            MarketType.AWAY_WIN: MLPrediction(probability=0.22, confidence=0.82),
        }

        # Set up bookmaker odds
        odds = BookmakerOddsInput(
            bookmaker="bet365",
            home_win=2.10,
            draw=3.40,
            away_win=3.50
        )

        # Analyze
        result = calculator.analyze_match(
            match_id="20260120_ARS_CHE",
            predictions=predictions,
            bookmaker_odds=[odds]
        )

        # Get intelligence brief output
        print(result.to_intelligence_brief())
    """

    def __init__(
        self,
        ev_threshold: float = EV_THRESHOLD_HIGH_VALUE,
        min_confidence: float = 0.50,
        max_kelly_fraction: float = 0.25
    ):
        """
        Initialize the ValueCalculator.

        Args:
            ev_threshold: Minimum EV to flag as high value (default 5%)
            min_confidence: Minimum ML confidence to consider (default 50%)
            max_kelly_fraction: Maximum Kelly stake fraction (default 25%)
        """
        self.ev_threshold = ev_threshold
        self.min_confidence = min_confidence
        self.max_kelly_fraction = max_kelly_fraction
        self.odds_analyzer = OddsAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ==========================================================================
    # CORE CALCULATION METHODS
    # ==========================================================================

    def calculate_expected_value(
        self,
        ml_probability: float,
        decimal_odds: float
    ) -> Tuple[float, float]:
        """
        Calculate expected value for a bet.

        EV = (ML probability x odds) - 1

        Args:
            ml_probability: Model's predicted probability (0-1)
            decimal_odds: Bookmaker's decimal odds

        Returns:
            Tuple of (EV as decimal, EV as percentage)

        Example:
            >>> calc = ValueCalculator()
            >>> ev, ev_pct = calc.calculate_expected_value(0.52, 2.10)
            >>> print(f"EV: {ev:.4f} ({ev_pct:.2f}%)")
            EV: 0.0920 (9.20%)
        """
        if not 0 <= ml_probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {ml_probability}")
        if decimal_odds < 1.0:
            raise ValueError(f"Decimal odds must be >= 1.0, got {decimal_odds}")

        ev = (ml_probability * decimal_odds) - 1.0
        ev_percentage = ev * 100

        return ev, ev_percentage

    def calculate_edge(
        self,
        ml_probability: float,
        implied_probability: float
    ) -> Tuple[float, float]:
        """
        Calculate the edge (difference) between ML and implied probabilities.

        Args:
            ml_probability: Model's predicted probability (0-1)
            implied_probability: Bookmaker's implied probability (0-1)

        Returns:
            Tuple of (edge as decimal, edge as percentage)
        """
        edge = ml_probability - implied_probability
        edge_percentage = edge * 100
        return edge, edge_percentage

    def calculate_kelly_criterion(
        self,
        ml_probability: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.

        Kelly % = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = decimal odds - 1 (net odds)

        Args:
            ml_probability: Model's predicted probability (0-1)
            decimal_odds: Bookmaker's decimal odds

        Returns:
            Kelly fraction as decimal (capped at max_kelly_fraction)
        """
        if ml_probability <= 0 or ml_probability >= 1:
            return 0.0

        b = decimal_odds - 1.0  # Net odds (profit per unit if win)
        if b <= 0:
            return 0.0

        q = 1.0 - ml_probability
        kelly = (ml_probability * b - q) / b

        # Don't bet if Kelly is negative (no edge)
        if kelly <= 0:
            return 0.0

        # Cap at maximum fraction to manage risk
        return min(kelly, self.max_kelly_fraction)

    def assess_risk_level(
        self,
        ml_probability: float,
        implied_probability: float,
        confidence: float,
        decimal_odds: float
    ) -> RiskLevel:
        """
        Assess the risk level of a betting opportunity.

        Risk assessment considers:
        - Odds level (higher odds = higher risk)
        - Confidence level (lower confidence = higher risk)
        - Edge size (small edge on low probability = higher risk)
        - Probability of outcome (lower probability = higher risk)

        Args:
            ml_probability: Model's predicted probability
            implied_probability: Bookmaker's implied probability
            confidence: Model's confidence in prediction
            decimal_odds: Bookmaker's decimal odds

        Returns:
            RiskLevel enum value
        """
        risk_score = 0

        # Factor 1: Odds level (higher odds = more variance)
        if decimal_odds >= 5.0:
            risk_score += 3
        elif decimal_odds >= 3.0:
            risk_score += 2
        elif decimal_odds >= 2.0:
            risk_score += 1

        # Factor 2: Model confidence
        if confidence < 0.50:
            risk_score += 3
        elif confidence < 0.65:
            risk_score += 2
        elif confidence < 0.75:
            risk_score += 1

        # Factor 3: Probability level (low probability events are riskier)
        if ml_probability < 0.20:
            risk_score += 2
        elif ml_probability < 0.35:
            risk_score += 1

        # Factor 4: Edge size relative to implied probability
        edge = ml_probability - implied_probability
        relative_edge = edge / implied_probability if implied_probability > 0 else 0
        if relative_edge > 0.30:  # Large relative edge - either great value or model error
            risk_score += 1

        # Map score to risk level
        if risk_score <= 1:
            return RiskLevel.VERY_LOW
        elif risk_score <= 3:
            return RiskLevel.LOW
        elif risk_score <= 5:
            return RiskLevel.MEDIUM
        elif risk_score <= 7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    # ==========================================================================
    # OPPORTUNITY ANALYSIS
    # ==========================================================================

    def analyze_single_market(
        self,
        market: MarketType,
        prediction: MLPrediction,
        decimal_odds: float,
        bookmaker: str,
        match_id: Optional[str] = None
    ) -> Optional[ValueOpportunity]:
        """
        Analyze a single market for value opportunity.

        Args:
            market: The market type being analyzed
            prediction: ML model's prediction
            decimal_odds: Best available decimal odds
            bookmaker: Name of bookmaker offering best odds
            match_id: Optional match identifier

        Returns:
            ValueOpportunity if there's positive EV, None otherwise
        """
        # Skip if confidence is below threshold
        if prediction.confidence < self.min_confidence:
            self.logger.debug(
                f"Skipping {market.value}: confidence {prediction.confidence:.2%} "
                f"below threshold {self.min_confidence:.2%}"
            )
            return None

        # Calculate implied probability from odds
        implied_prob = 1.0 / decimal_odds

        # Calculate EV
        ev, ev_percentage = self.calculate_expected_value(
            prediction.probability,
            decimal_odds
        )

        # Only return opportunities with positive EV
        if ev <= 0:
            return None

        # Calculate edge
        edge, edge_percentage = self.calculate_edge(
            prediction.probability,
            implied_prob
        )

        # Calculate Kelly criterion
        kelly = self.calculate_kelly_criterion(
            prediction.probability,
            decimal_odds
        )

        # Assess risk
        risk_level = self.assess_risk_level(
            prediction.probability,
            implied_prob,
            prediction.confidence,
            decimal_odds
        )

        # Determine if high value
        is_high_value = ev >= self.ev_threshold

        return ValueOpportunity(
            market=market,
            expected_value=ev,
            expected_value_percentage=ev_percentage,
            ml_probability=prediction.probability,
            implied_probability=implied_prob,
            edge=edge,
            edge_percentage=edge_percentage,
            decimal_odds=decimal_odds,
            bookmaker=bookmaker,
            confidence_level=prediction.get_confidence_level(),
            risk_level=risk_level,
            kelly_fraction=kelly,
            is_high_value=is_high_value,
            match_id=match_id,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def find_best_odds(
        self,
        market: MarketType,
        bookmaker_odds: List[BookmakerOddsInput]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Find the best odds across all bookmakers for a given market.

        Args:
            market: The market to find best odds for
            bookmaker_odds: List of bookmaker odds inputs

        Returns:
            Tuple of (best odds, bookmaker name) or (None, None) if not available
        """
        best_odds = None
        best_bookmaker = None

        for bm_odds in bookmaker_odds:
            odds = bm_odds.get_odds_for_market(market)
            if odds is not None and (best_odds is None or odds > best_odds):
                best_odds = odds
                best_bookmaker = bm_odds.bookmaker

        return best_odds, best_bookmaker

    def analyze_match(
        self,
        match_id: str,
        predictions: Dict[MarketType, MLPrediction],
        bookmaker_odds: List[BookmakerOddsInput]
    ) -> ValueAnalysisResult:
        """
        Perform complete value analysis for a match across all markets.

        Args:
            match_id: Unique match identifier
            predictions: Dict mapping market types to ML predictions
            bookmaker_odds: List of bookmaker odds inputs

        Returns:
            ValueAnalysisResult with all opportunities ranked by EV
        """
        opportunities: List[ValueOpportunity] = []
        markets_analyzed = 0

        # Analyze each market with predictions
        for market, prediction in predictions.items():
            # Find best odds for this market
            best_odds, best_bookmaker = self.find_best_odds(market, bookmaker_odds)

            if best_odds is None or best_bookmaker is None:
                self.logger.debug(f"No odds available for market {market.value}")
                continue

            markets_analyzed += 1

            # Analyze the market
            opportunity = self.analyze_single_market(
                market=market,
                prediction=prediction,
                decimal_odds=best_odds,
                bookmaker=best_bookmaker,
                match_id=match_id
            )

            if opportunity:
                opportunities.append(opportunity)

        # Sort by expected value (highest first)
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)

        # Count high value opportunities
        high_value_count = sum(1 for o in opportunities if o.is_high_value)

        # Get best opportunity
        best_opportunity = opportunities[0] if opportunities else None

        # Build summary
        summary = self._build_summary(opportunities, markets_analyzed)

        return ValueAnalysisResult(
            match_id=match_id,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            opportunities=opportunities,
            high_value_count=high_value_count,
            total_markets_analyzed=markets_analyzed,
            best_opportunity=best_opportunity,
            summary=summary
        )

    def _build_summary(
        self,
        opportunities: List[ValueOpportunity],
        markets_analyzed: int
    ) -> Dict[str, Any]:
        """Build a summary dictionary for the analysis result."""
        if not opportunities:
            return {
                "total_opportunities": 0,
                "markets_analyzed": markets_analyzed,
                "average_ev": 0.0,
                "max_ev": 0.0,
                "high_value_markets": [],
                "recommendation": "No value opportunities identified."
            }

        evs = [o.expected_value_percentage for o in opportunities]
        high_value_markets = [o.market.value for o in opportunities if o.is_high_value]

        # Generate recommendation
        if len(high_value_markets) >= 2:
            recommendation = (
                f"Multiple high-value opportunities detected. "
                f"Consider positions in: {', '.join(high_value_markets)}."
            )
        elif len(high_value_markets) == 1:
            best = opportunities[0]
            recommendation = (
                f"Single high-value opportunity in {best.market.value.replace('_', ' ')}. "
                f"EV: {best.expected_value_percentage:.1f}%, "
                f"Confidence: {best.confidence_level.value}."
            )
        else:
            recommendation = (
                "Value opportunities exist but none exceed high-value threshold. "
                "Proceed with caution."
            )

        return {
            "total_opportunities": len(opportunities),
            "markets_analyzed": markets_analyzed,
            "average_ev": round(sum(evs) / len(evs), 2),
            "max_ev": round(max(evs), 2),
            "min_ev": round(min(evs), 2),
            "high_value_markets": high_value_markets,
            "recommendation": recommendation,
            "risk_distribution": {
                "very_low": sum(1 for o in opportunities if o.risk_level == RiskLevel.VERY_LOW),
                "low": sum(1 for o in opportunities if o.risk_level == RiskLevel.LOW),
                "medium": sum(1 for o in opportunities if o.risk_level == RiskLevel.MEDIUM),
                "high": sum(1 for o in opportunities if o.risk_level == RiskLevel.HIGH),
                "very_high": sum(1 for o in opportunities if o.risk_level == RiskLevel.VERY_HIGH),
            }
        }

    # ==========================================================================
    # CONVENIENCE METHODS FOR INTEGRATION
    # ==========================================================================

    def analyze_from_dict(
        self,
        match_id: str,
        ml_predictions: Dict[str, Dict[str, float]],
        bookmaker_odds_list: List[Dict[str, Any]]
    ) -> ValueAnalysisResult:
        """
        Analyze match from dictionary inputs (for easier integration).

        Args:
            match_id: Unique match identifier
            ml_predictions: Dict with market names as keys, containing
                           'probability' and 'confidence' values
            bookmaker_odds_list: List of dicts with bookmaker odds

        Returns:
            ValueAnalysisResult

        Example:
            >>> calc = ValueCalculator()
            >>> result = calc.analyze_from_dict(
            ...     match_id="20260120_ARS_CHE",
            ...     ml_predictions={
            ...         "home_win": {"probability": 0.52, "confidence": 0.85},
            ...         "draw": {"probability": 0.26, "confidence": 0.80},
            ...         "away_win": {"probability": 0.22, "confidence": 0.82},
            ...     },
            ...     bookmaker_odds_list=[
            ...         {"bookmaker": "bet365", "home_win": 2.10, "draw": 3.40, "away_win": 3.50},
            ...         {"bookmaker": "skybet", "home_win": 2.15, "draw": 3.30, "away_win": 3.45},
            ...     ]
            ... )
        """
        # Convert ML predictions
        market_map = {
            "home_win": MarketType.HOME_WIN,
            "draw": MarketType.DRAW,
            "away_win": MarketType.AWAY_WIN,
            "over_2_5": MarketType.OVER_2_5,
            "under_2_5": MarketType.UNDER_2_5,
            "over_1_5": MarketType.OVER_1_5,
            "under_1_5": MarketType.UNDER_1_5,
            "btts_yes": MarketType.BTTS_YES,
            "btts_no": MarketType.BTTS_NO,
        }

        predictions: Dict[MarketType, MLPrediction] = {}
        for market_name, pred_data in ml_predictions.items():
            market_type = market_map.get(market_name)
            if market_type is None:
                self.logger.warning(f"Unknown market type: {market_name}")
                continue

            predictions[market_type] = MLPrediction(
                probability=pred_data.get("probability", 0.0),
                confidence=pred_data.get("confidence", 0.5),
                model_name=pred_data.get("model_name", "default_model")
            )

        # Convert bookmaker odds
        bookmaker_odds: List[BookmakerOddsInput] = []
        for bm_data in bookmaker_odds_list:
            bookmaker_odds.append(BookmakerOddsInput(
                bookmaker=bm_data.get("bookmaker", "unknown"),
                home_win=bm_data.get("home_win", 1.0),
                draw=bm_data.get("draw", 1.0),
                away_win=bm_data.get("away_win", 1.0),
                over_2_5=bm_data.get("over_2_5"),
                under_2_5=bm_data.get("under_2_5"),
                over_1_5=bm_data.get("over_1_5"),
                under_1_5=bm_data.get("under_1_5"),
                btts_yes=bm_data.get("btts_yes"),
                btts_no=bm_data.get("btts_no"),
            ))

        return self.analyze_match(match_id, predictions, bookmaker_odds)

    def quick_ev_check(
        self,
        ml_probability: float,
        decimal_odds: float
    ) -> Dict[str, Any]:
        """
        Quick EV calculation for a single bet.

        Args:
            ml_probability: Model's predicted probability
            decimal_odds: Bookmaker's decimal odds

        Returns:
            Dict with EV analysis
        """
        ev, ev_pct = self.calculate_expected_value(ml_probability, decimal_odds)
        implied_prob = 1.0 / decimal_odds
        edge, edge_pct = self.calculate_edge(ml_probability, implied_prob)
        kelly = self.calculate_kelly_criterion(ml_probability, decimal_odds)

        return {
            "expected_value": round(ev, 4),
            "expected_value_percentage": round(ev_pct, 2),
            "is_value_bet": ev > 0,
            "is_high_value": ev >= self.ev_threshold,
            "edge": round(edge, 4),
            "edge_percentage": round(edge_pct, 2),
            "implied_probability": round(implied_prob, 4),
            "ml_probability": round(ml_probability, 4),
            "kelly_fraction": round(kelly, 4),
            "recommended_stake_pct": round(kelly * 100, 2),
        }


# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

def rank_opportunities(
    opportunities: List[ValueOpportunity],
    sort_by: str = "ev"
) -> List[ValueOpportunity]:
    """
    Rank value opportunities by specified criteria.

    Args:
        opportunities: List of ValueOpportunity objects
        sort_by: Sorting criterion - "ev", "edge", "kelly", "confidence"

    Returns:
        Sorted list of opportunities
    """
    sort_keys = {
        "ev": lambda x: x.expected_value,
        "edge": lambda x: x.edge,
        "kelly": lambda x: x.kelly_fraction,
        "confidence": lambda x: (
            {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
            .get(x.confidence_level.value, 0)
        ),
    }

    key_func = sort_keys.get(sort_by, sort_keys["ev"])
    return sorted(opportunities, key=key_func, reverse=True)


def filter_by_risk(
    opportunities: List[ValueOpportunity],
    max_risk: RiskLevel = RiskLevel.MEDIUM
) -> List[ValueOpportunity]:
    """
    Filter opportunities by maximum acceptable risk level.

    Args:
        opportunities: List of ValueOpportunity objects
        max_risk: Maximum acceptable risk level

    Returns:
        Filtered list of opportunities
    """
    risk_order = [
        RiskLevel.VERY_LOW,
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.VERY_HIGH
    ]

    max_index = risk_order.index(max_risk)

    return [
        opp for opp in opportunities
        if risk_order.index(opp.risk_level) <= max_index
    ]


def filter_high_value_only(
    opportunities: List[ValueOpportunity]
) -> List[ValueOpportunity]:
    """
    Filter to only high value opportunities (EV > 5%).

    Args:
        opportunities: List of ValueOpportunity objects

    Returns:
        Filtered list containing only high value opportunities
    """
    return [opp for opp in opportunities if opp.is_high_value]


# ==========================================================================
# DEMO / TEST
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Arsenal Intelligence Brief - Value Calculator Demo")
    print("=" * 60)

    # Create calculator instance
    calculator = ValueCalculator()

    # Example 1: Quick EV check
    print("\n--- Quick EV Check ---")
    result = calculator.quick_ev_check(ml_probability=0.52, decimal_odds=2.10)
    print(f"ML Probability: {result['ml_probability']:.1%}")
    print(f"Implied Probability: {result['implied_probability']:.1%}")
    print(f"Expected Value: {result['expected_value_percentage']:.2f}%")
    print(f"Edge: {result['edge_percentage']:.2f}%")
    print(f"Is Value Bet: {result['is_value_bet']}")
    print(f"Is High Value (>5%): {result['is_high_value']}")
    print(f"Kelly Stake: {result['recommended_stake_pct']:.1f}% of bankroll")

    # Example 2: Full match analysis with multiple markets
    print("\n--- Full Match Analysis ---")
    print("Arsenal vs Chelsea - Match ID: 20260120_ARS_CHE")

    # ML model predictions
    ml_predictions = {
        "home_win": {"probability": 0.52, "confidence": 0.85, "model_name": "xgboost_v2"},
        "draw": {"probability": 0.26, "confidence": 0.78, "model_name": "xgboost_v2"},
        "away_win": {"probability": 0.22, "confidence": 0.80, "model_name": "xgboost_v2"},
        "over_2_5": {"probability": 0.58, "confidence": 0.72, "model_name": "goals_model"},
        "under_2_5": {"probability": 0.42, "confidence": 0.72, "model_name": "goals_model"},
    }

    # Bookmaker odds from multiple sources
    bookmaker_odds = [
        {
            "bookmaker": "Bet365",
            "home_win": 2.10,
            "draw": 3.40,
            "away_win": 3.50,
            "over_2_5": 1.85,
            "under_2_5": 1.95
        },
        {
            "bookmaker": "William Hill",
            "home_win": 2.15,
            "draw": 3.30,
            "away_win": 3.45,
            "over_2_5": 1.90,
            "under_2_5": 1.90
        },
        {
            "bookmaker": "Paddy Power",
            "home_win": 2.08,
            "draw": 3.50,
            "away_win": 3.40,
            "over_2_5": 1.87,
            "under_2_5": 1.93
        },
    ]

    # Run analysis
    analysis = calculator.analyze_from_dict(
        match_id="20260120_ARS_CHE",
        ml_predictions=ml_predictions,
        bookmaker_odds_list=bookmaker_odds
    )

    # Print intelligence brief output
    print(analysis.to_intelligence_brief())

    # Example 3: Filter and rank
    print("\n--- Filtered Results (Max Risk: Medium) ---")
    filtered = filter_by_risk(analysis.opportunities, max_risk=RiskLevel.MEDIUM)
    for opp in filtered:
        print(f"  {opp.to_brief_summary()}")

    # Example 4: Using dataclasses directly
    print("\n--- Direct Dataclass Usage ---")
    prediction = MLPrediction(probability=0.55, confidence=0.88)
    odds_input = BookmakerOddsInput(
        bookmaker="Example",
        home_win=2.00,
        draw=3.50,
        away_win=3.80
    )

    opportunity = calculator.analyze_single_market(
        market=MarketType.HOME_WIN,
        prediction=prediction,
        decimal_odds=odds_input.home_win,
        bookmaker=odds_input.bookmaker,
        match_id="example_match"
    )

    if opportunity:
        print(f"Market: {opportunity.market.value}")
        print(f"EV: {opportunity.expected_value_percentage:.2f}%")
        print(f"Risk: {opportunity.risk_level.value}")
        print(f"Kelly: {opportunity.kelly_fraction:.2%}")

    print("\n" + "=" * 60)
    print("Demo complete.")
