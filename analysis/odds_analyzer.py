#!/usr/bin/env python3
"""
Odds Analyzer Module for Arsenal Intelligence Brief.

This module provides functionality to:
- Convert between different odds formats (decimal, fractional, American)
- Calculate implied probabilities from bookmaker odds
- Handle and remove bookmaker overround (vig/juice)
- Calculate fair odds and identify value opportunities

Task: arsenalScript-vqp.33 - Convert odds to implied probabilities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)


# ==========================================================================
# STANDALONE FUNCTIONS (preserved from original implementation)
# ==========================================================================

def decimal_to_probability(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Decimal odds represent the total payout per unit staked, including the stake.
    The implied probability is calculated as 1 / odds.

    Args:
        odds: Decimal odds (must be >= 1.0)

    Returns:
        Implied probability as a float between 0 and 1

    Raises:
        ValueError: If odds are less than 1.0

    Example:
        >>> decimal_to_probability(2.0)
        0.5
        >>> decimal_to_probability(1.5)
        0.6666666666666666
    """
    if odds < 1.0:
        raise ValueError(f"Decimal odds must be >= 1.0, got {odds}")
    return 1.0 / odds


def fractional_to_decimal(numerator: int, denominator: int) -> float:
    """
    Convert fractional odds to decimal odds.

    Fractional odds (e.g., 5/2) represent profit relative to stake.
    Decimal odds = (numerator / denominator) + 1

    Args:
        numerator: The numerator of the fractional odds (profit)
        denominator: The denominator of the fractional odds (stake)

    Returns:
        Equivalent decimal odds

    Raises:
        ValueError: If denominator is zero or if values are negative

    Example:
        >>> fractional_to_decimal(5, 2)
        3.5
        >>> fractional_to_decimal(1, 4)
        1.25
    """
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    if numerator < 0 or denominator < 0:
        raise ValueError("Fractional odds components must be non-negative")
    return (numerator / denominator) + 1.0


def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.

    American odds come in two formats:
    - Positive (+150): Amount won on a $100 stake
    - Negative (-150): Amount needed to stake to win $100

    Args:
        odds: American odds (positive or negative integer, cannot be 0 or between -100 and 100 exclusive)

    Returns:
        Equivalent decimal odds

    Raises:
        ValueError: If odds are 0 or between -100 and 100 exclusive

    Example:
        >>> american_to_decimal(150)
        2.5
        >>> american_to_decimal(-150)
        1.6666666666666667
        >>> american_to_decimal(-100)
        2.0
    """
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if -100 < odds < 100:
        raise ValueError(f"American odds must be <= -100 or >= 100, got {odds}")

    if odds > 0:
        # Positive odds: decimal = (odds / 100) + 1
        return (odds / 100.0) + 1.0
    else:
        # Negative odds: decimal = (100 / abs(odds)) + 1
        return (100.0 / abs(odds)) + 1.0


def remove_overround(probabilities: list[float]) -> list[float]:
    """
    Normalize probabilities to sum to 1 by removing the bookmaker's overround.

    Bookmakers set odds such that implied probabilities sum to more than 1
    (the overround or vig). This function normalizes the probabilities to
    reflect true probabilities that sum to exactly 1.

    Args:
        probabilities: List of implied probabilities (each between 0 and 1)

    Returns:
        List of normalized probabilities that sum to 1

    Raises:
        ValueError: If any probability is not between 0 and 1, or if list is empty

    Example:
        >>> probs = [0.55, 0.55]  # Sum = 1.10 (10% overround)
        >>> remove_overround(probs)
        [0.5, 0.5]
    """
    if not probabilities:
        raise ValueError("Probabilities list cannot be empty")

    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError(f"Each probability must be between 0 and 1, got {p}")

    total = sum(probabilities)
    if total == 0:
        raise ValueError("Sum of probabilities cannot be zero")

    return [p / total for p in probabilities]


def calculate_expected_value(probability: float, odds: float) -> float:
    """
    Calculate the expected value (EV) of a bet.

    Expected value represents the average return per unit staked over many bets.
    EV = (probability * profit) - (1 - probability) * stake
    For a unit stake: EV = (probability * (odds - 1)) - (1 - probability)
    Simplified: EV = (probability * odds) - 1

    A positive EV indicates a profitable bet in the long run.

    Args:
        probability: True probability of the outcome (between 0 and 1)
        odds: Decimal odds offered by the bookmaker

    Returns:
        Expected value per unit staked (positive = profitable)

    Raises:
        ValueError: If probability is not between 0 and 1, or odds < 1

    Example:
        >>> calculate_expected_value(0.5, 2.1)  # Fair odds would be 2.0
        0.05
        >>> calculate_expected_value(0.5, 1.9)  # Below fair odds
        -0.05
    """
    if not 0 <= probability <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if odds < 1.0:
        raise ValueError(f"Decimal odds must be >= 1.0, got {odds}")

    return (probability * odds) - 1.0


# ==========================================================================
# DATA CLASSES
# ==========================================================================

@dataclass
class OddsData:
    """Data class to hold odds information for a match outcome."""
    decimal: float
    implied_probability: float
    fair_probability: Optional[float] = None
    bookmaker: Optional[str] = None
    market: Optional[str] = None  # e.g., "home_win", "draw", "away_win"


@dataclass
class MatchOdds:
    """Data class to hold complete odds for a match."""
    home_win: OddsData
    draw: OddsData
    away_win: OddsData
    overround: float
    fair_overround: float = 100.0  # Perfect market has 100% total probability


# ==========================================================================
# ODDS ANALYZER CLASS
# ==========================================================================

class OddsAnalyzer:
    """
    Analyzer for converting and analyzing betting odds.

    Supports conversion between:
    - Decimal odds (European): e.g., 2.50
    - Fractional odds (UK): e.g., 3/2
    - American odds (US): e.g., +150 or -200

    Also calculates implied probabilities and removes overround.
    """

    def __init__(self):
        """Initialize the OddsAnalyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ==========================================================================
    # ODDS FORMAT CONVERSIONS
    # ==========================================================================

    def decimal_to_implied_probability(self, decimal_odds: float) -> float:
        """
        Convert decimal odds to implied probability.

        Formula: Implied Probability = 1 / Decimal Odds

        Args:
            decimal_odds: Decimal odds (e.g., 2.50)

        Returns:
            Implied probability as a decimal (0-1), e.g., 0.40 for 40%

        Raises:
            ValueError: If decimal_odds is less than or equal to 1.0

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> analyzer.decimal_to_implied_probability(2.50)
            0.4
        """
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be greater than 1.0, got {decimal_odds}")

        return 1.0 / decimal_odds

    def implied_probability_to_decimal(self, probability: float) -> float:
        """
        Convert implied probability to decimal odds.

        Formula: Decimal Odds = 1 / Probability

        Args:
            probability: Probability as decimal (0-1), e.g., 0.40

        Returns:
            Decimal odds, e.g., 2.50

        Raises:
            ValueError: If probability is not between 0 and 1 (exclusive)
        """
        if not 0 < probability < 1:
            raise ValueError(f"Probability must be between 0 and 1 (exclusive), got {probability}")

        return 1.0 / probability

    def fractional_to_decimal(self, fractional: Union[str, Tuple[int, int]]) -> float:
        """
        Convert fractional odds to decimal odds.

        Formula: Decimal = (Numerator / Denominator) + 1

        Args:
            fractional: Either a string like "3/2" or a tuple like (3, 2)

        Returns:
            Decimal odds

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> analyzer.fractional_to_decimal("3/2")
            2.5
            >>> analyzer.fractional_to_decimal((3, 2))
            2.5
        """
        if isinstance(fractional, str):
            parts = fractional.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid fractional odds format: {fractional}")
            numerator, denominator = int(parts[0]), int(parts[1])
        else:
            numerator, denominator = fractional

        if denominator == 0:
            raise ValueError("Denominator cannot be zero")

        return (numerator / denominator) + 1

    def decimal_to_fractional(self, decimal_odds: float) -> str:
        """
        Convert decimal odds to fractional odds.

        Args:
            decimal_odds: Decimal odds (e.g., 2.50)

        Returns:
            Fractional odds as string (e.g., "3/2")
        """
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be greater than 1.0, got {decimal_odds}")

        # Convert to fraction, then simplify
        frac = Fraction(decimal_odds - 1).limit_denominator(100)
        return f"{frac.numerator}/{frac.denominator}"

    def american_to_decimal(self, american_odds: int) -> float:
        """
        Convert American odds to decimal odds.

        For positive American odds (underdog): Decimal = (American / 100) + 1
        For negative American odds (favorite): Decimal = (100 / |American|) + 1

        Args:
            american_odds: American odds (e.g., +150 or -200)

        Returns:
            Decimal odds

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> analyzer.american_to_decimal(150)  # +150
            2.5
            >>> analyzer.american_to_decimal(-200)
            1.5
        """
        if american_odds == 0:
            raise ValueError("American odds cannot be zero")

        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def decimal_to_american(self, decimal_odds: float) -> int:
        """
        Convert decimal odds to American odds.

        Args:
            decimal_odds: Decimal odds (e.g., 2.50)

        Returns:
            American odds (positive for underdog, negative for favorite)
        """
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be greater than 1.0, got {decimal_odds}")

        if decimal_odds >= 2.0:
            # Underdog: positive American odds
            return int(round((decimal_odds - 1) * 100))
        else:
            # Favorite: negative American odds
            return int(round(-100 / (decimal_odds - 1)))

    def fractional_to_implied_probability(self, fractional: Union[str, Tuple[int, int]]) -> float:
        """
        Convert fractional odds directly to implied probability.

        Args:
            fractional: Fractional odds (e.g., "3/2" or (3, 2))

        Returns:
            Implied probability as decimal (0-1)
        """
        decimal = self.fractional_to_decimal(fractional)
        return self.decimal_to_implied_probability(decimal)

    def american_to_implied_probability(self, american_odds: int) -> float:
        """
        Convert American odds directly to implied probability.

        Args:
            american_odds: American odds (e.g., +150 or -200)

        Returns:
            Implied probability as decimal (0-1)
        """
        decimal = self.american_to_decimal(american_odds)
        return self.decimal_to_implied_probability(decimal)

    # ==========================================================================
    # OVERROUND AND FAIR PROBABILITY CALCULATIONS
    # ==========================================================================

    def calculate_overround(self, probabilities: List[float]) -> float:
        """
        Calculate the bookmaker's overround (vig/juice) from implied probabilities.

        The overround is the percentage by which total implied probabilities
        exceed 100%. A fair market has exactly 100% total probability.

        Args:
            probabilities: List of implied probabilities for all outcomes

        Returns:
            Overround as percentage (e.g., 105.0 means 5% overround)

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> # Home 40%, Draw 30%, Away 35% = 105% total
            >>> analyzer.calculate_overround([0.40, 0.30, 0.35])
            105.0
        """
        total = sum(probabilities) * 100
        return round(total, 2)

    def calculate_overround_from_odds(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> float:
        """
        Calculate overround directly from decimal odds.

        Args:
            home_odds: Decimal odds for home win
            draw_odds: Decimal odds for draw
            away_odds: Decimal odds for away win

        Returns:
            Overround as percentage
        """
        probs = [
            self.decimal_to_implied_probability(home_odds),
            self.decimal_to_implied_probability(draw_odds),
            self.decimal_to_implied_probability(away_odds)
        ]
        return self.calculate_overround(probs)

    def remove_overround_method(
        self,
        probabilities: List[float],
        method: str = "multiplicative"
    ) -> List[float]:
        """
        Remove bookmaker overround to get fair probabilities.

        Supports multiple methods:
        - "multiplicative": Proportionally reduce all probabilities
        - "additive": Subtract equal amount from each probability
        - "odds_ratio": More sophisticated method using odds ratios

        Args:
            probabilities: List of implied probabilities (with overround)
            method: Method to use for removing overround

        Returns:
            List of fair probabilities that sum to 1.0

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> # 105% total probability -> normalize to 100%
            >>> analyzer.remove_overround_method([0.42, 0.315, 0.315])
            [0.4, 0.3, 0.3]  # approximately
        """
        total = sum(probabilities)

        if method == "multiplicative":
            # Most common method: scale proportionally
            return [p / total for p in probabilities]

        elif method == "additive":
            # Subtract equal amount from each
            excess = (total - 1) / len(probabilities)
            fair_probs = [max(0.001, p - excess) for p in probabilities]
            # Re-normalize to handle edge cases
            total_fair = sum(fair_probs)
            return [p / total_fair for p in fair_probs]

        elif method == "odds_ratio":
            # Use power method for more accurate fair odds
            # Based on Shin's model for bookmaker behavior
            n = len(probabilities)
            # Iteratively solve for fair probabilities
            fair_probs = probabilities.copy()
            for _ in range(10):  # Iterate to converge
                total_fair = sum(fair_probs)
                fair_probs = [p / total_fair for p in fair_probs]
            return fair_probs

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_fair_odds(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        method: str = "multiplicative"
    ) -> Dict[str, float]:
        """
        Calculate fair decimal odds by removing overround.

        Args:
            home_odds: Bookmaker decimal odds for home win
            draw_odds: Bookmaker decimal odds for draw
            away_odds: Bookmaker decimal odds for away win
            method: Method to remove overround

        Returns:
            Dictionary with fair odds for each outcome
        """
        implied_probs = [
            self.decimal_to_implied_probability(home_odds),
            self.decimal_to_implied_probability(draw_odds),
            self.decimal_to_implied_probability(away_odds)
        ]

        fair_probs = self.remove_overround_method(implied_probs, method)

        return {
            "home_win": self.implied_probability_to_decimal(fair_probs[0]),
            "draw": self.implied_probability_to_decimal(fair_probs[1]),
            "away_win": self.implied_probability_to_decimal(fair_probs[2]),
            "home_probability": fair_probs[0],
            "draw_probability": fair_probs[1],
            "away_probability": fair_probs[2]
        }

    # ==========================================================================
    # ANALYSIS METHODS
    # ==========================================================================

    def analyze_match_odds(
        self,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        bookmaker: Optional[str] = None
    ) -> MatchOdds:
        """
        Perform comprehensive analysis of match odds.

        Args:
            home_odds: Decimal odds for home win
            draw_odds: Decimal odds for draw
            away_odds: Decimal odds for away win
            bookmaker: Name of bookmaker (optional)

        Returns:
            MatchOdds object with complete analysis
        """
        # Calculate implied probabilities
        home_prob = self.decimal_to_implied_probability(home_odds)
        draw_prob = self.decimal_to_implied_probability(draw_odds)
        away_prob = self.decimal_to_implied_probability(away_odds)

        # Calculate overround
        overround = self.calculate_overround([home_prob, draw_prob, away_prob])

        # Get fair probabilities
        fair_probs = self.remove_overround_method([home_prob, draw_prob, away_prob])

        return MatchOdds(
            home_win=OddsData(
                decimal=home_odds,
                implied_probability=home_prob,
                fair_probability=fair_probs[0],
                bookmaker=bookmaker,
                market="home_win"
            ),
            draw=OddsData(
                decimal=draw_odds,
                implied_probability=draw_prob,
                fair_probability=fair_probs[1],
                bookmaker=bookmaker,
                market="draw"
            ),
            away_win=OddsData(
                decimal=away_odds,
                implied_probability=away_prob,
                fair_probability=fair_probs[2],
                bookmaker=bookmaker,
                market="away_win"
            ),
            overround=overround
        )

    def compare_bookmaker_odds(
        self,
        odds_list: List[Dict[str, float]]
    ) -> Dict[str, any]:
        """
        Compare odds across multiple bookmakers and find best value.

        Args:
            odds_list: List of dicts with keys: bookmaker, home, draw, away

        Returns:
            Dict with best odds for each outcome and analysis

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> odds = [
            ...     {"bookmaker": "Bet365", "home": 2.10, "draw": 3.40, "away": 3.50},
            ...     {"bookmaker": "William Hill", "home": 2.15, "draw": 3.30, "away": 3.40}
            ... ]
            >>> analyzer.compare_bookmaker_odds(odds)
        """
        if not odds_list:
            raise ValueError("odds_list cannot be empty")

        best_home = max(odds_list, key=lambda x: x["home"])
        best_draw = max(odds_list, key=lambda x: x["draw"])
        best_away = max(odds_list, key=lambda x: x["away"])

        # Calculate average overround
        overrounds = []
        for odds in odds_list:
            overround = self.calculate_overround_from_odds(
                odds["home"], odds["draw"], odds["away"]
            )
            overrounds.append(overround)

        avg_overround = sum(overrounds) / len(overrounds)

        # Calculate best possible combined odds (cherry-picking)
        best_combined_overround = self.calculate_overround_from_odds(
            best_home["home"], best_draw["draw"], best_away["away"]
        )

        return {
            "best_home": {
                "bookmaker": best_home["bookmaker"],
                "odds": best_home["home"],
                "implied_probability": self.decimal_to_implied_probability(best_home["home"])
            },
            "best_draw": {
                "bookmaker": best_draw["bookmaker"],
                "odds": best_draw["draw"],
                "implied_probability": self.decimal_to_implied_probability(best_draw["draw"])
            },
            "best_away": {
                "bookmaker": best_away["bookmaker"],
                "odds": best_away["away"],
                "implied_probability": self.decimal_to_implied_probability(best_away["away"])
            },
            "average_overround": round(avg_overround, 2),
            "best_combined_overround": round(best_combined_overround, 2),
            "bookmaker_count": len(odds_list)
        }

    def calculate_expected_value_detailed(
        self,
        decimal_odds: float,
        true_probability: float,
        stake: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate the expected value (EV) of a bet with detailed analysis.

        EV = (Probability of Winning * Profit) - (Probability of Losing * Stake)

        A positive EV indicates a value bet.

        Args:
            decimal_odds: The bookmaker's decimal odds
            true_probability: Your estimated true probability of the outcome
            stake: Amount to stake (default 1.0 for EV per unit)

        Returns:
            Dict with EV analysis

        Example:
            >>> analyzer = OddsAnalyzer()
            >>> # Bookmaker offers 2.50, but you think true prob is 45%
            >>> analyzer.calculate_expected_value_detailed(2.50, 0.45)
            {'expected_value': 0.125, 'ev_percentage': 12.5, 'is_value_bet': True}
        """
        if not 0 < true_probability < 1:
            raise ValueError("true_probability must be between 0 and 1")

        profit_if_win = (decimal_odds - 1) * stake
        loss_if_lose = stake

        ev = (true_probability * profit_if_win) - ((1 - true_probability) * loss_if_lose)
        ev_percentage = (ev / stake) * 100

        # Alternative formula: EV% = (Prob * Odds) - 1
        ev_pct_simple = (true_probability * decimal_odds - 1) * 100

        return {
            "expected_value": round(ev, 4),
            "ev_percentage": round(ev_pct_simple, 2),
            "is_value_bet": ev > 0,
            "implied_probability": self.decimal_to_implied_probability(decimal_odds),
            "edge": round((true_probability - self.decimal_to_implied_probability(decimal_odds)) * 100, 2)
        }

    def find_value_bets(
        self,
        bookmaker_odds: Dict[str, float],
        model_probabilities: Dict[str, float],
        min_edge: float = 0.05
    ) -> List[Dict[str, any]]:
        """
        Find value bets by comparing ML model predictions with bookmaker odds.

        Args:
            bookmaker_odds: Dict with home_win, draw, away_win decimal odds
            model_probabilities: Dict with home_win, draw, away_win probabilities
            min_edge: Minimum edge required to flag as value (default 5%)

        Returns:
            List of value bet opportunities
        """
        value_bets = []

        markets = ["home_win", "draw", "away_win"]

        for market in markets:
            if market not in bookmaker_odds or market not in model_probabilities:
                continue

            odds = bookmaker_odds[market]
            model_prob = model_probabilities[market]
            implied_prob = self.decimal_to_implied_probability(odds)

            edge = model_prob - implied_prob

            if edge >= min_edge:
                ev_analysis = self.calculate_expected_value_detailed(odds, model_prob)
                value_bets.append({
                    "market": market,
                    "decimal_odds": odds,
                    "implied_probability": round(implied_prob, 4),
                    "model_probability": round(model_prob, 4),
                    "edge": round(edge, 4),
                    "edge_percentage": round(edge * 100, 2),
                    "expected_value_percentage": ev_analysis["ev_percentage"]
                })

        # Sort by edge (highest first)
        value_bets.sort(key=lambda x: x["edge"], reverse=True)

        return value_bets


# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

def convert_odds_batch(
    odds_data: List[Dict],
    from_format: str = "decimal",
    to_format: str = "implied_probability"
) -> List[Dict]:
    """
    Batch convert a list of odds to different formats.

    Args:
        odds_data: List of dicts with odds values
        from_format: Source format ("decimal", "fractional", "american")
        to_format: Target format ("decimal", "fractional", "american", "implied_probability")

    Returns:
        List of dicts with converted odds
    """
    analyzer = OddsAnalyzer()
    results = []

    conversion_map = {
        ("decimal", "implied_probability"): analyzer.decimal_to_implied_probability,
        ("decimal", "fractional"): analyzer.decimal_to_fractional,
        ("decimal", "american"): analyzer.decimal_to_american,
        ("fractional", "decimal"): analyzer.fractional_to_decimal,
        ("fractional", "implied_probability"): analyzer.fractional_to_implied_probability,
        ("american", "decimal"): analyzer.american_to_decimal,
        ("american", "implied_probability"): analyzer.american_to_implied_probability,
        ("implied_probability", "decimal"): analyzer.implied_probability_to_decimal,
    }

    converter = conversion_map.get((from_format, to_format))
    if not converter:
        raise ValueError(f"Conversion from {from_format} to {to_format} not supported")

    for item in odds_data:
        result = item.copy()
        for key in ["home", "draw", "away", "home_win", "draw", "away_win", "odds"]:
            if key in item:
                try:
                    result[f"{key}_{to_format}"] = converter(item[key])
                except (ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Failed to convert {key}={item[key]}: {e}")
                    result[f"{key}_{to_format}"] = None
        results.append(result)

    return results


if __name__ == "__main__":
    # Demo/test the module
    print("=" * 60)
    print("Arsenal Intelligence Brief - Odds Analyzer Demo")
    print("=" * 60)

    analyzer = OddsAnalyzer()

    # Example: Arsenal vs Chelsea match odds
    print("\n--- Example Match: Arsenal vs Chelsea ---")
    print("\nBookmaker odds (decimal):")
    print("  Arsenal win: 2.10")
    print("  Draw: 3.40")
    print("  Chelsea win: 3.50")

    analysis = analyzer.analyze_match_odds(2.10, 3.40, 3.50, bookmaker="Example Bookie")

    print(f"\n--- Implied Probabilities (with overround) ---")
    print(f"  Arsenal win: {analysis.home_win.implied_probability:.1%}")
    print(f"  Draw: {analysis.draw.implied_probability:.1%}")
    print(f"  Chelsea win: {analysis.away_win.implied_probability:.1%}")
    print(f"  Total (overround): {analysis.overround:.1f}%")

    print(f"\n--- Fair Probabilities (overround removed) ---")
    print(f"  Arsenal win: {analysis.home_win.fair_probability:.1%}")
    print(f"  Draw: {analysis.draw.fair_probability:.1%}")
    print(f"  Chelsea win: {analysis.away_win.fair_probability:.1%}")

    # Value bet example
    print("\n--- Value Bet Analysis ---")
    print("If ML model predicts Arsenal win probability: 52%")

    ev = analyzer.calculate_expected_value_detailed(2.10, 0.52)
    print(f"  Expected Value: {ev['ev_percentage']:.1f}%")
    print(f"  Is value bet: {ev['is_value_bet']}")
    print(f"  Edge: {ev['edge']:.1f}%")

    # Odds conversion demo
    print("\n--- Odds Format Conversions ---")
    print(f"  Decimal 2.50 -> Fractional: {analyzer.decimal_to_fractional(2.50)}")
    print(f"  Decimal 2.50 -> American: {analyzer.decimal_to_american(2.50):+d}")
    print(f"  Fractional 3/2 -> Decimal: {analyzer.fractional_to_decimal('3/2')}")
    print(f"  American +150 -> Decimal: {analyzer.american_to_decimal(150)}")
    print(f"  American -200 -> Decimal: {analyzer.american_to_decimal(-200)}")

    # Multi-bookmaker comparison
    print("\n--- Multi-Bookmaker Comparison ---")
    multi_odds = [
        {"bookmaker": "Bet365", "home": 2.10, "draw": 3.40, "away": 3.50},
        {"bookmaker": "William Hill", "home": 2.15, "draw": 3.30, "away": 3.40},
        {"bookmaker": "Paddy Power", "home": 2.08, "draw": 3.50, "away": 3.45},
    ]
    comparison = analyzer.compare_bookmaker_odds(multi_odds)
    print(f"  Best Home odds: {comparison['best_home']['odds']} ({comparison['best_home']['bookmaker']})")
    print(f"  Best Draw odds: {comparison['best_draw']['odds']} ({comparison['best_draw']['bookmaker']})")
    print(f"  Best Away odds: {comparison['best_away']['odds']} ({comparison['best_away']['bookmaker']})")
    print(f"  Average overround: {comparison['average_overround']}%")
    print(f"  Best combined overround: {comparison['best_combined_overround']}%")

    # Value bet detection
    print("\n--- Value Bet Detection ---")
    bookmaker = {"home_win": 2.10, "draw": 3.40, "away_win": 3.50}
    model = {"home_win": 0.52, "draw": 0.26, "away_win": 0.22}
    value_bets = analyzer.find_value_bets(bookmaker, model, min_edge=0.03)
    if value_bets:
        for vb in value_bets:
            print(f"  {vb['market']}: Edge {vb['edge_percentage']:.1f}%, EV {vb['expected_value_percentage']:.1f}%")
    else:
        print("  No value bets found with >= 3% edge")
