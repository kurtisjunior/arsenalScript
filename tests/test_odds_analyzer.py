#!/usr/bin/env python3
"""
Unit tests for the odds_analyzer module.

Task: arsenalScript-vqp.33 - Convert odds to implied probabilities
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.odds_analyzer import (
    OddsAnalyzer,
    OddsData,
    MatchOdds,
    decimal_to_probability,
    fractional_to_decimal,
    american_to_decimal,
    remove_overround,
    calculate_expected_value,
    convert_odds_batch,
)


class TestStandaloneFunctions:
    """Tests for standalone utility functions."""

    def test_decimal_to_probability_basic(self):
        """Test basic decimal to probability conversion."""
        assert decimal_to_probability(2.0) == 0.5
        assert abs(decimal_to_probability(1.5) - 0.6666666666666666) < 1e-10
        assert decimal_to_probability(4.0) == 0.25

    def test_decimal_to_probability_invalid(self):
        """Test that invalid odds raise ValueError."""
        with pytest.raises(ValueError):
            decimal_to_probability(0.5)
        with pytest.raises(ValueError):
            decimal_to_probability(0.99)

    def test_fractional_to_decimal_basic(self):
        """Test basic fractional to decimal conversion."""
        assert fractional_to_decimal(5, 2) == 3.5
        assert fractional_to_decimal(1, 4) == 1.25
        assert fractional_to_decimal(1, 1) == 2.0  # Evens

    def test_fractional_to_decimal_invalid(self):
        """Test that invalid fractional odds raise ValueError."""
        with pytest.raises(ValueError):
            fractional_to_decimal(1, 0)  # Division by zero
        with pytest.raises(ValueError):
            fractional_to_decimal(-1, 2)  # Negative numerator
        with pytest.raises(ValueError):
            fractional_to_decimal(1, -2)  # Negative denominator

    def test_american_to_decimal_positive(self):
        """Test positive American odds conversion."""
        assert american_to_decimal(150) == 2.5
        assert american_to_decimal(100) == 2.0
        assert american_to_decimal(200) == 3.0

    def test_american_to_decimal_negative(self):
        """Test negative American odds conversion."""
        assert abs(american_to_decimal(-150) - 1.6666666666666667) < 1e-10
        assert american_to_decimal(-100) == 2.0
        assert american_to_decimal(-200) == 1.5

    def test_american_to_decimal_invalid(self):
        """Test that invalid American odds raise ValueError."""
        with pytest.raises(ValueError):
            american_to_decimal(0)
        with pytest.raises(ValueError):
            american_to_decimal(50)  # Between -100 and 100
        with pytest.raises(ValueError):
            american_to_decimal(-50)  # Between -100 and 100

    def test_remove_overround_basic(self):
        """Test basic overround removal."""
        probs = [0.55, 0.55]  # 110% total
        normalized = remove_overround(probs)
        assert abs(sum(normalized) - 1.0) < 1e-10
        assert normalized[0] == normalized[1] == 0.5

    def test_remove_overround_three_way(self):
        """Test overround removal for three-way market."""
        probs = [0.45, 0.28, 0.35]  # 108% total
        normalized = remove_overround(probs)
        assert abs(sum(normalized) - 1.0) < 1e-10

    def test_remove_overround_invalid(self):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            remove_overround([])  # Empty list
        with pytest.raises(ValueError):
            remove_overround([1.5, 0.5])  # Probability > 1
        with pytest.raises(ValueError):
            remove_overround([-0.1, 0.5])  # Negative probability

    def test_calculate_expected_value_positive(self):
        """Test positive EV calculation."""
        # Fair odds for 50% would be 2.0, so 2.1 is +EV
        ev = calculate_expected_value(0.5, 2.1)
        assert ev == pytest.approx(0.05, abs=0.001)

    def test_calculate_expected_value_negative(self):
        """Test negative EV calculation."""
        # Fair odds for 50% would be 2.0, so 1.9 is -EV
        ev = calculate_expected_value(0.5, 1.9)
        assert ev == pytest.approx(-0.05, abs=0.001)

    def test_calculate_expected_value_invalid(self):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            calculate_expected_value(1.5, 2.0)  # Probability > 1
        with pytest.raises(ValueError):
            calculate_expected_value(-0.1, 2.0)  # Negative probability
        with pytest.raises(ValueError):
            calculate_expected_value(0.5, 0.5)  # Odds < 1


class TestOddsAnalyzer:
    """Tests for the OddsAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an OddsAnalyzer instance."""
        return OddsAnalyzer()

    def test_decimal_to_implied_probability(self, analyzer):
        """Test decimal to implied probability conversion."""
        assert analyzer.decimal_to_implied_probability(2.5) == 0.4
        assert analyzer.decimal_to_implied_probability(2.0) == 0.5
        assert analyzer.decimal_to_implied_probability(4.0) == 0.25

    def test_decimal_to_implied_probability_invalid(self, analyzer):
        """Test that invalid odds raise ValueError."""
        with pytest.raises(ValueError):
            analyzer.decimal_to_implied_probability(1.0)
        with pytest.raises(ValueError):
            analyzer.decimal_to_implied_probability(0.5)

    def test_implied_probability_to_decimal(self, analyzer):
        """Test implied probability to decimal conversion."""
        assert analyzer.implied_probability_to_decimal(0.5) == 2.0
        assert analyzer.implied_probability_to_decimal(0.25) == 4.0
        assert analyzer.implied_probability_to_decimal(0.4) == 2.5

    def test_implied_probability_to_decimal_invalid(self, analyzer):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError):
            analyzer.implied_probability_to_decimal(0)
        with pytest.raises(ValueError):
            analyzer.implied_probability_to_decimal(1)
        with pytest.raises(ValueError):
            analyzer.implied_probability_to_decimal(1.5)

    def test_fractional_to_decimal_string(self, analyzer):
        """Test fractional string to decimal conversion."""
        assert analyzer.fractional_to_decimal("3/2") == 2.5
        assert analyzer.fractional_to_decimal("1/1") == 2.0
        assert analyzer.fractional_to_decimal("5/1") == 6.0

    def test_fractional_to_decimal_tuple(self, analyzer):
        """Test fractional tuple to decimal conversion."""
        assert analyzer.fractional_to_decimal((3, 2)) == 2.5
        assert analyzer.fractional_to_decimal((1, 1)) == 2.0
        assert analyzer.fractional_to_decimal((5, 1)) == 6.0

    def test_decimal_to_fractional(self, analyzer):
        """Test decimal to fractional conversion."""
        assert analyzer.decimal_to_fractional(2.5) == "3/2"
        assert analyzer.decimal_to_fractional(2.0) == "1/1"

    def test_american_to_decimal(self, analyzer):
        """Test American to decimal conversion."""
        assert analyzer.american_to_decimal(150) == 2.5
        assert analyzer.american_to_decimal(-200) == 1.5

    def test_decimal_to_american(self, analyzer):
        """Test decimal to American conversion."""
        assert analyzer.decimal_to_american(2.5) == 150
        assert analyzer.decimal_to_american(1.5) == -200

    def test_fractional_to_implied_probability(self, analyzer):
        """Test fractional to implied probability conversion."""
        prob = analyzer.fractional_to_implied_probability("3/2")
        assert prob == 0.4

    def test_american_to_implied_probability(self, analyzer):
        """Test American to implied probability conversion."""
        prob = analyzer.american_to_implied_probability(150)
        assert prob == 0.4

    def test_calculate_overround(self, analyzer):
        """Test overround calculation."""
        # Home 40%, Draw 30%, Away 35% = 105%
        overround = analyzer.calculate_overround([0.40, 0.30, 0.35])
        assert overround == 105.0

    def test_calculate_overround_from_odds(self, analyzer):
        """Test overround calculation from decimal odds."""
        # These odds should give approximately 105% overround
        overround = analyzer.calculate_overround_from_odds(2.10, 3.40, 3.50)
        assert 104 < overround < 106

    def test_remove_overround_method_multiplicative(self, analyzer):
        """Test multiplicative overround removal."""
        probs = [0.42, 0.315, 0.315]  # 105% total
        fair = analyzer.remove_overround_method(probs, "multiplicative")
        assert abs(sum(fair) - 1.0) < 1e-10

    def test_remove_overround_method_additive(self, analyzer):
        """Test additive overround removal."""
        probs = [0.42, 0.315, 0.315]  # 105% total
        fair = analyzer.remove_overround_method(probs, "additive")
        assert abs(sum(fair) - 1.0) < 1e-10

    def test_get_fair_odds(self, analyzer):
        """Test fair odds calculation."""
        fair = analyzer.get_fair_odds(2.10, 3.40, 3.50)
        assert "home_win" in fair
        assert "draw" in fair
        assert "away_win" in fair
        assert "home_probability" in fair
        # Fair probabilities should sum to 1
        assert abs(fair["home_probability"] + fair["draw_probability"] + fair["away_probability"] - 1.0) < 1e-10


class TestMatchOddsAnalysis:
    """Tests for match odds analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create an OddsAnalyzer instance."""
        return OddsAnalyzer()

    def test_analyze_match_odds(self, analyzer):
        """Test comprehensive match odds analysis."""
        analysis = analyzer.analyze_match_odds(2.10, 3.40, 3.50, bookmaker="Test")

        assert isinstance(analysis, MatchOdds)
        assert isinstance(analysis.home_win, OddsData)
        assert analysis.home_win.decimal == 2.10
        assert analysis.home_win.bookmaker == "Test"
        assert analysis.home_win.market == "home_win"
        assert 0 < analysis.home_win.implied_probability < 1
        assert 0 < analysis.home_win.fair_probability < 1
        assert analysis.overround > 100  # Should have some overround

    def test_compare_bookmaker_odds(self, analyzer):
        """Test multi-bookmaker comparison."""
        odds_list = [
            {"bookmaker": "Bet365", "home": 2.10, "draw": 3.40, "away": 3.50},
            {"bookmaker": "William Hill", "home": 2.15, "draw": 3.30, "away": 3.40},
            {"bookmaker": "Paddy Power", "home": 2.08, "draw": 3.50, "away": 3.45},
        ]

        comparison = analyzer.compare_bookmaker_odds(odds_list)

        assert comparison["best_home"]["bookmaker"] == "William Hill"
        assert comparison["best_home"]["odds"] == 2.15
        assert comparison["best_draw"]["bookmaker"] == "Paddy Power"
        assert comparison["best_draw"]["odds"] == 3.50
        assert comparison["best_away"]["bookmaker"] == "Bet365"
        assert comparison["best_away"]["odds"] == 3.50
        assert comparison["bookmaker_count"] == 3

    def test_compare_bookmaker_odds_empty(self, analyzer):
        """Test that empty odds list raises ValueError."""
        with pytest.raises(ValueError):
            analyzer.compare_bookmaker_odds([])


class TestExpectedValueCalculations:
    """Tests for expected value calculations."""

    @pytest.fixture
    def analyzer(self):
        """Create an OddsAnalyzer instance."""
        return OddsAnalyzer()

    def test_calculate_expected_value_detailed_positive(self, analyzer):
        """Test positive EV calculation with detailed output."""
        # If true prob is 50% and odds are 2.10, EV should be positive
        ev = analyzer.calculate_expected_value_detailed(2.10, 0.50)

        assert ev["is_value_bet"] == True
        assert ev["ev_percentage"] > 0
        assert ev["edge"] > 0

    def test_calculate_expected_value_detailed_negative(self, analyzer):
        """Test negative EV calculation with detailed output."""
        # If true prob is 40% and odds are 2.10, EV should be negative
        ev = analyzer.calculate_expected_value_detailed(2.10, 0.40)

        assert ev["is_value_bet"] == False
        assert ev["ev_percentage"] < 0
        assert ev["edge"] < 0

    def test_find_value_bets_with_edge(self, analyzer):
        """Test value bet detection with sufficient edge."""
        bookmaker_odds = {"home_win": 2.10, "draw": 3.40, "away_win": 3.50}
        model_probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}

        value_bets = analyzer.find_value_bets(bookmaker_odds, model_probs, min_edge=0.05)

        assert len(value_bets) >= 1
        # Home win should be flagged - model says 55% but implied is ~47.6%
        home_bet = next((vb for vb in value_bets if vb["market"] == "home_win"), None)
        assert home_bet is not None
        assert home_bet["edge"] > 0.05

    def test_find_value_bets_no_edge(self, analyzer):
        """Test value bet detection with no sufficient edge."""
        bookmaker_odds = {"home_win": 2.10, "draw": 3.40, "away_win": 3.50}
        # Model probabilities close to implied
        model_probs = {"home_win": 0.48, "draw": 0.29, "away_win": 0.23}

        value_bets = analyzer.find_value_bets(bookmaker_odds, model_probs, min_edge=0.10)

        # With 10% minimum edge, should find no value bets
        assert len(value_bets) == 0


class TestBatchConversion:
    """Tests for batch conversion utility."""

    def test_convert_odds_batch_decimal_to_probability(self):
        """Test batch conversion from decimal to probability."""
        odds_data = [
            {"home": 2.0, "draw": 3.0, "away": 4.0},
            {"home": 1.5, "draw": 4.0, "away": 6.0},
        ]

        results = convert_odds_batch(odds_data, "decimal", "implied_probability")

        assert len(results) == 2
        assert results[0]["home_implied_probability"] == 0.5
        assert results[0]["draw_implied_probability"] == pytest.approx(0.333, abs=0.01)


class TestDataClasses:
    """Tests for data classes."""

    def test_odds_data_creation(self):
        """Test OddsData creation."""
        data = OddsData(
            decimal=2.50,
            implied_probability=0.40,
            fair_probability=0.38,
            bookmaker="Test",
            market="home_win"
        )

        assert data.decimal == 2.50
        assert data.implied_probability == 0.40
        assert data.fair_probability == 0.38
        assert data.bookmaker == "Test"
        assert data.market == "home_win"

    def test_match_odds_creation(self):
        """Test MatchOdds creation."""
        home = OddsData(decimal=2.10, implied_probability=0.476)
        draw = OddsData(decimal=3.40, implied_probability=0.294)
        away = OddsData(decimal=3.50, implied_probability=0.286)

        match = MatchOdds(
            home_win=home,
            draw=draw,
            away_win=away,
            overround=105.6
        )

        assert match.home_win.decimal == 2.10
        assert match.draw.decimal == 3.40
        assert match.away_win.decimal == 3.50
        assert match.overround == 105.6
        assert match.fair_overround == 100.0  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
