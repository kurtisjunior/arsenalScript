#!/usr/bin/env python3
"""
ML Match Predictor for Arsenal Intelligence Brief

This module implements machine learning-based match outcome prediction
for Arsenal FC matches using historical data and various features.

Features engineered:
- Recent form (last 5 games)
- Head-to-head record
- Home/away advantage
- Injury impact factor
- Bookmaker odds (as features)

Model: Logistic Regression with softmax for multi-class classification
Validation: K-fold cross-validation with multiple metrics

Task: arsenalScript-vqp.23-27 - ML prediction system
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
HISTORICAL_DIR = DATA_DIR / "historical"

# Model version for tracking
MODEL_VERSION = "1.0.0"


@dataclass
class FormStats:
    """Statistics representing recent form over N games."""
    n_games: int = 5
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    clean_sheets: int = 0
    failed_to_score: int = 0

    @property
    def points(self) -> int:
        """Total points (W=3, D=1, L=0)."""
        return (self.wins * 3) + self.draws

    @property
    def max_points(self) -> int:
        """Maximum possible points."""
        return self.n_games * 3

    @property
    def points_per_game(self) -> float:
        """Average points per game."""
        return self.points / self.n_games if self.n_games > 0 else 0.0

    @property
    def goal_difference(self) -> int:
        """Goal difference."""
        return self.goals_scored - self.goals_conceded

    @property
    def goals_per_game(self) -> float:
        """Average goals scored per game."""
        return self.goals_scored / self.n_games if self.n_games > 0 else 0.0

    @property
    def goals_conceded_per_game(self) -> float:
        """Average goals conceded per game."""
        return self.goals_conceded / self.n_games if self.n_games > 0 else 0.0

    def to_feature_vector(self) -> List[float]:
        """Convert form stats to feature vector."""
        return [
            self.points_per_game,
            self.goals_per_game,
            self.goals_conceded_per_game,
            self.wins / self.n_games if self.n_games > 0 else 0.0,
            self.draws / self.n_games if self.n_games > 0 else 0.0,
            self.losses / self.n_games if self.n_games > 0 else 0.0,
            self.clean_sheets / self.n_games if self.n_games > 0 else 0.0,
            self.failed_to_score / self.n_games if self.n_games > 0 else 0.0,
        ]


@dataclass
class H2HStats:
    """Head-to-head statistics against a specific opponent."""
    opponent: str = ""
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0

    @property
    def win_rate(self) -> float:
        """Historical win rate."""
        return self.wins / self.matches_played if self.matches_played > 0 else 0.5

    @property
    def draw_rate(self) -> float:
        """Historical draw rate."""
        return self.draws / self.matches_played if self.matches_played > 0 else 0.25

    @property
    def loss_rate(self) -> float:
        """Historical loss rate."""
        return self.losses / self.matches_played if self.matches_played > 0 else 0.25

    def to_feature_vector(self) -> List[float]:
        """Convert H2H stats to feature vector."""
        return [
            self.win_rate,
            self.draw_rate,
            self.loss_rate,
            self.goals_scored / self.matches_played if self.matches_played > 0 else 1.5,
            self.goals_conceded / self.matches_played if self.matches_played > 0 else 1.5,
        ]


@dataclass
class MatchFeatures:
    """Complete feature set for a single match prediction."""
    # Form features (Arsenal)
    arsenal_form: FormStats = field(default_factory=FormStats)

    # Form features (Opponent)
    opponent_form: FormStats = field(default_factory=FormStats)

    # Head-to-head features
    h2h_stats: H2HStats = field(default_factory=H2HStats)

    # Home advantage
    is_home: bool = True

    # Injury impact (0 = no injuries, 1 = severe injuries)
    arsenal_injury_factor: float = 0.0
    opponent_injury_factor: float = 0.0

    # Bookmaker odds (as features)
    bookmaker_win_prob: float = 0.0
    bookmaker_draw_prob: float = 0.0
    bookmaker_loss_prob: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert all features to a numpy array for model input."""
        features = []

        # Arsenal form features (8)
        features.extend(self.arsenal_form.to_feature_vector())

        # Opponent form features (8)
        features.extend(self.opponent_form.to_feature_vector())

        # H2H features (5)
        features.extend(self.h2h_stats.to_feature_vector())

        # Home advantage (1)
        features.append(1.0 if self.is_home else 0.0)

        # Injury factors (2)
        features.append(1.0 - self.arsenal_injury_factor)  # Higher = better (less injury)
        features.append(self.opponent_injury_factor)  # Higher = worse for opponent

        # Bookmaker odds features (3)
        features.append(self.bookmaker_win_prob)
        features.append(self.bookmaker_draw_prob)
        features.append(self.bookmaker_loss_prob)

        return np.array(features, dtype=np.float64)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return list of feature names in order."""
        return [
            # Arsenal form (8)
            "arsenal_ppg",
            "arsenal_gpg",
            "arsenal_gcpg",
            "arsenal_win_rate",
            "arsenal_draw_rate",
            "arsenal_loss_rate",
            "arsenal_clean_sheet_rate",
            "arsenal_failed_to_score_rate",
            # Opponent form (8)
            "opponent_ppg",
            "opponent_gpg",
            "opponent_gcpg",
            "opponent_win_rate",
            "opponent_draw_rate",
            "opponent_loss_rate",
            "opponent_clean_sheet_rate",
            "opponent_failed_to_score_rate",
            # H2H (5)
            "h2h_win_rate",
            "h2h_draw_rate",
            "h2h_loss_rate",
            "h2h_gpg",
            "h2h_gcpg",
            # Home advantage (1)
            "is_home",
            # Injury factors (2)
            "arsenal_fitness",
            "opponent_injury_penalty",
            # Bookmaker odds (3)
            "bookie_win_prob",
            "bookie_draw_prob",
            "bookie_loss_prob",
        ]


@dataclass
class PredictionResult:
    """
    Result of a match prediction with confidence intervals.

    Provides win/draw/loss probabilities with uncertainty ranges
    calculated via bootstrap sampling.
    """
    win_prob: float
    draw_prob: float
    loss_prob: float

    # Confidence intervals (95%)
    win_ci_lower: float = 0.0
    win_ci_upper: float = 1.0
    draw_ci_lower: float = 0.0
    draw_ci_upper: float = 1.0
    loss_ci_lower: float = 0.0
    loss_ci_upper: float = 1.0

    # Prediction metadata
    confidence_level: float = 0.95
    model_version: str = MODEL_VERSION
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')

    @property
    def predicted_outcome(self) -> str:
        """Most likely outcome."""
        probs = {"W": self.win_prob, "D": self.draw_prob, "L": self.loss_prob}
        return max(probs, key=probs.get)

    @property
    def prediction_confidence(self) -> float:
        """Confidence in the predicted outcome."""
        return max(self.win_prob, self.draw_prob, self.loss_prob)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "probabilities": {
                "win": round(self.win_prob * 100, 2),
                "draw": round(self.draw_prob * 100, 2),
                "loss": round(self.loss_prob * 100, 2),
            },
            "confidence_intervals": {
                "win": {
                    "lower": round(self.win_ci_lower * 100, 2),
                    "upper": round(self.win_ci_upper * 100, 2),
                },
                "draw": {
                    "lower": round(self.draw_ci_lower * 100, 2),
                    "upper": round(self.draw_ci_upper * 100, 2),
                },
                "loss": {
                    "lower": round(self.loss_ci_lower * 100, 2),
                    "upper": round(self.loss_ci_upper * 100, 2),
                },
            },
            "predicted_outcome": self.predicted_outcome,
            "prediction_confidence": round(self.prediction_confidence * 100, 2),
            "confidence_level": self.confidence_level,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        """Human-readable prediction output."""
        return (
            f"Win:  {self.win_prob*100:.1f}% ({self.win_ci_lower*100:.1f}% - {self.win_ci_upper*100:.1f}%)\n"
            f"Draw: {self.draw_prob*100:.1f}% ({self.draw_ci_lower*100:.1f}% - {self.draw_ci_upper*100:.1f}%)\n"
            f"Loss: {self.loss_prob*100:.1f}% ({self.loss_ci_lower*100:.1f}% - {self.loss_ci_upper*100:.1f}%)"
        )


@dataclass
class CrossValidationResults:
    """Results from k-fold cross-validation."""
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    n_folds: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": {
                "mean": round(self.accuracy_mean, 4),
                "std": round(self.accuracy_std, 4),
            },
            "precision": {
                "mean": round(self.precision_mean, 4),
                "std": round(self.precision_std, 4),
            },
            "recall": {
                "mean": round(self.recall_mean, 4),
                "std": round(self.recall_std, 4),
            },
            "n_folds": self.n_folds,
        }

    def __str__(self) -> str:
        """Human-readable cross-validation results."""
        return (
            f"Cross-Validation Results ({self.n_folds}-fold):\n"
            f"  Accuracy:  {self.accuracy_mean:.4f} (+/- {self.accuracy_std:.4f})\n"
            f"  Precision: {self.precision_mean:.4f} (+/- {self.precision_std:.4f})\n"
            f"  Recall:    {self.recall_mean:.4f} (+/- {self.recall_std:.4f})"
        )


class MatchPredictor:
    """
    ML-based match outcome predictor for Arsenal FC.

    Uses logistic regression with softmax for multi-class classification
    (Win/Draw/Loss prediction) based on engineered features.

    Features:
    - Recent form (last 5 games) for both teams
    - Head-to-head record
    - Home/away advantage
    - Injury impact factors
    - Bookmaker odds as features

    Usage:
        # Training
        predictor = MatchPredictor()
        predictor.train(X_train, y_train)
        results = predictor.cross_validate(X, y, n_folds=5)
        predictor.save_model()

        # Prediction
        predictor = MatchPredictor.load_model()
        features = MatchFeatures(...)
        prediction = predictor.predict(features)
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        C: float = 1.0,
    ):
        """
        Initialize the match predictor.

        Args:
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for logistic regression
            C: Regularization strength (smaller = stronger regularization)
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C

        # Initialize model
        # Note: multi_class='multinomial' is now default in scikit-learn 1.7+
        self.model = LogisticRegression(
            solver='lbfgs',
            random_state=random_state,
            max_iter=max_iter,
            C=C,
        )

        # Feature scaler
        self.scaler = StandardScaler()

        # Training state
        self.is_trained = False
        self.feature_names = MatchFeatures.get_feature_names()
        self.classes_ = ['L', 'D', 'W']  # Loss, Draw, Win

        # Metadata
        self.training_timestamp: Optional[str] = None
        self.training_samples: int = 0
        self.cv_results: Optional[CrossValidationResults] = None

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _encode_outcome(self, result: str) -> int:
        """Encode result string to integer label."""
        mapping = {'L': 0, 'D': 1, 'W': 2}
        return mapping.get(result.upper(), 1)

    def _decode_outcome(self, label: int) -> str:
        """Decode integer label to result string."""
        mapping = {0: 'L', 1: 'D', 2: 'W'}
        return mapping.get(label, 'D')

    def prepare_features(
        self,
        match_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels from match data.

        Args:
            match_data: List of match dictionaries with features and results

        Returns:
            Tuple of (X features array, y labels array)
        """
        X_list = []
        y_list = []

        for match in match_data:
            # Extract features from match data
            features = self._extract_features_from_match(match)
            X_list.append(features.to_feature_vector())

            # Extract label
            result = match.get('result', 'D')
            y_list.append(self._encode_outcome(result))

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def _extract_features_from_match(
        self,
        match: Dict[str, Any]
    ) -> MatchFeatures:
        """
        Extract MatchFeatures from a match dictionary.

        Args:
            match: Dictionary containing match data

        Returns:
            MatchFeatures instance
        """
        features = MatchFeatures()

        # Arsenal form
        if 'arsenal_form' in match:
            form = match['arsenal_form']
            features.arsenal_form = FormStats(
                n_games=form.get('n_games', 5),
                wins=form.get('wins', 0),
                draws=form.get('draws', 0),
                losses=form.get('losses', 0),
                goals_scored=form.get('goals_scored', 0),
                goals_conceded=form.get('goals_conceded', 0),
                clean_sheets=form.get('clean_sheets', 0),
                failed_to_score=form.get('failed_to_score', 0),
            )

        # Opponent form
        if 'opponent_form' in match:
            form = match['opponent_form']
            features.opponent_form = FormStats(
                n_games=form.get('n_games', 5),
                wins=form.get('wins', 0),
                draws=form.get('draws', 0),
                losses=form.get('losses', 0),
                goals_scored=form.get('goals_scored', 0),
                goals_conceded=form.get('goals_conceded', 0),
                clean_sheets=form.get('clean_sheets', 0),
                failed_to_score=form.get('failed_to_score', 0),
            )

        # H2H stats
        if 'h2h' in match:
            h2h = match['h2h']
            features.h2h_stats = H2HStats(
                opponent=h2h.get('opponent', ''),
                matches_played=h2h.get('matches_played', 0),
                wins=h2h.get('wins', 0),
                draws=h2h.get('draws', 0),
                losses=h2h.get('losses', 0),
                goals_scored=h2h.get('goals_scored', 0),
                goals_conceded=h2h.get('goals_conceded', 0),
            )

        # Home advantage
        features.is_home = match.get('venue', 'home').lower() == 'home'

        # Injury factors
        features.arsenal_injury_factor = match.get('arsenal_injury_factor', 0.0)
        features.opponent_injury_factor = match.get('opponent_injury_factor', 0.0)

        # Bookmaker odds probabilities
        if 'bookmaker_odds' in match:
            odds = match['bookmaker_odds']
            features.bookmaker_win_prob = odds.get('win_prob', 0.33)
            features.bookmaker_draw_prob = odds.get('draw_prob', 0.33)
            features.bookmaker_loss_prob = odds.get('loss_prob', 0.33)

        return features

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True
    ) -> None:
        """
        Train the logistic regression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            normalize: Whether to normalize features using StandardScaler
        """
        self.logger.info(f"Training model with {X.shape[0]} samples, {X.shape[1]} features")

        # Normalize features
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Train model
        self.model.fit(X_scaled, y)

        # Update state
        self.is_trained = True
        self.training_timestamp = datetime.utcnow().isoformat() + 'Z'
        self.training_samples = X.shape[0]

        self.logger.info("Model training complete")

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> CrossValidationResults:
        """
        Perform k-fold cross-validation with multiple metrics.

        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of cross-validation folds

        Returns:
            CrossValidationResults with accuracy, precision, recall metrics
        """
        self.logger.info(f"Running {n_folds}-fold cross-validation")

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Create stratified k-fold splitter
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # Calculate metrics
        accuracy_scores = cross_val_score(
            self.model, X_scaled, y, cv=cv, scoring='accuracy'
        )

        # Custom scorers for multi-class
        precision_scorer = make_scorer(
            precision_score, average='weighted', zero_division=0
        )
        precision_scores = cross_val_score(
            self.model, X_scaled, y, cv=cv, scoring=precision_scorer
        )

        recall_scorer = make_scorer(
            recall_score, average='weighted', zero_division=0
        )
        recall_scores = cross_val_score(
            self.model, X_scaled, y, cv=cv, scoring=recall_scorer
        )

        results = CrossValidationResults(
            accuracy_mean=accuracy_scores.mean(),
            accuracy_std=accuracy_scores.std(),
            precision_mean=precision_scores.mean(),
            precision_std=precision_scores.std(),
            recall_mean=recall_scores.mean(),
            recall_std=recall_scores.std(),
            n_folds=n_folds,
        )

        self.cv_results = results
        self.logger.info(f"Cross-validation complete: Accuracy = {results.accuracy_mean:.4f}")

        return results

    def predict(
        self,
        features: Union[MatchFeatures, np.ndarray],
        n_bootstrap: int = 100
    ) -> PredictionResult:
        """
        Generate prediction with confidence intervals.

        Args:
            features: MatchFeatures instance or feature vector
            n_bootstrap: Number of bootstrap samples for confidence intervals

        Returns:
            PredictionResult with probabilities and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Convert features to array if needed
        if isinstance(features, MatchFeatures):
            X = features.to_feature_vector().reshape(1, -1)
        else:
            X = features.reshape(1, -1)

        # Normalize features
        X_scaled = self.scaler.transform(X)

        # Get base prediction probabilities
        proba = self.model.predict_proba(X_scaled)[0]

        # Map probabilities to outcomes (L=0, D=1, W=2)
        loss_prob = proba[0]
        draw_prob = proba[1]
        win_prob = proba[2]

        # Calculate confidence intervals using bootstrap
        bootstrap_probs = self._bootstrap_confidence_intervals(
            X_scaled, n_bootstrap
        )

        return PredictionResult(
            win_prob=win_prob,
            draw_prob=draw_prob,
            loss_prob=loss_prob,
            win_ci_lower=bootstrap_probs['win_lower'],
            win_ci_upper=bootstrap_probs['win_upper'],
            draw_ci_lower=bootstrap_probs['draw_lower'],
            draw_ci_upper=bootstrap_probs['draw_upper'],
            loss_ci_lower=bootstrap_probs['loss_lower'],
            loss_ci_upper=bootstrap_probs['loss_upper'],
            confidence_level=0.95,
            model_version=MODEL_VERSION,
        )

    def _bootstrap_confidence_intervals(
        self,
        X_scaled: np.ndarray,
        n_bootstrap: int = 100
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals using bootstrap sampling.

        Uses model parameter perturbation to simulate prediction uncertainty.

        Args:
            X_scaled: Scaled feature matrix
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with lower/upper bounds for each outcome
        """
        # Get base probabilities and coefficients
        base_proba = self.model.predict_proba(X_scaled)[0]
        coef = self.model.coef_.copy()
        intercept = self.model.intercept_.copy()

        bootstrap_probas = []

        # Generate bootstrap samples by perturbing model parameters
        np.random.seed(self.random_state)

        for _ in range(n_bootstrap):
            # Add noise to coefficients (proportional to their magnitude)
            noise_scale = 0.1  # 10% noise
            perturbed_coef = coef + np.random.normal(0, noise_scale, coef.shape) * np.abs(coef)

            # Calculate perturbed decision function
            decision = X_scaled @ perturbed_coef.T + intercept

            # Apply softmax to get probabilities
            exp_decision = np.exp(decision - np.max(decision))  # Numerical stability
            proba = exp_decision / exp_decision.sum()

            bootstrap_probas.append(proba[0])

        bootstrap_probas = np.array(bootstrap_probas)

        # Calculate 95% confidence intervals (2.5% - 97.5%)
        lower_percentile = 2.5
        upper_percentile = 97.5

        return {
            'loss_lower': np.percentile(bootstrap_probas[:, 0], lower_percentile),
            'loss_upper': np.percentile(bootstrap_probas[:, 0], upper_percentile),
            'draw_lower': np.percentile(bootstrap_probas[:, 1], lower_percentile),
            'draw_upper': np.percentile(bootstrap_probas[:, 1], upper_percentile),
            'win_lower': np.percentile(bootstrap_probas[:, 2], lower_percentile),
            'win_upper': np.percentile(bootstrap_probas[:, 2], upper_percentile),
        }

    def predict_from_raw_features(
        self,
        arsenal_form: Dict[str, Any],
        opponent_form: Dict[str, Any],
        h2h: Dict[str, Any],
        is_home: bool = True,
        arsenal_injury_factor: float = 0.0,
        opponent_injury_factor: float = 0.0,
        bookmaker_odds: Optional[Dict[str, float]] = None
    ) -> PredictionResult:
        """
        Convenience method to predict from raw feature dictionaries.

        Args:
            arsenal_form: Arsenal's recent form stats
            opponent_form: Opponent's recent form stats
            h2h: Head-to-head record
            is_home: Whether Arsenal is playing at home
            arsenal_injury_factor: Arsenal injury impact (0-1)
            opponent_injury_factor: Opponent injury impact (0-1)
            bookmaker_odds: Optional bookmaker probability estimates

        Returns:
            PredictionResult with probabilities and confidence intervals
        """
        # Build features
        features = MatchFeatures(
            arsenal_form=FormStats(
                n_games=arsenal_form.get('n_games', 5),
                wins=arsenal_form.get('wins', 0),
                draws=arsenal_form.get('draws', 0),
                losses=arsenal_form.get('losses', 0),
                goals_scored=arsenal_form.get('goals_scored', 0),
                goals_conceded=arsenal_form.get('goals_conceded', 0),
                clean_sheets=arsenal_form.get('clean_sheets', 0),
                failed_to_score=arsenal_form.get('failed_to_score', 0),
            ),
            opponent_form=FormStats(
                n_games=opponent_form.get('n_games', 5),
                wins=opponent_form.get('wins', 0),
                draws=opponent_form.get('draws', 0),
                losses=opponent_form.get('losses', 0),
                goals_scored=opponent_form.get('goals_scored', 0),
                goals_conceded=opponent_form.get('goals_conceded', 0),
                clean_sheets=opponent_form.get('clean_sheets', 0),
                failed_to_score=opponent_form.get('failed_to_score', 0),
            ),
            h2h_stats=H2HStats(
                opponent=h2h.get('opponent', ''),
                matches_played=h2h.get('matches_played', 0),
                wins=h2h.get('wins', 0),
                draws=h2h.get('draws', 0),
                losses=h2h.get('losses', 0),
                goals_scored=h2h.get('goals_scored', 0),
                goals_conceded=h2h.get('goals_conceded', 0),
            ),
            is_home=is_home,
            arsenal_injury_factor=arsenal_injury_factor,
            opponent_injury_factor=opponent_injury_factor,
            bookmaker_win_prob=bookmaker_odds.get('win_prob', 0.33) if bookmaker_odds else 0.33,
            bookmaker_draw_prob=bookmaker_odds.get('draw_prob', 0.33) if bookmaker_odds else 0.33,
            bookmaker_loss_prob=bookmaker_odds.get('loss_prob', 0.33) if bookmaker_odds else 0.33,
        )

        return self.predict(features)

    def save_model(
        self,
        filepath: Optional[str] = None,
        include_cv_results: bool = True
    ) -> str:
        """
        Save trained model to pickle file with metadata.

        Args:
            filepath: Output path (defaults to models/match_predictor.pkl)
            include_cv_results: Whether to include cross-validation results

        Returns:
            Path to saved model file
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = str(MODELS_DIR / "match_predictor.pkl")
        else:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Build model package with metadata
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'metadata': {
                'version': MODEL_VERSION,
                'feature_names': self.feature_names,
                'classes': self.classes_,
                'training_timestamp': self.training_timestamp,
                'training_samples': self.training_samples,
                'hyperparameters': {
                    'C': self.C,
                    'max_iter': self.max_iter,
                    'random_state': self.random_state,
                },
                'n_features': len(self.feature_names),
            }
        }

        if include_cv_results and self.cv_results:
            model_package['metadata']['cross_validation'] = self.cv_results.to_dict()

        # Save to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        self.logger.info(f"Model saved to {filepath}")

        return filepath

    @classmethod
    def load_model(cls, filepath: Optional[str] = None) -> 'MatchPredictor':
        """
        Load a trained model from pickle file.

        Args:
            filepath: Path to model file (defaults to models/match_predictor.pkl)

        Returns:
            MatchPredictor instance with loaded model
        """
        if filepath is None:
            filepath = str(MODELS_DIR / "match_predictor.pkl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model package
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        # Extract metadata
        metadata = model_package['metadata']
        hyperparams = metadata.get('hyperparameters', {})

        # Create predictor instance
        predictor = cls(
            random_state=hyperparams.get('random_state', 42),
            max_iter=hyperparams.get('max_iter', 1000),
            C=hyperparams.get('C', 1.0),
        )

        # Load trained components
        predictor.model = model_package['model']
        predictor.scaler = model_package['scaler']
        predictor.feature_names = metadata.get('feature_names', MatchFeatures.get_feature_names())
        predictor.classes_ = metadata.get('classes', ['L', 'D', 'W'])
        predictor.training_timestamp = metadata.get('training_timestamp')
        predictor.training_samples = metadata.get('training_samples', 0)
        predictor.is_trained = True

        # Load CV results if available
        if 'cross_validation' in metadata:
            cv_data = metadata['cross_validation']
            predictor.cv_results = CrossValidationResults(
                accuracy_mean=cv_data['accuracy']['mean'],
                accuracy_std=cv_data['accuracy']['std'],
                precision_mean=cv_data['precision']['mean'],
                precision_std=cv_data['precision']['std'],
                recall_mean=cv_data['recall']['mean'],
                recall_std=cv_data['recall']['std'],
                n_folds=cv_data['n_folds'],
            )

        logger.info(f"Model loaded from {filepath} (version: {metadata.get('version', 'unknown')})")

        return predictor

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.

        Returns:
            Dictionary with model details
        """
        info = {
            'is_trained': self.is_trained,
            'version': MODEL_VERSION,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'hyperparameters': {
                'C': self.C,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
            },
        }

        if self.is_trained:
            info['training_timestamp'] = self.training_timestamp
            info['training_samples'] = self.training_samples

            if self.cv_results:
                info['cross_validation'] = self.cv_results.to_dict()

        return info

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on model coefficients.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        # Average absolute coefficient across classes
        importance = np.mean(np.abs(self.model.coef_), axis=0)

        # Normalize to sum to 1
        importance = importance / importance.sum()

        return {
            name: round(imp, 4)
            for name, imp in zip(self.feature_names, importance)
        }


def generate_synthetic_training_data(
    n_samples: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for demonstration/testing.

    Creates realistic match feature data with correlated outcomes.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        Tuple of (X features, y labels)
    """
    np.random.seed(random_state)

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # Generate Arsenal form (generally good)
        arsenal_ppg = np.clip(np.random.normal(2.0, 0.5), 0.5, 3.0)
        arsenal_gpg = np.clip(np.random.normal(2.0, 0.7), 0.5, 4.0)
        arsenal_gcpg = np.clip(np.random.normal(1.0, 0.4), 0.2, 2.5)
        arsenal_wins = int(np.clip(np.random.normal(3, 1), 0, 5))
        arsenal_draws = int(np.clip(np.random.normal(1, 0.8), 0, 5 - arsenal_wins))
        arsenal_losses = 5 - arsenal_wins - arsenal_draws

        # Generate opponent form (variable)
        opponent_ppg = np.clip(np.random.normal(1.5, 0.6), 0.3, 2.8)
        opponent_gpg = np.clip(np.random.normal(1.5, 0.6), 0.3, 3.5)
        opponent_gcpg = np.clip(np.random.normal(1.3, 0.5), 0.3, 2.5)
        opponent_wins = int(np.clip(np.random.normal(2, 1.2), 0, 5))
        opponent_draws = int(np.clip(np.random.normal(1, 0.8), 0, 5 - opponent_wins))
        opponent_losses = 5 - opponent_wins - opponent_draws

        # H2H features (Arsenal historically strong)
        h2h_win_rate = np.clip(np.random.normal(0.5, 0.2), 0.1, 0.9)
        h2h_draw_rate = np.clip(np.random.normal(0.25, 0.1), 0.05, 0.4)
        h2h_loss_rate = 1.0 - h2h_win_rate - h2h_draw_rate
        h2h_gpg = np.clip(np.random.normal(1.8, 0.5), 0.5, 3.5)
        h2h_gcpg = np.clip(np.random.normal(1.2, 0.4), 0.3, 2.5)

        # Home advantage
        is_home = np.random.random() > 0.5

        # Injury factors
        arsenal_fitness = np.clip(np.random.normal(0.9, 0.1), 0.5, 1.0)
        opponent_injury = np.clip(np.random.normal(0.1, 0.1), 0.0, 0.5)

        # Bookmaker odds (correlated with features)
        form_advantage = (arsenal_ppg - opponent_ppg) / 3.0
        home_bonus = 0.1 if is_home else -0.1
        base_win_prob = 0.4 + form_advantage * 0.2 + home_bonus

        bookie_win = np.clip(base_win_prob + np.random.normal(0, 0.05), 0.15, 0.7)
        bookie_draw = np.clip(0.25 + np.random.normal(0, 0.05), 0.15, 0.35)
        bookie_loss = 1.0 - bookie_win - bookie_draw

        # Build feature vector
        features = [
            # Arsenal form (8)
            arsenal_ppg / 3.0,
            arsenal_gpg / 4.0,
            arsenal_gcpg / 3.0,
            arsenal_wins / 5.0,
            arsenal_draws / 5.0,
            arsenal_losses / 5.0,
            np.random.random() * 0.6,  # clean sheet rate
            np.random.random() * 0.3,  # failed to score rate
            # Opponent form (8)
            opponent_ppg / 3.0,
            opponent_gpg / 4.0,
            opponent_gcpg / 3.0,
            opponent_wins / 5.0,
            opponent_draws / 5.0,
            opponent_losses / 5.0,
            np.random.random() * 0.4,  # clean sheet rate
            np.random.random() * 0.35,  # failed to score rate
            # H2H (5)
            h2h_win_rate,
            h2h_draw_rate,
            h2h_loss_rate,
            h2h_gpg / 4.0,
            h2h_gcpg / 3.0,
            # Home (1)
            1.0 if is_home else 0.0,
            # Injury (2)
            arsenal_fitness,
            opponent_injury,
            # Bookmaker (3)
            bookie_win,
            bookie_draw,
            bookie_loss,
        ]

        X_list.append(features)

        # Generate outcome based on features (with some randomness)
        win_score = (
            form_advantage * 2.0 +
            (0.15 if is_home else -0.05) +
            (arsenal_fitness - 0.9) * 0.5 +
            opponent_injury * 0.3 +
            (h2h_win_rate - 0.5) * 0.5 +
            np.random.normal(0, 0.3)
        )

        # Determine outcome
        if win_score > 0.3:
            outcome = 2  # Win
        elif win_score > -0.2:
            outcome = 1  # Draw
        else:
            outcome = 0  # Loss

        y_list.append(outcome)

    return np.array(X_list), np.array(y_list)


def example_usage():
    """Demonstrate usage of the ML predictor."""

    print("=" * 60)
    print("Arsenal Intelligence Brief - ML Match Predictor Demo")
    print("=" * 60)

    # Generate synthetic training data
    print("\n--- Generating Training Data ---")
    X, y = generate_synthetic_training_data(n_samples=300, random_state=42)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: L={sum(y==0)}, D={sum(y==1)}, W={sum(y==2)}")

    # Create and train predictor
    print("\n--- Training Model ---")
    predictor = MatchPredictor(random_state=42)

    # Cross-validation
    cv_results = predictor.cross_validate(X, y, n_folds=5)
    print(cv_results)

    # Train on full data
    predictor.train(X, y)

    # Save model
    print("\n--- Saving Model ---")
    model_path = predictor.save_model()
    print(f"Model saved to: {model_path}")

    # Load model
    print("\n--- Loading Model ---")
    loaded_predictor = MatchPredictor.load_model(model_path)
    print(f"Model loaded successfully (version: {MODEL_VERSION})")

    # Make a prediction
    print("\n--- Example Prediction: Arsenal vs Chelsea (Home) ---")

    # Create example features
    arsenal_form = {
        'n_games': 5,
        'wins': 4,
        'draws': 1,
        'losses': 0,
        'goals_scored': 12,
        'goals_conceded': 3,
        'clean_sheets': 3,
        'failed_to_score': 0,
    }

    opponent_form = {
        'n_games': 5,
        'wins': 2,
        'draws': 2,
        'losses': 1,
        'goals_scored': 8,
        'goals_conceded': 6,
        'clean_sheets': 1,
        'failed_to_score': 1,
    }

    h2h = {
        'opponent': 'Chelsea',
        'matches_played': 10,
        'wins': 5,
        'draws': 3,
        'losses': 2,
        'goals_scored': 15,
        'goals_conceded': 10,
    }

    bookmaker_odds = {
        'win_prob': 0.45,
        'draw_prob': 0.28,
        'loss_prob': 0.27,
    }

    prediction = loaded_predictor.predict_from_raw_features(
        arsenal_form=arsenal_form,
        opponent_form=opponent_form,
        h2h=h2h,
        is_home=True,
        arsenal_injury_factor=0.1,
        opponent_injury_factor=0.15,
        bookmaker_odds=bookmaker_odds,
    )

    print(f"\nPrediction Results:")
    print(prediction)
    print(f"\nPredicted Outcome: {prediction.predicted_outcome}")
    print(f"Confidence: {prediction.prediction_confidence*100:.1f}%")

    # Show feature importance
    print("\n--- Feature Importance (Top 10) ---")
    importance = loaded_predictor.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in sorted_features:
        print(f"  {name}: {imp:.4f}")

    # Output as JSON
    print("\n--- JSON Output ---")
    print(json.dumps(prediction.to_dict(), indent=2))


if __name__ == "__main__":
    example_usage()
