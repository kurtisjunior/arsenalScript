#!/usr/bin/env python3
"""
Chart Generator for Arsenal Intelligence Brief

This module generates visualizations as base64-encoded images for embedding
in HTML reports. Uses matplotlib for chart generation with a professional
Arsenal-themed color scheme.

Charts available:
- Odds comparison bar chart
- Win probability gauge/meter
- Sentiment timeline
- Value opportunity comparison

Task: arsenalScript-vqp.37-41 - Reporting module
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Wedge, Circle
    import matplotlib.colors as mcolors
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Chart generation will be disabled.")


# Arsenal color scheme
ARSENAL_COLORS = {
    'red': '#EF0107',           # Arsenal primary red
    'gold': '#9C824A',          # Arsenal gold
    'navy': '#063672',          # Arsenal navy blue
    'white': '#FFFFFF',
    'light_gray': '#F5F5F5',
    'dark_gray': '#333333',
    'green': '#28A745',         # Positive/value
    'amber': '#FFC107',         # Warning/neutral
    'danger': '#DC3545',        # Negative/loss
}


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: int = 8
    height: int = 5
    dpi: int = 100
    font_family: str = 'Arial'
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    background_color: str = ARSENAL_COLORS['white']
    primary_color: str = ARSENAL_COLORS['red']
    secondary_color: str = ARSENAL_COLORS['navy']


class ChartGenerator:
    """
    Generator for Arsenal Intelligence Brief visualizations.

    Creates publication-quality charts as base64-encoded PNG images
    suitable for embedding in HTML email reports.

    Usage:
        generator = ChartGenerator()

        # Generate odds comparison chart
        odds_chart = generator.create_odds_comparison_chart(
            bookmaker_odds=[
                {"bookmaker": "Bet365", "home_win": 2.10, "draw": 3.40, "away_win": 3.50},
                {"bookmaker": "William Hill", "home_win": 2.15, "draw": 3.30, "away_win": 3.45},
            ],
            home_team="Arsenal",
            away_team="Chelsea"
        )

        # Embed in HTML
        html_img = f'<img src="data:image/png;base64,{odds_chart}" />'
    """

    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize the chart generator.

        Args:
            config: Optional ChartConfig for customization
        """
        self.config = config or ChartConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not available. Install with: pip install matplotlib"
            )
            return False
        return True

    def _fig_to_base64(self, fig: Any) -> str:
        """
        Convert matplotlib figure to base64 string.

        Args:
            fig: matplotlib Figure object

        Returns:
            Base64-encoded PNG string
        """
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=self.config.dpi,
            bbox_inches='tight',
            facecolor=self.config.background_color,
            edgecolor='none'
        )
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_base64

    def _setup_style(self, ax: Any) -> None:
        """Apply consistent styling to axes."""
        ax.set_facecolor(self.config.background_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(ARSENAL_COLORS['dark_gray'])
        ax.spines['bottom'].set_color(ARSENAL_COLORS['dark_gray'])
        ax.tick_params(colors=ARSENAL_COLORS['dark_gray'], labelsize=self.config.tick_size)

    def create_odds_comparison_chart(
        self,
        bookmaker_odds: List[Dict[str, Any]],
        home_team: str = "Arsenal",
        away_team: str = "Opponent",
        title: Optional[str] = None
    ) -> str:
        """
        Create a grouped bar chart comparing odds across bookmakers.

        Args:
            bookmaker_odds: List of dicts with bookmaker, home_win, draw, away_win
            home_team: Name of home team
            away_team: Name of away team
            title: Optional chart title

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Odds Comparison (matplotlib required)")

        if not bookmaker_odds:
            return self._placeholder_image("No odds data available")

        # Extract data
        bookmakers = [od.get('bookmaker', 'Unknown') for od in bookmaker_odds]
        home_odds = [od.get('home_win', 0) for od in bookmaker_odds]
        draw_odds = [od.get('draw', 0) for od in bookmaker_odds]
        away_odds = [od.get('away_win', 0) for od in bookmaker_odds]

        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height))
        self._setup_style(ax)

        # Bar positions
        x = np.arange(len(bookmakers))
        width = 0.25

        # Create bars
        bars1 = ax.bar(x - width, home_odds, width, label=f'{home_team} Win',
                       color=ARSENAL_COLORS['red'], edgecolor='white')
        bars2 = ax.bar(x, draw_odds, width, label='Draw',
                       color=ARSENAL_COLORS['gold'], edgecolor='white')
        bars3 = ax.bar(x + width, away_odds, width, label=f'{away_team} Win',
                       color=ARSENAL_COLORS['navy'], edgecolor='white')

        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=self.config.tick_size - 1,
                           color=ARSENAL_COLORS['dark_gray'])

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)

        # Labels and title
        ax.set_ylabel('Decimal Odds', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])
        ax.set_xlabel('Bookmaker', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])
        ax.set_xticks(x)
        ax.set_xticklabels(bookmakers, fontsize=self.config.tick_size)

        chart_title = title or f'Odds Comparison: {home_team} vs {away_team}'
        ax.set_title(chart_title, fontsize=self.config.title_size,
                    fontweight='bold', color=ARSENAL_COLORS['dark_gray'],
                    pad=15)

        # Legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.9,
                 fontsize=self.config.tick_size)

        # Set y-axis to start at 1 (minimum decimal odds)
        ax.set_ylim(bottom=1)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def create_win_probability_gauge(
        self,
        win_prob: float,
        draw_prob: float,
        loss_prob: float,
        confidence: Optional[float] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Create a semi-circular gauge showing win/draw/loss probabilities.

        Args:
            win_prob: Win probability (0-1)
            draw_prob: Draw probability (0-1)
            loss_prob: Loss probability (0-1)
            confidence: Optional model confidence level (0-1)
            title: Optional chart title

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Win Probability Gauge (matplotlib required)")

        # Validate probabilities
        total = win_prob + draw_prob + loss_prob
        if abs(total - 1.0) > 0.01:
            # Normalize if not summing to 1
            win_prob = win_prob / total
            draw_prob = draw_prob / total
            loss_prob = loss_prob / total

        # Create figure with white background
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height))
        ax.set_aspect('equal')
        ax.axis('off')

        # Semi-circle gauge parameters
        center = (0.5, 0.3)
        radius = 0.35

        # Convert probabilities to angles (semi-circle = 180 degrees)
        # Order: Loss (left), Draw (center), Win (right)
        angles = [loss_prob * 180, draw_prob * 180, win_prob * 180]
        colors = [ARSENAL_COLORS['danger'], ARSENAL_COLORS['amber'], ARSENAL_COLORS['green']]
        labels = [f'Loss\n{loss_prob*100:.1f}%', f'Draw\n{draw_prob*100:.1f}%', f'Win\n{win_prob*100:.1f}%']

        # Draw the gauge segments
        start_angle = 180  # Start from left (180 degrees)

        for angle, color, label in zip(angles, colors, labels):
            end_angle = start_angle - angle  # Move counter-clockwise

            # Create wedge
            wedge = Wedge(
                center, radius, end_angle, start_angle,
                facecolor=color, edgecolor='white', linewidth=2
            )
            ax.add_patch(wedge)

            # Calculate label position
            mid_angle = np.radians((start_angle + end_angle) / 2)
            label_x = center[0] + (radius * 0.65) * np.cos(mid_angle)
            label_y = center[1] + (radius * 0.65) * np.sin(mid_angle)

            # Add percentage label
            ax.text(label_x, label_y, label, ha='center', va='center',
                   fontsize=self.config.label_size, fontweight='bold',
                   color='white')

            start_angle = end_angle

        # Add center circle (for aesthetics)
        inner_circle = Circle(center, radius * 0.3, facecolor='white',
                             edgecolor=ARSENAL_COLORS['dark_gray'], linewidth=2)
        ax.add_patch(inner_circle)

        # Determine predicted outcome
        max_prob = max(win_prob, draw_prob, loss_prob)
        if max_prob == win_prob:
            predicted = 'WIN'
            pred_color = ARSENAL_COLORS['green']
        elif max_prob == draw_prob:
            predicted = 'DRAW'
            pred_color = ARSENAL_COLORS['amber']
        else:
            predicted = 'LOSS'
            pred_color = ARSENAL_COLORS['danger']

        # Add predicted outcome in center
        ax.text(center[0], center[1], predicted, ha='center', va='center',
               fontsize=self.config.title_size, fontweight='bold',
               color=pred_color)

        # Add confidence indicator if provided
        if confidence is not None:
            ax.text(center[0], center[1] - radius - 0.08,
                   f'Model Confidence: {confidence*100:.1f}%',
                   ha='center', va='top',
                   fontsize=self.config.tick_size,
                   color=ARSENAL_COLORS['dark_gray'])

        # Title
        chart_title = title or 'Match Outcome Probability'
        ax.text(center[0], center[1] + radius + 0.12, chart_title,
               ha='center', va='bottom',
               fontsize=self.config.title_size, fontweight='bold',
               color=ARSENAL_COLORS['dark_gray'])

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return self._fig_to_base64(fig)

    def create_sentiment_timeline(
        self,
        sentiment_data: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Create a line chart showing sentiment over time.

        Args:
            sentiment_data: List of dicts with date, score, source (optional)
            title: Optional chart title

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Sentiment Timeline (matplotlib required)")

        if not sentiment_data:
            return self._placeholder_image("No sentiment data available")

        # Sort by date
        sorted_data = sorted(sentiment_data, key=lambda x: x.get('date', ''))

        # Extract data
        dates = []
        scores = []
        sources = []

        for item in sorted_data:
            date_str = item.get('date', item.get('publish_date', ''))
            if date_str:
                try:
                    # Try to parse date
                    if 'T' in date_str:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    dates.append(dt)
                    scores.append(item.get('score', item.get('combined_score', 0)))
                    sources.append(item.get('source', 'Unknown'))
                except (ValueError, TypeError):
                    continue

        if not dates:
            return self._placeholder_image("Could not parse sentiment dates")

        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height))
        self._setup_style(ax)

        # Plot sentiment line
        ax.plot(dates, scores, color=ARSENAL_COLORS['red'], linewidth=2.5,
               marker='o', markersize=8, markerfacecolor=ARSENAL_COLORS['gold'],
               markeredgecolor=ARSENAL_COLORS['red'], markeredgewidth=2)

        # Add horizontal reference lines
        ax.axhline(y=0, color=ARSENAL_COLORS['dark_gray'], linestyle='--',
                  alpha=0.3, linewidth=1)
        ax.axhline(y=0.3, color=ARSENAL_COLORS['green'], linestyle=':',
                  alpha=0.5, linewidth=1, label='Positive threshold')
        ax.axhline(y=-0.3, color=ARSENAL_COLORS['danger'], linestyle=':',
                  alpha=0.5, linewidth=1, label='Negative threshold')

        # Fill areas
        ax.fill_between(dates, scores, 0, where=[s > 0 for s in scores],
                       color=ARSENAL_COLORS['green'], alpha=0.2)
        ax.fill_between(dates, scores, 0, where=[s < 0 for s in scores],
                       color=ARSENAL_COLORS['danger'], alpha=0.2)

        # Labels
        ax.set_ylabel('Sentiment Score', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])
        ax.set_xlabel('Date', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])

        # Format x-axis dates
        fig.autofmt_xdate()

        # Set y-axis limits
        ax.set_ylim(-1, 1)

        # Title
        chart_title = title or 'Sentiment Timeline'
        ax.set_title(chart_title, fontsize=self.config.title_size,
                    fontweight='bold', color=ARSENAL_COLORS['dark_gray'],
                    pad=15)

        # Add legend for reference lines
        ax.legend(loc='upper left', fontsize=self.config.tick_size - 1,
                 frameon=True, framealpha=0.9)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def create_value_opportunities_chart(
        self,
        opportunities: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Create a horizontal bar chart showing value betting opportunities.

        Args:
            opportunities: List of dicts with market, expected_value_percentage, edge_percentage
            title: Optional chart title

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Value Opportunities (matplotlib required)")

        if not opportunities:
            return self._placeholder_image("No value opportunities identified")

        # Extract data and sort by EV
        sorted_opps = sorted(opportunities,
                            key=lambda x: x.get('expected_value_percentage', 0),
                            reverse=True)[:6]  # Top 6 opportunities

        markets = [op.get('market', 'unknown').replace('_', ' ').title()
                  for op in sorted_opps]
        evs = [op.get('expected_value_percentage', 0) for op in sorted_opps]
        edges = [op.get('edge_percentage', 0) for op in sorted_opps]
        is_high_value = [op.get('is_high_value', False) for op in sorted_opps]

        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height))
        self._setup_style(ax)

        # Bar positions
        y_pos = np.arange(len(markets))

        # Color bars based on high value status
        colors = [ARSENAL_COLORS['green'] if hv else ARSENAL_COLORS['navy']
                 for hv in is_high_value]

        # Create horizontal bars
        bars = ax.barh(y_pos, evs, color=colors, edgecolor='white', height=0.6)

        # Add value labels
        for i, (bar, ev, edge) in enumerate(zip(bars, evs, edges)):
            width = bar.get_width()
            label = f'EV: {ev:.1f}% | Edge: {edge:.1f}%'

            # Position label inside or outside bar
            if width > 3:
                ax.text(width - 0.3, bar.get_y() + bar.get_height()/2,
                       label, ha='right', va='center',
                       fontsize=self.config.tick_size - 1,
                       color='white', fontweight='bold')
            else:
                ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center',
                       fontsize=self.config.tick_size - 1,
                       color=ARSENAL_COLORS['dark_gray'])

        # Labels
        ax.set_xlabel('Expected Value (%)', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(markets, fontsize=self.config.label_size)

        # Add 5% threshold line
        ax.axvline(x=5, color=ARSENAL_COLORS['gold'], linestyle='--',
                  linewidth=2, label='High Value Threshold (5%)')

        # Title
        chart_title = title or 'Value Betting Opportunities'
        ax.set_title(chart_title, fontsize=self.config.title_size,
                    fontweight='bold', color=ARSENAL_COLORS['dark_gray'],
                    pad=15)

        # Legend
        high_value_patch = mpatches.Patch(color=ARSENAL_COLORS['green'],
                                         label='High Value (EV > 5%)')
        standard_patch = mpatches.Patch(color=ARSENAL_COLORS['navy'],
                                       label='Standard Value')
        ax.legend(handles=[high_value_patch, standard_patch],
                 loc='lower right', fontsize=self.config.tick_size - 1,
                 frameon=True, framealpha=0.9)

        # Set x-axis to start at 0
        ax.set_xlim(left=0)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def create_probability_comparison_chart(
        self,
        ml_probs: Dict[str, float],
        bookmaker_probs: Dict[str, float],
        title: Optional[str] = None
    ) -> str:
        """
        Create a side-by-side bar chart comparing ML vs bookmaker probabilities.

        Args:
            ml_probs: Dict with win, draw, loss ML probabilities
            bookmaker_probs: Dict with win, draw, loss implied probabilities
            title: Optional chart title

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._placeholder_image("Probability Comparison (matplotlib required)")

        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.width, self.config.height))
        self._setup_style(ax)

        # Data
        outcomes = ['Win', 'Draw', 'Loss']
        ml_values = [
            ml_probs.get('win', ml_probs.get('win_prob', 0)) * 100,
            ml_probs.get('draw', ml_probs.get('draw_prob', 0)) * 100,
            ml_probs.get('loss', ml_probs.get('loss_prob', 0)) * 100
        ]
        bookie_values = [
            bookmaker_probs.get('win', bookmaker_probs.get('home_win', 0)) * 100,
            bookmaker_probs.get('draw', 0) * 100,
            bookmaker_probs.get('loss', bookmaker_probs.get('away_win', 0)) * 100
        ]

        # Bar positions
        x = np.arange(len(outcomes))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width/2, ml_values, width, label='ML Model',
                      color=ARSENAL_COLORS['red'], edgecolor='white')
        bars2 = ax.bar(x + width/2, bookie_values, width, label='Bookmaker Implied',
                      color=ARSENAL_COLORS['navy'], edgecolor='white')

        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=self.config.tick_size,
                           color=ARSENAL_COLORS['dark_gray'])

        add_labels(bars1)
        add_labels(bars2)

        # Labels
        ax.set_ylabel('Probability (%)', fontsize=self.config.label_size,
                     color=ARSENAL_COLORS['dark_gray'])
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes, fontsize=self.config.label_size)

        # Title
        chart_title = title or 'ML Model vs Bookmaker Probabilities'
        ax.set_title(chart_title, fontsize=self.config.title_size,
                    fontweight='bold', color=ARSENAL_COLORS['dark_gray'],
                    pad=15)

        # Legend
        ax.legend(loc='upper right', fontsize=self.config.tick_size,
                 frameon=True, framealpha=0.9)

        # Set y-axis limits
        ax.set_ylim(0, 100)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _placeholder_image(self, message: str) -> str:
        """
        Create a simple placeholder image with a message.

        Used when matplotlib is not available or data is missing.

        Args:
            message: Message to display in placeholder

        Returns:
            Base64-encoded PNG image string
        """
        if not MATPLOTLIB_AVAILABLE:
            # Return a minimal SVG as fallback
            svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">
                <rect width="100%" height="100%" fill="{ARSENAL_COLORS['light_gray']}"/>
                <text x="50%" y="50%" text-anchor="middle" fill="{ARSENAL_COLORS['dark_gray']}"
                      font-family="Arial" font-size="14">{message}</text>
            </svg>'''
            return base64.b64encode(svg.encode()).decode('utf-8')

        # Create simple placeholder with matplotlib
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha='center', va='center',
               fontsize=12, color=ARSENAL_COLORS['dark_gray'],
               transform=ax.transAxes)
        ax.set_facecolor(ARSENAL_COLORS['light_gray'])
        ax.axis('off')

        return self._fig_to_base64(fig)


# ==========================================================================
# DEMO / TEST
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Arsenal Intelligence Brief - Chart Generator Demo")
    print("=" * 60)

    if not MATPLOTLIB_AVAILABLE:
        print("\nmatplotlib not installed. Install with: pip install matplotlib")
        exit(1)

    generator = ChartGenerator()

    # Demo 1: Odds comparison chart
    print("\n--- Generating Odds Comparison Chart ---")
    odds_data = [
        {"bookmaker": "Bet365", "home_win": 2.10, "draw": 3.40, "away_win": 3.50},
        {"bookmaker": "William Hill", "home_win": 2.15, "draw": 3.30, "away_win": 3.45},
        {"bookmaker": "Paddy Power", "home_win": 2.08, "draw": 3.50, "away_win": 3.40},
    ]
    odds_chart = generator.create_odds_comparison_chart(
        odds_data, "Arsenal", "Chelsea"
    )
    print(f"Odds chart generated: {len(odds_chart)} bytes base64")

    # Demo 2: Win probability gauge
    print("\n--- Generating Win Probability Gauge ---")
    prob_gauge = generator.create_win_probability_gauge(
        win_prob=0.52, draw_prob=0.26, loss_prob=0.22, confidence=0.85
    )
    print(f"Probability gauge generated: {len(prob_gauge)} bytes base64")

    # Demo 3: Sentiment timeline
    print("\n--- Generating Sentiment Timeline ---")
    sentiment_data = [
        {"date": "2026-01-14", "score": 0.3, "source": "BBC Sport"},
        {"date": "2026-01-15", "score": 0.5, "source": "Arsenal.com"},
        {"date": "2026-01-16", "score": 0.2, "source": "The Guardian"},
        {"date": "2026-01-17", "score": -0.1, "source": "Sky Sports"},
        {"date": "2026-01-18", "score": 0.4, "source": "Arsenal.com"},
    ]
    sentiment_chart = generator.create_sentiment_timeline(sentiment_data)
    print(f"Sentiment timeline generated: {len(sentiment_chart)} bytes base64")

    # Demo 4: Value opportunities
    print("\n--- Generating Value Opportunities Chart ---")
    opportunities = [
        {"market": "home_win", "expected_value_percentage": 9.2,
         "edge_percentage": 4.5, "is_high_value": True},
        {"market": "over_2_5", "expected_value_percentage": 7.3,
         "edge_percentage": 3.8, "is_high_value": True},
        {"market": "draw", "expected_value_percentage": 3.5,
         "edge_percentage": 1.8, "is_high_value": False},
    ]
    value_chart = generator.create_value_opportunities_chart(opportunities)
    print(f"Value chart generated: {len(value_chart)} bytes base64")

    # Demo 5: Probability comparison
    print("\n--- Generating Probability Comparison Chart ---")
    ml_probs = {"win": 0.52, "draw": 0.26, "loss": 0.22}
    bookie_probs = {"win": 0.48, "draw": 0.29, "loss": 0.23}
    prob_chart = generator.create_probability_comparison_chart(ml_probs, bookie_probs)
    print(f"Probability comparison generated: {len(prob_chart)} bytes base64")

    print("\n" + "=" * 60)
    print("Demo complete. All charts generated as base64 PNG images.")
