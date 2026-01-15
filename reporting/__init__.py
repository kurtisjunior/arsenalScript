#!/usr/bin/env python3
"""
Reporting Module for Arsenal Intelligence Brief

This module provides report generation capabilities including:
- ReportBuilder: Main report generator aggregating all data sources
- ChartGenerator: Creates visualizations as base64 images
- HTML and Markdown output via Jinja2 templates

Usage:
    from reporting import ReportBuilder, ChartGenerator

    builder = ReportBuilder()
    report = builder.build_report(
        match_id="20260120_ARS_CHE",
        odds_data=odds_dict,
        ml_prediction=prediction_dict,
        sentiment_data=sentiment_dict,
        value_analysis=value_dict,
        news_data=news_dict
    )

    # Generate HTML output
    html = report.to_html()

    # Generate Markdown output
    markdown = report.to_markdown()
"""

from .report_builder import (
    ReportBuilder,
    IntelligenceBrief,
    ReportSection,
    DataCompleteness,
)

from .chart_generator import (
    ChartGenerator,
    ChartConfig,
)

__all__ = [
    'ReportBuilder',
    'IntelligenceBrief',
    'ReportSection',
    'DataCompleteness',
    'ChartGenerator',
    'ChartConfig',
]

__version__ = '1.0.0'
