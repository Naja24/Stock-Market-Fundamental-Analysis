"""
Indian Stock Market Analysis Platform
--------------------------------------
A comprehensive stock analysis tool featuring:
- Technical Analysis with AI predictions
- Fundamental Analysis
- Sentiment Analysis

Author: BTP Project
Version: 2.0
"""

__version__ = "2.0.0"
__author__ = "BTP Project Team"

from .technical import display_technical_analysis
from .fundamental import display_fundamental_analysis
from .sentiment import display_sentiment_analysis

__all__ = [
    'display_technical_analysis',
    'display_fundamental_analysis',
    'display_sentiment_analysis'
]