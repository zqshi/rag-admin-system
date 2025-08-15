"""
意图识别模块
Intent Recognition Module
"""

from .classifier import CustomerIntentClassifier, IntentCategory, Priority, IntentResult

__all__ = ['CustomerIntentClassifier', 'IntentCategory', 'Priority', 'IntentResult']