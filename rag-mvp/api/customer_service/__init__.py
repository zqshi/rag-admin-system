"""
客服RAG系统核心模块
Customer Service RAG System Core Module
"""

from .intent.classifier import CustomerIntentClassifier
from .conversation.manager import ConversationManager
# from .safety.generator import SafeCustomerServiceGenerator
# from .escalation.engine import EscalationEngine

__all__ = [
    'CustomerIntentClassifier',
    'ConversationManager'
    # 'SafeCustomerServiceGenerator',
    # 'EscalationEngine'
]

# 客服系统配置
CUSTOMER_SERVICE_CONFIG = {
    "response_time_sla": {
        "urgent": 60,      # 1分钟
        "high": 180,       # 3分钟
        "medium": 300,     # 5分钟
        "low": 600         # 10分钟
    },
    "confidence_threshold": {
        "auto_response": 0.85,
        "human_escalation": 0.6,
        "clarification_needed": 0.4
    },
    "safety_filters": {
        "enable_content_filter": True,
        "enable_sentiment_check": True,
        "enable_compliance_check": True
    }
}