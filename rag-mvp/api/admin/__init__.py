"""
管理后台模块
Admin Dashboard Module

提供企业级的客服系统管理能力
"""

from .document_manager import DocumentManager
# from .config_manager import ConfigManager
# from .analytics_manager import AnalyticsManager
# from .conversation_monitor import ConversationMonitor

__all__ = [
    'DocumentManager'
    # 'ConfigManager', 
    # 'AnalyticsManager',
    # 'ConversationMonitor'
]

# 管理后台配置
ADMIN_CONFIG = {
    "upload_limits": {
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "allowed_extensions": [".pdf", ".txt", ".md", ".doc", ".docx"],
        "max_files_per_batch": 10
    },
    "security": {
        "admin_session_timeout": 3600,  # 1小时
        "max_login_attempts": 5,
        "password_min_length": 8
    },
    "monitoring": {
        "metrics_retention_days": 30,
        "real_time_update_interval": 5,  # 5秒
        "alert_thresholds": {
            "response_time_ms": 3000,
            "error_rate_percent": 5,
            "memory_usage_percent": 80
        }
    }
}