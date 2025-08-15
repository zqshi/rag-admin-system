"""
客服意图识别器 - Customer Service Intent Classifier
这是客服系统的大脑，负责理解客户真正想要什么
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class IntentCategory(Enum):
    """客服意图分类"""
    TECHNICAL_SUPPORT = "technical_support"      # 技术支持
    BILLING_INQUIRY = "billing_inquiry"          # 账单咨询
    COMPLAINT = "complaint"                       # 投诉抱怨
    PRODUCT_INFO = "product_info"                 # 产品信息
    REFUND_REQUEST = "refund_request"             # 退款请求
    ORDER_STATUS = "order_status"                 # 订单状态
    ACCOUNT_ISSUE = "account_issue"               # 账号问题
    GENERAL_INQUIRY = "general_inquiry"           # 一般咨询
    GREETING = "greeting"                         # 打招呼
    EMERGENCY = "emergency"                       # 紧急情况

class Priority(Enum):
    """优先级"""
    URGENT = "urgent"      # 紧急 - 1分钟内响应
    HIGH = "high"          # 高 - 3分钟内响应
    MEDIUM = "medium"      # 中 - 5分钟内响应
    LOW = "low"            # 低 - 10分钟内响应

@dataclass
class IntentResult:
    """意图识别结果"""
    category: IntentCategory
    confidence: float
    priority: Priority
    keywords: List[str]
    emotion_indicators: List[str]
    escalation_needed: bool
    sla_seconds: int

class CustomerIntentClassifier:
    """客服意图识别器"""
    
    def __init__(self):
        """初始化意图识别器"""
        self.intent_keywords = self._load_intent_keywords()
        self.emotion_keywords = self._load_emotion_keywords()
        self.priority_rules = self._load_priority_rules()
        
    def _load_intent_keywords(self) -> Dict[IntentCategory, List[str]]:
        """加载意图关键词库"""
        return {
            IntentCategory.TECHNICAL_SUPPORT: [
                "不能用", "打不开", "登录不了", "卡住了", "错误", "bug", "故障", 
                "连接失败", "加载不出", "闪退", "崩溃", "无法访问", "技术问题",
                "系统异常", "网络错误", "服务器", "数据库", "API", "接口"
            ],
            IntentCategory.BILLING_INQUIRY: [
                "账单", "扣费", "收费", "价格", "费用", "付款", "支付", "发票", 
                "金额", "余额", "充值", "续费", "套餐", "优惠", "折扣", "免费",
                "欠费", "到期", "计费", "结算"
            ],
            IntentCategory.COMPLAINT: [
                "投诉", "抱怨", "不满意", "差评", "服务态度", "解决不了", "推诿",
                "敷衍", "不负责", "欺骗", "虚假", "误导", "坑人", "黑心",
                "垃圾", "太差", "恶心", "愤怒", "气死", "受不了"
            ],
            IntentCategory.REFUND_REQUEST: [
                "退款", "退钱", "退费", "撤销", "取消订单", "不想要", "退订",
                "退货", "返还", "追回", "申请退款", "退款流程", "退款时间"
            ],
            IntentCategory.ORDER_STATUS: [
                "订单", "发货", "物流", "快递", "配送", "到货", "签收", "运输",
                "跟踪", "查询状态", "进度", "什么时候", "多久", "预计时间"
            ],
            IntentCategory.ACCOUNT_ISSUE: [
                "账号", "密码", "登录", "注册", "身份验证", "验证码", "绑定",
                "解绑", "实名", "权限", "封号", "解封", "找回", "重置"
            ],
            IntentCategory.PRODUCT_INFO: [
                "功能", "怎么用", "如何", "教程", "使用方法", "操作步骤",
                "介绍", "详情", "参数", "规格", "配置", "兼容", "支持"
            ],
            IntentCategory.GREETING: [
                "你好", "您好", "hello", "hi", "在吗", "客服", "人工", "咨询"
            ],
            IntentCategory.EMERGENCY: [
                "紧急", "急", "马上", "立即", "火急", "十万火急", "救命",
                "重要", "严重", "危险", "损失", "影响业务", "无法工作"
            ]
        }
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """加载情感关键词"""
        return {
            "angry": ["生气", "愤怒", "气死", "火大", "受不了", "忍无可忍", "太过分"],
            "frustrated": ["郁闷", "无语", "崩溃", "绝望", "没办法", "搞不懂", "头疼"],
            "urgent": ["急", "马上", "立即", "赶紧", "快点", "等不及", "来不及"],
            "polite": ["请", "麻烦", "谢谢", "感谢", "辛苦", "打扰", "不好意思"],
            "satisfied": ["满意", "不错", "很好", "棒", "赞", "给力", "完美"]
        }
    
    def _load_priority_rules(self) -> Dict[IntentCategory, Priority]:
        """加载优先级规则"""
        return {
            IntentCategory.EMERGENCY: Priority.URGENT,
            IntentCategory.COMPLAINT: Priority.URGENT,
            IntentCategory.REFUND_REQUEST: Priority.HIGH,
            IntentCategory.BILLING_INQUIRY: Priority.HIGH,
            IntentCategory.TECHNICAL_SUPPORT: Priority.MEDIUM,
            IntentCategory.ORDER_STATUS: Priority.MEDIUM,
            IntentCategory.ACCOUNT_ISSUE: Priority.MEDIUM,
            IntentCategory.PRODUCT_INFO: Priority.LOW,
            IntentCategory.GENERAL_INQUIRY: Priority.LOW,
            IntentCategory.GREETING: Priority.LOW
        }
    
    async def classify(self, query: str, context: Optional[Dict] = None) -> IntentResult:
        """
        分类客户意图
        
        Args:
            query: 客户查询文本
            context: 对话上下文
        
        Returns:
            IntentResult: 意图识别结果
        """
        query_lower = query.lower()
        
        # 1. 关键词匹配评分
        keyword_scores = await self._calculate_keyword_scores(query_lower)
        
        # 2. 情感分析
        emotion_indicators = await self._analyze_emotion(query_lower)
        
        # 3. 确定主要意图
        primary_intent = max(keyword_scores.items(), key=lambda x: x[1])
        intent_category = primary_intent[0]
        confidence = primary_intent[1]
        
        # 4. 优先级调整
        priority = self._determine_priority(intent_category, emotion_indicators, context)
        
        # 5. 升级判断
        escalation_needed = await self._should_escalate(confidence, emotion_indicators, intent_category)
        
        # 6. SLA时间
        sla_seconds = self._get_sla_seconds(priority)
        
        # 7. 提取关键词
        matched_keywords = await self._extract_matched_keywords(query_lower, intent_category)
        
        result = IntentResult(
            category=intent_category,
            confidence=confidence,
            priority=priority,
            keywords=matched_keywords,
            emotion_indicators=emotion_indicators,
            escalation_needed=escalation_needed,
            sla_seconds=sla_seconds
        )
        
        logger.info(f"Intent classified: {intent_category.value} (confidence: {confidence:.3f}, priority: {priority.value})")
        return result
    
    async def _calculate_keyword_scores(self, query: str) -> Dict[IntentCategory, float]:
        """计算各意图的关键词匹配分数"""
        scores = {intent: 0.0 for intent in IntentCategory}
        
        for intent, keywords in self.intent_keywords.items():
            matched_count = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in query:
                    matched_count += 1
                    # 关键词长度越长，权重越高
                    weight = len(keyword) / 10.0
                    scores[intent] += weight
            
            # 归一化分数
            if total_keywords > 0:
                scores[intent] = min(scores[intent] / total_keywords, 1.0)
        
        # 如果没有明显匹配，默认为一般咨询
        if max(scores.values()) < 0.1:
            scores[IntentCategory.GENERAL_INQUIRY] = 0.5
            
        return scores
    
    async def _analyze_emotion(self, query: str) -> List[str]:
        """分析情感指标"""
        detected_emotions = []
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_emotions.append(emotion)
                    break
        
        return detected_emotions
    
    def _determine_priority(self, intent: IntentCategory, emotions: List[str], context: Optional[Dict]) -> Priority:
        """确定优先级"""
        base_priority = self.priority_rules.get(intent, Priority.MEDIUM)
        
        # 情感升级
        if "angry" in emotions or "frustrated" in emotions:
            if base_priority == Priority.LOW:
                return Priority.MEDIUM
            elif base_priority == Priority.MEDIUM:
                return Priority.HIGH
        
        if "urgent" in emotions:
            if base_priority in [Priority.LOW, Priority.MEDIUM]:
                return Priority.HIGH
            elif base_priority == Priority.HIGH:
                return Priority.URGENT
        
        # 上下文升级（如果是重复问题）
        if context and context.get("retry_count", 0) > 2:
            if base_priority == Priority.LOW:
                return Priority.MEDIUM
            elif base_priority == Priority.MEDIUM:
                return Priority.HIGH
        
        return base_priority
    
    async def _should_escalate(self, confidence: float, emotions: List[str], intent: IntentCategory) -> bool:
        """判断是否需要升级到人工"""
        # 置信度太低
        if confidence < 0.6:
            return True
        
        # 强烈负面情绪
        if "angry" in emotions:
            return True
        
        # 特定类型自动升级
        if intent in [IntentCategory.COMPLAINT, IntentCategory.EMERGENCY]:
            return True
        
        return False
    
    def _get_sla_seconds(self, priority: Priority) -> int:
        """获取SLA响应时间（秒）"""
        sla_mapping = {
            Priority.URGENT: 60,    # 1分钟
            Priority.HIGH: 180,     # 3分钟
            Priority.MEDIUM: 300,   # 5分钟
            Priority.LOW: 600       # 10分钟
        }
        return sla_mapping[priority]
    
    async def _extract_matched_keywords(self, query: str, intent: IntentCategory) -> List[str]:
        """提取匹配的关键词"""
        matched = []
        keywords = self.intent_keywords.get(intent, [])
        
        for keyword in keywords:
            if keyword in query:
                matched.append(keyword)
        
        return matched[:5]  # 最多返回5个关键词

# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test_classifier():
        classifier = CustomerIntentClassifier()
        
        test_queries = [
            "你好，我想咨询一下产品功能",
            "我的账单怎么这么贵？扣费有问题！",
            "登录不了，一直提示密码错误",
            "要求退款！这个产品太垃圾了！",
            "订单什么时候发货？",
            "紧急！系统崩溃了，影响业务！"
        ]
        
        for query in test_queries:
            result = await classifier.classify(query)
            print(f"查询: {query}")
            print(f"意图: {result.category.value}")
            print(f"置信度: {result.confidence:.3f}")
            print(f"优先级: {result.priority.value}")
            print(f"需要升级: {result.escalation_needed}")
            print("---")
    
    asyncio.run(test_classifier())