"""
对话管理器 - Conversation Manager
客服系统的记忆大脑，负责多轮对话的上下文管理和状态跟踪
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConversationStatus(Enum):
    """对话状态"""
    ACTIVE = "active"              # 活跃对话中
    WAITING_USER = "waiting_user"  # 等待用户回复
    WAITING_AGENT = "waiting_agent" # 等待人工客服
    RESOLVED = "resolved"          # 已解决
    ESCALATED = "escalated"        # 已升级
    ABANDONED = "abandoned"        # 用户中断

@dataclass
class ConversationTurn:
    """单轮对话记录"""
    timestamp: float
    user_message: str
    user_intent: str
    bot_response: str
    confidence: float
    sources: List[str]
    satisfaction_score: Optional[float] = None
    
@dataclass
class ConversationSession:
    """完整对话会话"""
    session_id: str
    start_time: float
    last_activity: float
    status: ConversationStatus
    customer_id: Optional[str]
    turns: List[ConversationTurn]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'last_activity': self.last_activity,
            'status': self.status.value,
            'customer_id': self.customer_id,
            'turns': [asdict(turn) for turn in self.turns],
            'context': self.context,
            'metadata': self.metadata
        }

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, session_timeout: int = 1800):  # 30分钟超时
        """
        初始化对话管理器
        
        Args:
            session_timeout: 会话超时时间（秒）
        """
        self.sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = session_timeout
        
    async def start_session(self, session_id: str, customer_id: Optional[str] = None) -> ConversationSession:
        """
        开始新的对话会话
        
        Args:
            session_id: 会话ID
            customer_id: 客户ID（可选）
        
        Returns:
            ConversationSession: 新的会话对象
        """
        current_time = time.time()
        
        session = ConversationSession(
            session_id=session_id,
            start_time=current_time,
            last_activity=current_time,
            status=ConversationStatus.ACTIVE,
            customer_id=customer_id,
            turns=[],
            context={
                'retry_count': 0,
                'escalation_requested': False,
                'satisfaction_collected': False,
                'dominant_intent': None,
                'unresolved_issues': []
            },
            metadata={
                'user_agent': None,
                'ip_address': None,
                'channel': 'web',  # web, mobile, api
                'language': 'zh-cn'
            }
        )
        
        self.sessions[session_id] = session
        logger.info(f"Started new conversation session: {session_id}")
        return session
    
    async def add_turn(self, 
                      session_id: str,
                      user_message: str,
                      user_intent: str,
                      bot_response: str,
                      confidence: float,
                      sources: List[str]) -> bool:
        """
        添加一轮对话记录
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            user_intent: 用户意图
            bot_response: 机器人回复
            confidence: 置信度
            sources: 信息来源
        
        Returns:
            bool: 是否成功添加
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        turn = ConversationTurn(
            timestamp=time.time(),
            user_message=user_message,
            user_intent=user_intent,
            bot_response=bot_response,
            confidence=confidence,
            sources=sources
        )
        
        session.turns.append(turn)
        session.last_activity = time.time()
        
        # 更新上下文
        await self._update_context(session, user_intent, confidence)
        
        logger.info(f"Added turn to session {session_id}: intent={user_intent}, confidence={confidence:.3f}")
        return True
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
        
        Returns:
            ConversationSession: 会话对象，如果不存在则返回None
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # 检查会话是否超时
        if time.time() - session.last_activity > self.session_timeout:
            session.status = ConversationStatus.ABANDONED
            logger.info(f"Session {session_id} marked as abandoned due to timeout")
        
        return session
    
    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        获取对话上下文
        
        Args:
            session_id: 会话ID
        
        Returns:
            Dict: 上下文信息
        """
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        # 构建丰富的上下文信息
        context = session.context.copy()
        
        # 添加对话历史摘要
        if session.turns:
            recent_turns = session.turns[-3:]  # 最近3轮对话
            context['recent_intents'] = [turn.user_intent for turn in recent_turns]
            context['recent_confidences'] = [turn.confidence for turn in recent_turns]
            context['conversation_length'] = len(session.turns)
        
        # 添加时间信息
        context['session_duration'] = time.time() - session.start_time
        context['last_activity'] = session.last_activity
        
        return context
    
    async def contextualize_query(self, session_id: str, current_query: str) -> str:
        """
        基于对话历史丰富当前查询
        
        Args:
            session_id: 会话ID
            current_query: 当前查询
        
        Returns:
            str: 上下文增强后的查询
        """
        session = await self.get_session(session_id)
        if not session or not session.turns:
            return current_query
        
        # 获取最近的对话内容
        recent_turns = session.turns[-2:]  # 最近2轮
        
        # 构建上下文增强查询
        context_parts = []
        
        # 添加之前的问题上下文
        for turn in recent_turns:
            if turn.user_intent not in ['greeting', 'general_inquiry']:
                context_parts.append(f"之前问题: {turn.user_message}")
        
        # 添加未解决的问题
        unresolved = session.context.get('unresolved_issues', [])
        if unresolved:
            context_parts.append(f"未解决问题: {', '.join(unresolved)}")
        
        if context_parts:
            contextualized = f"对话背景: {' | '.join(context_parts)}\n当前问题: {current_query}"
            logger.info(f"Contextualized query for session {session_id}")
            return contextualized
        
        return current_query
    
    async def update_session_status(self, session_id: str, status: ConversationStatus) -> bool:
        """
        更新会话状态
        
        Args:
            session_id: 会话ID
            status: 新状态
        
        Returns:
            bool: 是否成功更新
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        old_status = session.status
        session.status = status
        session.last_activity = time.time()
        
        logger.info(f"Session {session_id} status changed: {old_status.value} -> {status.value}")
        return True
    
    async def record_satisfaction(self, session_id: str, score: float, feedback: str = "") -> bool:
        """
        记录客户满意度
        
        Args:
            session_id: 会话ID
            score: 满意度评分 (1-5)
            feedback: 文字反馈
        
        Returns:
            bool: 是否成功记录
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # 更新最后一轮对话的满意度
        if session.turns:
            session.turns[-1].satisfaction_score = score
        
        # 更新会话元数据
        session.metadata['final_satisfaction'] = score
        session.metadata['satisfaction_feedback'] = feedback
        session.context['satisfaction_collected'] = True
        
        logger.info(f"Recorded satisfaction for session {session_id}: {score}/5")
        return True
    
    async def should_escalate(self, session_id: str) -> Tuple[bool, str]:
        """
        判断是否应该升级到人工客服
        
        Args:
            session_id: 会话ID
        
        Returns:
            Tuple[bool, str]: (是否需要升级, 升级原因)
        """
        session = await self.get_session(session_id)
        if not session:
            return False, "会话不存在"
        
        context = session.context
        
        # 规则1: 用户明确要求升级
        if context.get('escalation_requested', False):
            return True, "用户主动要求人工服务"
        
        # 规则2: 连续低置信度回答
        recent_turns = session.turns[-3:] if len(session.turns) >= 3 else session.turns
        if recent_turns:
            avg_confidence = sum(turn.confidence for turn in recent_turns) / len(recent_turns)
            if avg_confidence < 0.6:
                return True, "连续回答置信度过低"
        
        # 规则3: 重试次数过多
        if context.get('retry_count', 0) > 3:
            return True, "用户重复询问同类问题"
        
        # 规则4: 对话时间过长
        if len(session.turns) > 10:
            return True, "对话轮次过多，可能需要人工介入"
        
        # 规则5: 负面情绪累积
        complaint_count = sum(1 for turn in session.turns if 'complaint' in turn.user_intent)
        if complaint_count >= 2:
            return True, "检测到投诉情绪，需要人工处理"
        
        return False, ""
    
    async def _update_context(self, session: ConversationSession, intent: str, confidence: float):
        """更新会话上下文"""
        context = session.context
        
        # 更新重试计数
        if confidence < 0.7:
            context['retry_count'] = context.get('retry_count', 0) + 1
        else:
            context['retry_count'] = 0
        
        # 更新主导意图
        intent_counts = {}
        for turn in session.turns:
            intent_counts[turn.user_intent] = intent_counts.get(turn.user_intent, 0) + 1
        
        if intent_counts:
            context['dominant_intent'] = max(intent_counts.items(), key=lambda x: x[1])[0]
        
        # 检测升级请求
        if any(keyword in session.turns[-1].user_message.lower() for keyword in ['人工', '客服', '转接', '投诉']):
            context['escalation_requested'] = True
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话摘要统计
        
        Args:
            session_id: 会话ID
        
        Returns:
            Dict: 会话摘要
        """
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        turns_count = len(session.turns)
        avg_confidence = 0
        if session.turns:
            avg_confidence = sum(turn.confidence for turn in session.turns) / turns_count
        
        duration = time.time() - session.start_time
        
        # 意图分布
        intent_distribution = {}
        for turn in session.turns:
            intent_distribution[turn.user_intent] = intent_distribution.get(turn.user_intent, 0) + 1
        
        return {
            'session_id': session_id,
            'status': session.status.value,
            'duration_seconds': duration,
            'turns_count': turns_count,
            'average_confidence': avg_confidence,
            'intent_distribution': intent_distribution,
            'escalation_needed': await self.should_escalate(session_id),
            'satisfaction_score': session.metadata.get('final_satisfaction'),
            'retry_count': session.context.get('retry_count', 0)
        }
    
    async def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话
        
        Returns:
            int: 清理的会话数量
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if self.sessions[session_id].status == ConversationStatus.ACTIVE:
                self.sessions[session_id].status = ConversationStatus.ABANDONED
            del self.sessions[session_id]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test_conversation_manager():
        manager = ConversationManager()
        
        # 开始会话
        session = await manager.start_session("test_session_001", "customer_123")
        print(f"Started session: {session.session_id}")
        
        # 添加对话轮次
        await manager.add_turn(
            session_id="test_session_001",
            user_message="你好，我的订单什么时候发货？",
            user_intent="order_status",
            bot_response="您好！请提供您的订单号，我帮您查询发货状态。",
            confidence=0.9,
            sources=["order_faq.txt"]
        )
        
        await manager.add_turn(
            session_id="test_session_001",
            user_message="订单号是123456",
            user_intent="order_status",
            bot_response="您的订单123456已经发货，预计明天到达。",
            confidence=0.95,
            sources=["order_system.db"]
        )
        
        # 获取上下文
        context = await manager.get_context("test_session_001")
        print(f"Context: {context}")
        
        # 获取会话摘要
        summary = await manager.get_session_summary("test_session_001")
        print(f"Summary: {summary}")
        
        # 检查是否需要升级
        should_escalate, reason = await manager.should_escalate("test_session_001")
        print(f"Should escalate: {should_escalate}, Reason: {reason}")
    
    asyncio.run(test_conversation_manager())