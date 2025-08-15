#!/usr/bin/env python3
"""
客服RAG系统主API - Customer Service RAG System Main API
这是一个真正的客服级别的智能问答系统，不再是学术Demo
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# 导入原有模块
from document_processor import EnhancedDocumentProcessor
from llm_generator import RAGGenerator

# 导入新的客服模块
from customer_service.intent import CustomerIntentClassifier, IntentCategory, Priority
from customer_service.conversation import ConversationManager, ConversationStatus

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Customer Service RAG System",
    description="基于RAG的智能客服系统 - 不是Demo，是真正的商业级产品",
    version="3.0.0"
)

# 安全的CORS配置（不再是通配符！）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],  # 仅本地开发
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-Session-ID"],
)

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# 全局组件
doc_processor = None
rag_generator = None
intent_classifier = None
conversation_manager = None

# 请求模型
class CustomerQuery(BaseModel):
    """客服查询请求"""
    session_id: str
    message: str
    customer_id: Optional[str] = None
    channel: str = "web"  # web, mobile, api, phone
    
class CustomerResponse(BaseModel):
    """客服响应"""
    session_id: str
    response: str
    intent: str
    confidence: float
    priority: str
    sources: List[str]
    escalation_needed: bool
    suggested_actions: List[str]
    response_time_ms: int

class EscalationRequest(BaseModel):
    """升级请求"""
    session_id: str
    reason: str
    customer_message: str

class SatisfactionFeedback(BaseModel):
    """满意度反馈"""
    session_id: str
    score: int  # 1-5
    feedback: Optional[str] = ""

@app.on_event("startup")
async def startup_event():
    """启动事件 - 初始化客服系统"""
    global doc_processor, rag_generator, intent_classifier, conversation_manager
    
    print("=" * 60)
    print("🚀 客服RAG系统启动中...")
    print("=" * 60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"🗄️ 向量数据库: {CHROMA_DB_DIR}")
    print("=" * 60)
    
    try:
        # 1. 初始化文档处理器
        doc_processor = EnhancedDocumentProcessor(
            chroma_db_path=str(CHROMA_DB_DIR),
            embedding_model="paraphrase-MiniLM-L6-v2",
            collection_name="customer_service_kb"
        )
        print("✅ 文档处理器初始化完成")
        
        # 2. 初始化LLM生成器
        try:
            rag_generator = RAGGenerator(provider="openai")
            print("✅ LLM生成器初始化完成 (OpenAI)")
        except Exception as e:
            logger.warning(f"OpenAI初始化失败: {e}, 尝试Claude...")
            try:
                rag_generator = RAGGenerator(provider="claude")
                print("✅ LLM生成器初始化完成 (Claude)")
            except Exception as e:
                logger.warning(f"Claude初始化失败: {e}, 使用降级模式...")
                rag_generator = None
                print("⚠️ LLM生成器降级模式")
        
        # 3. 初始化意图识别器
        intent_classifier = CustomerIntentClassifier()
        print("✅ 意图识别器初始化完成")
        
        # 4. 初始化对话管理器
        conversation_manager = ConversationManager(session_timeout=1800)  # 30分钟
        print("✅ 对话管理器初始化完成")
        
        print("=" * 60)
        print("🎉 客服RAG系统启动完成!")
        print("🔧 系统特性:")
        print("   ✓ 智能意图识别")
        print("   ✓ 多轮对话管理")
        print("   ✓ 自动升级判断")
        print("   ✓ 客户满意度跟踪")
        print("   ✓ 安全响应生成")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "system": "Customer Service RAG System",
        "version": "3.0.0",
        "status": "online",
        "capabilities": [
            "智能意图识别",
            "多轮对话管理", 
            "知识库检索",
            "安全回答生成",
            "自动人工升级",
            "满意度跟踪"
        ]
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    # 检查各组件状态
    components_status = {
        "document_processor": doc_processor is not None,
        "intent_classifier": intent_classifier is not None,
        "conversation_manager": conversation_manager is not None,
        "llm_generator": rag_generator is not None
    }
    
    # 获取统计信息
    stats = {}
    if doc_processor:
        stats = doc_processor.get_statistics()
    
    # 活跃会话数
    active_sessions = 0
    if conversation_manager:
        active_sessions = len([s for s in conversation_manager.sessions.values() 
                             if s.status == ConversationStatus.ACTIVE])
    
    overall_health = all(components_status.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": components_status,
        "statistics": {
            **stats,
            "active_sessions": active_sessions
        },
        "service_level": {
            "intent_recognition": "enabled",
            "conversation_tracking": "enabled", 
            "auto_escalation": "enabled",
            "satisfaction_tracking": "enabled"
        }
    }

@app.post("/api/customer/chat", response_model=CustomerResponse)
async def customer_chat(query: CustomerQuery):
    """
    客服聊天接口 - 这是核心API
    """
    start_time = time.time()
    
    try:
        # 1. 获取或创建会话
        session = await conversation_manager.get_session(query.session_id)
        if not session:
            session = await conversation_manager.start_session(
                query.session_id, 
                query.customer_id
            )
            logger.info(f"Started new customer session: {query.session_id}")
        
        # 2. 意图识别
        context = await conversation_manager.get_context(query.session_id)
        intent_result = await intent_classifier.classify(query.message, context)
        
        logger.info(f"Intent classified: {intent_result.category.value} "
                   f"(confidence: {intent_result.confidence:.3f}, "
                   f"priority: {intent_result.priority.value})")
        
        # 3. 紧急升级检查
        if intent_result.escalation_needed:
            await conversation_manager.update_session_status(
                query.session_id, 
                ConversationStatus.WAITING_AGENT
            )
            
            return CustomerResponse(
                session_id=query.session_id,
                response="您的问题我将为您转接专业客服人员处理，请稍候...",
                intent=intent_result.category.value,
                confidence=intent_result.confidence,
                priority=intent_result.priority.value,
                sources=[],
                escalation_needed=True,
                suggested_actions=["等待人工客服接入"],
                response_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # 4. 上下文增强查询
        enhanced_query = await conversation_manager.contextualize_query(
            query.session_id, 
            query.message
        )
        
        # 5. 知识库检索
        search_results = doc_processor.search(
            query=enhanced_query,
            n_results=5
        )
        
        # 6. 生成回答
        response_text = "抱歉，我暂时无法为您提供准确的答案。"
        sources = []
        
        if search_results:
            if rag_generator:
                # 使用LLM生成智能回答
                llm_result = rag_generator.generate_answer(
                    query=enhanced_query,
                    retrieved_results=search_results,
                    max_tokens=500,
                    temperature=0.3  # 客服需要更保守的回答
                )
                
                if llm_result.get("success", False):
                    response_text = llm_result["answer"]
                    sources = llm_result.get("sources", [])
                else:
                    # LLM失败，使用简单拼接
                    response_text = await _generate_fallback_response(search_results, intent_result)
                    sources = [r['metadata'].get('source', 'Unknown') for r in search_results[:3]]
            else:
                # 无LLM可用，使用简单拼接
                response_text = await _generate_fallback_response(search_results, intent_result)
                sources = [r['metadata'].get('source', 'Unknown') for r in search_results[:3]]
        
        # 7. 客服格式化响应
        formatted_response = await _format_customer_service_response(
            response_text, 
            intent_result
        )
        
        # 8. 记录对话轮次
        await conversation_manager.add_turn(
            session_id=query.session_id,
            user_message=query.message,
            user_intent=intent_result.category.value,
            bot_response=formatted_response,
            confidence=intent_result.confidence,
            sources=sources
        )
        
        # 9. 检查是否需要升级
        should_escalate, escalation_reason = await conversation_manager.should_escalate(query.session_id)
        
        # 10. 生成建议操作
        suggested_actions = await _generate_suggested_actions(intent_result, should_escalate)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Customer query processed: session={query.session_id}, "
                   f"intent={intent_result.category.value}, "
                   f"time={response_time_ms}ms")
        
        return CustomerResponse(
            session_id=query.session_id,
            response=formatted_response,
            intent=intent_result.category.value,
            confidence=intent_result.confidence,
            priority=intent_result.priority.value,
            sources=sources,
            escalation_needed=should_escalate,
            suggested_actions=suggested_actions,
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        logger.error(f"Customer chat error: {str(e)}")
        
        # 错误情况下的安全响应
        return CustomerResponse(
            session_id=query.session_id,
            response="抱歉，系统暂时出现问题，我将为您转接人工客服。",
            intent="system_error",
            confidence=0.0,
            priority="urgent",
            sources=[],
            escalation_needed=True,
            suggested_actions=["系统升级到人工客服"],
            response_time_ms=int((time.time() - start_time) * 1000)
        )

@app.post("/api/customer/escalate")
async def escalate_to_human(request: EscalationRequest):
    """升级到人工客服"""
    try:
        success = await conversation_manager.update_session_status(
            request.session_id,
            ConversationStatus.ESCALATED
        )
        
        if success:
            logger.info(f"Session {request.session_id} escalated to human: {request.reason}")
            return {
                "success": True,
                "message": "已为您转接人工客服，请稍候...",
                "ticket_id": f"TK_{int(time.time())}_{request.session_id[:8]}"
            }
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
            
    except Exception as e:
        logger.error(f"Escalation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/customer/satisfaction")
async def record_satisfaction(feedback: SatisfactionFeedback):
    """记录客户满意度"""
    try:
        success = await conversation_manager.record_satisfaction(
            feedback.session_id,
            feedback.score,
            feedback.feedback
        )
        
        if success:
            logger.info(f"Satisfaction recorded: session={feedback.session_id}, score={feedback.score}")
            return {"success": True, "message": "感谢您的反馈！"}
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
            
    except Exception as e:
        logger.error(f"Satisfaction recording failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/customer/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    try:
        session = await conversation_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        summary = await conversation_manager.get_session_summary(session_id)
        return summary
        
    except Exception as e:
        logger.error(f"Get session info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/sessions")
async def list_active_sessions():
    """管理员查看活跃会话"""
    try:
        active_sessions = []
        for session_id, session in conversation_manager.sessions.items():
            if session.status == ConversationStatus.ACTIVE:
                summary = await conversation_manager.get_session_summary(session_id)
                active_sessions.append(summary)
        
        return {
            "total_active": len(active_sessions),
            "sessions": active_sessions
        }
        
    except Exception as e:
        logger.error(f"List sessions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数
async def _generate_fallback_response(search_results: List[Dict], intent_result) -> str:
    """生成降级响应"""
    if not search_results:
        return "抱歉，我在知识库中没有找到相关信息。请稍候，我为您转接专业客服。"
    
    # 根据意图生成不同风格的回答
    if intent_result.category == IntentCategory.COMPLAINT:
        response = "非常抱歉给您带来困扰。根据我找到的信息：\n\n"
    elif intent_result.category == IntentCategory.TECHNICAL_SUPPORT:
        response = "关于您的技术问题，我找到以下解决方案：\n\n"
    else:
        response = "根据您的咨询，我找到以下相关信息：\n\n"
    
    # 添加最相关的结果
    for i, result in enumerate(search_results[:2], 1):
        content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        response += f"{i}. {content}\n\n"
    
    response += "这些信息是否能帮到您？如需更详细的帮助，我可以为您转接专业客服。"
    return response

async def _format_customer_service_response(response: str, intent_result) -> str:
    """格式化客服标准响应"""
    # 添加客服礼貌用语
    if intent_result.category == IntentCategory.GREETING:
        return f"您好！很高兴为您服务。{response}"
    elif intent_result.category == IntentCategory.COMPLAINT:
        return f"非常抱歉给您带来不便。{response} 我们会认真对待您的反馈。"
    elif intent_result.priority == Priority.URGENT:
        return f"我理解这个问题的紧急性。{response} 如需立即处理，我可以为您转接专员。"
    else:
        return f"{response} 还有什么可以帮助您的吗？"

async def _generate_suggested_actions(intent_result, should_escalate: bool) -> List[str]:
    """生成建议操作"""
    actions = []
    
    if should_escalate:
        actions.append("转接人工客服")
    
    if intent_result.category == IntentCategory.ORDER_STATUS:
        actions.extend(["查看订单详情", "联系物流"])
    elif intent_result.category == IntentCategory.TECHNICAL_SUPPORT:
        actions.extend(["查看技术文档", "重试操作"])
    elif intent_result.category == IntentCategory.BILLING_INQUIRY:
        actions.extend(["查看账单详情", "联系财务"])
    
    actions.append("结束咨询")
    return actions

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动客服RAG系统...")
    uvicorn.run(app, host="0.0.0.0", port=8000)