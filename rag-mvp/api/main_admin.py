#!/usr/bin/env python3
"""
客服RAG管理后台API - Admin Dashboard API
企业级管理后台，不是简单的CRUD界面！
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

# 导入核心模块
from document_processor import EnhancedDocumentProcessor
from llm_generator import RAGGenerator
from customer_service.conversation import ConversationManager
from admin.document_manager import DocumentManager, SplitterConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Customer Service RAG Admin System",
    description="客服RAG系统管理后台 - 企业级管理能力",
    version="1.0.0"
)

# 安全的CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://127.0.0.1:8081"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
ADMIN_DB_PATH = DATA_DIR / "admin.db"

# 全局组件
doc_processor = None
doc_manager = None
conversation_manager = None

# 简单认证（生产环境需要更强的认证）
security = HTTPBasic()

def get_admin_user(credentials: HTTPBasicCredentials = Depends(security)):
    """管理员认证"""
    # 简单的用户名密码验证（生产环境应使用更安全的方式）
    if credentials.username == "admin" and credentials.password == "admin123":
        return credentials.username
    else:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# 请求模型
class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    splitter_config: str = "default_recursive"
    tags: Optional[List[str]] = []
    category: Optional[str] = None

class SplitterConfigRequest(BaseModel):
    """分片配置请求"""
    config_name: str
    config_type: str
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    language: str = "zh-cn"
    preserve_structure: bool = True

class SystemConfigRequest(BaseModel):
    """系统配置请求"""
    intent_threshold: float = 0.8
    escalation_threshold: float = 0.6
    max_conversation_turns: int = 10
    response_timeout_seconds: int = 30

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global doc_processor, doc_manager, conversation_manager
    
    print("=" * 60)
    print("🔧 客服RAG管理后台启动中...")
    print("=" * 60)
    
    try:
        # 创建必要目录
        DATA_DIR.mkdir(exist_ok=True)
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        # 初始化文档处理器
        doc_processor = EnhancedDocumentProcessor(
            chroma_db_path=str(CHROMA_DB_DIR),
            embedding_model="paraphrase-MiniLM-L6-v2",
            collection_name="customer_service_kb"
        )
        print("✅ 文档处理器初始化完成")
        
        # 初始化文档管理器
        doc_manager = DocumentManager(
            upload_dir=str(UPLOAD_DIR),
            db_path=str(ADMIN_DB_PATH),
            doc_processor=doc_processor
        )
        print("✅ 文档管理器初始化完成")
        
        # 初始化对话管理器（用于监控）
        conversation_manager = ConversationManager()
        print("✅ 对话管理器初始化完成")
        
        print("=" * 60)
        print("🎉 管理后台启动完成!")
        print("🔧 管理功能:")
        print("   ✓ 文档上传和管理")
        print("   ✓ 分片策略配置")
        print("   ✓ 系统参数调优")
        print("   ✓ 对话质量监控")
        print("   ✓ 数据统计分析")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"管理后台初始化失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径 - 返回管理后台Web界面"""
    from fastapi.responses import FileResponse
    return FileResponse(str(BASE_DIR / "admin_dashboard.html"))

@app.get("/api/admin/health")
async def admin_health_check(admin_user: str = Depends(get_admin_user)):
    """管理后台健康检查"""
    # 检查各组件状态
    components_status = {
        "document_processor": doc_processor is not None,
        "document_manager": doc_manager is not None,
        "conversation_manager": conversation_manager is not None,
        "database": ADMIN_DB_PATH.exists()
    }
    
    # 获取文档统计
    doc_stats = {}
    if doc_manager:
        doc_stats = await doc_manager.get_document_statistics()
    
    # 系统资源使用情况
    import psutil
    system_stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    
    overall_health = all(components_status.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": components_status,
        "document_statistics": doc_stats,
        "system_resources": system_stats,
        "admin_features": {
            "document_management": "enabled",
            "configuration_management": "enabled",
            "monitoring_dashboard": "enabled",
            "analytics_reports": "enabled"
        }
    }

# ==================== 文档管理接口 ====================

@app.post("/api/admin/documents/check-duplicate")
async def check_document_duplicate(
    file: UploadFile = File(...),
    admin_user: str = Depends(get_admin_user)
):
    """检查文档重复"""
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 验证文件大小（50MB限制）
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件大小超过50MB限制")
        
        # 检查重复
        result = await doc_manager.check_document_duplicates(
            filename=file.filename,
            file_content=file_content
        )
        
        return result
        
    except Exception as e:
        logger.error(f"检查文档重复失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    splitter_config: str = Form("default_recursive"),
    tags: str = Form("[]"),  # JSON string
    category: Optional[str] = Form(None),
    force_replace: bool = Form(False),
    admin_user: str = Depends(get_admin_user)
):
    """上传文档"""
    try:
        # 解析标签
        tags_list = json.loads(tags) if tags else []
        
        # 读取文件内容
        file_content = await file.read()
        
        # 验证文件大小（50MB限制）
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件大小超过50MB限制")
        
        # 验证文件类型
        allowed_extensions = ['.pdf', '.txt', '.md', '.doc', '.docx']
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=415, 
                detail=f"不支持的文件类型: {file_extension}"
            )
        
        # 上传并处理文档
        result = await doc_manager.upload_document(
            file_content=file_content,
            filename=file.filename,
            splitter_config_name=splitter_config,
            tags=tags_list,
            category=category,
            operator=admin_user,
            force_replace=force_replace
        )
        
        return result
        
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents")
async def list_documents(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),  # 逗号分隔的标签
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin_user: str = Depends(get_admin_user)
):
    """获取文档列表"""
    try:
        # 解析标签
        tags_list = tags.split(',') if tags else None
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        result = await doc_manager.get_documents_list(
            category=category,
            status=status,
            tags=tags_list,
            limit=page_size,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """删除文档"""
    try:
        result = await doc_manager.delete_document(doc_id, admin_user)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
        
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    splitter_config: Optional[str] = None,
    admin_user: str = Depends(get_admin_user)
):
    """重新处理文档"""
    try:
        result = await doc_manager.reprocess_document(
            doc_id=doc_id,
            new_splitter_config=splitter_config,
            operator=admin_user
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"重新处理文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents/{doc_id}/details")
async def get_document_details(
    doc_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """获取文档详情"""
    try:
        result = await doc_manager.get_document_details(doc_id)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
        
    except Exception as e:
        logger.error(f"获取文档详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """获取文档切片列表"""
    try:
        result = await doc_manager.get_document_chunks(doc_id)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
        
    except Exception as e:
        logger.error(f"获取文档切片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents/statistics")
async def get_document_statistics(admin_user: str = Depends(get_admin_user)):
    """获取文档统计信息"""
    try:
        stats = await doc_manager.get_document_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 分片配置管理 ====================

@app.get("/api/admin/splitter-configs")
async def get_splitter_configs(admin_user: str = Depends(get_admin_user)):
    """获取所有分片配置"""
    try:
        configs = await doc_manager.get_splitter_configs()
        return {
            "configs": configs,
            "total_count": len(configs)
        }
        
    except Exception as e:
        logger.error(f"获取分片配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/splitter-configs")
async def add_splitter_config(
    config_request: SplitterConfigRequest,
    admin_user: str = Depends(get_admin_user)
):
    """添加新的分片配置"""
    try:
        # 创建配置对象
        config = SplitterConfig(
            type=config_request.config_type,
            chunk_size=config_request.chunk_size,
            chunk_overlap=config_request.chunk_overlap,
            separators=config_request.separators,
            language=config_request.language,
            preserve_structure=config_request.preserve_structure
        )
        
        result = await doc_manager.add_splitter_config(
            config_name=config_request.config_name,
            config=config
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"添加分片配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 系统监控接口 ====================

@app.get("/api/admin/monitoring/conversations")
async def get_active_conversations(admin_user: str = Depends(get_admin_user)):
    """获取活跃对话监控"""
    try:
        active_sessions = []
        
        if conversation_manager and conversation_manager.sessions:
            for session_id, session in conversation_manager.sessions.items():
                if session.status.value == "active":
                    summary = await conversation_manager.get_session_summary(session_id)
                    active_sessions.append(summary)
        
        return {
            "active_conversations": active_sessions,
            "total_active": len(active_sessions),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取对话监控失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/monitoring/system-metrics")
async def get_system_metrics(admin_user: str = Depends(get_admin_user)):
    """获取系统性能指标"""
    try:
        import psutil
        
        # CPU和内存使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取文档处理统计
        doc_stats = await doc_manager.get_document_statistics()
        
        return {
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": round(disk.used / disk.total * 100, 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            },
            "document_metrics": doc_stats,
            "conversation_metrics": {
                "active_sessions": len(conversation_manager.sessions) if conversation_manager else 0,
                # 可以添加更多对话相关指标
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 数据分析接口 ====================

@app.get("/api/admin/analytics/usage-trends")
async def get_usage_trends(
    days: int = Query(7, ge=1, le=30),
    admin_user: str = Depends(get_admin_user)
):
    """获取使用趋势分析"""
    try:
        # 这里可以实现更复杂的趋势分析
        # 目前返回模拟数据
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 生成日期序列
        trends = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            trends.append({
                "date": date.strftime("%Y-%m-%d"),
                "document_uploads": 0,  # 实际应从数据库查询
                "conversations": 0,
                "queries": 0,
                "success_rate": 0.95
            })
        
        return {
            "trends": trends,
            "period": f"{days} days",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取使用趋势失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/analytics/quality-report")
async def get_quality_report(admin_user: str = Depends(get_admin_user)):
    """获取质量分析报告"""
    try:
        # 获取文档质量统计
        doc_stats = await doc_manager.get_document_statistics()
        
        return {
            "document_quality": {
                "average_score": doc_stats.get("average_quality_score", 0),
                "total_documents": doc_stats.get("total_documents", 0),
                "quality_distribution": {
                    "excellent": 0,  # 可以实现更详细的质量分布统计
                    "good": 0,
                    "fair": 0,
                    "poor": 0
                }
            },
            "conversation_quality": {
                "average_confidence": 0.85,  # 示例数据
                "escalation_rate": 0.15,
                "satisfaction_score": 4.2
            },
            "recommendations": [
                "考虑优化低质量文档的分片策略",
                "增加技术支持类文档的数量",
                "定期更新过时的文档内容"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取质量报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 配置管理接口 ====================

@app.get("/api/admin/config/system")
async def get_system_config(admin_user: str = Depends(get_admin_user)):
    """获取系统配置"""
    try:
        # 返回当前系统配置
        # 实际应该从数据库或配置文件读取
        config = {
            "intent_recognition": {
                "confidence_threshold": 0.8,
                "enable_emotion_detection": True,
                "supported_languages": ["zh-cn", "en-us"]
            },
            "conversation_management": {
                "max_turns": 10,
                "session_timeout_minutes": 30,
                "auto_escalation_enabled": True
            },
            "llm_generation": {
                "max_tokens": 500,
                "temperature": 0.3,
                "enable_safety_filter": True
            },
            "performance": {
                "response_timeout_seconds": 30,
                "max_concurrent_sessions": 100,
                "cache_enabled": True
            }
        }
        
        return {
            "config": config,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/config/system")
async def update_system_config(
    config_request: SystemConfigRequest,
    admin_user: str = Depends(get_admin_user)
):
    """更新系统配置"""
    try:
        # 这里应该验证配置参数并保存到数据库
        # 目前只返回成功消息
        
        logger.info(f"管理员 {admin_user} 更新了系统配置")
        
        return {
            "success": True,
            "message": "系统配置更新成功",
            "updated_by": admin_user,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"更新系统配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FAQ管理接口 ====================

@app.post("/api/admin/documents/{doc_id}/extract-faq")
async def extract_faq_from_document(
    doc_id: str,
    max_faqs: int = Query(10, ge=1, le=20),
    admin_user: str = Depends(get_admin_user)
):
    """从文档中抽取FAQ"""
    try:
        result = await doc_manager.extract_faqs_from_document(
            doc_id=doc_id,
            operator=admin_user,
            max_faqs=max_faqs
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"FAQ抽取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/faqs")
async def get_faqs_list(
    doc_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin_user: str = Depends(get_admin_user)
):
    """获取FAQ列表"""
    try:
        offset = (page - 1) * page_size
        
        result = await doc_manager.get_faqs_list(
            doc_id=doc_id,
            status=status,
            category=category,
            limit=page_size,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        logger.error(f"获取FAQ列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/admin/faqs/{faq_id}")
async def update_faq(
    faq_id: str,
    updates: dict,
    admin_user: str = Depends(get_admin_user)
):
    """更新FAQ"""
    try:
        result = await doc_manager.update_faq(
            faq_id=faq_id,
            updates=updates,
            operator=admin_user
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"更新FAQ失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/faqs/{faq_id}")
async def delete_faq(
    faq_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """删除FAQ"""
    try:
        result = await doc_manager.delete_faq(
            faq_id=faq_id,
            operator=admin_user
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"删除FAQ失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/faqs/statistics")
async def get_faq_statistics(admin_user: str = Depends(get_admin_user)):
    """获取FAQ统计信息"""
    try:
        stats = await doc_manager.get_faq_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"获取FAQ统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/faqs/cleanup-duplicates")
async def cleanup_duplicate_faqs(admin_user: str = Depends(get_admin_user)):
    """清理重复的FAQ"""
    try:
        result = await doc_manager.cleanup_duplicate_faqs()
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
        
    except Exception as e:
        logger.error(f"清理重复FAQ失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/faqs/extraction-status")
async def get_faq_extraction_status(admin_user: str = Depends(get_admin_user)):
    """获取FAQ抽取状态（用于前端置灰功能）"""
    try:
        result = await doc_manager.get_faq_extraction_status()
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
        
    except Exception as e:
        logger.error(f"获取FAQ抽取状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 挂载静态文件（放在所有API路由之后）
app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动客服RAG管理后台...")
    uvicorn.run(app, host="0.0.0.0", port=8001)