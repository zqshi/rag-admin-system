#!/usr/bin/env python3
"""
RAG增强版系统 - 集成LangChain和ChromaDB
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import logging

# 导入增强版文档处理器和LLM生成器
from document_processor import EnhancedDocumentProcessor
from llm_generator import RAGGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Enhanced System",
    description="增强版RAG系统 - 使用LangChain和ChromaDB",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "rag.db"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# 全局文档处理器和LLM生成器
doc_processor = None
rag_generator = None

# 请求模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None
    
class ProcessRequest(BaseModel):
    file_path: str
    splitter_type: str = "recursive"
    metadata: Optional[Dict[str, Any]] = None

class UpdateMetadataRequest(BaseModel):
    chunk_id: str
    metadata: Dict[str, Any]

# 响应模型
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    upload_time: str
    status: str
    chunk_count: int
    total_chars: int

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    processing_time: float

def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 增强版文档表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            status TEXT DEFAULT 'processing',
            chunk_count INTEGER DEFAULT 0,
            total_chars INTEGER DEFAULT 0,
            splitter_type TEXT DEFAULT 'recursive',
            metadata TEXT,
            processing_time REAL
        )
    ''')
    
    # 搜索日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            results_count INTEGER,
            processing_time REAL,
            timestamp TEXT NOT NULL,
            filter_used TEXT
        )
    ''')
    
    # 系统配置表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global doc_processor, rag_generator
    
    print("=" * 50)
    print("🚀 RAG增强版系统启动中...")
    print("=" * 50)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📤 上传目录: {UPLOAD_DIR}")
    print(f"🗄️ SQLite数据库: {DB_PATH}")
    print(f"🎯 ChromaDB目录: {CHROMA_DB_DIR}")
    print("=" * 50)
    
    # 初始化数据库
    init_database()
    
    # 初始化文档处理器
    doc_processor = EnhancedDocumentProcessor(
        chroma_db_path=str(CHROMA_DB_DIR),
        embedding_model="paraphrase-MiniLM-L6-v2",
        collection_name="rag_documents"
    )
    
    # 初始化LLM生成器 - 优先使用本地模型，降级到简单回答
    try:
        # 尝试本地LLM
        rag_generator = RAGGenerator(provider="local", model="llama2")
        print("✅ LLM生成器初始化完成！(本地模式)")
    except Exception as e:
        logger.warning(f"本地LLM初始化失败: {e}")
        try:
            # 尝试OpenAI
            rag_generator = RAGGenerator(provider="openai")
            print("✅ LLM生成器初始化完成！(OpenAI模式)")
        except Exception as e:
            logger.warning(f"OpenAI初始化失败: {e}")
            try:
                # 尝试Claude
                rag_generator = RAGGenerator(provider="claude")
                print("✅ LLM生成器初始化完成！(Claude模式)")
            except Exception as e:
                logger.warning(f"Claude初始化失败: {e}")
                rag_generator = None
                print("⚠️ LLM生成器初始化失败，将使用简单模式")
    
    print("✅ 文档处理器初始化完成！")
    print("✅ ChromaDB向量数据库就绪！")
    print("=" * 50)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG增强版系统",
        "version": "2.0.0",
        "features": [
            "LangChain智能文档切片",
            "ChromaDB向量数据库",
            "多种切片策略",
            "高级语义搜索"
        ]
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    stats = doc_processor.get_statistics() if doc_processor else {}
    
    # LLM状态检查
    llm_status = {
        "available": rag_generator is not None,
        "provider": rag_generator.provider if rag_generator else None,
        "model": getattr(rag_generator.generator, 'model', 'unknown') if rag_generator else None
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "chroma_db": "active",
        "llm": llm_status,
        "components": {
            "document_processor": "ready" if doc_processor else "not_initialized",
            "vector_db": "ready" if doc_processor else "not_initialized", 
            "llm_generator": "ready" if rag_generator else "fallback_mode"
        },
        "statistics": stats
    }

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    splitter_type: str = Query("recursive", description="切片类型: recursive, token, char")
):
    """上传并处理文档"""
    import time
    start_time = time.time()
    
    # 生成文档ID
    doc_id = hashlib.md5(f"{file.filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # 保存文件
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # 处理文档
    try:
        result = doc_processor.process_and_store(
            file_path=str(file_path),
            metadata={
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(content)
            },
            splitter_type=splitter_type
        )
        
        processing_time = time.time() - start_time
        
        # 保存到数据库
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (id, filename, file_path, upload_time, status, chunk_count, total_chars, splitter_type, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['doc_id'],
            file.filename,
            str(file_path),
            datetime.now().isoformat(),
            'completed',
            result['chunks_count'],
            result['total_chars'],
            splitter_type,
            processing_time
        ))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "doc_id": result['doc_id'],
            "filename": file.filename,
            "chunks_count": result['chunks_count'],
            "total_chars": result['total_chars'],
            "processing_time": processing_time,
            "splitter_type": splitter_type
        }
        
    except Exception as e:
        logger.error(f"处理文档失败: {str(e)}")
        # 清理文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """查询文档"""
    import time
    start_time = time.time()
    
    try:
        # 执行搜索
        results = doc_processor.search(
            query=request.query,
            n_results=request.top_k,
            where=request.filter
        )
        
        # 使用LLM生成答案
        if rag_generator and results:
            # 使用LLM生成器生成智能回答
            llm_result = rag_generator.generate_answer(
                query=request.query,
                retrieved_results=results,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = llm_result.get("answer", "生成回答时出现错误。")
            sources = llm_result.get("sources", [])
            
            # 记录LLM使用情况
            logger.info(f"LLM生成完成: model={llm_result.get('model')}, "
                       f"tokens={llm_result.get('tokens_used', 0)}, "
                       f"success={llm_result.get('success', False)}")
            
        elif results:
            # 降级：简单的文本拼接（无LLM可用时）
            contexts = []
            seen_contexts = set()
            
            for r in results:
                context = r['content'][:300].strip()
                context_hash = hash(context)
                if context_hash not in seen_contexts:
                    seen_contexts.add(context_hash)
                    contexts.append(context)
            
            sources = list(set([r['metadata'].get('source', 'Unknown') for r in results]))
            sources = [s.split('/')[-1] for s in sources]  # 只保留文件名
            
            if len(contexts) > 0:
                answer = f"基于检索到的 {len(results)} 个文档片段，找到以下相关信息：\n\n"
                for i, context in enumerate(contexts[:3], 1):
                    preview = context[:200] + "..." if len(context) > 200 else context
                    answer += f"**片段{i}**: {preview}\n\n"
                
                if len(contexts) > 3:
                    answer += f"...以及其他 {len(contexts)-3} 个相关片段。"
                    
                answer += "\n\n> ⚠️ 当前使用简单模式，建议配置LLM获得更智能的回答。"
            else:
                answer = "检索到相关文档，但内容处理出现问题。"
        else:
            answer = "抱歉，没有找到与您的问题相关的内容。请尝试其他关键词或上传相关文档。"
            sources = []
        
        processing_time = time.time() - start_time
        
        # 记录搜索日志
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO search_logs (query, results_count, processing_time, timestamp, filter_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            request.query,
            len(results),
            processing_time,
            datetime.now().isoformat(),
            json.dumps(request.filter) if request.filter else None
        ))
        conn.commit()
        conn.close()
        
        return {
            "query": request.query,
            "results": results,
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    """获取文档列表"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, filename, upload_time, status, chunk_count, total_chars, splitter_type, processing_time
        FROM documents
        ORDER BY upload_time DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    documents = []
    for row in rows:
        documents.append({
            "id": row[0],
            "filename": row[1],
            "upload_time": row[2],
            "status": row[3],
            "chunk_count": row[4],
            "total_chars": row[5],
            "splitter_type": row[6],
            "processing_time": row[7]
        })
    
    return documents

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    try:
        # 从向量数据库删除
        success = doc_processor.delete_document(doc_id)
        
        if success:
            # 从SQLite删除
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"文档 {doc_id} 已删除"}
        else:
            raise HTTPException(status_code=404, detail="文档不存在")
            
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """获取系统统计信息"""
    # 获取ChromaDB统计
    chroma_stats = doc_processor.get_statistics() if doc_processor else {}
    
    # 获取SQLite统计
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM documents')
    doc_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM search_logs')
    search_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(processing_time) FROM documents WHERE processing_time IS NOT NULL')
    avg_processing = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT SUM(chunk_count) FROM documents')
    total_chunks = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "documents_count": doc_count,
        "total_chunks": total_chunks,
        "search_count": search_count,
        "avg_processing_time": avg_processing,
        "chroma_db": chroma_stats
    }

@app.post("/api/reprocess")
async def reprocess_document(request: ProcessRequest):
    """重新处理已存在的文档"""
    try:
        # 检查文件是否存在
        if not Path(request.file_path).exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 重新处理
        result = doc_processor.process_and_store(
            file_path=request.file_path,
            metadata=request.metadata,
            splitter_type=request.splitter_type
        )
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"重新处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/metadata")
async def update_metadata(request: UpdateMetadataRequest):
    """更新文档片段元数据"""
    try:
        success = doc_processor.update_metadata(
            chunk_id=request.chunk_id,
            metadata=request.metadata
        )
        
        if success:
            return {"success": True, "message": "元数据已更新"}
        else:
            raise HTTPException(status_code=404, detail="片段不存在")
            
    except Exception as e:
        logger.error(f"更新元数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-logs")
async def get_search_logs(limit: int = Query(100, description="返回记录数")):
    """获取搜索日志"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT query, results_count, processing_time, timestamp, filter_used
        FROM search_logs
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            "query": row[0],
            "results_count": row[1],
            "processing_time": row[2],
            "timestamp": row[3],
            "filter_used": json.loads(row[4]) if row[4] else None
        })
    
    return logs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)