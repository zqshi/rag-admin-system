#!/usr/bin/env python3
"""
RAG离线版系统 - 无需网络连接的版本
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RAG Offline System",
    description="离线版RAG系统 - 无网络依赖",
    version="1.0.0"
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
DB_PATH = DATA_DIR / "rag_offline.db"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# 请求模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[Dict[str, Any]] = None

def init_database():
    """初始化数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 文档表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_time TEXT,
            status TEXT,
            file_size INTEGER,
            content TEXT
        )
    ''')
    
    # 搜索日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            results_count INTEGER,
            processing_time REAL,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("离线数据库初始化完成")

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print("=" * 50)
    print("🚀 RAG离线版系统启动中...")
    print("=" * 50)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📤 上传目录: {UPLOAD_DIR}")
    print(f"🗄️ 离线数据库: {DB_PATH}")
    print("=" * 50)
    
    # 初始化数据库
    init_database()
    
    print("✅ 离线系统初始化完成！")
    print("⚠️ 注意：当前为离线模式，功能有限")
    print("=" * 50)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG离线版系统",
        "version": "1.0.0",
        "mode": "offline",
        "features": [
            "文档上传",
            "基本文本搜索",
            "简单问答"
        ]
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "offline",
        "database": "connected",
        "llm": {
            "available": False,
            "provider": None,
            "model": "offline_mode"
        },
        "components": {
            "document_processor": "offline_mode",
            "vector_db": "disabled",
            "llm_generator": "disabled"
        }
    }

def read_text_file(file_path: Path) -> str:
    """读取文本文件内容"""
    try:
        # 尝试不同编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # 如果都失败了，用二进制模式读取并尽可能解码
        with open(file_path, 'rb') as f:
            content = f.read()
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return f"文件读取失败: {str(e)}"

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    splitter_type: str = Query("char", description="切片类型: char (字符分割)")
):
    """上传并处理文档"""
    import time
    start_time = time.time()
    
    logger.info(f"开始处理上传文件: {file.filename}")
    
    # 生成文档ID
    doc_id = hashlib.md5(f"{file.filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # 保存文件
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件保存成功: {file_path}")
        
        # 读取文件内容（仅支持文本文件）
        if file.filename.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css')):
            text_content = read_text_file(file_path)
        else:
            text_content = f"文件类型: {file.filename.split('.')[-1].upper()}\n文件大小: {len(content)} 字节\n注意: 离线模式暂不支持该文件类型的内容提取"
        
        processing_time = time.time() - start_time
        
        # 保存到数据库
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (id, filename, file_path, upload_time, status, file_size, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            file.filename,
            str(file_path),
            datetime.now().isoformat(),
            'completed',
            len(content),
            text_content[:10000]  # 限制内容长度
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"文档处理完成: {doc_id}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "file_size": len(content),
            "processing_time": processing_time,
            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
            "mode": "offline",
            "note": "离线模式：仅支持基本文本文件，无向量化处理"
        }
        
    except Exception as e:
        logger.error(f"处理文档失败: {str(e)}")
        # 清理文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """查询文档（离线简单搜索）"""
    import time
    start_time = time.time()
    
    try:
        # 从数据库获取所有文档
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, content FROM documents WHERE status = "completed"')
        documents = cursor.fetchall()
        conn.close()
        
        if not documents:
            return {
                "query": request.query,
                "results": [],
                "answer": "暂无文档可供搜索。请先上传一些文档。",
                "processing_time": time.time() - start_time,
                "mode": "offline"
            }
        
        # 简单关键词搜索
        query_lower = request.query.lower()
        results = []
        
        for doc_id, filename, content in documents:
            if content and query_lower in content.lower():
                # 找到匹配位置
                match_pos = content.lower().find(query_lower)
                start_pos = max(0, match_pos - 100)
                end_pos = min(len(content), match_pos + 200)
                snippet = content[start_pos:end_pos]
                
                results.append({
                    'id': doc_id,
                    'content': snippet,
                    'metadata': {
                        'source': filename,
                        'doc_id': doc_id
                    },
                    'distance': 0.1  # 模拟相似度
                })
        
        # 限制结果数量
        results = results[:request.top_k]
        
        # 生成简单回答
        if results:
            answer = f"在 {len(results)} 个文档片段中找到相关信息：\n\n"
            for i, result in enumerate(results[:3], 1):
                content_preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                answer += f"**片段{i}** (来源: {result['metadata']['source']}):\n{content_preview}\n\n"
            
            if len(results) > 3:
                answer += f"...以及其他 {len(results)-3} 个相关片段。"
                
            answer += "\n\n> ⚠️ 离线模式：基于关键词匹配，建议启用在线模式获得更智能的回答。"
        else:
            answer = f"未找到包含 '{request.query}' 的相关内容。请尝试其他关键词或上传更多文档。"
        
        processing_time = time.time() - start_time
        
        # 记录搜索日志
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO search_logs (query, results_count, processing_time, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            request.query,
            len(results),
            processing_time,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        
        return {
            "query": request.query,
            "results": results,
            "answer": answer,
            "processing_time": processing_time,
            "mode": "offline"
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
        SELECT id, filename, upload_time, status, file_size
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
            "file_size": row[4],
            "mode": "offline"
        })
    
    return documents

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # 获取文件路径
        cursor.execute('SELECT file_path FROM documents WHERE id = ?', (doc_id,))
        result = cursor.fetchone()
        
        if result:
            file_path = Path(result[0])
            if file_path.exists():
                file_path.unlink()
            
            # 从数据库删除
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"文档 {doc_id} 已删除"}
        else:
            conn.close()
            raise HTTPException(status_code=404, detail="文档不存在")
            
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """获取系统统计信息"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM documents')
    doc_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM search_logs')
    search_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(processing_time) FROM search_logs')
    avg_processing = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "documents_count": doc_count,
        "total_chunks": doc_count,  # 离线模式：文档即片段
        "search_count": search_count,
        "avg_processing_time": avg_processing,
        "mode": "offline",
        "chroma_db": {
            "status": "disabled",
            "mode": "offline"
        }
    }

@app.get("/api/search-logs")
async def get_search_logs(limit: int = Query(100, description="返回记录数")):
    """获取搜索日志"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT query, results_count, processing_time, timestamp
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
            "timestamp": row[3]
        })
    
    return logs

if __name__ == "__main__":
    import uvicorn
    print("🔧 启动离线版RAG系统...")
    uvicorn.run(app, host="0.0.0.0", port=8000)