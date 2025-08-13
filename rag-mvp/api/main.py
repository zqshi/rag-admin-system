#!/usr/bin/env python3
"""
RAG MVP系统 - 极简可运行版本
目标：2小时内实现文档上传、向量化、智能问答
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# 向量化和检索
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 文档处理
import PyPDF2

# 创建FastAPI应用
app = FastAPI(
    title="RAG MVP System",
    description="快速原型 - 文档智能问答系统",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "rag.db"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# 全局变量
model = None
index = None
dimension = 384  # MiniLM-L6-v2的输出维度

def init_vector_model():
    """初始化向量模型"""
    global model, index
    print("正在加载向量模型...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(dimension)
    print("向量模型加载完成！")

def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 文档表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            status TEXT DEFAULT 'processing',
            chunk_count INTEGER DEFAULT 0,
            total_chars INTEGER DEFAULT 0
        )
    ''')
    
    # 文档片段表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            content TEXT NOT NULL,
            position INTEGER NOT NULL,
            vector_id INTEGER,
            char_count INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    ''')
    
    # 搜索日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            result_count INTEGER,
            processing_time REAL,
            created_at TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库初始化完成！")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    简单的固定大小文本分片
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    text = text.strip()
    text_len = len(text)
    
    # 如果文本很短，直接作为一个片段
    if text_len <= chunk_size:
        return [text]
    
    start = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # 如果是最后一个片段，退出
        if end >= text_len:
            break
            
        start = end - overlap
    
    return chunks

def extract_pdf_text(file_path: Path) -> str:
    """提取PDF文本内容"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        print(f"跳过第{page_num + 1}页: {page_error}")
                        continue
                        
                print(f"成功提取PDF: {file_path.name}, 共{num_pages}页")
            except Exception as pdf_error:
                # 如果是加密PDF，尝试用空密码解密
                print(f"PDF可能被加密，尝试解密: {pdf_error}")
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt('')  # 尝试空密码
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    except:
                        raise HTTPException(status_code=400, detail="PDF文件被加密，无法读取")
                else:
                    raise
                    
    except Exception as e:
        print(f"PDF提取错误 ({file_path.name}): {str(e)}")
        # 返回错误信息而不是抛出异常，让用户知道问题
        return f"[PDF提取失败: {str(e)}]"
    
    return text if text else "[PDF内容为空或无法提取]"

def extract_text_from_file(file_path: Path, filename: str) -> str:
    """根据文件类型提取文本"""
    ext = filename.lower().split('.')[-1]
    
    if ext == 'pdf':
        return extract_pdf_text(file_path)
    elif ext in ['txt', 'md', 'text', 'markdown']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

# 数据模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    processing_time: float
    total_results: int

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_time: str
    status: str
    chunk_count: int
    total_chars: int

# API路由
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    init_database()
    init_vector_model()

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "RAG MVP System Running",
        "version": "0.1.0",
        "endpoints": {
            "upload": "/api/upload",
            "query": "/api/query", 
            "documents": "/api/documents",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_size": index.ntotal if index else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并处理文档"""
    start_time = datetime.now()
    
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    # 生成唯一ID
    doc_id = hashlib.md5(
        f"{file.filename}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    try:
        # 保存文件
        file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 记录到数据库
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT INTO documents 
               (id, filename, file_path, upload_time, status) 
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, file.filename, str(file_path), 
             datetime.now().isoformat(), "processing")
        )
        conn.commit()
        
        # 提取文本
        try:
            text = extract_text_from_file(file_path, file.filename)
        except Exception as e:
            cursor.execute(
                "UPDATE documents SET status = ? WHERE id = ?",
                ("failed", doc_id)
            )
            conn.commit()
            conn.close()
            return {
                "status": "error",
                "message": f"文档处理失败: {str(e)}",
                "doc_id": doc_id,
                "filename": file.filename
            }
        
        # 检查是否提取失败
        if text and text.startswith("[PDF提取失败"):
            cursor.execute(
                "UPDATE documents SET status = ? WHERE id = ?",
                ("failed", doc_id)
            )
            conn.commit()
            conn.close()
            return {
                "status": "error",
                "message": text,
                "doc_id": doc_id,
                "filename": file.filename
            }
        
        if not text or not text.strip() or text == "[PDF内容为空或无法提取]":
            cursor.execute(
                "UPDATE documents SET status = ? WHERE id = ?",
                ("empty", doc_id)
            )
            conn.commit()
            conn.close()
            return {
                "status": "warning",
                "message": "文档内容为空或无法提取",
                "doc_id": doc_id,
                "filename": file.filename
            }
        
        # 文本分片
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # 向量化并存储
        for i, chunk in enumerate(chunks):
            # 生成向量
            embedding = model.encode([chunk])[0]
            
            # 添加到FAISS索引
            vector_id = index.ntotal
            index.add(np.array([embedding], dtype=np.float32))
            
            # 存储到数据库
            cursor.execute(
                """INSERT INTO chunks 
                   (doc_id, content, position, vector_id, char_count) 
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, chunk, i, vector_id, len(chunk))
            )
        
        # 更新文档状态
        cursor.execute(
            """UPDATE documents 
               SET status = ?, chunk_count = ?, total_chars = ? 
               WHERE id = ?""",
            ("completed", len(chunks), len(text), doc_id)
        )
        conn.commit()
        conn.close()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_chars": len(text),
            "processing_time": f"{processing_time:.2f}秒"
        }
        
    except Exception as e:
        # 清理失败的上传
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """智能问答查询"""
    start_time = datetime.now()
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    # 检查是否有文档
    if index.ntotal == 0:
        return QueryResponse(
            answer="系统中还没有任何文档，请先上传文档。",
            sources=[],
            processing_time=0,
            total_results=0
        )
    
    try:
        # 查询向量化
        query_embedding = model.encode([request.query])[0]
        
        # FAISS搜索
        k = min(request.top_k, index.ntotal)
        distances, indices = index.search(
            np.array([query_embedding], dtype=np.float32), k
        )
        
        # 获取对应的文本片段
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        sources = []
        contexts = []
        
        for idx, distance in zip(indices[0], distances[0]):
            cursor.execute(
                """SELECT c.content, c.position, d.filename, c.char_count
                   FROM chunks c 
                   JOIN documents d ON c.doc_id = d.id 
                   WHERE c.vector_id = ?""",
                (int(idx),)
            )
            result = cursor.fetchone()
            
            if result:
                content, position, filename, char_count = result
                score = float(1 / (1 + distance))  # 转换为相似度
                
                sources.append({
                    "content": content[:300] + "..." if len(content) > 300 else content,
                    "filename": filename,
                    "position": position + 1,
                    "score": round(score, 3),
                    "char_count": char_count
                })
                contexts.append(content)
        
        # 记录搜索日志
        processing_time = (datetime.now() - start_time).total_seconds()
        cursor.execute(
            """INSERT INTO search_logs 
               (query, result_count, processing_time, created_at) 
               VALUES (?, ?, ?, ?)""",
            (request.query, len(sources), processing_time, 
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        # 生成答案（简化版，实际应接入LLM）
        if contexts:
            # 简单拼接最相关的内容作为答案
            answer = f"根据知识库中的 {len(contexts)} 个相关片段：\n\n"
            
            # 取前3个最相关的片段
            for i, ctx in enumerate(contexts[:3], 1):
                preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
                answer += f"{i}. {preview}\n\n"
                
            answer += f"\n💡 提示：这是基于向量相似度的检索结果，完整内容请查看源文件。"
        else:
            answer = "未找到与您查询相关的内容。"
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 3),
            total_results=len(sources)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """获取所有文档列表"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT id, filename, upload_time, status, chunk_count, total_chars 
           FROM documents 
           ORDER BY upload_time DESC"""
    )
    documents = cursor.fetchall()
    conn.close()
    
    return [
        DocumentInfo(
            id=doc[0],
            filename=doc[1],
            upload_time=doc[2],
            status=doc[3],
            chunk_count=doc[4] or 0,
            total_chars=doc[5] or 0
        )
        for doc in documents
    ]

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 检查文档是否存在
    cursor.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="文档不存在")
    
    file_path = Path(result[0])
    
    # 删除文件
    if file_path.exists():
        file_path.unlink()
    
    # 删除数据库记录
    cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    # 注意：FAISS索引中的向量无法单独删除，需要重建索引
    # 这是MVP版本的限制，生产版本应该使用支持删除的向量数据库
    
    return {"status": "success", "message": f"文档 {doc_id} 已删除"}

@app.get("/api/stats")
async def get_statistics():
    """获取系统统计信息"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # 文档统计
    cursor.execute("SELECT COUNT(*), SUM(chunk_count), SUM(total_chars) FROM documents WHERE status = 'completed'")
    doc_stats = cursor.fetchone()
    
    # 搜索统计
    cursor.execute("SELECT COUNT(*), AVG(processing_time) FROM search_logs")
    search_stats = cursor.fetchone()
    
    conn.close()
    
    return {
        "documents": {
            "total": doc_stats[0] or 0,
            "total_chunks": doc_stats[1] or 0,
            "total_chars": doc_stats[2] or 0
        },
        "searches": {
            "total": search_stats[0] or 0,
            "avg_time": round(search_stats[1], 3) if search_stats[1] else 0
        },
        "index": {
            "vectors": index.ntotal if index else 0,
            "dimension": dimension
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 RAG MVP系统启动中...")
    print("="*50)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📤 上传目录: {UPLOAD_DIR}")
    print(f"🗄️ 数据库: {DB_PATH}")
    print("="*50 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # 修改为False避免警告
        log_level="info"
    )