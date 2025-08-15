"""
文档管理器 - Document Manager
企业级知识库管理系统，不是简单的文件上传！
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import sqlite3

# 导入文档处理器
from document_processor import EnhancedDocumentProcessor

logger = logging.getLogger(__name__)

@dataclass
class DocumentInfo:
    """文档信息"""
    doc_id: str
    filename: str
    file_path: str
    file_size: int
    upload_time: str
    status: str  # processing, completed, failed, archived
    chunk_count: int
    total_chars: int
    splitter_type: str
    metadata: Dict[str, Any]
    processing_time: float
    quality_score: Optional[float] = None
    tags: List[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SplitterConfig:
    """分片配置"""
    type: str  # recursive, token, char, semantic
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    language: str = "zh-cn"
    preserve_structure: bool = True
    
class DocumentManager:
    """企业级文档管理器"""
    
    def __init__(self, 
                 upload_dir: str,
                 db_path: str,
                 doc_processor: EnhancedDocumentProcessor):
        """
        初始化文档管理器
        
        Args:
            upload_dir: 文档上传目录
            db_path: 数据库路径
            doc_processor: 文档处理器实例
        """
        self.upload_dir = Path(upload_dir)
        self.db_path = db_path
        self.doc_processor = doc_processor
        
        # 创建必要目录
        self.upload_dir.mkdir(exist_ok=True)
        (self.upload_dir / "archives").mkdir(exist_ok=True)  # 归档目录
        (self.upload_dir / "temp").mkdir(exist_ok=True)      # 临时目录
        
        # 初始化数据库
        self._init_admin_database()
        
        # 预定义分片配置
        self.splitter_configs = self._load_splitter_configs()
        
    def _init_admin_database(self):
        """初始化管理后台数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文档管理表（扩展版）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                upload_time TEXT,
                status TEXT DEFAULT 'processing',
                chunk_count INTEGER DEFAULT 0,
                total_chars INTEGER DEFAULT 0,
                splitter_type TEXT,
                metadata TEXT,
                processing_time REAL,
                quality_score REAL,
                tags TEXT,  -- JSON array
                category TEXT,
                created_by TEXT,
                updated_time TEXT
            )
        ''')
        
        # 分片配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS splitter_configs (
                config_id TEXT PRIMARY KEY,
                config_name TEXT NOT NULL,
                config_type TEXT NOT NULL,
                chunk_size INTEGER,
                chunk_overlap INTEGER,
                separators TEXT,  -- JSON array
                language TEXT DEFAULT 'zh-cn',
                preserve_structure BOOLEAN DEFAULT TRUE,
                created_time TEXT,
                is_default BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 文档操作日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_operations (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                operation_type TEXT,  -- upload, delete, reprocess, archive
                operator TEXT,
                operation_time TEXT,
                operation_details TEXT,  -- JSON
                success BOOLEAN
            )
        ''')
        
        # 文档质量评估表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_quality (
                doc_id TEXT PRIMARY KEY,
                readability_score REAL,
                completeness_score REAL,
                relevance_score REAL,
                structure_score REAL,
                overall_score REAL,
                quality_issues TEXT,  -- JSON array
                evaluation_time TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("管理后台数据库初始化完成")
    
    def _load_splitter_configs(self) -> Dict[str, SplitterConfig]:
        """加载分片配置预设"""
        return {
            "default_recursive": SplitterConfig(
                type="recursive",
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
            ),
            "technical_docs": SplitterConfig(
                type="recursive", 
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", "."]
            ),
            "faq_style": SplitterConfig(
                type="char",
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n问：", "\n答：", "\nQ:", "\nA:", "\n\n"]
            ),
            "legal_docs": SplitterConfig(
                type="recursive",
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n第", "\n条", "\n款", "\n项", "\n\n", "。"]
            ),
            "product_manual": SplitterConfig(
                type="recursive",
                chunk_size=600,
                chunk_overlap=120,
                separators=["\n步骤", "\n注意", "\n警告", "\n\n", "。"]
            )
        }
    
    async def upload_document(self, 
                            file_content: bytes,
                            filename: str,
                            splitter_config_name: str = "default_recursive",
                            tags: List[str] = None,
                            category: str = None,
                            operator: str = "admin") -> Dict[str, Any]:
        """
        上传并处理文档
        
        Args:
            file_content: 文件内容
            filename: 文件名
            splitter_config_name: 分片配置名称
            tags: 文档标签
            category: 文档分类
            operator: 操作者
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        
        # 生成文档ID
        doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # 保存文件
        file_path = self.upload_dir / f"{doc_id}_{filename}"
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # 获取分片配置
            splitter_config = self.splitter_configs.get(splitter_config_name, 
                                                       self.splitter_configs["default_recursive"])
            
            # 处理文档
            result = self.doc_processor.process_and_store(
                file_path=str(file_path),
                metadata={
                    "filename": filename,
                    "upload_time": datetime.now().isoformat(),
                    "file_size": len(file_content),
                    "tags": tags or [],
                    "category": category,
                    "operator": operator
                },
                splitter_type=splitter_config.type
            )
            
            processing_time = time.time() - start_time
            
            # 评估文档质量
            quality_score = await self._evaluate_document_quality(file_path, result)
            
            # 保存到管理数据库
            doc_info = DocumentInfo(
                doc_id=doc_id,
                filename=filename,
                file_path=str(file_path),
                file_size=len(file_content),
                upload_time=datetime.now().isoformat(),
                status='completed',
                chunk_count=result['chunks_count'],
                total_chars=result['total_chars'],
                splitter_type=splitter_config.type,
                metadata={
                    "tags": tags or [],
                    "category": category,
                    "operator": operator,
                    "splitter_config": splitter_config_name
                },
                processing_time=processing_time,
                quality_score=quality_score,
                tags=tags or [],
                category=category
            )
            
            await self._save_document_info(doc_info)
            
            # 记录操作日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="upload",
                operator=operator,
                operation_details={
                    "filename": filename,
                    "file_size": len(file_content),
                    "splitter_config": splitter_config_name,
                    "processing_time": processing_time,
                    "quality_score": quality_score
                },
                success=True
            )
            
            logger.info(f"文档上传成功: {filename} -> {doc_id}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "filename": filename,
                "chunks_count": result['chunks_count'],
                "total_chars": result['total_chars'],
                "processing_time": processing_time,
                "quality_score": quality_score,
                "message": "文档上传并处理成功"
            }
            
        except Exception as e:
            logger.error(f"文档上传失败: {str(e)}")
            
            # 清理文件
            if file_path.exists():
                file_path.unlink()
                
            # 记录失败日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="upload",
                operator=operator,
                operation_details={
                    "filename": filename,
                    "error": str(e)
                },
                success=False
            )
            
            return {
                "success": False,
                "error": str(e),
                "message": "文档上传失败"
            }
    
    async def get_documents_list(self, 
                               category: Optional[str] = None,
                               status: Optional[str] = None,
                               tags: Optional[List[str]] = None,
                               limit: int = 50,
                               offset: int = 0) -> Dict[str, Any]:
        """
        获取文档列表（支持筛选）
        
        Args:
            category: 分类筛选
            status: 状态筛选
            tags: 标签筛选
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict: 文档列表和统计信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询条件
        where_conditions = []
        params = []
        
        if category:
            where_conditions.append("category = ?")
            params.append(category)
            
        if status:
            where_conditions.append("status = ?")
            params.append(status)
            
        # 标签筛选（简单实现，可优化）
        if tags:
            for tag in tags:
                where_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 获取文档列表
        query = f'''
            SELECT * FROM admin_documents 
            WHERE {where_clause}
            ORDER BY upload_time DESC
            LIMIT ? OFFSET ?
        '''
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 获取总数
        count_query = f'''
            SELECT COUNT(*) FROM admin_documents 
            WHERE {where_clause}
        '''
        cursor.execute(count_query, params[:-2])  # 去掉LIMIT和OFFSET参数
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        # 转换为字典格式
        documents = []
        columns = [description[0] for description in cursor.description]
        
        for row in rows:
            doc_dict = dict(zip(columns, row))
            # 解析JSON字段
            if doc_dict['metadata']:
                doc_dict['metadata'] = json.loads(doc_dict['metadata'])
            if doc_dict['tags']:
                doc_dict['tags'] = json.loads(doc_dict['tags'])
            documents.append(doc_dict)
        
        return {
            "documents": documents,
            "total_count": total_count,
            "current_page": offset // limit + 1,
            "total_pages": (total_count + limit - 1) // limit,
            "has_next": offset + limit < total_count
        }
    
    async def delete_document(self, doc_id: str, operator: str = "admin") -> Dict[str, Any]:
        """
        删除文档（支持软删除和硬删除）
        
        Args:
            doc_id: 文档ID
            operator: 操作者
            
        Returns:
            Dict: 删除结果
        """
        try:
            # 从向量数据库删除
            success = self.doc_processor.delete_document(doc_id)
            
            if success:
                # 从管理数据库获取文档信息
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM admin_documents WHERE doc_id = ?', (doc_id,))
                doc_info = cursor.fetchone()
                
                if doc_info:
                    # 移动文件到归档目录（软删除）
                    file_path = Path(doc_info[2])  # file_path字段
                    if file_path.exists():
                        archive_path = self.upload_dir / "archives" / file_path.name
                        file_path.rename(archive_path)
                    
                    # 更新状态为已删除
                    cursor.execute('''
                        UPDATE admin_documents 
                        SET status = 'deleted', updated_time = ?
                        WHERE doc_id = ?
                    ''', (datetime.now().isoformat(), doc_id))
                    
                    conn.commit()
                    conn.close()
                    
                    # 记录操作日志
                    await self._log_operation(
                        doc_id=doc_id,
                        operation_type="delete",
                        operator=operator,
                        operation_details={
                            "filename": doc_info[1],  # filename字段
                            "archive_path": str(archive_path)
                        },
                        success=True
                    )
                    
                    logger.info(f"文档删除成功: {doc_id}")
                    return {
                        "success": True,
                        "message": f"文档 {doc_id} 已删除"
                    }
                else:
                    return {
                        "success": False,
                        "error": "文档记录不存在"
                    }
            else:
                return {
                    "success": False,
                    "error": "从向量数据库删除失败"
                }
                
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            
            # 记录失败日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="delete",
                operator=operator,
                operation_details={"error": str(e)},
                success=False
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def reprocess_document(self, 
                               doc_id: str,
                               new_splitter_config: str = None,
                               operator: str = "admin") -> Dict[str, Any]:
        """
        重新处理文档（使用新的分片策略）
        
        Args:
            doc_id: 文档ID
            new_splitter_config: 新的分片配置名称
            operator: 操作者
            
        Returns:
            Dict: 重处理结果
        """
        try:
            # 获取文档信息
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM admin_documents WHERE doc_id = ?', (doc_id,))
            doc_info = cursor.fetchone()
            conn.close()
            
            if not doc_info:
                return {
                    "success": False,
                    "error": "文档不存在"
                }
            
            file_path = doc_info[2]  # file_path字段
            filename = doc_info[1]   # filename字段
            
            # 如果文件在归档目录，先恢复
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                archive_path = self.upload_dir / "archives" / file_path_obj.name
                if archive_path.exists():
                    archive_path.rename(file_path_obj)
                else:
                    return {
                        "success": False,
                        "error": "源文件不存在"
                    }
            
            # 先删除现有向量数据
            self.doc_processor.delete_document(doc_id)
            
            # 获取分片配置
            if new_splitter_config:
                splitter_config = self.splitter_configs.get(new_splitter_config)
            else:
                # 使用原配置
                metadata = json.loads(doc_info[9]) if doc_info[9] else {}
                splitter_config_name = metadata.get("splitter_config", "default_recursive")
                splitter_config = self.splitter_configs.get(splitter_config_name)
            
            start_time = time.time()
            
            # 重新处理
            result = self.doc_processor.process_and_store(
                file_path=file_path,
                metadata={
                    "filename": filename,
                    "reprocess_time": datetime.now().isoformat(),
                    "operator": operator
                },
                splitter_type=splitter_config.type
            )
            
            processing_time = time.time() - start_time
            
            # 重新评估质量
            quality_score = await self._evaluate_document_quality(file_path_obj, result)
            
            # 更新数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE admin_documents 
                SET chunk_count = ?, total_chars = ?, processing_time = ?,
                    quality_score = ?, status = 'completed', updated_time = ?
                WHERE doc_id = ?
            ''', (
                result['chunks_count'],
                result['total_chars'],
                processing_time,
                quality_score,
                datetime.now().isoformat(),
                doc_id
            ))
            conn.commit()
            conn.close()
            
            # 记录操作日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="reprocess",
                operator=operator,
                operation_details={
                    "splitter_config": new_splitter_config,
                    "chunks_count": result['chunks_count'],
                    "processing_time": processing_time,
                    "quality_score": quality_score
                },
                success=True
            )
            
            logger.info(f"文档重处理成功: {doc_id}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_count": result['chunks_count'],
                "total_chars": result['total_chars'],
                "processing_time": processing_time,
                "quality_score": quality_score,
                "message": "文档重处理成功"
            }
            
        except Exception as e:
            logger.error(f"重处理文档失败: {str(e)}")
            
            # 记录失败日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="reprocess",
                operator=operator,
                operation_details={"error": str(e)},
                success=False
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_splitter_configs(self) -> Dict[str, Any]:
        """获取所有分片配置"""
        configs = {}
        for name, config in self.splitter_configs.items():
            configs[name] = {
                "name": name,
                "type": config.type,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "separators": config.separators,
                "language": config.language,
                "preserve_structure": config.preserve_structure
            }
        return configs
    
    async def add_splitter_config(self, 
                                config_name: str,
                                config: SplitterConfig) -> Dict[str, Any]:
        """添加新的分片配置"""
        try:
            self.splitter_configs[config_name] = config
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO splitter_configs 
                (config_id, config_name, config_type, chunk_size, chunk_overlap,
                 separators, language, preserve_structure, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config_name,
                config_name,
                config.type,
                config.chunk_size,
                config.chunk_overlap,
                json.dumps(config.separators),
                config.language,
                config.preserve_structure,
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"分片配置 {config_name} 添加成功"
            }
            
        except Exception as e:
            logger.error(f"添加分片配置失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _evaluate_document_quality(self, file_path: Path, process_result: Dict) -> float:
        """
        评估文档质量
        
        Args:
            file_path: 文件路径
            process_result: 处理结果
            
        Returns:
            float: 质量评分 (0-1)
        """
        try:
            # 简单的质量评估指标
            scores = []
            
            # 1. 内容完整性（基于字符数）
            total_chars = process_result.get('total_chars', 0)
            if total_chars > 1000:
                scores.append(1.0)
            elif total_chars > 500:
                scores.append(0.8)
            elif total_chars > 100:
                scores.append(0.6)
            else:
                scores.append(0.3)
            
            # 2. 分片合理性（分片数量与内容的比例）
            chunks_count = process_result.get('chunks_count', 0)
            if chunks_count > 0:
                avg_chunk_size = total_chars / chunks_count
                if 300 <= avg_chunk_size <= 800:  # 理想分片大小
                    scores.append(1.0)
                elif 200 <= avg_chunk_size <= 1000:
                    scores.append(0.8)
                else:
                    scores.append(0.6)
            else:
                scores.append(0.0)
            
            # 3. 文件类型适配性
            file_ext = file_path.suffix.lower()
            if file_ext in ['.txt', '.md']:
                scores.append(1.0)  # 文本文件质量最好
            elif file_ext in ['.pdf', '.doc', '.docx']:
                scores.append(0.8)  # 结构化文档
            else:
                scores.append(0.6)
            
            # 计算平均分
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return 0.5  # 默认中等质量
    
    async def _save_document_info(self, doc_info: DocumentInfo):
        """保存文档信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO admin_documents 
            (doc_id, filename, file_path, file_size, upload_time, status,
             chunk_count, total_chars, splitter_type, metadata, processing_time,
             quality_score, tags, category, created_by, updated_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_info.doc_id,
            doc_info.filename,
            doc_info.file_path,
            doc_info.file_size,
            doc_info.upload_time,
            doc_info.status,
            doc_info.chunk_count,
            doc_info.total_chars,
            doc_info.splitter_type,
            json.dumps(doc_info.metadata),
            doc_info.processing_time,
            doc_info.quality_score,
            json.dumps(doc_info.tags or []),
            doc_info.category,
            doc_info.metadata.get('operator', 'admin'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _log_operation(self, 
                           doc_id: str,
                           operation_type: str,
                           operator: str,
                           operation_details: Dict,
                           success: bool):
        """记录操作日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO document_operations 
            (doc_id, operation_type, operator, operation_time, operation_details, success)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            operation_type,
            operator,
            datetime.now().isoformat(),
            json.dumps(operation_details),
            success
        ))
        
        conn.commit()
        conn.close()
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基础统计
        cursor.execute('SELECT COUNT(*) FROM admin_documents WHERE status != "deleted"')
        total_docs = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(chunk_count) FROM admin_documents WHERE status = "completed"')
        total_chunks = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(quality_score) FROM admin_documents WHERE quality_score IS NOT NULL')
        avg_quality = cursor.fetchone()[0] or 0
        
        # 按状态分组
        cursor.execute('SELECT status, COUNT(*) FROM admin_documents GROUP BY status')
        status_stats = dict(cursor.fetchall())
        
        # 按分类分组
        cursor.execute('SELECT category, COUNT(*) FROM admin_documents WHERE category IS NOT NULL GROUP BY category')
        category_stats = dict(cursor.fetchall())
        
        # 按分片类型分组
        cursor.execute('SELECT splitter_type, COUNT(*) FROM admin_documents GROUP BY splitter_type')
        splitter_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "average_quality_score": round(avg_quality, 3) if avg_quality else 0,
            "status_distribution": status_stats,
            "category_distribution": category_stats,
            "splitter_distribution": splitter_stats,
            "last_updated": datetime.now().isoformat()
        }


# 测试代码
if __name__ == "__main__":
    import asyncio
    from document_processor import EnhancedDocumentProcessor
    
    async def test_document_manager():
        # 初始化组件
        doc_processor = EnhancedDocumentProcessor(
            chroma_db_path="./test_chroma_admin",
            collection_name="admin_test"
        )
        
        doc_manager = DocumentManager(
            upload_dir="./test_uploads",
            db_path="./test_admin.db",
            doc_processor=doc_processor
        )
        
        # 测试获取配置
        configs = await doc_manager.get_splitter_configs()
        print(f"分片配置: {list(configs.keys())}")
        
        # 测试统计信息
        stats = await doc_manager.get_document_statistics()
        print(f"文档统计: {stats}")
    
    asyncio.run(test_document_manager())