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
from utils.file_deduplication import FileDeduplication

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

@dataclass
class FAQItem:
    """FAQ条目"""
    faq_id: str
    question: str
    answer: str
    doc_id: str
    created_time: str
    status: str = "active"  # active, inactive, deleted
    quality_score: Optional[float] = None
    extracted_by: str = "llm"  # llm, manual
    category: Optional[str] = None
    tags: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
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
        
        # 初始化文件去重处理器
        self.file_deduplication = FileDeduplication(
            db_path=self.db_path,
            uploads_dir=str(self.upload_dir)
        )
        
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
        
        # FAQ管理表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faqs (
                faq_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                created_time TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                quality_score REAL,
                extracted_by TEXT DEFAULT 'llm',
                category TEXT,
                tags TEXT,  -- JSON array
                metadata TEXT,  -- JSON object
                updated_time TEXT,
                FOREIGN KEY (doc_id) REFERENCES admin_documents (doc_id)
            )
        ''')
        
        # FAQ抽取任务表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faq_extraction_tasks (
                task_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
                created_time TEXT NOT NULL,
                started_time TEXT,
                completed_time TEXT,
                extracted_count INTEGER DEFAULT 0,
                error_message TEXT,
                extraction_params TEXT,  -- JSON object
                FOREIGN KEY (doc_id) REFERENCES admin_documents (doc_id)
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
                            operator: str = "admin",
                            force_replace: bool = False) -> Dict[str, Any]:
        """
        上传并处理文档
        
        Args:
            file_content: 文件内容
            filename: 文件名
            splitter_config_name: 分片配置名称
            tags: 文档标签
            category: 文档分类
            operator: 操作者
            force_replace: 是否强制替换重复文档
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        
        # 首先检查文档重复
        is_duplicate, duplicates = self.file_deduplication.check_duplicates(
            filename=filename,
            file_content=file_content,
            strict_mode=True  # 使用严格模式，只检查完全相同的文件
        )
        
        if is_duplicate and not force_replace:
            # 记录重复检测
            for dup in duplicates:
                self.file_deduplication.record_duplicate(
                    original_fingerprint_id=dup['fingerprint_id'],
                    duplicate_filename=filename,
                    duplicate_size=len(file_content),
                    duplicate_hash=self.file_deduplication.calculate_file_hash(file_content),
                    action_taken='blocked',
                    metadata={'duplicate_type': dup['duplicate_type']}
                )
            
            return {
                "success": False,
                "error": "duplicate_file",
                "message": "文档已存在",
                "duplicates": duplicates,
                "suggestion": "该文档已存在于系统中，如需更新请选择强制替换选项"
            }
        
        # 保存文件（临时文件名）
        temp_doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        file_path = self.upload_dir / f"{temp_doc_id}_{filename}"
        
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
            
            # 使用document_processor生成的真实doc_id
            doc_id = result['doc_id']
            
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
            
            # 注册文件到去重系统
            fingerprint_id = self.file_deduplication.register_file(
                filename=filename,
                file_size=len(file_content),
                file_content=file_content,
                doc_id=doc_id,
                file_path=str(file_path),
                text_content=None,  # 可以后续优化添加文本内容
                metadata={
                    "operator": operator,
                    "category": category,
                    "tags": tags or [],
                    "processing_time": processing_time,
                    "quality_score": quality_score
                }
            )
            
            # 如果是强制替换，处理旧文档
            if force_replace and is_duplicate:
                await self._handle_duplicate_replacement(duplicates, doc_id, operator)
            
            # 记录操作日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="upload" if not force_replace else "replace",
                operator=operator,
                operation_details={
                    "filename": filename,
                    "file_size": len(file_content),
                    "splitter_config": splitter_config_name,
                    "processing_time": processing_time,
                    "quality_score": quality_score,
                    "fingerprint_id": fingerprint_id,
                    "is_replacement": force_replace and is_duplicate
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
        # 立即获取列名，避免被后续查询覆盖
        columns = [description[0] for description in cursor.description]
        
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
        
        for row in rows:
            doc_dict = dict(zip(columns, row))
            
            # 解析JSON字段，处理空值
            if 'metadata' in doc_dict and doc_dict['metadata']:
                try:
                    doc_dict['metadata'] = json.loads(doc_dict['metadata'])
                except (json.JSONDecodeError, TypeError):
                    doc_dict['metadata'] = {}
            else:
                doc_dict['metadata'] = {}
            
            if 'tags' in doc_dict and doc_dict['tags']:
                try:
                    doc_dict['tags'] = json.loads(doc_dict['tags'])
                except (json.JSONDecodeError, TypeError):
                    doc_dict['tags'] = []
            else:
                doc_dict['tags'] = []
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
    
    async def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """
        获取文档详情
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Dict: 文档详情信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM admin_documents WHERE doc_id = ?', (doc_id,))
            doc_info = cursor.fetchone()
            
            if not doc_info:
                conn.close()
                return {
                    "success": False,
                    "error": "文档不存在"
                }
            
            # 获取列名
            columns = [description[0] for description in cursor.description]
            doc_dict = dict(zip(columns, doc_info))
            
            # 解析JSON字段
            if doc_dict['metadata']:
                try:
                    doc_dict['metadata'] = json.loads(doc_dict['metadata'])
                except (json.JSONDecodeError, TypeError):
                    doc_dict['metadata'] = {}
            else:
                doc_dict['metadata'] = {}
            
            if doc_dict['tags']:
                try:
                    doc_dict['tags'] = json.loads(doc_dict['tags'])
                except (json.JSONDecodeError, TypeError):
                    doc_dict['tags'] = []
            else:
                doc_dict['tags'] = []
            
            conn.close()
            
            # 检查文件是否存在
            file_path = Path(doc_dict['file_path'])
            file_exists = file_path.exists()
            
            # 如果文件不存在，检查归档目录
            if not file_exists:
                archive_path = self.upload_dir / "archives" / file_path.name
                if archive_path.exists():
                    file_path = archive_path
                    file_exists = True
            
            return {
                "success": True,
                "document": doc_dict,
                "file_exists": file_exists,
                "file_path": str(file_path),
                "file_size_mb": round(doc_dict['file_size'] / (1024 * 1024), 2) if doc_dict['file_size'] else 0
            }
            
        except Exception as e:
            logger.error(f"获取文档详情失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_document_chunks(self, doc_id: str) -> Dict[str, Any]:
        """
        获取文档的所有切片
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Dict: 切片列表和统计信息
        """
        try:
            # 使用document_processor获取切片
            chunks = self.doc_processor.get_document_chunks(doc_id)
            
            # 如果没找到，可能是doc_id不一致，尝试通过文件路径重新生成
            if not chunks:
                # 从数据库获取文件路径
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT file_path FROM admin_documents WHERE doc_id = ?', (doc_id,))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    file_path = result[0]
                    # 用document_processor的方式重新生成doc_id
                    import hashlib
                    actual_doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
                    logger.info(f"Doc ID不匹配，尝试使用文件路径生成的ID: {actual_doc_id}")
                    
                    # 用新的doc_id重试
                    chunks = self.doc_processor.get_document_chunks(actual_doc_id)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "未找到文档切片"
                }
            
            # 计算统计信息
            total_length = sum(chunk['content_length'] for chunk in chunks)
            avg_length = total_length / len(chunks) if chunks else 0
            
            return {
                "success": True,
                "chunks": chunks,
                "statistics": {
                    "total_chunks": len(chunks),
                    "total_characters": total_length,
                    "average_chunk_length": round(avg_length, 2),
                    "doc_id": doc_id
                }
            }
            
        except Exception as e:
            logger.error(f"获取文档切片失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

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
    
    # ==================== FAQ管理方法 ====================
    
    async def extract_faqs_from_document(self, 
                                       doc_id: str, 
                                       operator: str = "admin",
                                       max_faqs: int = 10) -> Dict[str, Any]:
        """
        从文档中抽取FAQ
        
        Args:
            doc_id: 文档ID
            operator: 操作者
            max_faqs: 最大FAQ数量
            
        Returns:
            Dict: 抽取结果
        """
        task_id = None
        try:
            # 导入FAQ抽取器
            from .faq_extractor import FAQExtractor
            
            # 创建抽取任务记录
            task_id = hashlib.md5(f"faq_{doc_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 记录抽取任务
            cursor.execute('''
                INSERT INTO faq_extraction_tasks 
                (task_id, doc_id, status, created_time, extraction_params)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                task_id,
                doc_id,
                'processing',
                datetime.now().isoformat(),
                json.dumps({"max_faqs": max_faqs, "operator": operator})
            ))
            conn.commit()
            
            # 更新任务开始时间
            cursor.execute('''
                UPDATE faq_extraction_tasks 
                SET started_time = ?, status = ?
                WHERE task_id = ?
            ''', (datetime.now().isoformat(), 'processing', task_id))
            conn.commit()
            conn.close()
            
            # 获取文档切片
            chunks_result = await self.get_document_chunks(doc_id)
            if not chunks_result["success"]:
                await self._update_extraction_task(task_id, 'failed', error_message="无法获取文档切片")
                return {
                    "success": False,
                    "error": "无法获取文档切片"
                }
            
            chunks = chunks_result["chunks"]
            
            # 初始化FAQ抽取器
            extractor = FAQExtractor(llm_provider="local")  # 使用本地开源大模型进行FAQ抽取
            
            # 抽取FAQ
            faqs = await extractor.extract_faqs_from_chunks(chunks, doc_id, max_faqs)
            
            # 保存FAQ到数据库
            saved_count = 0
            for faq in faqs:
                success = await self._save_faq(faq)
                if success:
                    saved_count += 1
            
            # 更新任务状态
            await self._update_extraction_task(
                task_id, 
                'completed', 
                extracted_count=saved_count,
                completed_time=datetime.now().isoformat()
            )
            
            # 记录操作日志
            await self._log_operation(
                doc_id=doc_id,
                operation_type="extract_faq",
                operator=operator,
                operation_details={
                    "task_id": task_id,
                    "extracted_count": saved_count,
                    "max_faqs": max_faqs
                },
                success=True
            )
            
            logger.info(f"文档{doc_id}的FAQ抽取完成，成功保存{saved_count}个FAQ")
            
            return {
                "success": True,
                "task_id": task_id,
                "extracted_count": saved_count,
                "faqs": faqs,
                "message": f"成功抽取{saved_count}个FAQ"
            }
            
        except Exception as e:
            logger.error(f"FAQ抽取失败: {str(e)}")
            if task_id:
                await self._update_extraction_task(task_id, 'failed', error_message=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_faqs_list(self, 
                          doc_id: Optional[str] = None,
                          status: Optional[str] = None,
                          category: Optional[str] = None,
                          limit: int = 50,
                          offset: int = 0) -> Dict[str, Any]:
        """
        获取FAQ列表
        
        Args:
            doc_id: 文档ID筛选
            status: 状态筛选
            category: 分类筛选
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict: FAQ列表和统计信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建查询条件
            where_conditions = []
            params = []
            
            if doc_id:
                where_conditions.append("doc_id = ?")
                params.append(doc_id)
            
            if status:
                where_conditions.append("status = ?")
                params.append(status)
                
            if category:
                where_conditions.append("category = ?")
                params.append(category)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 获取FAQ列表
            query = f'''
                SELECT f.*, d.filename as doc_filename
                FROM faqs f
                LEFT JOIN admin_documents d ON f.doc_id = d.doc_id
                WHERE {where_clause}
                ORDER BY f.created_time DESC
                LIMIT ? OFFSET ?
            '''
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # 获取总数
            count_query = f'''
                SELECT COUNT(*) FROM faqs 
                WHERE {where_clause}
            '''
            cursor.execute(count_query, params[:-2])
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            # 转换为字典格式
            faqs = []
            for row in rows:
                faq_dict = dict(zip(columns, row))
                
                # 解析JSON字段
                if faq_dict['tags']:
                    try:
                        faq_dict['tags'] = json.loads(faq_dict['tags'])
                    except (json.JSONDecodeError, TypeError):
                        faq_dict['tags'] = []
                else:
                    faq_dict['tags'] = []
                
                if faq_dict['metadata']:
                    try:
                        faq_dict['metadata'] = json.loads(faq_dict['metadata'])
                    except (json.JSONDecodeError, TypeError):
                        faq_dict['metadata'] = {}
                else:
                    faq_dict['metadata'] = {}
                
                faqs.append(faq_dict)
            
            return {
                "faqs": faqs,
                "total_count": total_count,
                "current_page": offset // limit + 1,
                "total_pages": (total_count + limit - 1) // limit,
                "has_next": offset + limit < total_count
            }
            
        except Exception as e:
            logger.error(f"获取FAQ列表失败: {str(e)}")
            return {
                "faqs": [],
                "total_count": 0,
                "error": str(e)
            }
    
    async def update_faq(self, 
                        faq_id: str, 
                        updates: Dict[str, Any], 
                        operator: str = "admin") -> Dict[str, Any]:
        """
        更新FAQ
        
        Args:
            faq_id: FAQ ID
            updates: 更新数据
            operator: 操作者
            
        Returns:
            Dict: 更新结果
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建更新SQL
            update_fields = []
            params = []
            
            allowed_fields = ['question', 'answer', 'status', 'category', 'tags', 'quality_score']
            
            for field, value in updates.items():
                if field in allowed_fields:
                    if field in ['tags']:
                        update_fields.append(f"{field} = ?")
                        params.append(json.dumps(value) if value else None)
                    else:
                        update_fields.append(f"{field} = ?")
                        params.append(value)
            
            if not update_fields:
                return {
                    "success": False,
                    "error": "没有有效的更新字段"
                }
            
            # 添加更新时间
            update_fields.append("updated_time = ?")
            params.append(datetime.now().isoformat())
            params.append(faq_id)
            
            update_sql = f'''
                UPDATE faqs 
                SET {", ".join(update_fields)}
                WHERE faq_id = ?
            '''
            
            cursor.execute(update_sql, params)
            
            if cursor.rowcount == 0:
                conn.close()
                return {
                    "success": False,
                    "error": "FAQ不存在"
                }
            
            conn.commit()
            conn.close()
            
            logger.info(f"FAQ {faq_id} 更新成功")
            return {
                "success": True,
                "message": "FAQ更新成功"
            }
            
        except Exception as e:
            logger.error(f"更新FAQ失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_faq(self, faq_id: str, operator: str = "admin") -> Dict[str, Any]:
        """
        删除FAQ
        
        Args:
            faq_id: FAQ ID
            operator: 操作者
            
        Returns:
            Dict: 删除结果
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 软删除：更新状态为deleted
            cursor.execute('''
                UPDATE faqs 
                SET status = 'deleted', updated_time = ?
                WHERE faq_id = ?
            ''', (datetime.now().isoformat(), faq_id))
            
            if cursor.rowcount == 0:
                conn.close()
                return {
                    "success": False,
                    "error": "FAQ不存在"
                }
            
            conn.commit()
            conn.close()
            
            logger.info(f"FAQ {faq_id} 删除成功")
            return {
                "success": True,
                "message": "FAQ删除成功"
            }
            
        except Exception as e:
            logger.error(f"删除FAQ失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_faq_statistics(self) -> Dict[str, Any]:
        """获取FAQ统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 基础统计
            cursor.execute('SELECT COUNT(*) FROM faqs WHERE status = "active"')
            total_faqs = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(quality_score) FROM faqs WHERE quality_score IS NOT NULL AND status = "active"')
            avg_quality = cursor.fetchone()[0] or 0
            
            # 按文档分组
            cursor.execute('''
                SELECT d.filename, COUNT(f.faq_id) as faq_count
                FROM faqs f
                JOIN admin_documents d ON f.doc_id = d.doc_id
                WHERE f.status = 'active'
                GROUP BY f.doc_id, d.filename
                ORDER BY faq_count DESC
                LIMIT 10
            ''')
            top_docs = cursor.fetchall()
            
            # 按分类分组
            cursor.execute('''
                SELECT category, COUNT(*) as count
                FROM faqs 
                WHERE status = 'active' AND category IS NOT NULL
                GROUP BY category
            ''')
            category_stats = dict(cursor.fetchall())
            
            # 按抽取方式分组
            cursor.execute('''
                SELECT extracted_by, COUNT(*) as count
                FROM faqs 
                WHERE status = 'active'
                GROUP BY extracted_by
            ''')
            extraction_stats = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_faqs": total_faqs,
                "average_quality_score": round(avg_quality, 3),
                "top_documents": [{"filename": row[0], "faq_count": row[1]} for row in top_docs],
                "category_distribution": category_stats,
                "extraction_method_distribution": extraction_stats,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取FAQ统计失败: {str(e)}")
            return {
                "total_faqs": 0,
                "error": str(e)
            }
    
    async def _save_faq(self, faq: Dict[str, Any]) -> bool:
        """保存FAQ到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO faqs 
                (faq_id, question, answer, doc_id, created_time, status, quality_score, 
                 extracted_by, category, tags, metadata, updated_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                faq['faq_id'],
                faq['question'],
                faq['answer'],
                faq['doc_id'],
                faq['created_time'],
                faq.get('status', 'active'),
                faq.get('quality_score'),
                faq.get('extracted_by', 'llm'),
                faq.get('category'),
                json.dumps(faq.get('tags', [])),
                json.dumps(faq.get('metadata', {})),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"保存FAQ失败: {str(e)}")
            return False
    
    async def _update_extraction_task(self, 
                                    task_id: str, 
                                    status: str, 
                                    extracted_count: int = 0,
                                    error_message: str = None,
                                    completed_time: str = None):
        """更新抽取任务状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = ["status = ?"]
            params = [status]
            
            if extracted_count > 0:
                update_fields.append("extracted_count = ?")
                params.append(extracted_count)
            
            if error_message:
                update_fields.append("error_message = ?")
                params.append(error_message)
            
            if completed_time:
                update_fields.append("completed_time = ?")
                params.append(completed_time)
            
            params.append(task_id)
            
            cursor.execute(f'''
                UPDATE faq_extraction_tasks 
                SET {", ".join(update_fields)}
                WHERE task_id = ?
            ''', params)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"更新抽取任务失败: {str(e)}")
    
    async def _handle_duplicate_replacement(self, duplicates: List[Dict], new_doc_id: str, operator: str):
        """
        处理重复文档的替换逻辑
        
        Args:
            duplicates: 重复文档列表
            new_doc_id: 新文档ID
            operator: 操作者
        """
        try:
            for dup in duplicates:
                old_doc_id = dup.get('doc_id')
                if old_doc_id and old_doc_id != new_doc_id:
                    logger.info(f"删除被替换的文档: {old_doc_id}")
                    
                    # 删除旧文档（软删除）
                    await self.delete_document(old_doc_id, operator)
                    
                    # 记录替换操作
                    await self._log_operation(
                        doc_id=old_doc_id,
                        operation_type="replaced",
                        operator=operator,
                        operation_details={
                            "replaced_by": new_doc_id,
                            "reason": "duplicate_replacement"
                        },
                        success=True
                    )
        except Exception as e:
            logger.error(f"处理重复文档替换失败: {str(e)}")
    
    async def check_document_duplicates(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """
        检查文档重复（独立的API方法）
        
        Args:
            filename: 文件名
            file_content: 文件内容
            
        Returns:
            Dict: 重复检查结果
        """
        try:
            is_duplicate, duplicates = self.file_deduplication.check_duplicates(
                filename=filename,
                file_content=file_content,
                strict_mode=True
            )
            
            return {
                "success": True,
                "is_duplicate": is_duplicate,
                "duplicates": duplicates,
                "duplicate_count": len(duplicates),
                "message": "文档已存在，可选择强制替换" if is_duplicate else "文档检查通过"
            }
        except Exception as e:
            logger.error(f"检查文档重复失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cleanup_duplicate_faqs(self) -> Dict[str, Any]:
        """
        清理重复的FAQ，仅保留最近时间的
        
        Returns:
            Dict: 清理结果
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查找重复的FAQ（基于问题内容）
            cursor.execute('''
                SELECT question, COUNT(*) as count, GROUP_CONCAT(faq_id) as faq_ids,
                       GROUP_CONCAT(created_time) as created_times
                FROM faqs 
                WHERE status != 'deleted'
                GROUP BY TRIM(LOWER(question))
                HAVING COUNT(*) > 1
            ''')
            
            duplicate_groups = cursor.fetchall()
            total_removed = 0
            
            for question, count, faq_ids_str, created_times_str in duplicate_groups:
                faq_ids = faq_ids_str.split(',')
                created_times = created_times_str.split(',')
                
                # 创建(faq_id, created_time)对并按时间排序
                faq_time_pairs = list(zip(faq_ids, created_times))
                faq_time_pairs.sort(key=lambda x: x[1], reverse=True)  # 最新的在前
                
                # 保留最新的，删除其他的
                for i, (faq_id, _) in enumerate(faq_time_pairs):
                    if i > 0:  # 跳过第一个（最新的）
                        cursor.execute('''
                            UPDATE faqs 
                            SET status = 'deleted', updated_time = ?
                            WHERE faq_id = ?
                        ''', (datetime.now().isoformat(), faq_id))
                        total_removed += 1
                        
                        logger.info(f"删除重复FAQ: {faq_id}")
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"成功清理了{total_removed}个重复FAQ",
                "removed_count": total_removed,
                "duplicate_groups": len(duplicate_groups)
            }
            
        except Exception as e:
            logger.error(f"清理重复FAQ失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_faq_extraction_status(self) -> Dict[str, Any]:
        """
        获取FAQ挖掘状态，用于前端置灰功能
        
        Returns:
            Dict: 文档的FAQ挖掘状态
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有已完成FAQ抽取的文档
            cursor.execute('''
                SELECT DISTINCT doc_id, COUNT(task_id) as task_count,
                       MAX(completed_time) as last_extraction
                FROM faq_extraction_tasks 
                WHERE status = 'completed'
                GROUP BY doc_id
            ''')
            
            completed_extractions = {}
            for doc_id, task_count, last_extraction in cursor.fetchall():
                completed_extractions[doc_id] = {
                    "extraction_count": task_count,
                    "last_extraction": last_extraction,
                    "status": "completed"
                }
            
            # 获取正在进行的抽取任务
            cursor.execute('''
                SELECT DISTINCT doc_id, status, created_time
                FROM faq_extraction_tasks 
                WHERE status IN ('pending', 'processing')
            ''')
            
            for doc_id, status, created_time in cursor.fetchall():
                if doc_id not in completed_extractions:
                    completed_extractions[doc_id] = {}
                completed_extractions[doc_id]["current_status"] = status
                completed_extractions[doc_id]["current_task_created"] = created_time
            
            conn.close()
            
            return {
                "success": True,
                "extraction_status": completed_extractions
            }
            
        except Exception as e:
            logger.error(f"获取FAQ抽取状态失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
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