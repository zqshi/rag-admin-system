#!/usr/bin/env python3
"""
文件上传防重复工具
File Upload Deduplication Utility

功能：
1. 基于文件内容的MD5哈希检测重复
2. 基于文件名和大小的快速检测
3. 文件指纹存储和比对
4. 支持多种检测策略
"""

import os
import hashlib
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class FileDeduplication:
    """文件去重处理器"""
    
    def __init__(self, db_path: str, uploads_dir: str):
        """
        初始化文件去重器
        
        Args:
            db_path: 数据库文件路径
            uploads_dir: 上传目录路径
        """
        self.db_path = Path(db_path)
        self.uploads_dir = Path(uploads_dir)
        
        # 确保目录存在
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化去重数据库表"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 文件指纹表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                content_hash TEXT,
                upload_time TEXT NOT NULL,
                file_path TEXT,
                doc_id TEXT,
                metadata TEXT,
                is_deleted BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 重复文件记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_fingerprint_id INTEGER,
                duplicate_filename TEXT,
                duplicate_size INTEGER,
                duplicate_hash TEXT,
                detection_time TEXT,
                action_taken TEXT,  -- blocked, renamed, merged
                metadata TEXT,
                FOREIGN KEY (original_fingerprint_id) REFERENCES file_fingerprints (id)
            )
        ''')
        
        # 创建索引提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON file_fingerprints(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON file_fingerprints(content_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename_size ON file_fingerprints(filename, file_size)')
        
        conn.commit()
        conn.close()
        
        logger.info("文件去重数据库初始化完成")
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """
        计算文件内容的MD5哈希
        
        Args:
            file_content: 文件二进制内容
            
        Returns:
            文件的MD5哈希值
        """
        return hashlib.md5(file_content).hexdigest()
    
    def calculate_content_hash(self, text_content: str) -> str:
        """
        计算文本内容的哈希（用于检测内容相似性）
        
        Args:
            text_content: 文本内容
            
        Returns:
            内容哈希值
        """
        # 标准化文本内容（去除空白字符、统一换行）
        normalized = ''.join(text_content.split()).lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    def check_duplicate_by_hash(self, file_content: bytes) -> Optional[Dict[str, Any]]:
        """
        基于文件哈希检测重复
        
        Args:
            file_content: 文件二进制内容
            
        Returns:
            如果重复，返回原始文件信息；否则返回None
        """
        file_hash = self.calculate_file_hash(file_content)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_size, upload_time, file_path, doc_id, metadata
            FROM file_fingerprints
            WHERE file_hash = ? AND is_deleted = FALSE
        ''', (file_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'fingerprint_id': result[0],
                'filename': result[1],
                'file_size': result[2],
                'upload_time': result[3],
                'file_path': result[4],
                'doc_id': result[5],
                'metadata': json.loads(result[6]) if result[6] else {},
                'duplicate_type': 'exact_hash'
            }
        
        return None
    
    def check_duplicate_by_name_size(self, filename: str, file_size: int) -> List[Dict[str, Any]]:
        """
        基于文件名和大小检测潜在重复
        
        Args:
            filename: 文件名
            file_size: 文件大小
            
        Returns:
            潜在重复文件列表
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 检查完全相同的文件名和大小
        cursor.execute('''
            SELECT id, filename, file_hash, upload_time, file_path, doc_id
            FROM file_fingerprints
            WHERE filename = ? AND file_size = ? AND is_deleted = FALSE
        ''', (filename, file_size))
        
        exact_matches = cursor.fetchall()
        
        # 检查相似文件名（去除扩展名和前缀后缀）
        base_name = Path(filename).stem
        cursor.execute('''
            SELECT id, filename, file_size, file_hash, upload_time, file_path, doc_id
            FROM file_fingerprints
            WHERE filename LIKE ? AND ABS(file_size - ?) < 1024 AND is_deleted = FALSE
        ''', (f'%{base_name}%', file_size))
        
        similar_matches = cursor.fetchall()
        conn.close()
        
        results = []
        
        # 处理完全匹配
        for match in exact_matches:
            results.append({
                'fingerprint_id': match[0],
                'filename': match[1],
                'file_hash': match[2],
                'upload_time': match[3],
                'file_path': match[4],
                'doc_id': match[5],
                'duplicate_type': 'exact_name_size',
                'similarity': 1.0
            })
        
        # 处理相似匹配
        for match in similar_matches:
            if match[0] not in [r['fingerprint_id'] for r in results]:  # 避免重复
                similarity = self._calculate_name_similarity(filename, match[1])
                if similarity > 0.7:  # 相似度阈值
                    results.append({
                        'fingerprint_id': match[0],
                        'filename': match[1],
                        'file_size': match[2],
                        'file_hash': match[3],
                        'upload_time': match[4],
                        'file_path': match[5],
                        'doc_id': match[6],
                        'duplicate_type': 'similar_name',
                        'similarity': similarity
                    })
        
        return results
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        计算文件名相似度
        
        Args:
            name1: 文件名1
            name2: 文件名2
            
        Returns:
            相似度（0-1）
        """
        # 简单的编辑距离相似度计算
        name1 = Path(name1).stem.lower()
        name2 = Path(name2).stem.lower()
        
        if name1 == name2:
            return 1.0
        
        # 使用集合交集计算相似度
        set1 = set(name1)
        set2 = set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def register_file(self, filename: str, file_size: int, file_content: bytes, 
                     doc_id: Optional[str] = None, file_path: Optional[str] = None,
                     text_content: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        注册新文件到去重数据库
        
        Args:
            filename: 文件名
            file_size: 文件大小
            file_content: 文件二进制内容
            doc_id: 文档ID
            file_path: 文件存储路径
            text_content: 文本内容（用于内容去重）
            metadata: 额外元数据
            
        Returns:
            文件指纹ID
        """
        file_hash = self.calculate_file_hash(file_content)
        content_hash = self.calculate_content_hash(text_content) if text_content else None
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO file_fingerprints 
            (filename, file_size, file_hash, content_hash, upload_time, file_path, doc_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, 
            file_size, 
            file_hash, 
            content_hash,
            datetime.now().isoformat(),
            file_path,
            doc_id,
            json.dumps(metadata) if metadata else None
        ))
        
        fingerprint_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"文件已注册: {filename} (指纹ID: {fingerprint_id})")
        return str(fingerprint_id)
    
    def record_duplicate(self, original_fingerprint_id: int, duplicate_filename: str,
                        duplicate_size: int, duplicate_hash: str, action_taken: str,
                        metadata: Optional[Dict] = None):
        """
        记录重复文件检测结果
        
        Args:
            original_fingerprint_id: 原始文件指纹ID
            duplicate_filename: 重复文件名
            duplicate_size: 重复文件大小
            duplicate_hash: 重复文件哈希
            action_taken: 采取的行动
            metadata: 额外信息
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO duplicate_files 
            (original_fingerprint_id, duplicate_filename, duplicate_size, 
             duplicate_hash, detection_time, action_taken, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            original_fingerprint_id,
            duplicate_filename,
            duplicate_size,
            duplicate_hash,
            datetime.now().isoformat(),
            action_taken,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def check_duplicates(self, filename: str, file_content: bytes, 
                        text_content: Optional[str] = None,
                        strict_mode: bool = True) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        综合检测文件重复
        
        Args:
            filename: 文件名
            file_content: 文件二进制内容
            text_content: 文本内容
            strict_mode: 严格模式（只检测完全相同的文件）
            
        Returns:
            (是否重复, 重复文件列表)
        """
        file_size = len(file_content)
        duplicates = []
        
        # 1. 基于文件哈希的精确检测（最高优先级）
        hash_duplicate = self.check_duplicate_by_hash(file_content)
        if hash_duplicate:
            duplicates.append(hash_duplicate)
            return True, duplicates
        
        # 2. 如果不是严格模式，进行更多检测
        if not strict_mode:
            # 基于文件名和大小的检测
            name_size_duplicates = self.check_duplicate_by_name_size(filename, file_size)
            duplicates.extend(name_size_duplicates)
            
            # 基于内容相似性的检测（如果有文本内容）
            if text_content:
                content_duplicates = self._check_content_similarity(text_content)
                duplicates.extend(content_duplicates)
        
        return len(duplicates) > 0, duplicates
    
    def _check_content_similarity(self, text_content: str) -> List[Dict[str, Any]]:
        """
        检测内容相似的文件
        
        Args:
            text_content: 文本内容
            
        Returns:
            相似文件列表
        """
        content_hash = self.calculate_content_hash(text_content)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_size, upload_time, file_path, doc_id
            FROM file_fingerprints
            WHERE content_hash = ? AND is_deleted = FALSE
        ''', (content_hash,))
        
        results = cursor.fetchall()
        conn.close()
        
        similar_files = []
        for result in results:
            similar_files.append({
                'fingerprint_id': result[0],
                'filename': result[1],
                'file_size': result[2],
                'upload_time': result[3],
                'file_path': result[4],
                'doc_id': result[5],
                'duplicate_type': 'content_similarity',
                'similarity': 1.0  # 内容哈希完全匹配
            })
        
        return similar_files
    
    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """
        获取重复文件统计信息
        
        Returns:
            统计信息字典
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 总文件数
        cursor.execute('SELECT COUNT(*) FROM file_fingerprints WHERE is_deleted = FALSE')
        total_files = cursor.fetchone()[0]
        
        # 重复检测记录数
        cursor.execute('SELECT COUNT(*) FROM duplicate_files')
        duplicate_records = cursor.fetchone()[0]
        
        # 按行动分组统计
        cursor.execute('''
            SELECT action_taken, COUNT(*) 
            FROM duplicate_files 
            GROUP BY action_taken
        ''')
        actions_stats = dict(cursor.fetchall())
        
        # 最近的重复检测
        cursor.execute('''
            SELECT duplicate_filename, detection_time, action_taken
            FROM duplicate_files
            ORDER BY detection_time DESC
            LIMIT 10
        ''')
        recent_duplicates = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_registered_files': total_files,
            'duplicate_detections': duplicate_records,
            'actions_statistics': actions_stats,
            'recent_duplicates': [
                {
                    'filename': dup[0],
                    'detection_time': dup[1],
                    'action': dup[2]
                }
                for dup in recent_duplicates
            ]
        }
    
    def cleanup_deleted_files(self) -> int:
        """
        清理已删除文件的记录
        
        Returns:
            清理的记录数
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 标记不存在的文件为已删除
        cursor.execute('''
            SELECT id, file_path FROM file_fingerprints 
            WHERE is_deleted = FALSE AND file_path IS NOT NULL
        ''')
        
        file_records = cursor.fetchall()
        cleanup_count = 0
        
        for record_id, file_path in file_records:
            if file_path and not Path(file_path).exists():
                cursor.execute('''
                    UPDATE file_fingerprints 
                    SET is_deleted = TRUE 
                    WHERE id = ?
                ''', (record_id,))
                cleanup_count += 1
        
        conn.commit()
        conn.close()
        
        if cleanup_count > 0:
            logger.info(f"清理了 {cleanup_count} 个已删除文件的记录")
        
        return cleanup_count


# 使用示例和工具函数

def create_deduplication_middleware(dedup_handler: FileDeduplication):
    """
    创建FastAPI中间件用于文件上传去重
    
    Args:
        dedup_handler: 文件去重处理器
        
    Returns:
        去重检查函数
    """
    
    async def check_file_duplicate(filename: str, file_content: bytes, 
                                  text_content: Optional[str] = None,
                                  strict_mode: bool = True) -> Dict[str, Any]:
        """
        检查文件重复的中间件函数
        
        Returns:
            检查结果字典
        """
        is_duplicate, duplicates = dedup_handler.check_duplicates(
            filename, file_content, text_content, strict_mode
        )
        
        if is_duplicate and duplicates:
            # 记录重复检测
            for dup in duplicates:
                dedup_handler.record_duplicate(
                    original_fingerprint_id=dup['fingerprint_id'],
                    duplicate_filename=filename,
                    duplicate_size=len(file_content),
                    duplicate_hash=dedup_handler.calculate_file_hash(file_content),
                    action_taken='blocked',
                    metadata={'duplicate_type': dup['duplicate_type']}
                )
            
            return {
                'allowed': False,
                'reason': 'duplicate_file',
                'duplicates': duplicates,
                'suggestion': f"文件 '{filename}' 与已上传的文件重复",
                'details': f"检测到 {len(duplicates)} 个重复文件"
            }
        
        return {
            'allowed': True,
            'reason': 'unique_file',
            'duplicates': [],
            'suggestion': '文件检查通过，可以上传'
        }
    
    return check_file_duplicate


# 命令行工具
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="文件去重工具")
    parser.add_argument('--db-path', required=True, help='数据库文件路径')
    parser.add_argument('--uploads-dir', required=True, help='上传目录路径')
    parser.add_argument('--action', choices=['stats', 'cleanup'], default='stats', 
                       help='执行的操作')
    
    args = parser.parse_args()
    
    dedup = FileDeduplication(args.db_path, args.uploads_dir)
    
    if args.action == 'stats':
        stats = dedup.get_duplicate_statistics()
        print("=== 文件去重统计 ===")
        print(f"注册文件总数: {stats['total_registered_files']}")
        print(f"重复检测次数: {stats['duplicate_detections']}")
        print(f"操作统计: {stats['actions_statistics']}")
        print("\n最近的重复检测:")
        for dup in stats['recent_duplicates']:
            print(f"  {dup['filename']} - {dup['action']} ({dup['detection_time']})")
    
    elif args.action == 'cleanup':
        count = dedup.cleanup_deleted_files()
        print(f"清理了 {count} 个已删除文件的记录")