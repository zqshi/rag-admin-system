"""
RAG系统架构改进 - 核心实现示例
包含5个关键改进点的具体代码实现
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import redis.asyncio as redis
import numpy as np
from collections import defaultdict


# ==================== 1. 向量索引版本管理 ====================

@dataclass
class IndexVersion:
    """索引版本信息"""
    id: str
    name: str
    model: str
    dimension: int
    status: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorIndexVersionManager:
    """
    向量索引版本管理器
    支持多版本并存、灰度发布、无缝迁移
    """
    
    def __init__(self, vector_db, redis_client):
        self.vector_db = vector_db
        self.redis = redis_client
        self.versions = {}
        self.routing_rules = {}
        
    async def create_version(
        self,
        version_name: str,
        embedding_model: str,
        dimension: int
    ) -> IndexVersion:
        """创建新索引版本"""
        version_id = f"v_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 创建向量数据库集合
        collection_name = f"index_{version_id}"
        await self.vector_db.create_collection(
            name=collection_name,
            dimension=dimension,
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        
        # 记录版本信息
        version = IndexVersion(
            id=version_id,
            name=version_name,
            model=embedding_model,
            dimension=dimension,
            status="building",
            created_at=datetime.now()
        )
        
        self.versions[version_id] = version
        
        # 存储到Redis
        await self.redis.hset(
            "index_versions",
            version_id,
            json.dumps({
                "name": version_name,
                "model": embedding_model,
                "dimension": dimension,
                "status": "building",
                "created_at": version.created_at.isoformat()
            })
        )
        
        return version
    
    async def migrate_data(
        self,
        source_version: str,
        target_version: str,
        batch_size: int = 100
    ):
        """迁移数据到新版本"""
        source_collection = f"index_{source_version}"
        target_collection = f"index_{target_version}"
        
        # 获取源版本和目标版本信息
        source_info = self.versions[source_version]
        target_info = self.versions[target_version]
        
        # 分批迁移
        offset = 0
        total_migrated = 0
        
        while True:
            # 从源版本读取数据
            docs = await self.vector_db.query(
                collection=source_collection,
                expr=f"id >= {offset}",
                limit=batch_size
            )
            
            if not docs:
                break
            
            # 重新生成向量（使用新模型）
            new_vectors = []
            for doc in docs:
                # 这里应该调用新的embedding模型
                # 示例：使用模拟的向量转换
                new_vector = await self._regenerate_embedding(
                    doc['text'],
                    target_info.model
                )
                new_vectors.append(new_vector)
            
            # 插入到目标版本
            await self.vector_db.insert(
                collection=target_collection,
                vectors=new_vectors,
                metadata=[doc['metadata'] for doc in docs]
            )
            
            total_migrated += len(docs)
            offset += batch_size
            
            # 更新迁移进度
            await self.redis.hset(
                f"migration_progress:{source_version}:{target_version}",
                "processed",
                str(total_migrated)
            )
            
            # 避免过载
            await asyncio.sleep(0.1)
        
        # 更新版本状态
        self.versions[target_version].status = "ready"
        await self.redis.hset(
            "index_versions",
            target_version,
            json.dumps({"status": "ready"})
        )
    
    async def setup_gray_release(
        self,
        old_version: str,
        new_version: str,
        traffic_percentage: float = 0.1
    ):
        """配置灰度发布"""
        rule = {
            "old_version": old_version,
            "new_version": new_version,
            "percentage": traffic_percentage,
            "start_time": datetime.now().isoformat()
        }
        
        await self.redis.set(
            "gray_release_rule",
            json.dumps(rule),
            ex=86400  # 24小时过期
        )
        
        self.routing_rules = rule
        
    async def route_search(self, user_id: str, query: str) -> str:
        """根据灰度规则路由搜索请求"""
        if not self.routing_rules:
            # 没有灰度规则，使用默认版本
            return await self._get_default_version()
        
        # 基于用户ID的一致性哈希
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        threshold = int(self.routing_rules['percentage'] * (2**128))
        
        if hash_value < threshold:
            # 路由到新版本
            return self.routing_rules['new_version']
        else:
            # 路由到旧版本
            return self.routing_rules['old_version']
    
    async def _regenerate_embedding(self, text: str, model: str) -> List[float]:
        """使用新模型重新生成向量（示例）"""
        # 实际应该调用对应的embedding模型API
        # 这里返回模拟数据
        np.random.seed(hash(text + model) % 2**32)
        return np.random.randn(768).tolist()
    
    async def _get_default_version(self) -> str:
        """获取默认版本"""
        active_version = await self.redis.get("active_index_version")
        return active_version.decode() if active_version else "v_default"


# ==================== 2. 多模态处理 ====================

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalChunk:
    """多模态数据块"""
    chunk_id: str
    modality: ModalityType
    content: Any
    text_representation: str
    embeddings: List[float]
    metadata: Dict[str, Any]


class MultiModalProcessor:
    """
    多模态数据处理器
    统一处理文本、图片、表格、音视频等多种模态
    """
    
    def __init__(self):
        self.processors = {
            ModalityType.TEXT: self._process_text,
            ModalityType.IMAGE: self._process_image,
            ModalityType.TABLE: self._process_table,
            ModalityType.AUDIO: self._process_audio,
            ModalityType.VIDEO: self._process_video
        }
    
    async def process_document(
        self,
        document_path: str,
        doc_type: str
    ) -> List[MultiModalChunk]:
        """处理多模态文档"""
        chunks = []
        
        # 检测文档中的不同模态
        modalities = await self._detect_modalities(document_path, doc_type)
        
        # 并行处理各个模态
        tasks = []
        for modality_info in modalities:
            processor = self.processors[modality_info['type']]
            tasks.append(processor(modality_info['content']))
        
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        for result in results:
            chunks.extend(result)
        
        return chunks
    
    async def _detect_modalities(
        self,
        document_path: str,
        doc_type: str
    ) -> List[Dict]:
        """检测文档中的不同模态内容"""
        modalities = []
        
        if doc_type == "pdf":
            # PDF可能包含文本、图片、表格
            # 这里应该使用PDF解析库
            modalities.append({
                "type": ModalityType.TEXT,
                "content": "示例文本内容"
            })
            modalities.append({
                "type": ModalityType.IMAGE,
                "content": b"示例图片数据"
            })
            modalities.append({
                "type": ModalityType.TABLE,
                "content": [["列1", "列2"], ["数据1", "数据2"]]
            })
        
        return modalities
    
    async def _process_text(self, content: str) -> List[MultiModalChunk]:
        """处理文本"""
        chunks = []
        
        # 文本分片
        text_chunks = self._split_text(content, chunk_size=512, overlap=128)
        
        for i, text in enumerate(text_chunks):
            embedding = await self._generate_text_embedding(text)
            
            chunk = MultiModalChunk(
                chunk_id=f"text_{uuid.uuid4().hex[:8]}",
                modality=ModalityType.TEXT,
                content=text,
                text_representation=text,
                embeddings=embedding,
                metadata={"chunk_index": i}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _process_image(self, image_data: bytes) -> List[MultiModalChunk]:
        """处理图片"""
        # OCR提取文字
        ocr_text = await self._ocr_extract(image_data)
        
        # 生成图片描述
        caption = await self._generate_caption(image_data)
        
        # 合并文本表示
        text_repr = f"[图片描述] {caption}\n[OCR文字] {ocr_text}"
        
        # 生成向量
        embedding = await self._generate_text_embedding(text_repr)
        
        chunk = MultiModalChunk(
            chunk_id=f"image_{uuid.uuid4().hex[:8]}",
            modality=ModalityType.IMAGE,
            content=image_data,
            text_representation=text_repr,
            embeddings=embedding,
            metadata={
                "has_ocr": bool(ocr_text),
                "caption": caption
            }
        )
        
        return [chunk]
    
    async def _process_table(self, table_data: List[List]) -> List[MultiModalChunk]:
        """处理表格"""
        # 转换为结构化文本
        headers = table_data[0] if table_data else []
        rows = table_data[1:] if len(table_data) > 1 else []
        
        # 生成表格描述
        text_repr = f"表格包含{len(headers)}列，{len(rows)}行数据。\n"
        text_repr += f"列名：{', '.join(headers)}\n"
        
        # 添加示例数据
        if rows:
            text_repr += f"示例数据：{dict(zip(headers, rows[0]))}"
        
        # 生成向量
        embedding = await self._generate_text_embedding(text_repr)
        
        chunk = MultiModalChunk(
            chunk_id=f"table_{uuid.uuid4().hex[:8]}",
            modality=ModalityType.TABLE,
            content=table_data,
            text_representation=text_repr,
            embeddings=embedding,
            metadata={
                "columns": headers,
                "row_count": len(rows)
            }
        )
        
        return [chunk]
    
    async def _process_audio(self, audio_data: bytes) -> List[MultiModalChunk]:
        """处理音频"""
        # 语音转文字
        transcript = await self._transcribe_audio(audio_data)
        
        # 生成向量
        embedding = await self._generate_text_embedding(transcript)
        
        chunk = MultiModalChunk(
            chunk_id=f"audio_{uuid.uuid4().hex[:8]}",
            modality=ModalityType.AUDIO,
            content=audio_data,
            text_representation=transcript,
            embeddings=embedding,
            metadata={"duration": 60}  # 示例时长
        )
        
        return [chunk]
    
    async def _process_video(self, video_data: bytes) -> List[MultiModalChunk]:
        """处理视频"""
        chunks = []
        
        # 提取关键帧
        keyframes = await self._extract_keyframes(video_data)
        
        # 提取字幕
        subtitles = await self._extract_subtitles(video_data)
        
        # 为每个片段创建chunk
        for i, (frame, subtitle) in enumerate(zip(keyframes, subtitles)):
            text_repr = f"[时间{i*10}s] {subtitle}"
            embedding = await self._generate_text_embedding(text_repr)
            
            chunk = MultiModalChunk(
                chunk_id=f"video_{uuid.uuid4().hex[:8]}",
                modality=ModalityType.VIDEO,
                content={"frame": frame, "subtitle": subtitle},
                text_representation=text_repr,
                embeddings=embedding,
                metadata={"timestamp": i * 10}
            )
            chunks.append(chunk)
        
        return chunks
    
    # 辅助方法（示例实现）
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分片"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    async def _generate_text_embedding(self, text: str) -> List[float]:
        """生成文本向量（示例）"""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).tolist()
    
    async def _ocr_extract(self, image_data: bytes) -> str:
        """OCR提取（示例）"""
        return "OCR提取的文字内容"
    
    async def _generate_caption(self, image_data: bytes) -> str:
        """生成图片描述（示例）"""
        return "一张包含文档内容的图片"
    
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """音频转文字（示例）"""
        return "音频转写的文字内容"
    
    async def _extract_keyframes(self, video_data: bytes) -> List[bytes]:
        """提取关键帧（示例）"""
        return [b"frame1", b"frame2", b"frame3"]
    
    async def _extract_subtitles(self, video_data: bytes) -> List[str]:
        """提取字幕（示例）"""
        return ["字幕片段1", "字幕片段2", "字幕片段3"]


# ==================== 3. 实时协作冲突处理 ====================

@dataclass
class Lock:
    """分布式锁"""
    key: str
    value: str
    owner: str
    acquired_at: datetime
    ttl: int


class CollaborationLockManager:
    """
    协作锁管理器
    基于Redis实现的分布式锁
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.locks = {}
        
    async def acquire_lock(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        ttl: int = 300
    ) -> Optional[Lock]:
        """获取资源锁"""
        lock_key = f"lock:{resource_type}:{resource_id}"
        lock_value = f"{user_id}:{uuid.uuid4()}"
        
        # 尝试获取锁（原子操作）
        acquired = await self.redis.set(
            lock_key,
            lock_value,
            nx=True,  # 仅当不存在时设置
            ex=ttl    # 过期时间
        )
        
        if acquired:
            lock = Lock(
                key=lock_key,
                value=lock_value,
                owner=user_id,
                acquired_at=datetime.now(),
                ttl=ttl
            )
            self.locks[lock_key] = lock
            
            # 发布锁定事件
            await self.redis.publish(
                f"lock_events:{resource_type}:{resource_id}",
                json.dumps({
                    "event": "locked",
                    "user": user_id,
                    "timestamp": datetime.now().isoformat()
                })
            )
            
            return lock
        
        return None
    
    async def release_lock(self, lock: Lock) -> bool:
        """释放锁"""
        # Lua脚本确保原子性：只有持有者才能释放
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,  # number of keys
            lock.key,  # KEYS[1]
            lock.value  # ARGV[1]
        )
        
        if result:
            del self.locks[lock.key]
            
            # 发布释放事件
            await self.redis.publish(
                f"lock_events:{lock.key}",
                json.dumps({
                    "event": "released",
                    "user": lock.owner,
                    "timestamp": datetime.now().isoformat()
                })
            )
            
            return True
        
        return False
    
    async def extend_lock(self, lock: Lock, additional_ttl: int) -> bool:
        """延长锁的持有时间"""
        # 验证锁的所有权
        current_value = await self.redis.get(lock.key)
        if current_value and current_value.decode() == lock.value:
            # 延长过期时间
            await self.redis.expire(lock.key, lock.ttl + additional_ttl)
            lock.ttl += additional_ttl
            return True
        return False
    
    async def get_lock_info(
        self,
        resource_type: str,
        resource_id: str
    ) -> Optional[Dict]:
        """获取锁信息"""
        lock_key = f"lock:{resource_type}:{resource_id}"
        lock_value = await self.redis.get(lock_key)
        
        if lock_value:
            ttl = await self.redis.ttl(lock_key)
            user_id = lock_value.decode().split(':')[0]
            
            return {
                "locked": True,
                "owner": user_id,
                "ttl_remaining": ttl
            }
        
        return {"locked": False}


class ConflictResolver:
    """
    冲突解决器
    实现操作转换(OT)和三路合并
    """
    
    async def transform_operation(
        self,
        op1: Dict,
        op2: Dict
    ) -> Tuple[Dict, Dict]:
        """
        操作转换
        将两个并发操作转换为可顺序执行的操作
        """
        if op1['type'] == 'insert' and op2['type'] == 'insert':
            if op1['position'] < op2['position']:
                # op1在前，op2位置后移
                return op1, {
                    **op2,
                    'position': op2['position'] + len(op1['text'])
                }
            else:
                # op2在前，op1位置后移
                return {
                    **op1,
                    'position': op1['position'] + len(op2['text'])
                }, op2
                
        elif op1['type'] == 'delete' and op2['type'] == 'delete':
            # 处理删除冲突
            start1, end1 = op1['range']
            start2, end2 = op2['range']
            
            if end1 <= start2:
                # 无重叠，op1在前
                return op1, {
                    **op2,
                    'range': [start2 - (end1 - start1), end2 - (end1 - start1)]
                }
            elif end2 <= start1:
                # 无重叠，op2在前
                return {
                    **op1,
                    'range': [start1 - (end2 - start2), end1 - (end2 - start2)]
                }, op2
            else:
                # 有重叠，需要合并
                merged_start = min(start1, start2)
                merged_end = max(end1, end2)
                return {
                    'type': 'delete',
                    'range': [merged_start, merged_end]
                }, {'type': 'noop'}  # 第二个操作变为空操作
        
        # 其他情况的默认处理
        return op1, op2
    
    async def three_way_merge(
        self,
        base: str,
        mine: str,
        theirs: str
    ) -> Dict:
        """
        三路合并
        类似Git的合并策略
        """
        # 计算差异
        diff_mine = self._compute_diff(base, mine)
        diff_theirs = self._compute_diff(base, theirs)
        
        # 尝试自动合并
        merged_text = base
        conflicts = []
        
        # 应用非冲突的更改
        for change in diff_mine + diff_theirs:
            if not self._has_conflict(change, diff_mine, diff_theirs):
                merged_text = self._apply_change(merged_text, change)
            else:
                conflicts.append({
                    'position': change['position'],
                    'mine': self._get_change_text(mine, change),
                    'theirs': self._get_change_text(theirs, change)
                })
        
        return {
            'merged': merged_text,
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts
        }
    
    def _compute_diff(self, text1: str, text2: str) -> List[Dict]:
        """计算文本差异（简化版）"""
        # 实际应使用Myers差异算法
        diffs = []
        
        # 简单的逐字符比较（示例）
        min_len = min(len(text1), len(text2))
        for i in range(min_len):
            if text1[i] != text2[i]:
                diffs.append({
                    'type': 'change',
                    'position': i,
                    'old': text1[i],
                    'new': text2[i]
                })
        
        # 处理长度差异
        if len(text2) > len(text1):
            diffs.append({
                'type': 'insert',
                'position': min_len,
                'text': text2[min_len:]
            })
        elif len(text1) > len(text2):
            diffs.append({
                'type': 'delete',
                'position': min_len,
                'length': len(text1) - min_len
            })
        
        return diffs
    
    def _has_conflict(
        self,
        change: Dict,
        diff1: List[Dict],
        diff2: List[Dict]
    ) -> bool:
        """检查是否有冲突"""
        # 简化版：检查位置是否重叠
        for other in diff1 + diff2:
            if other != change and abs(other['position'] - change['position']) < 5:
                return True
        return False
    
    def _apply_change(self, text: str, change: Dict) -> str:
        """应用更改"""
        if change['type'] == 'insert':
            return text[:change['position']] + change['text'] + text[change['position']:]
        elif change['type'] == 'delete':
            return text[:change['position']] + text[change['position'] + change['length']:]
        elif change['type'] == 'change':
            return text[:change['position']] + change['new'] + text[change['position'] + 1:]
        return text
    
    def _get_change_text(self, text: str, change: Dict) -> str:
        """获取更改的文本"""
        if change['type'] == 'insert':
            return change['text']
        elif change['type'] == 'delete':
            return f"[删除了{change['length']}个字符]"
        elif change['type'] == 'change':
            return change['new']
        return ""


# ==================== 4. 数据版本控制 ====================

@dataclass
class Commit:
    """版本提交"""
    id: str
    doc_id: str
    parent: Optional[str]
    author: str
    message: str
    timestamp: datetime
    content_hash: str
    diff: Optional[str] = None


class DocumentVersionControl:
    """
    文档版本控制系统
    类似Git的实现
    """
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.commits = {}
        self.heads = {}  # 分支头部指针
        
    async def commit(
        self,
        doc_id: str,
        content: str,
        author: str,
        message: str,
        branch: str = "main"
    ) -> Commit:
        """创建新提交"""
        # 获取父提交
        parent_id = self.heads.get(f"{doc_id}:{branch}")
        
        # 计算内容哈希
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # 计算差异
        diff = None
        if parent_id:
            parent_commit = self.commits[parent_id]
            parent_content = await self.storage.get_content(parent_commit.content_hash)
            diff = self._compute_diff(parent_content, content)
        
        # 创建提交对象
        commit_id = hashlib.sha1(
            f"{doc_id}{parent_id or ''}{author}{message}{content_hash}".encode()
        ).hexdigest()
        
        commit = Commit(
            id=commit_id,
            doc_id=doc_id,
            parent=parent_id,
            author=author,
            message=message,
            timestamp=datetime.now(),
            content_hash=content_hash,
            diff=diff
        )
        
        # 存储提交
        self.commits[commit_id] = commit
        await self.storage.save_content(content_hash, content)
        await self.storage.save_commit(commit)
        
        # 更新分支指针
        self.heads[f"{doc_id}:{branch}"] = commit_id
        
        return commit
    
    async def checkout(
        self,
        doc_id: str,
        target: str  # commit_id or branch name
    ) -> str:
        """检出特定版本"""
        # 解析目标
        if target in ["main", "master", "develop"]:  # 分支名
            commit_id = self.heads.get(f"{doc_id}:{target}")
        else:  # commit id
            commit_id = target
        
        if not commit_id or commit_id not in self.commits:
            raise ValueError(f"Invalid target: {target}")
        
        # 重建内容
        commit = self.commits[commit_id]
        content = await self.storage.get_content(commit.content_hash)
        
        return content
    
    async def diff(
        self,
        doc_id: str,
        from_ref: str,
        to_ref: str
    ) -> str:
        """比较两个版本"""
        from_content = await self.checkout(doc_id, from_ref)
        to_content = await self.checkout(doc_id, to_ref)
        
        return self._compute_diff(from_content, to_content)
    
    async def merge(
        self,
        doc_id: str,
        source_branch: str,
        target_branch: str = "main"
    ) -> Commit:
        """合并分支"""
        source_commit_id = self.heads.get(f"{doc_id}:{source_branch}")
        target_commit_id = self.heads.get(f"{doc_id}:{target_branch}")
        
        if not source_commit_id or not target_commit_id:
            raise ValueError("Invalid branch")
        
        # 找到共同祖先
        common_ancestor = await self._find_common_ancestor(
            source_commit_id,
            target_commit_id
        )
        
        # 三路合并
        base_content = await self.checkout(doc_id, common_ancestor)
        source_content = await self.checkout(doc_id, source_commit_id)
        target_content = await self.checkout(doc_id, target_commit_id)
        
        merged_content = await self._three_way_merge(
            base_content,
            source_content,
            target_content
        )
        
        # 创建合并提交
        merge_commit = await self.commit(
            doc_id=doc_id,
            content=merged_content,
            author="system",
            message=f"Merge {source_branch} into {target_branch}",
            branch=target_branch
        )
        
        return merge_commit
    
    async def get_history(
        self,
        doc_id: str,
        branch: str = "main",
        limit: int = 10
    ) -> List[Commit]:
        """获取提交历史"""
        history = []
        commit_id = self.heads.get(f"{doc_id}:{branch}")
        
        while commit_id and len(history) < limit:
            commit = self.commits.get(commit_id)
            if not commit:
                break
            
            history.append(commit)
            commit_id = commit.parent
        
        return history
    
    def _compute_diff(self, text1: str, text2: str) -> str:
        """计算差异（简化版）"""
        # 实际应使用更复杂的差异算法
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        diff_lines = []
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                diff_lines.append(f"@@ -{i},{i} +{i},{i} @@")
                diff_lines.append(f"- {line1}")
                diff_lines.append(f"+ {line2}")
        
        return '\n'.join(diff_lines)
    
    async def _find_common_ancestor(
        self,
        commit1: str,
        commit2: str
    ) -> str:
        """找到两个提交的共同祖先"""
        # 构建祖先集合
        ancestors1 = set()
        current = commit1
        while current:
            ancestors1.add(current)
            commit = self.commits.get(current)
            current = commit.parent if commit else None
        
        # 查找共同祖先
        current = commit2
        while current:
            if current in ancestors1:
                return current
            commit = self.commits.get(current)
            current = commit.parent if commit else None
        
        return None
    
    async def _three_way_merge(
        self,
        base: str,
        source: str,
        target: str
    ) -> str:
        """三路合并算法"""
        # 简化实现，实际应处理冲突
        if base == target:
            return source
        elif base == source:
            return target
        else:
            # 简单合并策略：使用source的内容
            return source


# ==================== 5. 成本控制 ====================

@dataclass
class TokenBudget:
    """Token预算配置"""
    daily_limit: int
    monthly_limit: int
    cost_limit_usd: float
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95


@dataclass
class TokenUsage:
    """Token使用记录"""
    user_id: str
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class TokenBudgetController:
    """
    Token预算控制器
    精细化成本管理
    """
    
    PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015}
    }
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.budgets = {}
        self.usage_cache = defaultdict(list)
        
    async def set_budget(
        self,
        user_id: str,
        budget: TokenBudget
    ):
        """设置用户预算"""
        self.budgets[user_id] = budget
        
        # 存储到Redis
        await self.redis.hset(
            f"budget:{user_id}",
            mapping={
                "daily_limit": budget.daily_limit,
                "monthly_limit": budget.monthly_limit,
                "cost_limit_usd": budget.cost_limit_usd,
                "warning_threshold": budget.warning_threshold,
                "critical_threshold": budget.critical_threshold
            }
        )
    
    async def check_budget(
        self,
        user_id: str,
        estimated_tokens: int,
        model: str
    ) -> Tuple[bool, Optional[str]]:
        """检查预算是否充足"""
        budget = self.budgets.get(user_id)
        if not budget:
            return True, None  # 无预算限制
        
        # 计算预估成本
        estimated_cost = self.calculate_cost(estimated_tokens, model)
        
        # 获取当前使用量
        daily_usage = await self.get_daily_usage(user_id)
        monthly_usage = await self.get_monthly_usage(user_id)
        
        # 检查限制
        if daily_usage['tokens'] + estimated_tokens > budget.daily_limit:
            return False, "Daily token limit exceeded"
        
        if monthly_usage['tokens'] + estimated_tokens > budget.monthly_limit:
            return False, "Monthly token limit exceeded"
        
        if monthly_usage['cost'] + estimated_cost > budget.cost_limit_usd:
            return False, "Monthly cost limit exceeded"
        
        # 检查预警
        utilization = monthly_usage['cost'] / budget.cost_limit_usd
        if utilization > budget.critical_threshold:
            await self.send_alert(
                user_id,
                f"CRITICAL: Budget utilization at {utilization:.1%}"
            )
        elif utilization > budget.warning_threshold:
            await self.send_alert(
                user_id,
                f"WARNING: Budget utilization at {utilization:.1%}"
            )
        
        return True, None
    
    async def record_usage(
        self,
        user_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """记录Token使用"""
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(total_tokens, model)
        
        usage = TokenUsage(
            user_id=user_id,
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost
        )
        
        # 缓存最近使用记录
        self.usage_cache[user_id].append(usage)
        
        # 存储到Redis（按天聚合）
        date_key = datetime.now().strftime("%Y%m%d")
        await self.redis.hincrby(
            f"usage:{user_id}:{date_key}",
            "tokens",
            total_tokens
        )
        await self.redis.hincrbyfloat(
            f"usage:{user_id}:{date_key}",
            "cost",
            cost
        )
        
        # 设置过期时间（35天）
        await self.redis.expire(f"usage:{user_id}:{date_key}", 35 * 86400)
    
    def calculate_cost(
        self,
        tokens: int,
        model: str
    ) -> float:
        """计算成本"""
        if model not in self.PRICING:
            return 0.0
        
        # 简化计算：使用平均价格
        price_per_1k = (
            self.PRICING[model]['prompt'] + 
            self.PRICING[model]['completion']
        ) / 2
        
        return (tokens / 1000) * price_per_1k
    
    async def get_daily_usage(
        self,
        user_id: str
    ) -> Dict[str, float]:
        """获取今日使用量"""
        date_key = datetime.now().strftime("%Y%m%d")
        usage = await self.redis.hgetall(f"usage:{user_id}:{date_key}")
        
        return {
            'tokens': int(usage.get(b'tokens', 0)),
            'cost': float(usage.get(b'cost', 0))
        }
    
    async def get_monthly_usage(
        self,
        user_id: str
    ) -> Dict[str, float]:
        """获取本月使用量"""
        total_tokens = 0
        total_cost = 0.0
        
        # 遍历本月所有天
        today = datetime.now()
        for day in range(1, today.day + 1):
            date_key = today.replace(day=day).strftime("%Y%m%d")
            usage = await self.redis.hgetall(f"usage:{user_id}:{date_key}")
            
            total_tokens += int(usage.get(b'tokens', 0))
            total_cost += float(usage.get(b'cost', 0))
        
        return {
            'tokens': total_tokens,
            'cost': total_cost
        }
    
    async def optimize_request(
        self,
        request: Dict,
        user_id: str
    ) -> Dict:
        """优化请求以降低成本"""
        optimized = request.copy()
        
        # 策略1：使用更便宜的模型
        if request['model'] == 'gpt-4':
            # 评估任务复杂度
            if self._is_simple_task(request['prompt']):
                optimized['model'] = 'gpt-3.5-turbo'
        
        # 策略2：减少max_tokens
        if 'max_tokens' in request:
            optimized['max_tokens'] = min(request['max_tokens'], 2000)
        
        # 策略3：降低temperature
        if request.get('temperature', 1.0) > 0.7:
            optimized['temperature'] = 0.7
        
        # 策略4：使用缓存
        cache_key = hashlib.md5(
            f"{request['model']}{request['prompt']}".encode()
        ).hexdigest()
        
        cached = await self.redis.get(f"cache:{cache_key}")
        if cached:
            optimized['use_cache'] = True
            optimized['cached_response'] = cached.decode()
        
        return optimized
    
    def _is_simple_task(self, prompt: str) -> bool:
        """判断是否为简单任务"""
        # 简单的启发式规则
        simple_keywords = ['翻译', '总结', '列表', '定义']
        return any(keyword in prompt for keyword in simple_keywords)
    
    async def send_alert(self, user_id: str, message: str):
        """发送预算告警"""
        # 发布告警事件
        await self.redis.publish(
            f"budget_alerts:{user_id}",
            json.dumps({
                "user_id": user_id,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
        )
        
        print(f"Alert for {user_id}: {message}")


# ==================== 使用示例 ====================

async def main():
    """主函数 - 演示各个组件的使用"""
    
    # 初始化Redis连接
    redis_client = await redis.create_redis_pool('redis://localhost')
    
    print("=== RAG系统架构改进示例 ===\n")
    
    # 1. 向量索引版本管理
    print("1. 向量索引版本管理示例:")
    index_manager = VectorIndexVersionManager(None, redis_client)
    
    # 创建新版本
    v1 = await index_manager.create_version(
        version_name="v1.0-bge-base",
        embedding_model="bge-base-zh",
        dimension=768
    )
    print(f"   创建索引版本: {v1.id}")
    
    v2 = await index_manager.create_version(
        version_name="v2.0-bge-large",
        embedding_model="bge-large-zh",
        dimension=1024
    )
    print(f"   创建索引版本: {v2.id}")
    
    # 配置灰度发布
    await index_manager.setup_gray_release(v1.id, v2.id, 0.2)
    print(f"   配置灰度发布: 20%流量到新版本")
    
    # 2. 多模态处理
    print("\n2. 多模态处理示例:")
    processor = MultiModalProcessor()
    
    # 处理包含多种模态的文档
    chunks = await processor.process_document(
        document_path="/path/to/document.pdf",
        doc_type="pdf"
    )
    print(f"   处理文档，生成{len(chunks)}个多模态chunks")
    for chunk in chunks[:3]:
        print(f"   - {chunk.modality.value}: {chunk.chunk_id}")
    
    # 3. 协作锁管理
    print("\n3. 实时协作锁管理示例:")
    lock_manager = CollaborationLockManager(redis_client)
    
    # 用户A获取锁
    lock_a = await lock_manager.acquire_lock(
        resource_type="document",
        resource_id="doc_123",
        user_id="user_a"
    )
    if lock_a:
        print(f"   用户A获取锁成功: {lock_a.key}")
    
    # 用户B尝试获取同一资源的锁
    lock_b = await lock_manager.acquire_lock(
        resource_type="document",
        resource_id="doc_123",
        user_id="user_b"
    )
    if not lock_b:
        print(f"   用户B获取锁失败（资源已被锁定）")
    
    # 释放锁
    if lock_a:
        released = await lock_manager.release_lock(lock_a)
        print(f"   用户A释放锁: {released}")
    
    # 4. 版本控制
    print("\n4. 文档版本控制示例:")
    version_control = DocumentVersionControl(None)
    
    # 创建初始提交
    commit1 = await version_control.commit(
        doc_id="doc_456",
        content="这是文档的初始内容",
        author="张三",
        message="初始提交"
    )
    print(f"   创建提交: {commit1.id[:8]}... by {commit1.author}")
    
    # 修改并提交
    commit2 = await version_control.commit(
        doc_id="doc_456",
        content="这是文档的更新内容\n添加了新的一行",
        author="李四",
        message="添加新内容"
    )
    print(f"   创建提交: {commit2.id[:8]}... by {commit2.author}")
    
    # 查看历史
    history = await version_control.get_history("doc_456")
    print(f"   提交历史:")
    for commit in history:
        print(f"   - {commit.id[:8]}: {commit.message}")
    
    # 5. 成本控制
    print("\n5. Token预算控制示例:")
    budget_controller = TokenBudgetController(redis_client)
    
    # 设置用户预算
    await budget_controller.set_budget(
        user_id="user_123",
        budget=TokenBudget(
            daily_limit=100000,
            monthly_limit=3000000,
            cost_limit_usd=500.0
        )
    )
    print(f"   设置用户预算: 日限10万tokens，月限300万tokens")
    
    # 检查预算
    can_proceed, reason = await budget_controller.check_budget(
        user_id="user_123",
        estimated_tokens=5000,
        model="gpt-4"
    )
    print(f"   预算检查: {'通过' if can_proceed else f'拒绝 - {reason}'}")
    
    # 记录使用
    await budget_controller.record_usage(
        user_id="user_123",
        model="gpt-4",
        prompt_tokens=3000,
        completion_tokens=2000
    )
    
    # 获取使用统计
    daily = await budget_controller.get_daily_usage("user_123")
    monthly = await budget_controller.get_monthly_usage("user_123")
    print(f"   今日使用: {daily['tokens']} tokens, ${daily['cost']:.2f}")
    print(f"   本月使用: {monthly['tokens']} tokens, ${monthly['cost']:.2f}")
    
    # 关闭Redis连接
    redis_client.close()
    await redis_client.wait_closed()
    
    print("\n=== 示例运行完成 ===")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())