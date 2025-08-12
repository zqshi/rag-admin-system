# RAG系统后端TDD设计文档

**版本**: 1.0  
**技术栈**: Python/FastAPI + Go/Gin + PostgreSQL + Milvus  
**日期**: 2025年8月12日

---

## 一、后端TDD架构

### 1.1 测试分层策略

```
后端测试金字塔
├── 单元测试 (70%)
│   ├── 业务逻辑层
│   ├── 数据访问层
│   ├── 工具函数
│   └── 算法模块
├── 集成测试 (20%)
│   ├── API端点测试
│   ├── 数据库事务测试
│   ├── 消息队列测试
│   └── 缓存层测试
├── 契约测试 (5%)
│   ├── 前后端契约
│   └── 微服务间契约
└── 系统测试 (5%)
    ├── 端到端流程
    ├── 性能测试
    └── 压力测试
```

### 1.2 测试环境架构

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  test-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: rag_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
    tmpfs:
      - /var/lib/postgresql/data
      
  test-redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6380:6379"
      
  test-milvus:
    image: milvusdb/milvus:v2.3.0
    environment:
      ETCD_USE_EMBED: true
      COMMON_STORAGETYPE: local
    ports:
      - "19531:19530"
      
  test-kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9093
    ports:
      - "9093:9092"
```

---

## 二、Python服务TDD实现

### 2.1 文档处理服务

#### 步骤1：编写失败测试 🔴

```python
# tests/test_document_processor.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from io import BytesIO

from app.services.document_processor import DocumentProcessor
from app.models.document import Document, DocumentStatus, ChunkingStrategy
from app.exceptions import ParseError, ValidationError


class TestDocumentProcessor:
    """文档处理器测试套件"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return DocumentProcessor()
    
    @pytest.fixture
    def valid_pdf(self):
        """创建有效的PDF文件"""
        content = b"%PDF-1.4\n%fake pdf content"
        return BytesIO(content)
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM服务"""
        with patch('app.services.llm_service.LLMService') as mock:
            mock_instance = mock.return_value
            mock_instance.generate_embedding = AsyncMock(
                return_value=[[0.1] * 1536]
            )
            yield mock_instance
    
    # ============ 文档解析测试 ============
    
    @pytest.mark.asyncio
    async def test_should_parse_valid_pdf_document(self, processor, valid_pdf):
        """应该成功解析有效的PDF文档"""
        # Given
        document = Document(
            id="doc_123",
            name="test.pdf",
            content=valid_pdf.read(),
            type="application/pdf",
            size=1024
        )
        
        # When
        result = await processor.parse(document)
        
        # Then
        assert result.status == DocumentStatus.PARSED
        assert result.text_content is not None
        assert len(result.text_content) > 0
        assert result.metadata['pages'] > 0
        assert result.parse_time < 5.0  # 解析时间小于5秒
    
    @pytest.mark.asyncio
    async def test_should_reject_corrupted_file(self, processor):
        """应该拒绝损坏的文件"""
        # Given
        document = Document(
            id="doc_124",
            name="corrupted.pdf",
            content=b"invalid pdf content",
            type="application/pdf"
        )
        
        # When/Then
        with pytest.raises(ParseError) as exc_info:
            await processor.parse(document)
        
        assert "无法解析PDF文件" in str(exc_info.value)
        assert exc_info.value.document_id == "doc_124"
    
    @pytest.mark.asyncio
    async def test_should_handle_multiple_file_formats(self, processor):
        """应该处理多种文件格式"""
        formats = [
            ("test.pdf", "application/pdf", b"%PDF-1.4"),
            ("test.docx", "application/vnd.openxmlformats", b"PK\x03\x04"),
            ("test.txt", "text/plain", b"Plain text content"),
            ("test.md", "text/markdown", b"# Markdown content"),
            ("test.html", "text/html", b"<html><body>HTML</body></html>")
        ]
        
        for name, mime_type, content in formats:
            document = Document(
                id=f"doc_{name}",
                name=name,
                content=content,
                type=mime_type
            )
            
            result = await processor.parse(document)
            
            assert result.status == DocumentStatus.PARSED
            assert result.text_content is not None
    
    # ============ 文档分片测试 ============
    
    @pytest.mark.asyncio
    async def test_should_chunk_document_with_overlap(self, processor):
        """应该使用重叠策略分片文档"""
        # Given
        text = " ".join([f"Sentence {i}." for i in range(100)])
        document = Document(
            id="doc_125",
            text_content=text,
            status=DocumentStatus.PARSED
        )
        
        strategy = ChunkingStrategy(
            method="sliding_window",
            chunk_size=100,
            overlap=20
        )
        
        # When
        chunks = await processor.chunk(document, strategy)
        
        # Then
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 100 for chunk in chunks)
        
        # 验证重叠
        for i in range(len(chunks) - 1):
            current_end = chunks[i].content[-20:]
            next_start = chunks[i + 1].content[:20]
            assert current_end in chunks[i + 1].content
    
    @pytest.mark.asyncio
    async def test_should_preserve_semantic_boundaries(self, processor):
        """应该保持语义边界完整"""
        # Given
        text = """
        第一章：介绍
        这是第一章的内容，包含重要信息。
        
        第二章：方法
        这是第二章的内容，描述具体方法。
        
        第三章：结论
        这是第三章的内容，总结要点。
        """
        
        document = Document(id="doc_126", text_content=text)
        strategy = ChunkingStrategy(method="semantic", chunk_size=150)
        
        # When
        chunks = await processor.chunk(document, strategy)
        
        # Then
        # 每个章节应该在独立的chunk中
        assert any("第一章" in chunk.content for chunk in chunks)
        assert any("第二章" in chunk.content for chunk in chunks)
        assert any("第三章" in chunk.content for chunk in chunks)
        
        # 章节不应该被拆分
        for chunk in chunks:
            if "第一章" in chunk.content:
                assert "第一章的内容" in chunk.content
    
    # ============ 向量化测试 ============
    
    @pytest.mark.asyncio
    async def test_should_generate_embeddings_for_chunks(
        self, processor, mock_llm_service
    ):
        """应该为文档片段生成向量"""
        # Given
        chunks = [
            {"id": "chunk_1", "content": "First chunk"},
            {"id": "chunk_2", "content": "Second chunk"},
            {"id": "chunk_3", "content": "Third chunk"}
        ]
        
        # When
        embeddings = await processor.generate_embeddings(chunks)
        
        # Then
        assert len(embeddings) == 3
        assert all(len(emb.vector) == 1536 for emb in embeddings)
        assert all(emb.model == "text-embedding-ada-002" for emb in embeddings)
        
        # 验证调用次数
        assert mock_llm_service.generate_embedding.call_count == 3
    
    @pytest.mark.asyncio
    async def test_should_batch_embedding_requests(self, processor, mock_llm_service):
        """应该批量处理向量化请求"""
        # Given
        chunks = [{"id": f"chunk_{i}", "content": f"Content {i}"} 
                 for i in range(100)]
        
        # When
        embeddings = await processor.generate_embeddings(
            chunks, 
            batch_size=10
        )
        
        # Then
        assert len(embeddings) == 100
        # 应该调用10次（100个chunks，每批10个）
        assert mock_llm_service.generate_embedding.call_count == 10
    
    # ============ 质量评估测试 ============
    
    @pytest.mark.asyncio
    async def test_should_evaluate_chunk_quality(self, processor):
        """应该评估文档片段质量"""
        # Given
        high_quality_chunk = {
            "content": "This is a well-structured paragraph with clear information about the topic. It contains relevant keywords and maintains coherent flow.",
            "tokens": 25
        }
        
        low_quality_chunk = {
            "content": "This... um... like... you know...",
            "tokens": 6
        }
        
        # When
        high_score = await processor.evaluate_quality(high_quality_chunk)
        low_score = await processor.evaluate_quality(low_quality_chunk)
        
        # Then
        assert high_score > 0.7
        assert low_score < 0.3
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
```

#### 步骤2：实现最小代码 🟢

```python
# app/services/document_processor.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import hashlib

from app.models.document import Document, DocumentStatus, ChunkingStrategy
from app.services.llm_service import LLMService
from app.services.parser_factory import ParserFactory
from app.exceptions import ParseError, ValidationError


class DocumentProcessor:
    """文档处理核心服务"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.parser_factory = ParserFactory()
    
    async def parse(self, document: Document) -> Document:
        """解析文档内容"""
        start_time = datetime.now()
        
        try:
            # 获取对应的解析器
            parser = self.parser_factory.get_parser(document.type)
            
            # 解析文档
            text_content = await parser.parse(document.content)
            metadata = await parser.extract_metadata(document.content)
            
            # 更新文档状态
            document.text_content = text_content
            document.metadata = metadata
            document.status = DocumentStatus.PARSED
            document.parse_time = (datetime.now() - start_time).total_seconds()
            
            return document
            
        except Exception as e:
            raise ParseError(
                f"无法解析{document.type}文件",
                document_id=document.id,
                original_error=str(e)
            )
    
    async def chunk(
        self, 
        document: Document, 
        strategy: ChunkingStrategy
    ) -> List[Dict[str, Any]]:
        """文档分片"""
        if strategy.method == "sliding_window":
            return self._sliding_window_chunk(
                document.text_content,
                strategy.chunk_size,
                strategy.overlap
            )
        elif strategy.method == "semantic":
            return self._semantic_chunk(
                document.text_content,
                strategy.chunk_size
            )
        else:
            raise ValueError(f"不支持的分片策略: {strategy.method}")
    
    def _sliding_window_chunk(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Dict[str, Any]]:
        """滑动窗口分片"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunks.append({
                "content": chunk_content,
                "start": start,
                "end": end,
                "id": hashlib.md5(chunk_content.encode()).hexdigest()
            })
            
            start += chunk_size - overlap
            
        return chunks
    
    def _semantic_chunk(
        self, 
        text: str, 
        chunk_size: int
    ) -> List[Dict[str, Any]]:
        """语义分片"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "id": hashlib.md5(current_chunk.encode()).hexdigest()
                    })
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "id": hashlib.md5(current_chunk.encode()).hexdigest()
            })
        
        return chunks
    
    async def generate_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """生成向量嵌入"""
        embeddings = []
        
        # 批量处理
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk["content"] for chunk in batch]
            
            # 并发生成向量
            vectors = await self.llm_service.generate_embedding(batch_texts)
            
            for chunk, vector in zip(batch, vectors):
                embeddings.append({
                    "chunk_id": chunk["id"],
                    "vector": vector,
                    "model": "text-embedding-ada-002"
                })
        
        return embeddings
    
    async def evaluate_quality(self, chunk: Dict[str, Any]) -> float:
        """评估片段质量"""
        content = chunk["content"]
        tokens = chunk.get("tokens", len(content.split()))
        
        # 质量评分规则
        score = 0.0
        
        # 长度评分
        if tokens >= 20:
            score += 0.3
        elif tokens >= 10:
            score += 0.2
        elif tokens >= 5:
            score += 0.1
        
        # 内容质量评分
        if not any(filler in content.lower() 
                  for filler in ["um", "uh", "like", "you know"]):
            score += 0.3
        
        # 结构评分
        if "." in content and len(content.split(".")) > 1:
            score += 0.2
        
        # 信息密度评分
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        if total_words > 0:
            diversity = unique_words / total_words
            score += diversity * 0.2
        
        return min(score, 1.0)
```

#### 步骤3：重构优化 🔄

```python
# app/services/document_processor_optimized.py
from typing import List, Dict, Any, Protocol
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np

from app.models.document import Document
from app.services.quality_evaluator import QualityEvaluator
from app.utils.metrics import track_performance


class ChunkingStrategy(Protocol):
    """分片策略协议"""
    async def chunk(self, text: str) -> List[Dict[str, Any]]:
        ...


class OptimizedDocumentProcessor:
    """优化的文档处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.quality_evaluator = QualityEvaluator()
        self._cache = {}
    
    @track_performance
    async def process_batch(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """批量处理文档"""
        # 并发处理
        tasks = [self.process_single(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed = []
        failed = []
        
        for doc, result in zip(documents, results):
            if isinstance(result, Exception):
                failed.append((doc, result))
            else:
                processed.append(result)
        
        # 重试失败的文档
        if failed and self.config.get('retry_enabled', True):
            retry_results = await self._retry_failed(failed)
            processed.extend(retry_results)
        
        return processed
    
    async def process_single(self, document: Document) -> Document:
        """处理单个文档"""
        # 检查缓存
        cache_key = self._get_cache_key(document)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 处理流水线
        pipeline = [
            self._validate,
            self._parse,
            self._chunk,
            self._generate_embeddings,
            self._evaluate_quality,
            self._index
        ]
        
        result = document
        for step in pipeline:
            result = await step(result)
        
        # 缓存结果
        self._cache[cache_key] = result
        
        return result
    
    @lru_cache(maxsize=128)
    def _get_cache_key(self, document: Document) -> str:
        """生成缓存键"""
        return hashlib.sha256(
            f"{document.id}:{document.content_hash}".encode()
        ).hexdigest()
    
    async def _retry_failed(
        self, 
        failed: List[Tuple[Document, Exception]]
    ) -> List[Document]:
        """重试失败的文档"""
        retry_results = []
        max_retries = self.config.get('max_retries', 3)
        
        for doc, error in failed:
            for attempt in range(max_retries):
                try:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                    result = await self.process_single(doc)
                    retry_results.append(result)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"文档 {doc.id} 处理失败: {e}")
        
        return retry_results
```

### 2.2 检索服务测试（Go）

```go
// search_service_test.go
package services

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/suite"
)

type SearchServiceTestSuite struct {
    suite.Suite
    service      *SearchService
    mockVectorDB *MockVectorDB
    mockCache    *MockCache
    mockReranker *MockReranker
}

func (suite *SearchServiceTestSuite) SetupTest() {
    suite.mockVectorDB = new(MockVectorDB)
    suite.mockCache = new(MockCache)
    suite.mockReranker = new(MockReranker)
    
    suite.service = NewSearchService(
        suite.mockVectorDB,
        suite.mockCache,
        suite.mockReranker,
    )
}

// ============ 向量检索测试 ============

func (suite *SearchServiceTestSuite) TestVectorSearch_Success() {
    // Given
    ctx := context.Background()
    query := Query{
        Text:  "如何配置RAG系统",
        TopK:  10,
        Threshold: 0.7,
    }
    
    expectedDocs := []Document{
        {ID: "doc1", Score: 0.95, Content: "RAG配置指南"},
        {ID: "doc2", Score: 0.85, Content: "系统设置说明"},
    }
    
    suite.mockVectorDB.On("Search", ctx, query).Return(expectedDocs, nil)
    
    // When
    results, err := suite.service.VectorSearch(ctx, query)
    
    // Then
    assert.NoError(suite.T(), err)
    assert.Len(suite.T(), results, 2)
    assert.Equal(suite.T(), expectedDocs[0].ID, results[0].ID)
    suite.mockVectorDB.AssertExpectations(suite.T())
}

func (suite *SearchServiceTestSuite) TestVectorSearch_WithCache() {
    // Given
    ctx := context.Background()
    query := Query{Text: "cached query"}
    cacheKey := suite.service.getCacheKey(query)
    
    cachedResults := []Document{
        {ID: "cached1", Score: 0.9},
    }
    
    suite.mockCache.On("Get", cacheKey).Return(cachedResults, nil)
    
    // When
    results, err := suite.service.VectorSearch(ctx, query)
    
    // Then
    assert.NoError(suite.T(), err)
    assert.Equal(suite.T(), cachedResults, results)
    
    // 不应该调用VectorDB
    suite.mockVectorDB.AssertNotCalled(suite.T(), "Search")
}

// ============ 混合检索测试 ============

func (suite *SearchServiceTestSuite) TestHybridSearch_CombinesResults() {
    // Given
    ctx := context.Background()
    query := HybridQuery{
        Text: "test query",
        VectorWeight: 0.6,
        KeywordWeight: 0.4,
    }
    
    vectorResults := []Document{
        {ID: "vec1", Score: 0.9},
        {ID: "vec2", Score: 0.8},
        {ID: "common", Score: 0.7},
    }
    
    keywordResults := []Document{
        {ID: "key1", Score: 0.85},
        {ID: "key2", Score: 0.75},
        {ID: "common", Score: 0.65},
    }
    
    suite.mockVectorDB.On("Search", mock.Anything, mock.Anything).
        Return(vectorResults, nil)
    
    suite.mockKeywordDB.On("Search", mock.Anything, mock.Anything).
        Return(keywordResults, nil)
    
    // When
    results, err := suite.service.HybridSearch(ctx, query)
    
    // Then
    assert.NoError(suite.T(), err)
    
    // 验证结果融合
    assert.Len(suite.T(), results, 5) // 去重后的结果
    
    // 验证分数融合（RRF算法）
    commonDoc := findDocByID(results, "common")
    assert.NotNil(suite.T(), commonDoc)
    
    // common文档应该有更高的综合分数
    assert.Greater(suite.T(), commonDoc.Score, 0.7)
}

func (suite *SearchServiceTestSuite) TestHybridSearch_Timeout() {
    // Given
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()
    
    query := HybridQuery{Text: "test"}
    
    // 模拟慢查询
    suite.mockVectorDB.On("Search", mock.Anything, mock.Anything).
        Run(func(args mock.Arguments) {
            time.Sleep(200 * time.Millisecond)
        }).
        Return(nil, context.DeadlineExceeded)
    
    // When
    _, err := suite.service.HybridSearch(ctx, query)
    
    // Then
    assert.Error(suite.T(), err)
    assert.Contains(suite.T(), err.Error(), "timeout")
}

// ============ 重排序测试 ============

func (suite *SearchServiceTestSuite) TestRerank_ImproveRelevance() {
    // Given
    candidates := []Document{
        {ID: "doc1", Score: 0.8, Content: "相关内容"},
        {ID: "doc2", Score: 0.9, Content: "不太相关"},
        {ID: "doc3", Score: 0.7, Content: "非常相关"},
    }
    
    query := "查找相关内容"
    
    // 模拟重排序后的结果
    rerankedScores := map[string]float64{
        "doc1": 0.95,  // 提升
        "doc2": 0.6,   // 降低
        "doc3": 0.98,  // 大幅提升
    }
    
    suite.mockReranker.On("Rerank", candidates, query).
        Return(func(docs []Document, q string) []Document {
            for i := range docs {
                docs[i].RerankScore = rerankedScores[docs[i].ID]
            }
            // 按新分数排序
            sort.Slice(docs, func(i, j int) bool {
                return docs[i].RerankScore > docs[j].RerankScore
            })
            return docs
        }, nil)
    
    // When
    results := suite.service.Rerank(candidates, query)
    
    // Then
    assert.Equal(suite.T(), "doc3", results[0].ID) // 最相关的排第一
    assert.Equal(suite.T(), "doc1", results[1].ID)
    assert.Equal(suite.T(), "doc2", results[2].ID) // 不相关的排最后
}

// ============ 并发测试 ============

func (suite *SearchServiceTestSuite) TestConcurrentSearches() {
    // Given
    numRequests := 100
    ctx := context.Background()
    
    suite.mockVectorDB.On("Search", mock.Anything, mock.Anything).
        Return([]Document{{ID: "test"}}, nil)
    
    // When
    results := make(chan error, numRequests)
    
    for i := 0; i < numRequests; i++ {
        go func(id int) {
            query := Query{Text: fmt.Sprintf("query_%d", id)}
            _, err := suite.service.VectorSearch(ctx, query)
            results <- err
        }(i)
    }
    
    // Then
    for i := 0; i < numRequests; i++ {
        err := <-results
        assert.NoError(suite.T(), err)
    }
    
    // 验证调用次数
    suite.mockVectorDB.AssertNumberOfCalls(suite.T(), "Search", numRequests)
}

// ============ 性能基准测试 ============

func BenchmarkVectorSearch(b *testing.B) {
    service := setupBenchmarkService()
    ctx := context.Background()
    query := Query{Text: "benchmark query", TopK: 10}
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := service.VectorSearch(ctx, query)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}

func BenchmarkHybridSearch(b *testing.B) {
    service := setupBenchmarkService()
    ctx := context.Background()
    query := HybridQuery{
        Text: "benchmark query",
        VectorWeight: 0.6,
        KeywordWeight: 0.4,
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := service.HybridSearch(ctx, query)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func TestSearchServiceSuite(t *testing.T) {
    suite.Run(t, new(SearchServiceTestSuite))
}
```

---

## 三、数据访问层TDD

### 3.1 Repository层测试

```python
# tests/test_document_repository.py
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.repositories.document_repository import DocumentRepository
from app.models.database import Base, DocumentModel
from app.exceptions import NotFoundError, DuplicateError


class TestDocumentRepository:
    """文档仓库测试"""
    
    @pytest.fixture(scope="function")
    def db_session(self):
        """创建测试数据库会话"""
        # 使用内存数据库
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
        Base.metadata.drop_all(engine)
    
    @pytest.fixture
    def repository(self, db_session):
        """创建仓库实例"""
        return DocumentRepository(db_session)
    
    @pytest.fixture
    def sample_document(self):
        """创建示例文档"""
        return DocumentModel(
            id="doc_001",
            name="test.pdf",
            type="application/pdf",
            size=1024,
            status="pending",
            created_at=datetime.utcnow()
        )
    
    # ============ CRUD测试 ============
    
    def test_should_create_document(self, repository, sample_document):
        """应该创建文档"""
        # When
        created = repository.create(sample_document)
        
        # Then
        assert created.id == "doc_001"
        assert repository.count() == 1
    
    def test_should_prevent_duplicate_documents(self, repository, sample_document):
        """应该防止重复文档"""
        # Given
        repository.create(sample_document)
        
        # When/Then
        duplicate = DocumentModel(id="doc_001", name="duplicate.pdf")
        with pytest.raises(DuplicateError):
            repository.create(duplicate)
    
    def test_should_find_document_by_id(self, repository, sample_document):
        """应该通过ID查找文档"""
        # Given
        repository.create(sample_document)
        
        # When
        found = repository.find_by_id("doc_001")
        
        # Then
        assert found is not None
        assert found.name == "test.pdf"
    
    def test_should_raise_when_document_not_found(self, repository):
        """找不到文档时应该抛出异常"""
        # When/Then
        with pytest.raises(NotFoundError) as exc_info:
            repository.find_by_id("non_existent")
        
        assert "Document non_existent not found" in str(exc_info.value)
    
    def test_should_update_document(self, repository, sample_document):
        """应该更新文档"""
        # Given
        repository.create(sample_document)
        
        # When
        updates = {"status": "processed", "chunks_count": 10}
        updated = repository.update("doc_001", updates)
        
        # Then
        assert updated.status == "processed"
        assert updated.chunks_count == 10
        assert updated.updated_at > sample_document.created_at
    
    def test_should_delete_document(self, repository, sample_document):
        """应该删除文档"""
        # Given
        repository.create(sample_document)
        assert repository.count() == 1
        
        # When
        repository.delete("doc_001")
        
        # Then
        assert repository.count() == 0
        with pytest.raises(NotFoundError):
            repository.find_by_id("doc_001")
    
    # ============ 查询测试 ============
    
    def test_should_filter_documents_by_status(self, repository):
        """应该按状态过滤文档"""
        # Given
        docs = [
            DocumentModel(id=f"doc_{i}", status=status)
            for i, status in enumerate(["pending", "processed", "pending", "failed"])
        ]
        for doc in docs:
            repository.create(doc)
        
        # When
        pending = repository.find_by_status("pending")
        processed = repository.find_by_status("processed")
        
        # Then
        assert len(pending) == 2
        assert len(processed) == 1
        assert all(d.status == "pending" for d in pending)
    
    def test_should_paginate_results(self, repository):
        """应该分页返回结果"""
        # Given
        for i in range(25):
            repository.create(DocumentModel(id=f"doc_{i:03d}", name=f"file_{i}.pdf"))
        
        # When
        page1 = repository.find_all(page=1, per_page=10)
        page2 = repository.find_all(page=2, per_page=10)
        page3 = repository.find_all(page=3, per_page=10)
        
        # Then
        assert len(page1.items) == 10
        assert len(page2.items) == 10
        assert len(page3.items) == 5
        assert page1.total == 25
        assert page1.pages == 3
    
    def test_should_search_documents(self, repository):
        """应该搜索文档"""
        # Given
        docs = [
            DocumentModel(id="1", name="Python教程.pdf"),
            DocumentModel(id="2", name="Java指南.docx"),
            DocumentModel(id="3", name="Python入门.txt"),
        ]
        for doc in docs:
            repository.create(doc)
        
        # When
        results = repository.search("Python")
        
        # Then
        assert len(results) == 2
        assert all("Python" in r.name for r in results)
    
    # ============ 事务测试 ============
    
    def test_should_rollback_on_error(self, repository):
        """错误时应该回滚事务"""
        # Given
        initial_count = repository.count()
        
        # When
        try:
            with repository.transaction():
                repository.create(DocumentModel(id="tx_1", name="test1.pdf"))
                repository.create(DocumentModel(id="tx_2", name="test2.pdf"))
                # 故意触发错误
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Then
        assert repository.count() == initial_count  # 事务回滚
    
    def test_should_commit_successful_transaction(self, repository):
        """成功时应该提交事务"""
        # Given
        initial_count = repository.count()
        
        # When
        with repository.transaction():
            repository.create(DocumentModel(id="tx_1", name="test1.pdf"))
            repository.create(DocumentModel(id="tx_2", name="test2.pdf"))
        
        # Then
        assert repository.count() == initial_count + 2
    
    # ============ 批量操作测试 ============
    
    def test_should_bulk_insert(self, repository):
        """应该批量插入"""
        # Given
        documents = [
            DocumentModel(id=f"bulk_{i}", name=f"file_{i}.pdf")
            for i in range(100)
        ]
        
        # When
        start_time = datetime.now()
        repository.bulk_insert(documents)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Then
        assert repository.count() == 100
        assert duration < 1.0  # 批量插入应该很快
    
    def test_should_bulk_update(self, repository):
        """应该批量更新"""
        # Given
        docs = [DocumentModel(id=f"doc_{i}", status="pending") for i in range(10)]
        repository.bulk_insert(docs)
        
        # When
        ids = [f"doc_{i}" for i in range(5)]
        repository.bulk_update(ids, {"status": "processed"})
        
        # Then
        processed = repository.find_by_status("processed")
        pending = repository.find_by_status("pending")
        
        assert len(processed) == 5
        assert len(pending) == 5
```

### 3.2 缓存层测试

```python
# tests/test_cache_service.py
import pytest
import asyncio
from unittest.mock import Mock, patch
import redis
import json

from app.services.cache_service import CacheService, CacheKey


class TestCacheService:
    """缓存服务测试"""
    
    @pytest.fixture
    def redis_mock(self):
        """Mock Redis客户端"""
        with patch('redis.asyncio.Redis') as mock:
            yield mock.return_value
    
    @pytest.fixture
    def cache_service(self, redis_mock):
        """创建缓存服务"""
        return CacheService(redis_mock)
    
    # ============ 基础操作测试 ============
    
    @pytest.mark.asyncio
    async def test_should_set_and_get_value(self, cache_service, redis_mock):
        """应该设置和获取值"""
        # Given
        key = "test_key"
        value = {"data": "test_value"}
        ttl = 300
        
        redis_mock.set = Mock(return_value=True)
        redis_mock.get = Mock(return_value=json.dumps(value))
        
        # When
        await cache_service.set(key, value, ttl)
        result = await cache_service.get(key)
        
        # Then
        assert result == value
        redis_mock.set.assert_called_once()
        redis_mock.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_should_return_none_for_missing_key(self, cache_service, redis_mock):
        """缺失的键应该返回None"""
        # Given
        redis_mock.get = Mock(return_value=None)
        
        # When
        result = await cache_service.get("missing_key")
        
        # Then
        assert result is None
    
    @pytest.mark.asyncio
    async def test_should_delete_key(self, cache_service, redis_mock):
        """应该删除键"""
        # Given
        redis_mock.delete = Mock(return_value=1)
        
        # When
        deleted = await cache_service.delete("test_key")
        
        # Then
        assert deleted is True
        redis_mock.delete.assert_called_once_with("test_key")
    
    # ============ TTL测试 ============
    
    @pytest.mark.asyncio
    async def test_should_expire_after_ttl(self, cache_service, redis_mock):
        """应该在TTL后过期"""
        # Given
        redis_mock.setex = Mock(return_value=True)
        
        # When
        await cache_service.set("temp_key", "value", ttl=60)
        
        # Then
        redis_mock.setex.assert_called_once_with("temp_key", 60, '"value"')
    
    @pytest.mark.asyncio
    async def test_should_refresh_ttl_on_access(self, cache_service, redis_mock):
        """访问时应该刷新TTL"""
        # Given
        redis_mock.get = Mock(return_value='"value"')
        redis_mock.expire = Mock(return_value=True)
        
        # When
        await cache_service.get_and_refresh("key", ttl=300)
        
        # Then
        redis_mock.expire.assert_called_once_with("key", 300)
    
    # ============ 批量操作测试 ============
    
    @pytest.mark.asyncio
    async def test_should_batch_get(self, cache_service, redis_mock):
        """应该批量获取"""
        # Given
        keys = ["key1", "key2", "key3"]
        values = ['"val1"', '"val2"', None]
        redis_mock.mget = Mock(return_value=values)
        
        # When
        results = await cache_service.mget(keys)
        
        # Then
        assert results == {"key1": "val1", "key2": "val2", "key3": None}
    
    @pytest.mark.asyncio
    async def test_should_batch_set(self, cache_service, redis_mock):
        """应该批量设置"""
        # Given
        data = {"key1": "val1", "key2": "val2"}
        redis_mock.mset = Mock(return_value=True)
        
        # When
        await cache_service.mset(data, ttl=300)
        
        # Then
        redis_mock.mset.assert_called_once()
    
    # ============ 缓存策略测试 ============
    
    @pytest.mark.asyncio
    async def test_cache_aside_pattern(self, cache_service, redis_mock):
        """测试Cache-Aside模式"""
        # Given
        redis_mock.get = Mock(return_value=None)  # 缓存未命中
        redis_mock.set = Mock(return_value=True)
        
        async def load_from_db():
            return {"id": 1, "data": "from_db"}
        
        # When
        result = await cache_service.get_or_load(
            "cache_key",
            load_from_db,
            ttl=600
        )
        
        # Then
        assert result == {"id": 1, "data": "from_db"}
        redis_mock.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_should_prevent_cache_stampede(self, cache_service):
        """应该防止缓存雪崩"""
        # Given
        call_count = 0
        
        async def expensive_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return f"result_{call_count}"
        
        # When - 并发请求相同的键
        tasks = [
            cache_service.get_or_load(
                "same_key",
                expensive_operation,
                ttl=60
            )
            for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Then - 只应该调用一次
        assert call_count == 1
        assert all(r == "result_1" for r in results)
```

---

## 四、API契约测试

### 4.1 契约定义

```python
# tests/contracts/document_api_contract.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """文档上传请求契约"""
    file: bytes = Field(..., description="文件内容")
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., pattern="^(pdf|docx|txt|md)$")
    metadata: Optional[dict] = Field(default={})


class DocumentUploadResponse(BaseModel):
    """文档上传响应契约"""
    id: str
    status: str = Field(..., pattern="^(pending|processing|completed|failed)$")
    message: str
    created_at: datetime


class SearchRequest(BaseModel):
    """搜索请求契约"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[dict] = None


class SearchResponse(BaseModel):
    """搜索响应契约"""
    results: List[dict]
    total: int
    took_ms: float
```

### 4.2 契约测试

```python
# tests/test_api_contracts.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from tests.contracts.document_api_contract import *


class TestAPIContracts:
    """API契约测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_document_upload_contract(self, client):
        """测试文档上传契约"""
        # Given
        request_data = {
            "name": "test.pdf",
            "type": "pdf",
            "metadata": {"source": "test"}
        }
        
        files = {
            "file": ("test.pdf", b"PDF content", "application/pdf")
        }
        
        # When
        response = client.post(
            "/api/v1/documents",
            data=request_data,
            files=files
        )
        
        # Then
        assert response.status_code == 201
        
        # 验证响应契约
        data = response.json()
        validated = DocumentUploadResponse(**data)
        
        assert validated.id
        assert validated.status in ["pending", "processing"]
        assert validated.created_at
    
    def test_search_contract(self, client):
        """测试搜索契约"""
        # Given
        request = SearchRequest(
            query="test query",
            top_k=5,
            threshold=0.8
        )
        
        # When
        response = client.post(
            "/api/v1/search",
            json=request.dict()
        )
        
        # Then
        assert response.status_code == 200
        
        # 验证响应契约
        data = response.json()
        validated = SearchResponse(**data)
        
        assert isinstance(validated.results, list)
        assert validated.total >= 0
        assert validated.took_ms >= 0
    
    def test_contract_validation_error(self, client):
        """测试契约验证错误"""
        # Given - 无效的请求
        invalid_request = {
            "query": "",  # 空查询
            "top_k": 1000,  # 超出范围
            "threshold": 2.0  # 无效阈值
        }
        
        # When
        response = client.post(
            "/api/v1/search",
            json=invalid_request
        )
        
        # Then
        assert response.status_code == 422
        errors = response.json()["detail"]
        
        # 应该包含所有验证错误
        error_fields = [e["loc"][-1] for e in errors]
        assert "query" in error_fields
        assert "top_k" in error_fields
        assert "threshold" in error_fields
```

---

## 五、性能测试

### 5.1 负载测试

```python
# tests/performance/load_test.py
import asyncio
import aiohttp
import time
from statistics import mean, stdev
from typing import List
import pytest


class LoadTester:
    """负载测试器"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session: aiohttp.ClientSession) -> dict:
        """执行单个请求"""
        start = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/search",
                json={"query": "test query", "top_k": 10}
            ) as response:
                await response.json()
                latency = (time.time() - start) * 1000
                
                return {
                    "status": response.status,
                    "latency_ms": latency,
                    "success": response.status == 200
                }
        except Exception as e:
            return {
                "status": 0,
                "latency_ms": (time.time() - start) * 1000,
                "success": False,
                "error": str(e)
            }
    
    async def run_load_test(
        self,
        concurrent_users: int,
        duration_seconds: int
    ) -> dict:
        """运行负载测试"""
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + duration_seconds
            tasks = []
            
            while time.time() < end_time:
                # 创建并发请求
                batch = [
                    self.single_request(session)
                    for _ in range(concurrent_users)
                ]
                
                results = await asyncio.gather(*batch)
                self.results.extend(results)
                
                # 短暂延迟避免过载
                await asyncio.sleep(0.1)
        
        return self.analyze_results()
    
    def analyze_results(self) -> dict:
        """分析测试结果"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        return {
            "total_requests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) * 100,
            "latency": {
                "mean": mean(latencies) if latencies else 0,
                "stdev": stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "p50": self.percentile(latencies, 50),
                "p95": self.percentile(latencies, 95),
                "p99": self.percentile(latencies, 99)
            }
        }
    
    def percentile(self, data: List[float], p: int) -> float:
        """计算百分位数"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.performance
class TestLoadPerformance:
    """负载性能测试"""
    
    @pytest.mark.asyncio
    async def test_normal_load(self):
        """测试正常负载"""
        tester = LoadTester("http://localhost:8000")
        
        results = await tester.run_load_test(
            concurrent_users=10,
            duration_seconds=30
        )
        
        # 验证性能指标
        assert results["success_rate"] > 99
        assert results["latency"]["p95"] < 500  # 95%请求小于500ms
        assert results["latency"]["p99"] < 1000  # 99%请求小于1s
    
    @pytest.mark.asyncio
    async def test_peak_load(self):
        """测试峰值负载"""
        tester = LoadTester("http://localhost:8000")
        
        results = await tester.run_load_test(
            concurrent_users=100,
            duration_seconds=60
        )
        
        # 峰值负载下的性能要求
        assert results["success_rate"] > 95
        assert results["latency"]["p95"] < 2000
        assert results["latency"]["p99"] < 5000
    
    @pytest.mark.asyncio
    async def test_stress_test(self):
        """压力测试 - 找出系统极限"""
        tester = LoadTester("http://localhost:8000")
        
        for users in [10, 50, 100, 200, 500]:
            results = await tester.run_load_test(
                concurrent_users=users,
                duration_seconds=10
            )
            
            print(f"\n并发用户: {users}")
            print(f"成功率: {results['success_rate']:.2f}%")
            print(f"P95延迟: {results['latency']['p95']:.2f}ms")
            
            # 找到系统崩溃点
            if results["success_rate"] < 90:
                print(f"系统在 {users} 并发用户时开始降级")
                break
```

### 5.2 内存和CPU测试

```python
# tests/performance/resource_test.py
import psutil
import threading
import time
import pytest


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.samples = []
        
        def monitor():
            while self.monitoring:
                self.samples.append({
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                    "threads": self.process.num_threads(),
                    "connections": len(self.process.connections())
                })
                time.sleep(0.1)
        
        thread = threading.Thread(target=monitor)
        thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        time.sleep(0.2)
    
    def get_stats(self):
        """获取统计数据"""
        if not self.samples:
            return {}
        
        cpu_values = [s["cpu_percent"] for s in self.samples]
        memory_values = [s["memory_mb"] for s in self.samples]
        
        return {
            "cpu": {
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values)
            },
            "memory": {
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "leak": memory_values[-1] - memory_values[0]
            },
            "threads": {
                "max": max(s["threads"] for s in self.samples)
            }
        }


@pytest.mark.resource
class TestResourceUsage:
    """资源使用测试"""
    
    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # 模拟处理大量文档
        for _ in range(100):
            # 处理文档的代码
            pass
        
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        
        # 验证资源使用
        assert stats["memory"]["max"] < 500  # 最大内存小于500MB
        assert stats["memory"]["leak"] < 50   # 内存泄漏小于50MB
        assert stats["cpu"]["avg"] < 80       # 平均CPU使用率小于80%
```

---

## 六、CI/CD集成

### 6.1 GitHub Actions配置

```yaml
# .github/workflows/backend-test.yml
name: Backend Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
          
      milvus:
        image: milvusdb/milvus:v2.3.0
        ports:
          - 19530:19530
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-test.txt
          
      - name: Run linting
        run: |
          flake8 app/ --max-line-length=100
          black app/ --check
          mypy app/
          
      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=app --cov-report=xml
          
      - name: Run integration tests
        run: |
          pytest tests/integration -v
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/test
          REDIS_URL: redis://localhost:6379
          
      - name: Run contract tests
        run: |
          pytest tests/contracts -v
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: backend
          
      - name: Run performance tests
        if: github.event_name == 'pull_request'
        run: |
          pytest tests/performance -v -m "not stress"
          
      - name: Generate test report
        if: always()
        run: |
          pytest --html=report.html --self-contained-html
          
      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            report.html
            coverage.xml
            .coverage
```

### 6.2 测试质量门禁

```yaml
# sonar-project.properties
sonar.projectKey=rag-system-backend
sonar.sources=app
sonar.tests=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.xunit.reportPath=test-results/*.xml

# 质量门禁标准
sonar.qualitygate.wait=true
sonar.coverage.threshold=85
sonar.duplications.threshold=3
sonar.issues.threshold=0
```

---

## 七、测试最佳实践总结

### 7.1 测试编写原则

```python
# tests/best_practices.md
"""
后端TDD最佳实践

1. 测试独立性
   - 每个测试独立运行
   - 不依赖测试执行顺序
   - 使用fixture管理测试数据

2. 测试命名
   - test_should_[expected_behavior]_when_[condition]
   - 清晰描述测试意图

3. 测试结构
   - Arrange: 准备测试数据
   - Act: 执行被测代码
   - Assert: 验证结果

4. Mock使用
   - 只Mock外部依赖
   - 保持Mock简单
   - 验证Mock调用

5. 性能考虑
   - 单元测试 < 100ms
   - 集成测试 < 1s
   - 使用并行执行

6. 数据管理
   - 使用工厂模式创建测试数据
   - 测试后清理数据
   - 避免硬编码
"""
```

### 7.2 TDD工作流

```mermaid
graph TD
    A[需求分析] --> B[编写测试用例]
    B --> C[运行测试-失败]
    C --> D[编写最小实现]
    D --> E[运行测试-通过]
    E --> F[重构代码]
    F --> G[运行测试-确保通过]
    G --> H{所有需求完成?}
    H -->|否| B
    H -->|是| I[代码评审]
    I --> J[合并代码]
```

---

这份完整的TDD设计文档涵盖了前后端的测试策略、实施方法和最佳实践。关键要点：

1. **测试先行**：先写测试，再写代码
2. **分层测试**：单元、集成、契约、性能
3. **自动化**：CI/CD集成，质量门禁
4. **可维护**：清晰的命名，独立的测试
5. **高覆盖**：核心功能100%，整体85%+

建议团队严格执行TDD流程，确保代码质量和系统稳定性。