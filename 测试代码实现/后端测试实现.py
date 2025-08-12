# 后端API和服务测试实现示例
# 包含单元测试、集成测试、性能测试

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis
from faker import Faker

# 测试应用和模型导入（示例）
from app.main import app
from app.services.document_processor import DocumentProcessor
from app.services.search_service import SearchService
from app.services.vector_service import VectorService
from app.models.document import Document, DocumentStatus, Chunk
from app.repositories.document_repository import DocumentRepository
from app.utils.exceptions import ParseError, ValidationError

# ==================== 测试配置 ====================
fake = Faker()

# 测试数据库配置
TEST_DATABASE_URL = "postgresql://test:test@localhost:5433/rag_test"
test_engine = create_engine(TEST_DATABASE_URL)
TestSessionLocal = sessionmaker(bind=test_engine)

# 测试Redis配置
test_redis = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)

# 测试客户端
client = TestClient(app)

# ==================== 测试数据工厂 ====================
class TestDataFactory:
    """测试数据生成工厂"""
    
    @staticmethod
    def create_document(**kwargs) -> Document:
        """创建测试文档"""
        default = {
            'id': f'doc_{fake.uuid4()[:8]}',
            'name': fake.file_name(extension='pdf'),
            'type': 'application/pdf',
            'size': fake.random_int(1024, 10485760),
            'content': fake.text(max_nb_chars=1000),
            'status': DocumentStatus.PENDING,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        default.update(kwargs)
        return Document(**default)
    
    @staticmethod
    def create_chunk(**kwargs) -> Dict[str, Any]:
        """创建测试片段"""
        default = {
            'id': f'chunk_{fake.uuid4()[:8]}',
            'doc_id': f'doc_{fake.uuid4()[:8]}',
            'content': fake.text(max_nb_chars=500),
            'position': fake.random_int(0, 100),
            'tokens': fake.random_int(50, 500),
            'embedding': np.random.rand(1536).tolist(),
            'quality_score': round(np.random.uniform(0.3, 1.0), 2)
        }
        default.update(kwargs)
        return default
    
    @staticmethod
    def create_pdf_content(pages: int = 10) -> bytes:
        """创建模拟PDF内容"""
        content = b"%PDF-1.4\n"
        for i in range(pages):
            content += f"Page {i+1}: {fake.text()}\n".encode()
        content += b"%%EOF"
        return content
    
    @staticmethod
    def create_search_query(**kwargs) -> Dict[str, Any]:
        """创建搜索查询"""
        default = {
            'query': fake.sentence(),
            'top_k': 10,
            'threshold': 0.7,
            'filters': {}
        }
        default.update(kwargs)
        return default

# ==================== 文档处理服务测试 ====================
@pytest.mark.asyncio
class TestDocumentProcessor:
    """文档处理器测试套件"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return DocumentProcessor()
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM服务"""
        with patch('app.services.llm_service.LLMService') as mock:
            instance = mock.return_value
            instance.generate_embedding = AsyncMock(
                return_value=np.random.rand(1536).tolist()
            )
            instance.evaluate_quality = AsyncMock(
                return_value=0.85
            )
            yield instance
    
    # ========== TC_DOC_008: PDF文档解析 ==========
    async def test_parse_valid_pdf_document(self, processor, mock_llm_service):
        """测试有效PDF文档解析"""
        # Given
        pdf_content = TestDataFactory.create_pdf_content(pages=5)
        document = TestDataFactory.create_document(
            content=pdf_content,
            type='application/pdf'
        )
        
        # When
        result = await processor.parse(document)
        
        # Then
        assert result.status == DocumentStatus.PARSED
        assert result.text_content is not None
        assert len(result.text_content) > 0
        assert result.metadata['pages'] == 5
        assert result.parse_time < 5.0
        assert 'error' not in result.metadata
    
    async def test_reject_corrupted_pdf(self, processor):
        """测试拒绝损坏的PDF文件"""
        # Given
        corrupted_content = b"This is not a valid PDF"
        document = TestDataFactory.create_document(
            content=corrupted_content,
            type='application/pdf'
        )
        
        # When/Then
        with pytest.raises(ParseError) as exc_info:
            await processor.parse(document)
        
        assert "无法解析PDF文件" in str(exc_info.value)
        assert exc_info.value.document_id == document.id
    
    async def test_parse_multiple_formats(self, processor):
        """测试多种文件格式解析"""
        formats = [
            ('test.pdf', 'application/pdf', b'%PDF-1.4\ntest'),
            ('test.docx', 'application/vnd.openxmlformats', b'PK\x03\x04'),
            ('test.txt', 'text/plain', b'Plain text content'),
            ('test.md', 'text/markdown', b'# Markdown Title\nContent'),
            ('test.html', 'text/html', b'<html><body>Test</body></html>')
        ]
        
        for name, mime_type, content in formats:
            document = TestDataFactory.create_document(
                name=name,
                type=mime_type,
                content=content
            )
            
            result = await processor.parse(document)
            
            assert result.status == DocumentStatus.PARSED
            assert result.text_content is not None
    
    # ========== TC_DOC_011: 滑动窗口分片 ==========
    async def test_sliding_window_chunking(self, processor):
        """测试滑动窗口分片策略"""
        # Given
        text = " ".join([f"Sentence {i}." for i in range(100)])
        document = TestDataFactory.create_document(
            text_content=text,
            status=DocumentStatus.PARSED
        )
        
        chunk_config = {
            'method': 'sliding_window',
            'chunk_size': 100,
            'overlap': 20
        }
        
        # When
        chunks = await processor.chunk(document, chunk_config)
        
        # Then
        assert len(chunks) > 1
        assert all(len(chunk['content']) <= 100 for chunk in chunks)
        
        # 验证重叠
        for i in range(len(chunks) - 1):
            current_end = chunks[i]['content'][-20:]
            next_content = chunks[i + 1]['content']
            assert current_end in next_content, "Chunks should overlap"
    
    async def test_semantic_chunking(self, processor):
        """测试语义分片策略"""
        # Given
        text = """
        第一章：介绍
        这是第一章的内容，包含了重要的介绍信息。
        本章详细说明了系统的背景和目标。
        
        第二章：方法论
        这是第二章的内容，描述了具体的实施方法。
        包括了详细的步骤和流程说明。
        
        第三章：总结
        这是第三章的内容，对全文进行了总结。
        提出了未来的发展方向和建议。
        """
        
        document = TestDataFactory.create_document(
            text_content=text,
            status=DocumentStatus.PARSED
        )
        
        chunk_config = {
            'method': 'semantic',
            'chunk_size': 200
        }
        
        # When
        chunks = await processor.chunk(document, chunk_config)
        
        # Then
        # 验证章节完整性
        assert any("第一章" in chunk['content'] and "第一章的内容" in chunk['content'] 
                  for chunk in chunks)
        assert any("第二章" in chunk['content'] and "第二章的内容" in chunk['content'] 
                  for chunk in chunks)
        assert any("第三章" in chunk['content'] and "第三章的内容" in chunk['content'] 
                  for chunk in chunks)
    
    # ========== TC_DOC_014: 向量生成 ==========
    async def test_generate_embeddings(self, processor, mock_llm_service):
        """测试向量嵌入生成"""
        # Given
        chunks = [
            TestDataFactory.create_chunk(content="First chunk content"),
            TestDataFactory.create_chunk(content="Second chunk content"),
            TestDataFactory.create_chunk(content="Third chunk content")
        ]
        
        # When
        embeddings = await processor.generate_embeddings(chunks)
        
        # Then
        assert len(embeddings) == 3
        assert all(len(emb['vector']) == 1536 for emb in embeddings)
        assert all(isinstance(emb['vector'], list) for emb in embeddings)
        assert mock_llm_service.generate_embedding.call_count == 3
    
    async def test_batch_embedding_generation(self, processor, mock_llm_service):
        """测试批量向量生成"""
        # Given
        chunks = [TestDataFactory.create_chunk() for _ in range(100)]
        
        # When
        start_time = time.time()
        embeddings = await processor.generate_embeddings(chunks, batch_size=10)
        process_time = time.time() - start_time
        
        # Then
        assert len(embeddings) == 100
        assert process_time < 10  # 应该在10秒内完成
        # 验证批量调用（100个chunks，每批10个 = 10次调用）
        assert mock_llm_service.generate_embedding.call_count == 10
    
    # ========== TC_DOC_013: 质量评估 ==========
    async def test_chunk_quality_evaluation(self, processor):
        """测试片段质量评估"""
        # Given
        high_quality_chunk = TestDataFactory.create_chunk(
            content="This is a well-structured paragraph with clear and comprehensive information about the topic. It contains relevant keywords and maintains coherent flow throughout the text.",
            tokens=30
        )
        
        low_quality_chunk = TestDataFactory.create_chunk(
            content="Um... like... you know...",
            tokens=5
        )
        
        # When
        high_score = await processor.evaluate_quality(high_quality_chunk)
        low_score = await processor.evaluate_quality(low_quality_chunk)
        
        # Then
        assert 0.7 <= high_score <= 1.0
        assert 0.0 <= low_score <= 0.3
        assert high_score > low_score

# ==================== 搜索服务测试 ====================
@pytest.mark.asyncio
class TestSearchService:
    """搜索服务测试套件"""
    
    @pytest.fixture
    def search_service(self):
        """创建搜索服务实例"""
        return SearchService()
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock向量数据库"""
        with patch('app.services.vector_db.VectorDB') as mock:
            instance = mock.return_value
            instance.search = AsyncMock(return_value=[
                {'id': 'doc1', 'score': 0.95, 'content': 'Relevant content'},
                {'id': 'doc2', 'score': 0.85, 'content': 'Another match'}
            ])
            yield instance
    
    # ========== TC_SEARCH_001: 基础向量搜索 ==========
    async def test_basic_vector_search(self, search_service, mock_vector_db):
        """测试基础向量搜索功能"""
        # Given
        query = TestDataFactory.create_search_query(
            query="如何配置RAG系统",
            top_k=10,
            threshold=0.7
        )
        
        # When
        start_time = time.time()
        results = await search_service.vector_search(query)
        response_time = (time.time() - start_time) * 1000
        
        # Then
        assert len(results) > 0
        assert all(r['score'] >= 0.7 for r in results)
        assert results[0]['score'] >= results[1]['score']  # 按分数排序
        assert response_time < 200  # 响应时间小于200ms
    
    # ========== TC_SEARCH_002: 混合检索 ==========
    async def test_hybrid_search(self, search_service):
        """测试混合检索功能"""
        # Given
        query = TestDataFactory.create_search_query(
            query="test query",
            vector_weight=0.6,
            keyword_weight=0.4
        )
        
        with patch.object(search_service, 'vector_search', new_callable=AsyncMock) as mock_vector:
            with patch.object(search_service, 'keyword_search', new_callable=AsyncMock) as mock_keyword:
                mock_vector.return_value = [
                    {'id': 'vec1', 'score': 0.9},
                    {'id': 'vec2', 'score': 0.8},
                    {'id': 'common', 'score': 0.7}
                ]
                
                mock_keyword.return_value = [
                    {'id': 'key1', 'score': 0.85},
                    {'id': 'key2', 'score': 0.75},
                    {'id': 'common', 'score': 0.65}
                ]
                
                # When
                results = await search_service.hybrid_search(query)
                
                # Then
                assert len(results) == 5  # 去重后的结果
                
                # 验证分数融合
                common_result = next((r for r in results if r['id'] == 'common'), None)
                assert common_result is not None
                assert common_result['score'] > 0.7  # 融合后分数应该更高
    
    # ========== TC_SEARCH_005: 并发搜索测试 ==========
    async def test_concurrent_search_handling(self, search_service, mock_vector_db):
        """测试并发搜索处理"""
        # Given
        num_concurrent = 100
        queries = [
            TestDataFactory.create_search_query(query=f"query_{i}")
            for i in range(num_concurrent)
        ]
        
        # When
        start_time = time.time()
        tasks = [search_service.vector_search(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Then
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) >= 99  # 至少99%成功
        assert len(failed) <= 1  # 最多1%失败
        assert total_time < 10  # 100个并发请求在10秒内完成
    
    # ========== TC_SEARCH_007: 缓存机制 ==========
    async def test_search_cache_mechanism(self, search_service):
        """测试搜索缓存机制"""
        # Given
        query = TestDataFactory.create_search_query(query="cached query")
        
        # 第一次搜索
        start1 = time.time()
        result1 = await search_service.vector_search(query)
        time1 = (time.time() - start1) * 1000
        
        # 第二次搜索（应该命中缓存）
        start2 = time.time()
        result2 = await search_service.vector_search(query)
        time2 = (time.time() - start2) * 1000
        
        # Then
        assert result1 == result2  # 结果相同
        assert time2 < time1 / 10  # 缓存响应时间远小于首次查询
        assert time2 < 10  # 缓存响应小于10ms

# ==================== 数据库仓库测试 ====================
class TestDocumentRepository:
    """文档仓库测试套件"""
    
    @pytest.fixture(scope="function")
    def db_session(self):
        """创建测试数据库会话"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # 使用内存数据库进行测试
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
    
    # ========== TC_DB_001: 事务一致性 ==========
    def test_transaction_consistency(self, repository):
        """测试事务一致性"""
        # Given
        initial_count = repository.count()
        
        # When - 尝试在事务中执行操作，然后回滚
        try:
            with repository.transaction():
                doc1 = TestDataFactory.create_document(id='tx_1')
                doc2 = TestDataFactory.create_document(id='tx_2')
                
                repository.create(doc1)
                repository.create(doc2)
                
                # 故意触发错误
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Then - 事务应该回滚
        assert repository.count() == initial_count
        assert repository.find_by_id('tx_1') is None
        assert repository.find_by_id('tx_2') is None
    
    # ========== TC_DB_002: 并发写入 ==========
    def test_concurrent_writes(self, repository):
        """测试并发写入"""
        # Given
        num_threads = 10
        docs_per_thread = 100
        
        def write_documents(thread_id):
            docs = [
                TestDataFactory.create_document(
                    id=f'doc_{thread_id}_{i}'
                )
                for i in range(docs_per_thread)
            ]
            for doc in docs:
                repository.create(doc)
        
        # When
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(write_documents, i)
                for i in range(num_threads)
            ]
            for future in futures:
                future.result()
        
        # Then
        total_docs = repository.count()
        assert total_docs == num_threads * docs_per_thread
        
        # 验证无重复ID
        all_docs = repository.find_all()
        doc_ids = [doc.id for doc in all_docs]
        assert len(doc_ids) == len(set(doc_ids))
    
    def test_pagination(self, repository):
        """测试分页功能"""
        # Given - 创建25个文档
        for i in range(25):
            doc = TestDataFactory.create_document(id=f'doc_{i:03d}')
            repository.create(doc)
        
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
    
    def test_bulk_operations(self, repository):
        """测试批量操作"""
        # Given
        documents = [
            TestDataFactory.create_document(id=f'bulk_{i}')
            for i in range(1000)
        ]
        
        # When - 批量插入
        start_time = time.time()
        repository.bulk_insert(documents)
        insert_time = time.time() - start_time
        
        # Then
        assert repository.count() == 1000
        assert insert_time < 1.0  # 批量插入应该在1秒内完成
        
        # When - 批量更新
        ids = [f'bulk_{i}' for i in range(500)]
        repository.bulk_update(ids, {'status': DocumentStatus.PROCESSED})
        
        # Then
        processed = repository.find_by_status(DocumentStatus.PROCESSED)
        assert len(processed) == 500

# ==================== API端点测试 ====================
class TestAPIEndpoints:
    """API端点测试套件"""
    
    # ========== TC_DOC_001: 文档上传API ==========
    def test_document_upload_endpoint(self):
        """测试文档上传端点"""
        # Given
        files = {
            'file': ('test.pdf', TestDataFactory.create_pdf_content(), 'application/pdf')
        }
        data = {
            'metadata': json.dumps({'source': 'test', 'tags': ['test']})
        }
        
        # When
        response = client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data
        )
        
        # Then
        assert response.status_code == 201
        result = response.json()
        assert 'id' in result
        assert result['status'] in ['pending', 'processing']
        assert 'message' in result
    
    def test_upload_invalid_file_format(self):
        """测试上传无效格式文件"""
        # Given
        files = {
            'file': ('malware.exe', b'executable content', 'application/x-msdownload')
        }
        
        # When
        response = client.post("/api/v1/documents/upload", files=files)
        
        # Then
        assert response.status_code == 400
        assert 'error' in response.json()
        assert '不支持' in response.json()['error']
    
    # ========== TC_SEARCH_001: 搜索API ==========
    def test_search_endpoint(self):
        """测试搜索端点"""
        # Given
        search_request = {
            'query': 'test search query',
            'top_k': 10,
            'threshold': 0.7,
            'filters': {'type': 'pdf'}
        }
        
        # When
        response = client.post(
            "/api/v1/search",
            json=search_request
        )
        
        # Then
        assert response.status_code == 200
        result = response.json()
        assert 'results' in result
        assert 'total' in result
        assert 'took_ms' in result
        assert isinstance(result['results'], list)
    
    def test_search_with_invalid_params(self):
        """测试无效参数搜索"""
        # Given
        invalid_request = {
            'query': '',  # 空查询
            'top_k': 1000,  # 超出范围
            'threshold': 2.0  # 无效阈值
        }
        
        # When
        response = client.post("/api/v1/search", json=invalid_request)
        
        # Then
        assert response.status_code == 422
        errors = response.json()['detail']
        assert any(e['loc'][-1] == 'query' for e in errors)
        assert any(e['loc'][-1] == 'top_k' for e in errors)
        assert any(e['loc'][-1] == 'threshold' for e in errors)
    
    # ========== 策略配置API ==========
    def test_strategy_configuration_endpoint(self):
        """测试策略配置端点"""
        # Given
        strategy_config = {
            'recall_threshold': 0.8,
            'rerank_threshold': 0.85,
            'max_chunks': 5,
            'temperature': 0.1
        }
        
        # When
        response = client.put(
            "/api/v1/strategy/config",
            json=strategy_config,
            headers={'Authorization': 'Bearer admin_token'}
        )
        
        # Then
        assert response.status_code == 200
        result = response.json()
        assert result['recall_threshold'] == 0.8
        assert result['status'] == 'updated'

# ==================== 性能测试 ====================
@pytest.mark.performance
class TestPerformance:
    """性能测试套件"""
    
    # ========== TC_PERF_001: 负载测试 ==========
    @pytest.mark.asyncio
    async def test_standard_load(self):
        """测试标准负载性能"""
        # Given
        concurrent_users = 100
        duration = 30  # 秒
        target_tps = 500
        
        async def single_request():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/api/v1/search",
                    json={'query': fake.sentence(), 'top_k': 10}
                )
                return response.status_code, response.elapsed.total_seconds()
        
        # When
        start_time = time.time()
        request_count = 0
        errors = 0
        latencies = []
        
        while time.time() - start_time < duration:
            tasks = [single_request() for _ in range(concurrent_users)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    status, latency = result
                    if status != 200:
                        errors += 1
                    latencies.append(latency * 1000)  # 转换为毫秒
                    request_count += 1
            
            await asyncio.sleep(0.1)  # 避免过载
        
        # Then
        total_time = time.time() - start_time
        actual_tps = request_count / total_time
        error_rate = errors / request_count * 100 if request_count > 0 else 100
        
        # 计算延迟百分位数
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]
        p99_latency = latencies[int(len(latencies) * 0.99)]
        
        # 验证性能指标
        assert actual_tps > target_tps, f"TPS {actual_tps} 低于目标 {target_tps}"
        assert p95_latency < 1000, f"P95延迟 {p95_latency}ms 超过1秒"
        assert p99_latency < 2000, f"P99延迟 {p99_latency}ms 超过2秒"
        assert error_rate < 0.1, f"错误率 {error_rate}% 超过0.1%"
    
    # ========== TC_PERF_003: 数据容量测试 ==========
    def test_large_scale_data_handling(self):
        """测试大规模数据处理"""
        # Given
        num_documents = 10000
        num_vectors = 100000
        
        # 创建大量测试数据
        with TestSessionLocal() as session:
            repo = DocumentRepository(session)
            
            # 批量插入文档
            batch_size = 1000
            for i in range(0, num_documents, batch_size):
                docs = [
                    TestDataFactory.create_document(id=f'scale_{j}')
                    for j in range(i, min(i + batch_size, num_documents))
                ]
                repo.bulk_insert(docs)
            
            # When - 测试查询性能
            start_time = time.time()
            
            # 执行各种查询
            results1 = repo.find_all(page=1, per_page=100)
            results2 = repo.find_by_status(DocumentStatus.PENDING)
            results3 = repo.search("test")
            
            query_time = time.time() - start_time
            
            # Then
            assert results1.total == num_documents
            assert query_time < 1.0  # 查询应该在1秒内完成
    
    # ========== 内存泄漏测试 ==========
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # 获取初始内存使用
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量操作
        processor = DocumentProcessor()
        for i in range(1000):
            doc = TestDataFactory.create_document()
            await processor.parse(doc)
            
            # 每100次操作后进行垃圾回收
            if i % 100 == 0:
                gc.collect()
        
        # 最终垃圾回收
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 验证内存增长
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100, f"内存增长 {memory_growth}MB 超过100MB"

# ==================== 安全测试 ====================
@pytest.mark.security
class TestSecurity:
    """安全测试套件"""
    
    # ========== TC_SEC_001: 认证测试 ==========
    def test_authentication_required(self):
        """测试需要认证的端点"""
        # Given - 无认证信息的请求
        
        # When
        response = client.post("/api/v1/admin/users")
        
        # Then
        assert response.status_code == 401
        assert 'error' in response.json()
    
    def test_invalid_token(self):
        """测试无效令牌"""
        # Given
        headers = {'Authorization': 'Bearer invalid_token_12345'}
        
        # When
        response = client.get("/api/v1/admin/users", headers=headers)
        
        # Then
        assert response.status_code == 401
    
    # ========== TC_SEC_004: SQL注入测试 ==========
    def test_sql_injection_prevention(self):
        """测试SQL注入防护"""
        # Given - SQL注入测试向量
        injection_vectors = [
            "' OR '1'='1",
            "'; DROP TABLE documents--",
            "1; DELETE FROM users WHERE 1=1--",
            "' UNION SELECT * FROM users--"
        ]
        
        for vector in injection_vectors:
            # When
            response = client.post(
                "/api/v1/search",
                json={'query': vector, 'top_k': 10}
            )
            
            # Then
            assert response.status_code in [200, 400]  # 正常响应或参数错误
            # 不应该有数据库错误
            if response.status_code == 200:
                assert 'error' not in response.json()
    
    # ========== TC_SEC_005: XSS防护测试 ==========
    def test_xss_prevention(self):
        """测试XSS攻击防护"""
        # Given - XSS测试向量
        xss_vectors = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>"
        ]
        
        for vector in xss_vectors:
            # When
            response = client.post(
                "/api/v1/documents",
                json={'name': vector, 'content': 'test'}
            )
            
            # Then
            if response.status_code == 200:
                result = response.json()
                # 验证返回的内容已转义
                assert '<script>' not in str(result)
                assert 'javascript:' not in str(result)
    
    def test_rate_limiting(self):
        """测试速率限制"""
        # Given - 快速发送大量请求
        num_requests = 100
        
        # When
        responses = []
        for _ in range(num_requests):
            response = client.get("/api/v1/documents")
            responses.append(response.status_code)
        
        # Then - 应该有部分请求被限流
        rate_limited = responses.count(429)
        assert rate_limited > 0, "应该触发速率限制"

# ==================== 集成测试 ====================
@pytest.mark.integration
class TestIntegration:
    """集成测试套件"""
    
    # ========== TC_E2E_001: 完整文档处理流程 ==========
    @pytest.mark.asyncio
    async def test_complete_document_processing_flow(self):
        """测试完整的文档处理流程"""
        # 1. 上传文档
        files = {
            'file': ('integration_test.pdf', 
                    TestDataFactory.create_pdf_content(pages=3),
                    'application/pdf')
        }
        
        upload_response = client.post(
            "/api/v1/documents/upload",
            files=files
        )
        assert upload_response.status_code == 201
        doc_id = upload_response.json()['id']
        
        # 2. 等待处理完成
        max_wait = 30  # 秒
        start_time = time.time()
        processed = False
        
        while time.time() - start_time < max_wait:
            status_response = client.get(f"/api/v1/documents/{doc_id}")
            if status_response.status_code == 200:
                status = status_response.json()['status']
                if status == 'processed':
                    processed = True
                    break
            await asyncio.sleep(1)
        
        assert processed, "文档处理超时"
        
        # 3. 获取文档片段
        chunks_response = client.get(f"/api/v1/documents/{doc_id}/chunks")
        assert chunks_response.status_code == 200
        chunks = chunks_response.json()
        assert len(chunks) > 0
        
        # 4. 执行搜索测试
        search_response = client.post(
            "/api/v1/search",
            json={'query': 'test query', 'top_k': 10}
        )
        assert search_response.status_code == 200
        search_results = search_response.json()
        assert 'results' in search_results
        
        # 5. 验证端到端流程成功
        assert doc_id in [r.get('doc_id') for r in search_results.get('results', [])]
    
    # ========== 第三方服务集成测试 ==========
    @pytest.mark.asyncio
    async def test_third_party_service_integration(self):
        """测试第三方服务集成"""
        # Given - Mock第三方服务
        with patch('app.services.openai_service.OpenAIService') as mock_openai:
            mock_openai.return_value.generate_embedding = AsyncMock(
                return_value=np.random.rand(1536).tolist()
            )
            
            with patch('app.services.s3_service.S3Service') as mock_s3:
                mock_s3.return_value.upload_file = AsyncMock(
                    return_value={'url': 'https://s3.example.com/file.pdf'}
                )
                
                # When - 执行需要第三方服务的操作
                processor = DocumentProcessor()
                doc = TestDataFactory.create_document()
                
                result = await processor.process_with_external_services(doc)
                
                # Then
                assert result is not None
                assert mock_openai.return_value.generate_embedding.called
                assert mock_s3.return_value.upload_file.called

# ==================== 测试工具和辅助函数 ====================
class TestHelpers:
    """测试辅助工具"""
    
    @staticmethod
    def wait_for_condition(condition_func, timeout=10, interval=0.1):
        """等待条件满足"""
        start = time.time()
        while time.time() - start < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False
    
    @staticmethod
    def generate_test_report(results):
        """生成测试报告"""
        report = {
            'total': len(results),
            'passed': sum(1 for r in results if r['status'] == 'passed'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'skipped': sum(1 for r in results if r['status'] == 'skipped'),
            'duration': sum(r.get('duration', 0) for r in results),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        report['pass_rate'] = (report['passed'] / report['total'] * 100) if report['total'] > 0 else 0
        
        return report
    
    @staticmethod
    async def cleanup_test_data(session):
        """清理测试数据"""
        # 删除所有测试文档
        session.query(Document).filter(
            Document.id.like('test_%') | 
            Document.id.like('doc_%')
        ).delete()
        
        # 清理缓存
        test_redis.flushdb()
        
        session.commit()

# ==================== 测试配置和钩子 ====================
@pytest.fixture(scope="session")
def setup_test_environment():
    """设置测试环境"""
    # 创建测试数据库
    Base.metadata.create_all(test_engine)
    
    # 初始化测试数据
    with TestSessionLocal() as session:
        # 创建测试用户、角色等
        pass
    
    yield
    
    # 清理
    Base.metadata.drop_all(test_engine)
    test_redis.flushall()

@pytest.fixture(autouse=True)
def reset_mocks():
    """每个测试后重置所有mocks"""
    yield
    patch.stopall()

def pytest_configure(config):
    """Pytest配置"""
    config.addinivalue_line(
        "markers", "performance: 性能测试"
    )
    config.addinivalue_line(
        "markers", "security: 安全测试"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试"
    )

# ==================== 运行测试 ====================
if __name__ == "__main__":
    # 运行所有测试
    pytest.main([
        __file__,
        "-v",  # 详细输出
        "--tb=short",  # 简短的traceback
        "--cov=app",  # 代码覆盖率
        "--cov-report=html",  # HTML覆盖率报告
        "--cov-report=term-missing",  # 终端显示未覆盖行
        "-n", "4",  # 并行执行（4个进程）
        "--maxfail=5",  # 失败5个后停止
        "--durations=10",  # 显示最慢的10个测试
    ])