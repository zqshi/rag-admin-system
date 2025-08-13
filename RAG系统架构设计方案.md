# RAG系统架构设计方案

**版本**：1.0  
**日期**：2025年8月13日  
**架构师**：系统架构团队

---

## 一、架构原则

### 核心理念
1. **渐进式演进**：从MVP开始，逐步扩展功能
2. **实用主义**：优先解决实际问题，避免过度设计
3. **快速迭代**：2周一个版本，持续交付价值
4. **数据驱动**：基于用户反馈和指标优化

---

## 二、整体架构演进路线

### Phase 1: MVP版本（2周）

#### 系统架构图
```mermaid
graph TB
    subgraph "前端层"
        A[Web界面<br/>Next.js]
    end
    
    subgraph "应用层"
        B[API服务<br/>FastAPI]
        C[文档处理器<br/>Python]
    end
    
    subgraph "数据层"
        D[SQLite<br/>元数据]
        E[FAISS<br/>向量索引]
        F[本地文件<br/>文档存储]
    end
    
    A --> B
    B --> C
    B --> D
    C --> E
    C --> F
```

#### 核心功能
- 文档上传（PDF/TXT/MD）
- 自动分片处理
- 向量化存储
- 简单问答接口

#### 技术栈
```yaml
后端:
  - FastAPI 0.100+
  - SQLite
  - FAISS
  - Sentence-Transformers
  - PyPDF2/python-docx

前端:
  - Next.js 14
  - TailwindCSS
  - shadcn/ui

部署:
  - Docker
  - Nginx
```

---

### Phase 2: 基础功能版（4周）

#### 系统架构图
```mermaid
graph TB
    subgraph "接入层"
        LB[Nginx<br/>负载均衡]
    end
    
    subgraph "前端层"
        WEB[Web管理端<br/>Next.js]
        API_DOC[API文档<br/>Swagger]
    end
    
    subgraph "应用层"
        AUTH[认证服务]
        DOC_API[文档服务]
        SEARCH_API[搜索服务]
        PROCESS[处理服务]
    end
    
    subgraph "数据层"
        PG[(PostgreSQL<br/>业务数据)]
        QDRANT[(Qdrant<br/>向量数据)]
        MINIO[MinIO<br/>对象存储]
    end
    
    LB --> WEB
    LB --> API_DOC
    WEB --> AUTH
    WEB --> DOC_API
    WEB --> SEARCH_API
    DOC_API --> PROCESS
    AUTH --> PG
    DOC_API --> PG
    SEARCH_API --> QDRANT
    PROCESS --> MINIO
    PROCESS --> QDRANT
```

#### 新增功能
- 用户认证与权限管理
- FAQ管理
- 文档版本控制
- 批量处理能力
- 基础监控面板

---

### Phase 3: 企业级版本（8周）

#### 系统架构图
```mermaid
graph TB
    subgraph "接入层"
        CDN[CDN]
        WAF[WAF]
        LB[负载均衡器]
    end
    
    subgraph "网关层"
        GW[API Gateway<br/>Kong/Apisix]
    end
    
    subgraph "应用层"
        subgraph "核心服务"
            AUTH[认证中心]
            DOC[文档服务]
            SEARCH[搜索服务]
            STRATEGY[策略服务]
        end
        
        subgraph "AI服务"
            NLP[NLP处理]
            RANK[重排序服务]
            LLM[LLM网关]
        end
        
        subgraph "支撑服务"
            TASK[任务调度<br/>Celery]
            MONITOR[监控服务]
        end
    end
    
    subgraph "缓存层"
        REDIS[(Redis<br/>缓存/会话)]
    end
    
    subgraph "数据层"
        PG[(PostgreSQL<br/>主库)]
        PG_R[(PostgreSQL<br/>从库)]
        QDRANT[(Qdrant<br/>集群)]
        ES[(Elasticsearch<br/>全文检索)]
        S3[S3兼容存储]
    end
    
    subgraph "消息层"
        MQ[RabbitMQ/Kafka]
    end
    
    CDN --> WAF
    WAF --> LB
    LB --> GW
    GW --> AUTH
    GW --> DOC
    GW --> SEARCH
    GW --> STRATEGY
    DOC --> NLP
    SEARCH --> RANK
    STRATEGY --> LLM
    DOC --> TASK
    TASK --> MQ
    AUTH --> REDIS
    DOC --> PG
    SEARCH --> QDRANT
    SEARCH --> ES
    NLP --> S3
```

---

## 三、核心模块设计

### 3.1 文档处理模块

#### 处理流程
```mermaid
sequenceDiagram
    participant User
    participant API
    participant Parser
    participant Chunker
    participant Embedder
    participant Storage
    
    User->>API: 上传文档
    API->>Parser: 解析文档
    Parser->>Chunker: 文本分片
    Chunker->>Embedder: 向量化
    Embedder->>Storage: 存储
    Storage-->>API: 返回成功
    API-->>User: 处理完成
```

#### 分片策略
```python
class ChunkStrategy:
    """分片策略配置"""
    
    STRATEGIES = {
        "fixed": {
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "semantic": {
            "max_chunk_size": 1000,
            "min_chunk_size": 100,
            "separator": ["\n\n", "\n", "。", "！", "？"]
        },
        "sliding_window": {
            "window_size": 512,
            "stride": 256
        }
    }
```

### 3.2 检索服务模块

#### 混合检索架构
```mermaid
graph LR
    A[用户Query] --> B[Query理解]
    B --> C[关键词提取]
    B --> D[向量化]
    C --> E[BM25检索]
    D --> F[向量检索]
    E --> G[融合排序]
    F --> G
    G --> H[重排序]
    H --> I[结果返回]
```

#### 检索配置
```yaml
retrieval:
  semantic:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    top_k: 20
    threshold: 0.7
  
  keyword:
    algorithm: "bm25"
    top_k: 10
    boost: 1.2
  
  fusion:
    method: "rrf"  # Reciprocal Rank Fusion
    weights:
      semantic: 0.6
      keyword: 0.4
  
  rerank:
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 5
```

### 3.3 策略干预模块

#### 策略配置模型
```python
@dataclass
class StrategyConfig:
    """策略配置数据模型"""
    
    # 召回策略
    recall_config: dict = field(default_factory=lambda: {
        "semantic_threshold": 0.75,
        "keyword_weight": 0.4,
        "max_candidates": 50
    })
    
    # 排序策略
    ranking_config: dict = field(default_factory=lambda: {
        "rerank_threshold": 0.6,
        "diversity_factor": 0.2,
        "recency_boost": 1.1
    })
    
    # LLM配置
    llm_config: dict = field(default_factory=lambda: {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2048,
        "system_prompt": "你是一个专业的知识助手..."
    })
```

---

## 四、数据模型设计

### 核心实体关系
```mermaid
erDiagram
    USER ||--o{ DOCUMENT : uploads
    USER ||--o{ FAQ : manages
    USER ||--o{ SEARCH_LOG : generates
    
    DOCUMENT ||--o{ CHUNK : contains
    DOCUMENT ||--o{ VERSION : has
    DOCUMENT }|--|| KNOWLEDGE_BASE : belongs_to
    
    CHUNK ||--o{ EMBEDDING : has
    CHUNK }o--o{ FAQ : references
    
    STRATEGY_CONFIG ||--|| KNOWLEDGE_BASE : applies_to
    SEARCH_LOG }|--|| CHUNK : retrieves
    
    USER {
        uuid id PK
        string email
        string role
        timestamp created_at
    }
    
    DOCUMENT {
        uuid id PK
        string title
        string source_type
        string file_path
        jsonb metadata
        timestamp created_at
    }
    
    CHUNK {
        uuid id PK
        uuid document_id FK
        int position
        text content
        jsonb metadata
        timestamp created_at
    }
    
    EMBEDDING {
        uuid id PK
        uuid chunk_id FK
        vector embedding
        string model_name
    }
    
    FAQ {
        uuid id PK
        text question
        text answer
        uuid source_doc_id FK
        float confidence
    }
    
    STRATEGY_CONFIG {
        uuid id PK
        uuid kb_id FK
        jsonb recall_config
        jsonb ranking_config
        jsonb llm_config
        boolean is_active
    }
```

---

## 五、API设计规范

### RESTful API示例
```yaml
# 文档管理API
POST   /api/v1/documents          # 上传文档
GET    /api/v1/documents          # 文档列表
GET    /api/v1/documents/{id}     # 文档详情
PUT    /api/v1/documents/{id}     # 更新文档
DELETE /api/v1/documents/{id}     # 删除文档

# 搜索API
POST   /api/v1/search             # 执行搜索
GET    /api/v1/search/history     # 搜索历史

# 策略配置API
GET    /api/v1/strategies         # 策略列表
POST   /api/v1/strategies         # 创建策略
PUT    /api/v1/strategies/{id}    # 更新策略
```

### 请求/响应示例
```json
// 搜索请求
{
  "query": "如何配置RAG系统的分片策略",
  "knowledge_base_id": "kb_123",
  "top_k": 5,
  "filters": {
    "document_type": ["pdf", "markdown"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  }
}

// 搜索响应
{
  "status": "success",
  "data": {
    "answer": "RAG系统的分片策略配置...",
    "sources": [
      {
        "document_id": "doc_456",
        "chunk_id": "chunk_789",
        "content": "分片策略包括固定大小分片...",
        "score": 0.92,
        "metadata": {
          "page": 15,
          "section": "3.2"
        }
      }
    ],
    "search_id": "search_abc",
    "latency_ms": 234
  }
}
```

---

## 六、部署架构

### 容器化部署（Docker Compose）
```yaml
version: '3.8'

services:
  # API服务
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/rag
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - db
      - qdrant
    volumes:
      - ./data/uploads:/app/uploads
  
  # 前端服务
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://api:8000
  
  # PostgreSQL
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=rag
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Qdrant向量数据库
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
  
  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
      - frontend

volumes:
  postgres_data:
  qdrant_data:
```

---

## 七、性能指标设计

### 关键性能指标（KPI）
| 指标类别 | 指标名称 | 目标值 | 监控方式 |
|---------|---------|-------|---------|
| 延迟指标 | P50响应时间 | < 200ms | Prometheus |
| | P95响应时间 | < 500ms | Prometheus |
| | P99响应时间 | < 1000ms | Prometheus |
| 吞吐量 | QPS | > 100 | Grafana |
| | 并发用户数 | > 500 | Grafana |
| 准确性 | 检索准确率 | > 85% | 自定义指标 |
| | 答案相关性 | > 80% | 用户反馈 |
| 可用性 | 系统可用率 | > 99.9% | Uptime监控 |
| | 错误率 | < 1% | ELK Stack |

### 监控架构
```mermaid
graph TB
    subgraph "应用层"
        APP[应用服务]
    end
    
    subgraph "采集层"
        PROM[Prometheus<br/>指标采集]
        LOKI[Loki<br/>日志采集]
        TRACE[Jaeger<br/>链路追踪]
    end
    
    subgraph "存储层"
        TSDB[(时序数据库)]
        LOG_DB[(日志存储)]
        TRACE_DB[(追踪存储)]
    end
    
    subgraph "展示层"
        GRAFANA[Grafana<br/>可视化]
        ALERT[AlertManager<br/>告警]
    end
    
    APP --> PROM
    APP --> LOKI
    APP --> TRACE
    PROM --> TSDB
    LOKI --> LOG_DB
    TRACE --> TRACE_DB
    TSDB --> GRAFANA
    LOG_DB --> GRAFANA
    TRACE_DB --> GRAFANA
    GRAFANA --> ALERT
```

---

## 八、安全架构设计

### 安全防护体系
```mermaid
graph TB
    subgraph "网络安全"
        WAF[Web应用防火墙]
        DDoS[DDoS防护]
        SSL[SSL/TLS加密]
    end
    
    subgraph "应用安全"
        AUTH[身份认证<br/>OAuth2.0/JWT]
        RBAC[权限控制<br/>RBAC]
        AUDIT[审计日志]
    end
    
    subgraph "数据安全"
        ENCRYPT[数据加密<br/>AES-256]
        BACKUP[数据备份]
        PRIVACY[隐私保护]
    end
    
    subgraph "合规性"
        GDPR[GDPR合规]
        ISO[ISO 27001]
        PEN[渗透测试]
    end
```

### 安全措施清单
- [ ] API速率限制
- [ ] SQL注入防护
- [ ] XSS/CSRF防护
- [ ] 敏感数据脱敏
- [ ] 密钥管理（Vault）
- [ ] 容器安全扫描
- [ ] 代码安全审计

---

## 九、实施计划

### 里程碑计划
```mermaid
gantt
    title RAG系统实施计划
    dateFormat  YYYY-MM-DD
    section Phase 1 MVP
    需求分析        :a1, 2025-01-15, 2d
    基础框架搭建    :a2, after a1, 3d
    文档处理实现    :a3, after a2, 3d
    检索功能开发    :a4, after a3, 3d
    前端界面开发    :a5, after a2, 5d
    集成测试        :a6, after a4, 2d
    
    section Phase 2 基础版
    用户系统        :b1, after a6, 5d
    FAQ管理         :b2, after b1, 4d
    批量处理        :b3, after b2, 3d
    监控面板        :b4, after b3, 3d
    性能优化        :b5, after b4, 3d
    
    section Phase 3 企业版
    微服务改造      :c1, after b5, 10d
    高级策略        :c2, after c1, 7d
    AI能力增强      :c3, after c2, 7d
    安全加固        :c4, after c3, 5d
    生产部署        :c5, after c4, 3d
```

---

## 十、风险评估与应对

### 技术风险
| 风险项 | 概率 | 影响 | 应对措施 |
|-------|-----|------|---------|
| 向量检索性能瓶颈 | 中 | 高 | 采用分布式索引，优化向量维度 |
| LLM调用成本过高 | 高 | 中 | 实施缓存策略，使用开源模型 |
| 数据质量问题 | 高 | 高 | 建立数据清洗pipeline |
| 系统扩展性不足 | 低 | 高 | 预留扩展接口，模块化设计 |

### 业务风险
- **用户采纳度低**：MVP快速迭代，收集反馈
- **知识更新不及时**：建立定期更新机制
- **答案准确性不足**：人工审核+用户反馈循环

---

## 十一、成本估算

### 基础设施成本（月度）
```yaml
开发环境:
  - 服务器: 2台 * 4核8G = ¥400
  - 存储: 100GB SSD = ¥50
  - 带宽: 5Mbps = ¥100
  小计: ¥550/月

生产环境:
  - 应用服务器: 4台 * 8核16G = ¥2000
  - 数据库服务器: 2台 * 16核32G = ¥2400
  - 向量数据库: 2台 * 8核16G = ¥1200
  - 存储: 1TB SSD = ¥300
  - CDN: ¥500
  - 带宽: 100Mbps = ¥1000
  小计: ¥7400/月

AI服务:
  - OpenAI API: ¥3000/月（预估）
  - 向量模型部署: ¥1000/月
  小计: ¥4000/月

总计: ¥11,950/月
```

---

## 十二、总结与建议

### 核心价值
1. **渐进式交付**：2周见效果，持续优化
2. **技术债务控制**：避免过度设计，保持简洁
3. **用户驱动**：基于反馈快速迭代
4. **成本可控**：从小规模开始，按需扩展

### 下一步行动
1. **立即开始**：搭建MVP开发环境
2. **组建团队**：2名后端+1名前端+1名算法
3. **制定规范**：代码规范、API规范、文档规范
4. **建立DevOps**：CI/CD流水线
5. **用户试点**：选择5-10个种子用户

### 成功关键
- ✅ 保持专注，不贪大求全
- ✅ 快速迭代，小步快跑
- ✅ 数据驱动，量化决策
- ✅ 用户至上，价值导向

---

> **架构不是设计出来的，是演进出来的。**  
> 让我们从第一行代码开始，构建一个真正有价值的RAG系统。