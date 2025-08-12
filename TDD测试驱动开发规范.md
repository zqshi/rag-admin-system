# RAG系统 TDD测试驱动开发规范

**版本**: 1.0  
**日期**: 2025年8月12日  
**团队**: Engineering Team

---

## 一、TDD核心理念

### 1.1 红-绿-重构循环

```mermaid
graph LR
    A[编写失败测试 🔴] --> B[编写最小代码 🟢]
    B --> C[重构优化 🔄]
    C --> A
```

### 1.2 TDD基本原则

1. **测试先行**：在编写功能代码前先编写测试
2. **小步迭代**：每次只编写一个小测试
3. **快速反馈**：测试运行时间 < 100ms
4. **100%覆盖**：所有业务逻辑必须有测试覆盖
5. **持续重构**：保持代码整洁

### 1.3 测试金字塔

```
         /\
        /  \  E2E测试 (5%)
       /    \  - 用户旅程
      /      \  - 关键流程
     /--------\
    /          \ 集成测试 (15%)
   /            \ - API测试
  /              \ - 数据库测试
 /________________\
/                  \ 单元测试 (80%)
                     - 纯函数
                     - 业务逻辑
                     - 工具函数
```

---

## 二、测试命名规范

### 2.1 测试文件命名

```
# 单元测试
*.test.ts / *.spec.ts

# 集成测试  
*.integration.test.ts

# E2E测试
*.e2e.test.ts
```

### 2.2 测试用例命名

```typescript
// Given-When-Then模式
describe('DocumentProcessor', () => {
  describe('when processing a valid PDF', () => {
    it('should extract text content successfully', () => {
      // Given: 有效的PDF文件
      // When: 调用处理方法
      // Then: 成功提取文本
    });
    
    it('should generate correct chunks with overlap', () => {
      // Given: 文档内容
      // When: 分片处理
      // Then: 生成正确的重叠片段
    });
  });
  
  describe('when processing invalid file', () => {
    it('should throw ParseError for corrupted file', () => {
      // Given: 损坏的文件
      // When: 尝试处理
      // Then: 抛出解析错误
    });
  });
});
```

---

## 三、测试优先级策略

### 3.1 P0 - 核心功能（必须100%覆盖）
- 文档解析
- 向量生成
- 检索算法
- 用户认证

### 3.2 P1 - 重要功能（≥90%覆盖）
- FAQ管理
- 策略配置
- 数据同步
- 错误处理

### 3.3 P2 - 辅助功能（≥80%覆盖）
- 统计分析
- 日志记录
- 缓存逻辑

---

## 四、测试数据管理

### 4.1 测试数据工厂

```typescript
// factories/document.factory.ts
export class DocumentFactory {
  static createValidPDF(): Document {
    return {
      id: faker.datatype.uuid(),
      name: 'test.pdf',
      content: Buffer.from('...'),
      metadata: {
        size: 1024,
        type: 'application/pdf',
        created: new Date()
      }
    };
  }
  
  static createWithChunks(count: number): Document {
    return {
      ...this.createValidPDF(),
      chunks: Array.from({ length: count }, (_, i) => ({
        id: `chunk_${i}`,
        content: faker.lorem.paragraph(),
        position: i,
        tokens: faker.datatype.number({ min: 100, max: 500 })
      }))
    };
  }
}
```

### 4.2 测试数据隔离

```yaml
# docker-compose.test.yml
services:
  test_db:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    tmpfs:
      - /var/lib/postgresql/data  # 内存数据库，提升速度
      
  test_redis:
    image: redis:7-alpine
    command: redis-server --save ""  # 禁用持久化
    
  test_milvus:
    image: milvusdb/milvus:v2.3.0
    environment:
      ETCD_USE_EMBED: true
      COMMON_STORAGETYPE: local
```

---

## 五、Mock策略

### 5.1 外部服务Mock

```typescript
// mocks/llm.mock.ts
export const mockLLMService = {
  generateEmbedding: jest.fn().mockResolvedValue({
    vector: Array(1536).fill(0.1),
    tokens: 256,
    model: 'text-embedding-ada-002'
  }),
  
  generateAnswer: jest.fn().mockResolvedValue({
    answer: 'Mocked response',
    tokens: 150,
    confidence: 0.95
  })
};

// 使用Mock
beforeEach(() => {
  jest.clearAllMocks();
});

it('should generate embeddings for document', async () => {
  const result = await processor.process(document);
  
  expect(mockLLMService.generateEmbedding).toHaveBeenCalledWith(
    expect.objectContaining({
      text: document.content,
      model: 'text-embedding-ada-002'
    })
  );
});
```

### 5.2 时间Mock

```typescript
// 固定时间测试
beforeEach(() => {
  jest.useFakeTimers();
  jest.setSystemTime(new Date('2025-01-01'));
});

afterEach(() => {
  jest.useRealTimers();
});

it('should expire cache after TTL', () => {
  cache.set('key', 'value', 3600);
  
  jest.advanceTimersByTime(3601 * 1000);
  
  expect(cache.get('key')).toBeNull();
});
```

---

## 六、测试覆盖率要求

### 6.1 覆盖率标准

```json
{
  "jest": {
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 85,
        "lines": 85,
        "statements": 85
      },
      "src/core/**/*.ts": {
        "branches": 95,
        "functions": 95,
        "lines": 95,
        "statements": 95
      }
    }
  }
}
```

### 6.2 覆盖率报告

```bash
# 生成覆盖率报告
npm run test:coverage

# 覆盖率门禁
npm run test:coverage:check

# HTML报告
open coverage/lcov-report/index.html
```

---

## 七、性能测试规范

### 7.1 基准测试

```typescript
// benchmarks/search.bench.ts
describe('Search Performance', () => {
  const benchmark = new Benchmark();
  
  it('should complete vector search within 100ms', async () => {
    const result = await benchmark.run(async () => {
      await searchService.vectorSearch(query, { limit: 10 });
    }, { iterations: 100 });
    
    expect(result.mean).toBeLessThan(100);
    expect(result.p95).toBeLessThan(150);
    expect(result.p99).toBeLessThan(200);
  });
  
  it('should handle 1000 concurrent requests', async () => {
    const promises = Array(1000).fill(null).map(() => 
      searchService.search(randomQuery())
    );
    
    const start = Date.now();
    await Promise.all(promises);
    const duration = Date.now() - start;
    
    expect(duration).toBeLessThan(5000); // 5秒内完成
  });
});
```

### 7.2 内存泄漏测试

```typescript
// memory/leak.test.ts
describe('Memory Leak Detection', () => {
  it('should not leak memory during document processing', async () => {
    const initialMemory = process.memoryUsage().heapUsed;
    
    // 处理1000个文档
    for (let i = 0; i < 1000; i++) {
      const doc = DocumentFactory.createLarge();
      await processor.process(doc);
    }
    
    // 强制垃圾回收
    if (global.gc) global.gc();
    
    const finalMemory = process.memoryUsage().heapUsed;
    const memoryGrowth = finalMemory - initialMemory;
    
    // 内存增长不应超过100MB
    expect(memoryGrowth).toBeLessThan(100 * 1024 * 1024);
  });
});
```

---

## 八、CI/CD集成

### 8.1 GitHub Actions配置

```yaml
name: Test Pipeline

on: [push, pull_request]

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
          
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Run unit tests
        run: npm run test:unit
        
      - name: Run integration tests
        run: npm run test:integration
        
      - name: Check coverage
        run: npm run test:coverage:check
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          
      - name: Performance tests
        run: npm run test:performance
        if: github.event_name == 'pull_request'
```

### 8.2 测试并行化

```json
{
  "scripts": {
    "test:parallel": "jest --maxWorkers=4",
    "test:shard": "jest --shard=1/4"
  }
}
```

---

## 九、测试报告模板

### 9.1 日报模板

```markdown
## 测试日报 - 2025-01-15

### 执行情况
- 总用例数：1,234
- 通过：1,220
- 失败：14
- 跳过：0

### 覆盖率
- 行覆盖率：87.3% (+0.5%)
- 分支覆盖率：82.1% (-0.2%)
- 函数覆盖率：89.7% (+1.1%)

### 失败用例分析
1. `DocumentProcessor > PDF parsing` - 环境问题
2. `SearchService > concurrent requests` - 性能退化

### 今日新增
- 单元测试：23个
- 集成测试：5个
```

---

## 十、最佳实践

### 10.1 DO's ✅

1. **保持测试独立**：每个测试应该能单独运行
2. **使用AAA模式**：Arrange-Act-Assert
3. **测试行为而非实现**：关注输入输出
4. **及时清理**：使用afterEach清理状态
5. **有意义的断言**：使用具体的匹配器

### 10.2 DON'Ts ❌

1. **避免测试私有方法**：只测试公共API
2. **不要过度Mock**：保持适度的集成
3. **避免随机数据**：使用固定的测试数据
4. **不要忽略失败测试**：立即修复
5. **避免过长测试**：单个测试 < 20行

### 10.3 代码示例

```typescript
// ❌ 错误示例
it('should work', () => {
  const result = service.process(data);
  expect(result).toBeTruthy();
});

// ✅ 正确示例
it('should extract 5 chunks from 1000-word document with 200-word overlap', () => {
  // Arrange
  const document = DocumentFactory.create({ wordCount: 1000 });
  const config = { chunkSize: 300, overlap: 200 };
  
  // Act
  const chunks = chunkService.split(document, config);
  
  // Assert
  expect(chunks).toHaveLength(5);
  expect(chunks[0].wordCount).toBe(300);
  expect(chunks[1].content.startsWith(
    document.content.substring(100) // 验证重叠
  )).toBe(true);
});
```

---

## 附录：常用测试工具

| 工具 | 用途 | 命令 |
|------|------|------|
| Jest | 单元测试框架 | `npm test` |
| Supertest | API测试 | `npm run test:api` |
| Playwright | E2E测试 | `npm run test:e2e` |
| K6 | 负载测试 | `k6 run load-test.js` |
| SonarQube | 代码质量 | `sonar-scanner` |
| Stryker | 变异测试 | `npx stryker run` |