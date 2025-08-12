// 前端组件测试实现示例
// 文档上传组件完整测试套件

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import '@testing-library/jest-dom';

import { DocumentUploader } from '@/components/DocumentUploader';
import { SearchBar } from '@/components/SearchBar';
import { DocumentList } from '@/components/DocumentList';
import { ChunkEditor } from '@/components/ChunkEditor';
import { StrategyConfig } from '@/components/StrategyConfig';

// ==================== 测试数据工厂 ====================
class TestDataFactory {
  static createFile(name: string, size: number = 1024, type: string = 'application/pdf'): File {
    const content = new Array(size).fill('a').join('');
    return new File([content], name, { type });
  }

  static createDocument(overrides = {}) {
    return {
      id: 'doc_' + Math.random().toString(36).substr(2, 9),
      name: 'test.pdf',
      type: 'application/pdf',
      size: 1024,
      status: 'pending',
      createdAt: new Date().toISOString(),
      ...overrides
    };
  }

  static createChunk(overrides = {}) {
    return {
      id: 'chunk_' + Math.random().toString(36).substr(2, 9),
      content: 'This is a test chunk content',
      position: 0,
      tokens: 10,
      quality: 0.85,
      ...overrides
    };
  }
}

// ==================== Mock服务器设置 ====================
const server = setupServer(
  rest.post('/api/documents/upload', (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        id: 'doc_123',
        status: 'processing',
        message: 'Upload successful'
      })
    );
  }),

  rest.get('/api/documents', (req, res, ctx) => {
    return res(
      ctx.json({
        items: [
          TestDataFactory.createDocument({ id: 'doc_1', name: 'file1.pdf' }),
          TestDataFactory.createDocument({ id: 'doc_2', name: 'file2.docx' })
        ],
        total: 2,
        page: 1,
        pageSize: 10
      })
    );
  }),

  rest.post('/api/search', (req, res, ctx) => {
    return res(
      ctx.json({
        results: [
          { id: 'result_1', content: 'Matching content', score: 0.95 },
          { id: 'result_2', content: 'Another match', score: 0.85 }
        ],
        total: 2,
        took: 125
      })
    );
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// ==================== 文档上传组件测试 ====================
describe('DocumentUploader Component Tests', () => {
  describe('TC_DOC_001: Single File Upload', () => {
    it('should successfully upload a single PDF file', async () => {
      const onUploadComplete = jest.fn();
      
      render(
        <DocumentUploader
          onUploadComplete={onUploadComplete}
          maxSize={100}
          accept={['.pdf', '.docx']}
        />
      );

      // 创建测试文件
      const file = TestDataFactory.createFile('test.pdf', 2048);
      
      // 查找上传输入
      const input = screen.getByLabelText(/选择文件|拖拽文件/i);
      
      // 上传文件
      await userEvent.upload(input, file);
      
      // 验证文件显示在列表中
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
      
      // 点击上传按钮
      const uploadButton = screen.getByRole('button', { name: /确认上传/i });
      await userEvent.click(uploadButton);
      
      // 等待上传完成
      await waitFor(() => {
        expect(onUploadComplete).toHaveBeenCalledWith(
          expect.objectContaining({
            id: 'doc_123',
            status: 'processing'
          })
        );
      });
      
      // 验证成功提示
      expect(screen.getByText(/上传成功/i)).toBeInTheDocument();
    });
  });

  describe('TC_DOC_002: Batch Upload', () => {
    it('should handle multiple files upload with progress', async () => {
      const onProgress = jest.fn();
      
      render(
        <DocumentUploader
          multiple={true}
          onProgress={onProgress}
        />
      );

      // 创建多个文件
      const files = [
        TestDataFactory.createFile('doc1.pdf'),
        TestDataFactory.createFile('doc2.docx', 1024, 'application/vnd.openxmlformats'),
        TestDataFactory.createFile('doc3.txt', 512, 'text/plain')
      ];

      const input = screen.getByLabelText(/选择文件/i);
      await userEvent.upload(input, files);

      // 验证所有文件都在队列中
      expect(screen.getByText('doc1.pdf')).toBeInTheDocument();
      expect(screen.getByText('doc2.docx')).toBeInTheDocument();
      expect(screen.getByText('doc3.txt')).toBeInTheDocument();
      
      // 验证显示文件数量
      expect(screen.getByText(/3个文件待上传/i)).toBeInTheDocument();

      // 开始上传
      const uploadButton = screen.getByRole('button', { name: /批量上传/i });
      await userEvent.click(uploadButton);

      // 验证进度回调
      await waitFor(() => {
        expect(onProgress).toHaveBeenCalled();
      });
    });
  });

  describe('TC_DOC_003: File Size Validation', () => {
    it('should reject files exceeding size limit', async () => {
      render(
        <DocumentUploader maxSize={1} /> // 1MB限制
      );

      // 创建超大文件 (2MB)
      const largeFile = TestDataFactory.createFile('large.pdf', 2 * 1024 * 1024);
      
      const input = screen.getByLabelText(/选择文件/i);
      await userEvent.upload(input, largeFile);

      // 验证错误提示
      expect(screen.getByText(/文件大小不能超过1MB/i)).toBeInTheDocument();
      
      // 验证文件未添加到队列
      expect(screen.queryByText('large.pdf')).not.toBeInTheDocument();
    });
  });

  describe('TC_DOC_004: Invalid Format Rejection', () => {
    it('should reject unsupported file formats', async () => {
      render(
        <DocumentUploader accept={['.pdf', '.docx']} />
      );

      const invalidFile = new File(['content'], 'malware.exe', {
        type: 'application/x-msdownload'
      });

      const input = screen.getByLabelText(/选择文件/i);
      await userEvent.upload(input, invalidFile);

      expect(screen.getByText(/文件格式不支持/i)).toBeInTheDocument();
      expect(screen.queryByText('malware.exe')).not.toBeInTheDocument();
    });
  });

  describe('TC_DOC_006: Drag and Drop Upload', () => {
    it('should support drag and drop file upload', async () => {
      render(<DocumentUploader />);

      const dropzone = screen.getByTestId('dropzone');
      const file = TestDataFactory.createFile('drag-test.pdf');

      // 模拟拖拽进入
      fireEvent.dragEnter(dropzone);
      expect(dropzone).toHaveClass('dropzone-active');

      // 模拟放下文件
      fireEvent.drop(dropzone, {
        dataTransfer: {
          files: [file],
          items: [{ kind: 'file', type: file.type, getAsFile: () => file }],
          types: ['Files']
        }
      });

      // 验证文件被添加
      await waitFor(() => {
        expect(screen.getByText('drag-test.pdf')).toBeInTheDocument();
      });

      // 验证拖拽状态重置
      expect(dropzone).not.toHaveClass('dropzone-active');
    });
  });
});

// ==================== 搜索组件测试 ====================
describe('SearchBar Component Tests', () => {
  describe('TC_SEARCH_001: Basic Search', () => {
    it('should perform search with debouncing', async () => {
      const onSearch = jest.fn();
      
      render(
        <SearchBar
          onSearch={onSearch}
          debounceMs={300}
        />
      );

      const input = screen.getByPlaceholderText(/搜索文档/i);
      
      // 快速输入
      await userEvent.type(input, 'test query');
      
      // 立即检查：不应该调用
      expect(onSearch).not.toHaveBeenCalled();
      
      // 等待防抖
      await waitFor(
        () => {
          expect(onSearch).toHaveBeenCalledWith('test query');
        },
        { timeout: 400 }
      );
      
      // 只调用一次
      expect(onSearch).toHaveBeenCalledTimes(1);
    });
  });

  describe('Search Suggestions', () => {
    it('should display and highlight search suggestions', async () => {
      const suggestions = [
        'React Testing Library',
        'Jest Testing',
        'Testing Best Practices'
      ];

      render(
        <SearchBar
          suggestions={suggestions}
          enableSuggestions={true}
        />
      );

      const input = screen.getByPlaceholderText(/搜索/i);
      
      // 输入触发建议
      await userEvent.type(input, 'Test');
      
      // 等待建议出现
      await waitFor(() => {
        suggestions.forEach(suggestion => {
          if (suggestion.includes('Test')) {
            const element = screen.getByText(new RegExp(suggestion, 'i'));
            expect(element).toBeInTheDocument();
          }
        });
      });

      // 验证高亮
      const highlighted = screen.getAllByClassName('highlight');
      expect(highlighted.length).toBeGreaterThan(0);
    });
  });

  describe('Search History', () => {
    it('should save and display search history', async () => {
      render(<SearchBar enableHistory={true} maxHistory={5} />);
      
      const input = screen.getByPlaceholderText(/搜索/i);
      
      // 执行多次搜索
      const searches = ['query1', 'query2', 'query3'];
      
      for (const query of searches) {
        await userEvent.clear(input);
        await userEvent.type(input, query);
        await userEvent.keyboard('{Enter}');
      }

      // 点击输入框显示历史
      await userEvent.click(input);
      
      // 验证历史记录显示
      const historyDropdown = screen.getByTestId('search-history');
      
      searches.reverse().forEach(query => {
        expect(within(historyDropdown).getByText(query)).toBeInTheDocument();
      });
    });
  });
});

// ==================== 文档列表组件测试 ====================
describe('DocumentList Component Tests', () => {
  describe('Document Display and Pagination', () => {
    it('should display documents with pagination', async () => {
      const documents = Array.from({ length: 25 }, (_, i) =>
        TestDataFactory.createDocument({
          id: `doc_${i}`,
          name: `document_${i}.pdf`
        })
      );

      render(
        <DocumentList
          documents={documents}
          pageSize={10}
        />
      );

      // 验证第一页显示10个文档
      const firstPageDocs = screen.getAllByTestId(/document-item/i);
      expect(firstPageDocs).toHaveLength(10);
      
      // 验证分页器
      expect(screen.getByText('1')).toHaveClass('active');
      expect(screen.getByText('2')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
      
      // 切换到第二页
      await userEvent.click(screen.getByText('2'));
      
      // 验证第二页内容
      await waitFor(() => {
        expect(screen.getByText('document_10.pdf')).toBeInTheDocument();
      });
    });
  });

  describe('Document Actions', () => {
    it('should handle document actions correctly', async () => {
      const onView = jest.fn();
      const onEdit = jest.fn();
      const onDelete = jest.fn();
      
      const documents = [
        TestDataFactory.createDocument({ id: 'doc_1', name: 'test.pdf' })
      ];

      render(
        <DocumentList
          documents={documents}
          onView={onView}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      );

      const docRow = screen.getByTestId('document-item-doc_1');
      
      // 测试查看
      const viewBtn = within(docRow).getByLabelText(/查看/i);
      await userEvent.click(viewBtn);
      expect(onView).toHaveBeenCalledWith('doc_1');
      
      // 测试编辑
      const editBtn = within(docRow).getByLabelText(/编辑/i);
      await userEvent.click(editBtn);
      expect(onEdit).toHaveBeenCalledWith('doc_1');
      
      // 测试删除（需要确认）
      const deleteBtn = within(docRow).getByLabelText(/删除/i);
      await userEvent.click(deleteBtn);
      
      // 确认对话框
      const confirmBtn = screen.getByRole('button', { name: /确认删除/i });
      await userEvent.click(confirmBtn);
      
      expect(onDelete).toHaveBeenCalledWith('doc_1');
    });
  });

  describe('Batch Operations', () => {
    it('should support batch selection and operations', async () => {
      const onBatchDelete = jest.fn();
      const documents = Array.from({ length: 5 }, (_, i) =>
        TestDataFactory.createDocument({ id: `doc_${i}` })
      );

      render(
        <DocumentList
          documents={documents}
          enableBatchOps={true}
          onBatchDelete={onBatchDelete}
        />
      );

      // 全选
      const selectAllCheckbox = screen.getByLabelText(/全选/i);
      await userEvent.click(selectAllCheckbox);
      
      // 验证所有项被选中
      const checkboxes = screen.getAllByRole('checkbox', { name: /选择文档/i });
      checkboxes.forEach(checkbox => {
        expect(checkbox).toBeChecked();
      });
      
      // 批量删除
      const batchDeleteBtn = screen.getByRole('button', { name: /批量删除/i });
      await userEvent.click(batchDeleteBtn);
      
      // 确认
      const confirmBtn = await screen.findByRole('button', { name: /确认删除5个文档/i });
      await userEvent.click(confirmBtn);
      
      expect(onBatchDelete).toHaveBeenCalledWith([
        'doc_0', 'doc_1', 'doc_2', 'doc_3', 'doc_4'
      ]);
    });
  });
});

// ==================== 片段编辑器测试 ====================
describe('ChunkEditor Component Tests', () => {
  describe('TC_DOC_011: Chunk Display and Edit', () => {
    it('should display and allow editing of document chunks', async () => {
      const chunks = [
        TestDataFactory.createChunk({ id: 'chunk_1', position: 0 }),
        TestDataFactory.createChunk({ id: 'chunk_2', position: 1 }),
        TestDataFactory.createChunk({ id: 'chunk_3', position: 2 })
      ];

      const onSave = jest.fn();

      render(
        <ChunkEditor
          documentId="doc_123"
          chunks={chunks}
          onSave={onSave}
        />
      );

      // 验证所有片段显示
      expect(screen.getAllByTestId(/chunk-item/i)).toHaveLength(3);
      
      // 选择一个片段
      const firstChunk = screen.getByTestId('chunk-item-chunk_1');
      await userEvent.click(firstChunk);
      
      // 验证片段详情显示
      expect(screen.getByText(/片段 #1/i)).toBeInTheDocument();
      expect(screen.getByText(/position: 0/i)).toBeInTheDocument();
      
      // 编辑片段内容
      const editButton = screen.getByRole('button', { name: /编辑/i });
      await userEvent.click(editButton);
      
      const textarea = screen.getByRole('textbox', { name: /片段内容/i });
      await userEvent.clear(textarea);
      await userEvent.type(textarea, 'Updated chunk content');
      
      // 保存更改
      const saveButton = screen.getByRole('button', { name: /保存/i });
      await userEvent.click(saveButton);
      
      expect(onSave).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            id: 'chunk_1',
            content: 'Updated chunk content'
          })
        ])
      );
    });
  });

  describe('Chunk Merging', () => {
    it('should allow merging adjacent chunks', async () => {
      const chunks = [
        TestDataFactory.createChunk({ 
          id: 'chunk_1', 
          content: 'First chunk',
          position: 0 
        }),
        TestDataFactory.createChunk({ 
          id: 'chunk_2', 
          content: 'Second chunk',
          position: 1 
        })
      ];

      const onMerge = jest.fn();

      render(
        <ChunkEditor
          chunks={chunks}
          onMerge={onMerge}
        />
      );

      // 选择多个片段
      const chunk1 = screen.getByTestId('chunk-item-chunk_1');
      const chunk2 = screen.getByTestId('chunk-item-chunk_2');
      
      // Ctrl+Click多选
      await userEvent.click(chunk1);
      await userEvent.click(chunk2, { ctrlKey: true });
      
      // 点击合并按钮
      const mergeButton = screen.getByRole('button', { name: /合并/i });
      expect(mergeButton).not.toBeDisabled();
      await userEvent.click(mergeButton);
      
      // 确认合并
      const confirmButton = screen.getByRole('button', { name: /确认合并/i });
      await userEvent.click(confirmButton);
      
      expect(onMerge).toHaveBeenCalledWith(['chunk_1', 'chunk_2']);
    });
  });

  describe('Chunk Splitting', () => {
    it('should allow splitting a chunk at cursor position', async () => {
      const chunk = TestDataFactory.createChunk({
        id: 'chunk_1',
        content: 'This is a long chunk content that needs to be split into two parts.'
      });

      const onSplit = jest.fn();

      render(
        <ChunkEditor
          chunks={[chunk]}
          onSplit={onSplit}
        />
      );

      // 选择片段
      const chunkItem = screen.getByTestId('chunk-item-chunk_1');
      await userEvent.click(chunkItem);
      
      // 进入编辑模式
      const editButton = screen.getByRole('button', { name: /编辑/i });
      await userEvent.click(editButton);
      
      // 在文本中设置光标位置
      const textarea = screen.getByRole('textbox');
      textarea.setSelectionRange(30, 30); // 在"that"之前
      
      // 点击拆分按钮
      const splitButton = screen.getByRole('button', { name: /拆分/i });
      await userEvent.click(splitButton);
      
      expect(onSplit).toHaveBeenCalledWith('chunk_1', 30);
    });
  });
});

// ==================== 策略配置组件测试 ====================
describe('StrategyConfig Component Tests', () => {
  describe('TC_STRATEGY_001: Recall Threshold Configuration', () => {
    it('should configure and apply recall threshold', async () => {
      const onSave = jest.fn();
      
      render(
        <StrategyConfig
          initialConfig={{
            recallThreshold: 0.7,
            rerankThreshold: 0.8,
            maxChunks: 5
          }}
          onSave={onSave}
        />
      );

      // 调整召回阈值
      const thresholdSlider = screen.getByLabelText(/召回阈值/i);
      
      // 模拟滑动到0.85
      fireEvent.change(thresholdSlider, { target: { value: '0.85' } });
      
      // 验证显示更新
      expect(screen.getByText('0.85')).toBeInTheDocument();
      
      // 保存配置
      const saveButton = screen.getByRole('button', { name: /保存配置/i });
      await userEvent.click(saveButton);
      
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          recallThreshold: 0.85
        })
      );
    });
  });

  describe('TC_STRATEGY_002: Multi-path Recall Weights', () => {
    it('should configure weights for multiple recall paths', async () => {
      const onSave = jest.fn();
      
      render(
        <StrategyConfig
          enableMultiPath={true}
          onSave={onSave}
        />
      );

      // 设置向量召回权重
      const vectorWeight = screen.getByLabelText(/向量召回权重/i);
      await userEvent.clear(vectorWeight);
      await userEvent.type(vectorWeight, '60');
      
      // 设置BM25权重
      const bm25Weight = screen.getByLabelText(/BM25权重/i);
      await userEvent.clear(bm25Weight);
      await userEvent.type(bm25Weight, '30');
      
      // 设置知识图谱权重
      const kgWeight = screen.getByLabelText(/知识图谱权重/i);
      await userEvent.clear(kgWeight);
      await userEvent.type(kgWeight, '10');
      
      // 验证总和为100
      const totalDisplay = screen.getByTestId('total-weight');
      expect(totalDisplay).toHaveTextContent('100%');
      
      // 保存
      const saveButton = screen.getByRole('button', { name: /保存/i });
      await userEvent.click(saveButton);
      
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          vectorWeight: 0.6,
          bm25Weight: 0.3,
          kgWeight: 0.1
        })
      );
    });
  });

  describe('A/B Testing Configuration', () => {
    it('should create and configure A/B test', async () => {
      const onCreateABTest = jest.fn();
      
      render(
        <StrategyConfig
          enableABTesting={true}
          onCreateABTest={onCreateABTest}
        />
      );

      // 打开A/B测试配置
      const abTestButton = screen.getByRole('button', { name: /创建A\/B测试/i });
      await userEvent.click(abTestButton);
      
      // 填写测试名称
      const testNameInput = screen.getByLabelText(/测试名称/i);
      await userEvent.type(testNameInput, '召回策略优化测试');
      
      // 设置流量分配
      const trafficSlider = screen.getByLabelText(/B组流量/i);
      fireEvent.change(trafficSlider, { target: { value: '20' } });
      
      // 选择测试指标
      const metricsCheckboxes = screen.getAllByRole('checkbox', { name: /指标/i });
      await userEvent.click(metricsCheckboxes[0]); // 点击率
      await userEvent.click(metricsCheckboxes[1]); // 准确率
      
      // 创建测试
      const createButton = screen.getByRole('button', { name: /开始测试/i });
      await userEvent.click(createButton);
      
      expect(onCreateABTest).toHaveBeenCalledWith({
        name: '召回策略优化测试',
        trafficPercentage: 20,
        metrics: ['clickRate', 'accuracy'],
        status: 'running'
      });
    });
  });
});

// ==================== 性能测试 ====================
describe('Performance Tests', () => {
  describe('TC_PERF_001: Component Render Performance', () => {
    it('should render large document list efficiently', async () => {
      const documents = Array.from({ length: 1000 }, (_, i) =>
        TestDataFactory.createDocument({ id: `doc_${i}` })
      );

      const startTime = performance.now();
      
      render(
        <DocumentList
          documents={documents}
          virtualScroll={true}
        />
      );
      
      const renderTime = performance.now() - startTime;
      
      // 渲染时间应小于100ms
      expect(renderTime).toBeLessThan(100);
      
      // 验证虚拟滚动：只渲染可见项
      const visibleItems = screen.getAllByTestId(/document-item/i);
      expect(visibleItems.length).toBeLessThan(50); // 远少于1000
    });
  });

  describe('Memory Leak Detection', () => {
    it('should cleanup properly on unmount', async () => {
      const { unmount } = render(<DocumentUploader />);
      
      // 获取初始内存（如果可用）
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // 执行多次挂载/卸载
      for (let i = 0; i < 100; i++) {
        const { unmount: cleanup } = render(<DocumentUploader />);
        cleanup();
      }
      
      // 检查内存增长
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const memoryGrowth = finalMemory - initialMemory;
      
      // 内存增长应该很小
      if (initialMemory > 0) {
        expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024); // 10MB
      }
      
      unmount();
    });
  });
});

// ==================== 集成测试 ====================
describe('Integration Tests', () => {
  describe('TC_E2E_001: Complete Document Processing Flow', () => {
    it('should complete full document upload and search flow', async () => {
      // 模拟完整的用户流程
      const { container } = render(
        <div>
          <DocumentUploader />
          <DocumentList documents={[]} />
          <SearchBar />
        </div>
      );

      // 1. 上传文档
      const file = TestDataFactory.createFile('integration-test.pdf');
      const input = within(container).getByLabelText(/选择文件/i);
      await userEvent.upload(input, file);
      
      // 2. 确认上传
      const uploadBtn = within(container).getByRole('button', { name: /确认上传/i });
      await userEvent.click(uploadBtn);
      
      // 3. 等待处理完成
      await waitFor(() => {
        expect(within(container).getByText(/处理完成/i)).toBeInTheDocument();
      }, { timeout: 5000 });
      
      // 4. 执行搜索
      const searchInput = within(container).getByPlaceholderText(/搜索/i);
      await userEvent.type(searchInput, 'test query');
      
      // 5. 验证搜索结果
      await waitFor(() => {
        expect(within(container).getByText(/找到.*结果/i)).toBeInTheDocument();
      });
    });
  });
});

// ==================== 辅助测试工具 ====================
export const TestUtils = {
  // 等待异步操作完成
  waitForAsync: async (callback: () => boolean, timeout = 5000) => {
    const startTime = Date.now();
    while (!callback() && Date.now() - startTime < timeout) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (!callback()) {
      throw new Error('Async operation timeout');
    }
  },

  // 模拟网络延迟
  simulateNetworkDelay: (ms: number) => {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  // 验证无障碍性
  checkAccessibility: (container: HTMLElement) => {
    // 检查ARIA标签
    const buttons = container.querySelectorAll('button');
    buttons.forEach(button => {
      expect(button).toHaveAttribute('aria-label');
    });

    // 检查键盘导航
    const focusableElements = container.querySelectorAll(
      'button, input, select, textarea, a[href], [tabindex]'
    );
    focusableElements.forEach(element => {
      expect(Number(element.getAttribute('tabindex'))).toBeGreaterThanOrEqual(-1);
    });
  }
};