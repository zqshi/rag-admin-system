# RAG系统前端TDD设计文档

**版本**: 1.0  
**技术栈**: React 18 + TypeScript + Jest + React Testing Library  
**日期**: 2025年8月12日

---

## 一、前端TDD架构

### 1.1 测试架构分层

```
前端测试金字塔
├── 组件单元测试 (60%)
│   ├── 纯展示组件
│   ├── Hooks测试
│   └── 工具函数
├── 组件集成测试 (25%)
│   ├── 页面组件
│   ├── 状态管理
│   └── 路由测试
├── E2E测试 (10%)
│   ├── 用户流程
│   └── 关键路径
└── 视觉回归测试 (5%)
    └── UI快照
```

### 1.2 测试环境配置

```typescript
// jest.config.ts
export default {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/test/setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/src/test/mocks/fileMock.ts'
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.tsx',
    '!src/test/**'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

---

## 二、核心组件TDD实现

### 2.1 文档上传组件

#### 步骤1：编写失败测试 🔴

```typescript
// DocumentUploader.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentUploader } from './DocumentUploader';

describe('DocumentUploader', () => {
  const mockOnUpload = jest.fn();
  const mockOnProgress = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  describe('文件验证', () => {
    it('应该接受有效的文件格式', async () => {
      render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          accept={['.pdf', '.docx']}
        />
      );
      
      const file = new File(['content'], 'test.pdf', { 
        type: 'application/pdf' 
      });
      const input = screen.getByLabelText(/拖拽文件/i);
      
      await userEvent.upload(input, file);
      
      expect(mockOnUpload).toHaveBeenCalledWith([file]);
      expect(screen.queryByText(/文件格式不支持/i)).not.toBeInTheDocument();
    });
    
    it('应该拒绝无效的文件格式', async () => {
      render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          accept={['.pdf']}
        />
      );
      
      const file = new File(['content'], 'test.exe', { 
        type: 'application/x-msdownload' 
      });
      const input = screen.getByLabelText(/拖拽文件/i);
      
      await userEvent.upload(input, file);
      
      expect(mockOnUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/文件格式不支持/i)).toBeInTheDocument();
    });
    
    it('应该限制文件大小', async () => {
      render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          maxSize={1} // 1MB
        />
      );
      
      // 创建2MB文件
      const largeContent = new Array(2 * 1024 * 1024).fill('a').join('');
      const file = new File([largeContent], 'large.pdf', { 
        type: 'application/pdf' 
      });
      
      const input = screen.getByLabelText(/拖拽文件/i);
      await userEvent.upload(input, file);
      
      expect(mockOnUpload).not.toHaveBeenCalled();
      expect(screen.getByText(/文件大小不能超过1MB/i)).toBeInTheDocument();
    });
  });
  
  describe('上传进度', () => {
    it('应该显示上传进度条', async () => {
      const { rerender } = render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          onProgress={mockOnProgress}
        />
      );
      
      const file = new File(['content'], 'test.pdf');
      const input = screen.getByLabelText(/拖拽文件/i);
      
      await userEvent.upload(input, file);
      
      // 模拟进度更新
      mockOnProgress.mockImplementation((progress) => {
        rerender(
          <DocumentUploader 
            onUpload={mockOnUpload}
            onProgress={mockOnProgress}
            progress={progress}
          />
        );
      });
      
      // 触发50%进度
      mockOnProgress({ percent: 50, file: 'test.pdf' });
      
      await waitFor(() => {
        const progressBar = screen.getByRole('progressbar');
        expect(progressBar).toHaveAttribute('aria-valuenow', '50');
      });
    });
  });
  
  describe('拖拽上传', () => {
    it('应该支持拖拽上传文件', async () => {
      render(<DocumentUploader onUpload={mockOnUpload} />);
      
      const dropzone = screen.getByTestId('dropzone');
      const file = new File(['content'], 'test.pdf');
      
      // 模拟拖拽事件
      fireEvent.dragEnter(dropzone);
      expect(dropzone).toHaveClass('dropzone-active');
      
      fireEvent.drop(dropzone, {
        dataTransfer: {
          files: [file],
          items: [{ kind: 'file', getAsFile: () => file }]
        }
      });
      
      await waitFor(() => {
        expect(mockOnUpload).toHaveBeenCalledWith([file]);
      });
    });
  });
  
  describe('批量上传', () => {
    it('应该支持批量上传多个文件', async () => {
      render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          multiple={true}
        />
      );
      
      const files = [
        new File(['content1'], 'test1.pdf'),
        new File(['content2'], 'test2.docx'),
        new File(['content3'], 'test3.txt')
      ];
      
      const input = screen.getByLabelText(/拖拽文件/i);
      await userEvent.upload(input, files);
      
      expect(mockOnUpload).toHaveBeenCalledWith(files);
      expect(screen.getByText(/3个文件待上传/i)).toBeInTheDocument();
    });
    
    it('应该能取消单个文件上传', async () => {
      render(
        <DocumentUploader 
          onUpload={mockOnUpload}
          multiple={true}
        />
      );
      
      const files = [
        new File(['content1'], 'test1.pdf'),
        new File(['content2'], 'test2.pdf')
      ];
      
      const input = screen.getByLabelText(/拖拽文件/i);
      await userEvent.upload(input, files);
      
      // 取消第一个文件
      const cancelButtons = screen.getAllByLabelText(/取消上传/i);
      await userEvent.click(cancelButtons[0]);
      
      expect(screen.getByText(/1个文件待上传/i)).toBeInTheDocument();
      expect(screen.queryByText('test1.pdf')).not.toBeInTheDocument();
      expect(screen.getByText('test2.pdf')).toBeInTheDocument();
    });
  });
});
```

#### 步骤2：实现最小代码 🟢

```typescript
// DocumentUploader.tsx
import React, { useState, useCallback } from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadFile, UploadProps } from 'antd/es/upload';

interface DocumentUploaderProps {
  multiple?: boolean;
  maxSize?: number; // MB
  accept?: string[];
  onUpload: (files: File[]) => Promise<void>;
  onProgress?: (progress: { percent: number; file: string }) => void;
  progress?: { percent: number; file: string };
}

export const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  multiple = false,
  maxSize = 100,
  accept = ['.pdf', '.docx', '.xlsx', '.txt', '.md'],
  onUpload,
  onProgress,
  progress
}) => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  
  const validateFile = useCallback((file: File): boolean => {
    // 验证文件格式
    const isValidType = accept.some(ext => 
      file.name.toLowerCase().endsWith(ext)
    );
    
    if (!isValidType) {
      message.error('文件格式不支持');
      return false;
    }
    
    // 验证文件大小
    const isValidSize = file.size / 1024 / 1024 <= maxSize;
    
    if (!isValidSize) {
      message.error(`文件大小不能超过${maxSize}MB`);
      return false;
    }
    
    return true;
  }, [accept, maxSize]);
  
  const handleUpload = useCallback(async (files: File[]) => {
    const validFiles = files.filter(validateFile);
    
    if (validFiles.length === 0) return;
    
    setFileList(validFiles.map(file => ({
      uid: `-${Date.now()}-${file.name}`,
      name: file.name,
      status: 'uploading',
      originFileObj: file
    })));
    
    try {
      await onUpload(validFiles);
      
      setFileList(prev => prev.map(file => ({
        ...file,
        status: 'done'
      })));
      
      message.success('上传成功');
    } catch (error) {
      setFileList(prev => prev.map(file => ({
        ...file,
        status: 'error'
      })));
      
      message.error('上传失败');
    }
  }, [validateFile, onUpload]);
  
  const uploadProps: UploadProps = {
    name: 'file',
    multiple,
    fileList,
    accept: accept.join(','),
    beforeUpload: (file) => {
      const isValid = validateFile(file as File);
      if (isValid) {
        handleUpload([file as File]);
      }
      return false; // 阻止自动上传
    },
    onRemove: (file) => {
      setFileList(prev => prev.filter(f => f.uid !== file.uid));
    },
    showUploadList: {
      showDownloadIcon: false,
      showPreviewIcon: false
    }
  };
  
  return (
    <div className="document-uploader">
      <Upload.Dragger
        {...uploadProps}
        data-testid="dropzone"
        className={isDragging ? 'dropzone-active' : ''}
        onDragEnter={() => setIsDragging(true)}
        onDragLeave={() => setIsDragging(false)}
        onDrop={() => setIsDragging(false)}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          <label htmlFor="file-upload">拖拽文件到此处或点击选择</label>
        </p>
        <p className="ant-upload-hint">
          支持格式: {accept.join(', ')} | 最大: {maxSize}MB
        </p>
      </Upload.Dragger>
      
      {fileList.length > 0 && (
        <div className="upload-status">
          {multiple ? `${fileList.length}个文件待上传` : fileList[0]?.name}
        </div>
      )}
      
      {progress && (
        <div 
          role="progressbar"
          aria-valuenow={progress.percent}
          aria-valuemin={0}
          aria-valuemax={100}
          style={{ width: `${progress.percent}%` }}
        />
      )}
    </div>
  );
};
```

#### 步骤3：重构优化 🔄

```typescript
// hooks/useFileUpload.ts
export const useFileUpload = (config: UploadConfig) => {
  const [uploadQueue, setUploadQueue] = useState<UploadTask[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  
  const addToQueue = useCallback((files: File[]) => {
    const tasks = files.map(file => ({
      id: uuidv4(),
      file,
      progress: 0,
      status: 'pending' as const,
      startTime: null,
      endTime: null,
      error: null
    }));
    
    setUploadQueue(prev => [...prev, ...tasks]);
    return tasks.map(t => t.id);
  }, []);
  
  const processQueue = useCallback(async () => {
    if (isUploading) return;
    
    setIsUploading(true);
    
    const pendingTasks = uploadQueue.filter(t => t.status === 'pending');
    
    for (const task of pendingTasks) {
      try {
        await processTask(task);
      } catch (error) {
        handleTaskError(task, error);
      }
    }
    
    setIsUploading(false);
  }, [uploadQueue, isUploading]);
  
  return {
    uploadQueue,
    addToQueue,
    processQueue,
    isUploading,
    cancelUpload,
    retryUpload,
    clearQueue
  };
};
```

### 2.2 搜索组件TDD

#### 测试用例

```typescript
// SearchBar.test.tsx
describe('SearchBar Component', () => {
  describe('输入验证', () => {
    it('应该防抖处理搜索输入', async () => {
      const mockOnSearch = jest.fn();
      render(<SearchBar onSearch={mockOnSearch} debounceMs={300} />);
      
      const input = screen.getByPlaceholderText(/搜索文档/i);
      
      // 快速输入多个字符
      await userEvent.type(input, 'test');
      
      // 立即验证：不应该触发搜索
      expect(mockOnSearch).not.toHaveBeenCalled();
      
      // 等待防抖时间
      await waitFor(() => {
        expect(mockOnSearch).toHaveBeenCalledWith('test');
      }, { timeout: 400 });
      
      // 应该只调用一次
      expect(mockOnSearch).toHaveBeenCalledTimes(1);
    });
    
    it('应该高亮搜索建议中的匹配文本', async () => {
      const suggestions = ['React Testing', 'Vue Testing', 'Angular Testing'];
      
      render(
        <SearchBar 
          suggestions={suggestions}
          enableSuggestions={true}
        />
      );
      
      const input = screen.getByPlaceholderText(/搜索/i);
      await userEvent.type(input, 'React');
      
      const suggestion = await screen.findByText(/React/);
      const highlighted = suggestion.querySelector('.highlight');
      
      expect(highlighted).toHaveTextContent('React');
      expect(highlighted).toHaveStyle({ fontWeight: 'bold' });
    });
    
    it('应该支持快捷键触发搜索', async () => {
      const mockOnSearch = jest.fn();
      render(<SearchBar onSearch={mockOnSearch} />);
      
      const input = screen.getByPlaceholderText(/搜索/i);
      await userEvent.type(input, 'test query');
      
      // 按下 Cmd/Ctrl + Enter
      await userEvent.keyboard('{Control>}{Enter}{/Control}');
      
      expect(mockOnSearch).toHaveBeenCalledWith('test query');
    });
  });
  
  describe('搜索历史', () => {
    it('应该保存最近10条搜索历史', async () => {
      const { rerender } = render(<SearchBar enableHistory={true} />);
      
      const input = screen.getByPlaceholderText(/搜索/i);
      
      // 执行11次搜索
      for (let i = 1; i <= 11; i++) {
        await userEvent.clear(input);
        await userEvent.type(input, `query ${i}`);
        await userEvent.keyboard('{Enter}');
      }
      
      // 点击输入框显示历史
      await userEvent.click(input);
      
      const historyItems = screen.getAllByTestId('history-item');
      expect(historyItems).toHaveLength(10);
      
      // 验证最新的在最前
      expect(historyItems[0]).toHaveTextContent('query 11');
      
      // 验证最旧的被移除
      expect(screen.queryByText('query 1')).not.toBeInTheDocument();
    });
    
    it('应该能清除搜索历史', async () => {
      render(<SearchBar enableHistory={true} />);
      
      // 添加搜索历史
      const input = screen.getByPlaceholderText(/搜索/i);
      await userEvent.type(input, 'test');
      await userEvent.keyboard('{Enter}');
      
      // 显示历史
      await userEvent.click(input);
      expect(screen.getByText('test')).toBeInTheDocument();
      
      // 清除历史
      const clearButton = screen.getByLabelText(/清除历史/i);
      await userEvent.click(clearButton);
      
      expect(screen.queryByText('test')).not.toBeInTheDocument();
      expect(screen.getByText(/无搜索历史/i)).toBeInTheDocument();
    });
  });
});
```

### 2.3 状态管理TDD

```typescript
// store/documentStore.test.ts
import { renderHook, act } from '@testing-library/react';
import { useDocumentStore } from './documentStore';
import { DocumentFactory } from '@/test/factories';

describe('Document Store', () => {
  describe('文档CRUD操作', () => {
    it('应该添加新文档到store', () => {
      const { result } = renderHook(() => useDocumentStore());
      
      const newDoc = DocumentFactory.create();
      
      act(() => {
        result.current.addDocument(newDoc);
      });
      
      expect(result.current.documents).toHaveLength(1);
      expect(result.current.documents[0]).toEqual(newDoc);
      expect(result.current.getDocumentById(newDoc.id)).toEqual(newDoc);
    });
    
    it('应该批量更新文档状态', () => {
      const { result } = renderHook(() => useDocumentStore());
      
      const docs = DocumentFactory.createMany(3);
      
      act(() => {
        docs.forEach(doc => result.current.addDocument(doc));
      });
      
      act(() => {
        result.current.batchUpdateStatus(
          docs.map(d => d.id),
          'processing'
        );
      });
      
      result.current.documents.forEach(doc => {
        expect(doc.status).toBe('processing');
      });
    });
    
    it('应该支持乐观更新和回滚', async () => {
      const { result } = renderHook(() => useDocumentStore());
      
      const doc = DocumentFactory.create({ name: 'original.pdf' });
      
      act(() => {
        result.current.addDocument(doc);
      });
      
      // 乐观更新
      act(() => {
        result.current.optimisticUpdate(doc.id, { name: 'updated.pdf' });
      });
      
      expect(result.current.getDocumentById(doc.id)?.name).toBe('updated.pdf');
      
      // 模拟API失败，回滚
      act(() => {
        result.current.rollbackUpdate(doc.id);
      });
      
      expect(result.current.getDocumentById(doc.id)?.name).toBe('original.pdf');
    });
  });
  
  describe('分页和过滤', () => {
    it('应该正确分页文档列表', () => {
      const { result } = renderHook(() => useDocumentStore());
      
      const docs = DocumentFactory.createMany(25);
      
      act(() => {
        docs.forEach(doc => result.current.addDocument(doc));
      });
      
      act(() => {
        result.current.setPage(2);
        result.current.setPageSize(10);
      });
      
      const paginatedDocs = result.current.getPaginatedDocuments();
      
      expect(paginatedDocs.items).toHaveLength(10);
      expect(paginatedDocs.total).toBe(25);
      expect(paginatedDocs.page).toBe(2);
      expect(paginatedDocs.items[0]).toEqual(docs[10]);
    });
    
    it('应该按类型过滤文档', () => {
      const { result } = renderHook(() => useDocumentStore());
      
      act(() => {
        result.current.addDocument(DocumentFactory.create({ type: 'pdf' }));
        result.current.addDocument(DocumentFactory.create({ type: 'docx' }));
        result.current.addDocument(DocumentFactory.create({ type: 'pdf' }));
      });
      
      act(() => {
        result.current.setFilter({ type: 'pdf' });
      });
      
      const filtered = result.current.getFilteredDocuments();
      
      expect(filtered).toHaveLength(2);
      filtered.forEach(doc => {
        expect(doc.type).toBe('pdf');
      });
    });
  });
});
```

---

## 三、集成测试

### 3.1 页面级测试

```typescript
// pages/KnowledgeManagement.test.tsx
import { render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { KnowledgeManagement } from './KnowledgeManagement';
import { server } from '@/test/mocks/server';
import { rest } from 'msw';

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });
  
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {component}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Knowledge Management Page', () => {
  it('应该完成完整的文档上传流程', async () => {
    renderWithProviders(<KnowledgeManagement />);
    
    // 1. 初始状态：显示空状态
    expect(screen.getByText(/暂无文档/i)).toBeInTheDocument();
    
    // 2. 点击上传按钮
    const uploadButton = screen.getByRole('button', { name: /上传文档/i });
    await userEvent.click(uploadButton);
    
    // 3. 选择文件
    const file = new File(['test content'], 'test.pdf', {
      type: 'application/pdf'
    });
    
    const input = screen.getByLabelText(/选择文件/i);
    await userEvent.upload(input, file);
    
    // 4. 确认上传
    const confirmButton = screen.getByRole('button', { name: /确认上传/i });
    await userEvent.click(confirmButton);
    
    // 5. 等待上传完成
    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
      expect(screen.getByText(/已索引/i)).toBeInTheDocument();
    });
    
    // 6. 验证文档出现在列表中
    const documentRow = screen.getByRole('row', { name: /test.pdf/i });
    expect(documentRow).toBeInTheDocument();
    
    // 7. 验证操作按钮
    within(documentRow).getByRole('button', { name: /查看/i });
    within(documentRow).getByRole('button', { name: /编辑/i });
    within(documentRow).getByRole('button', { name: /删除/i });
  });
  
  it('应该处理并发上传和错误恢复', async () => {
    // 模拟第二个文件上传失败
    server.use(
      rest.post('/api/documents', (req, res, ctx) => {
        const fileName = req.headers.get('x-file-name');
        if (fileName === 'error.pdf') {
          return res(ctx.status(500), ctx.json({ error: '解析失败' }));
        }
        return res(ctx.json({ id: '123', status: 'success' }));
      })
    );
    
    renderWithProviders(<KnowledgeManagement />);
    
    const files = [
      new File(['content1'], 'success1.pdf'),
      new File(['content2'], 'error.pdf'),
      new File(['content3'], 'success2.pdf')
    ];
    
    // 批量上传
    const input = screen.getByLabelText(/选择文件/i);
    await userEvent.upload(input, files);
    
    // 等待处理完成
    await waitFor(() => {
      // 成功的文件应该显示
      expect(screen.getByText('success1.pdf')).toBeInTheDocument();
      expect(screen.getByText('success2.pdf')).toBeInTheDocument();
      
      // 失败的文件应该显示错误
      const errorRow = screen.getByRole('row', { name: /error.pdf/i });
      expect(within(errorRow).getByText(/解析失败/i)).toBeInTheDocument();
      
      // 应该有重试按钮
      expect(within(errorRow).getByRole('button', { name: /重试/i }))
        .toBeInTheDocument();
    });
  });
});
```

### 3.2 API集成测试

```typescript
// api/documentApi.test.ts
import { documentApi } from './documentApi';
import { server } from '@/test/mocks/server';
import { rest } from 'msw';

describe('Document API Integration', () => {
  describe('uploadDocument', () => {
    it('应该正确处理分片上传', async () => {
      const largeFile = new File(
        [new Array(10 * 1024 * 1024).fill('a').join('')],
        'large.pdf'
      );
      
      const onProgress = jest.fn();
      
      const result = await documentApi.uploadDocument(largeFile, {
        onProgress,
        chunkSize: 2 * 1024 * 1024 // 2MB chunks
      });
      
      // 验证分片数量
      expect(onProgress).toHaveBeenCalledTimes(5);
      
      // 验证进度回调
      expect(onProgress).toHaveBeenNthCalledWith(1, { percent: 20 });
      expect(onProgress).toHaveBeenNthCalledWith(5, { percent: 100 });
      
      // 验证结果
      expect(result).toMatchObject({
        id: expect.any(String),
        status: 'completed',
        chunks: 5
      });
    });
    
    it('应该自动重试失败的分片', async () => {
      let attemptCount = 0;
      
      server.use(
        rest.post('/api/upload/chunk', (req, res, ctx) => {
          attemptCount++;
          if (attemptCount === 2) {
            return res.once(ctx.status(500));
          }
          return res(ctx.json({ success: true }));
        })
      );
      
      const file = new File(['content'], 'test.pdf');
      
      const result = await documentApi.uploadDocument(file, {
        maxRetries: 3,
        retryDelay: 100
      });
      
      expect(result.status).toBe('completed');
      expect(attemptCount).toBeGreaterThan(2); // 包含重试
    });
  });
});
```

---

## 四、E2E测试

### 4.1 Playwright配置

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] }
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] }
    }
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI
  }
});
```

### 4.2 E2E测试用例

```typescript
// e2e/documentFlow.spec.ts
import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/LoginPage';
import { DocumentPage } from './pages/DocumentPage';

test.describe('文档管理完整流程', () => {
  test.beforeEach(async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('admin@test.com', 'password');
  });
  
  test('应该完成文档上传到问答的完整流程', async ({ page }) => {
    const docPage = new DocumentPage(page);
    
    // 1. 上传文档
    await docPage.goto();
    await docPage.uploadFile('fixtures/sample.pdf');
    
    // 2. 等待处理完成
    await expect(docPage.getDocumentStatus('sample.pdf'))
      .toHaveText('已索引', { timeout: 30000 });
    
    // 3. 进入问答测试
    await page.goto('/qa-test');
    
    // 4. 输入相关问题
    await page.fill('[data-testid="question-input"]', 
      '文档中提到的主要功能是什么？');
    
    await page.click('[data-testid="submit-question"]');
    
    // 5. 验证答案
    const answer = page.locator('[data-testid="answer-content"]');
    await expect(answer).toBeVisible();
    await expect(answer).toContainText('sample.pdf');
    
    // 6. 验证引用
    const references = page.locator('[data-testid="references"]');
    await expect(references).toContainText('sample.pdf - 第1页');
  });
  
  test('应该支持批量操作', async ({ page }) => {
    const docPage = new DocumentPage(page);
    await docPage.goto();
    
    // 上传多个文档
    const files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf'];
    for (const file of files) {
      await docPage.uploadFile(`fixtures/${file}`);
    }
    
    // 选择所有文档
    await page.click('[data-testid="select-all"]');
    
    // 批量删除
    await page.click('[data-testid="bulk-delete"]');
    
    // 确认对话框
    await page.click('button:has-text("确认删除")');
    
    // 验证删除成功
    await expect(page.locator('[data-testid="document-list"]'))
      .toHaveText(/暂无文档/);
  });
});
```

---

## 五、性能测试

### 5.1 组件性能测试

```typescript
// performance/DocumentList.perf.test.tsx
import { measureRender } from '@/test/utils/performance';
import { DocumentList } from '@/components/DocumentList';
import { DocumentFactory } from '@/test/factories';

describe('DocumentList Performance', () => {
  it('应该在100ms内渲染1000个文档', async () => {
    const documents = DocumentFactory.createMany(1000);
    
    const renderTime = await measureRender(
      <DocumentList documents={documents} />
    );
    
    expect(renderTime).toBeLessThan(100);
  });
  
  it('应该高效处理虚拟滚动', async () => {
    const documents = DocumentFactory.createMany(10000);
    
    const { container } = render(
      <DocumentList 
        documents={documents}
        virtualScroll={true}
      />
    );
    
    // 验证DOM节点数量（虚拟滚动应该只渲染可见项）
    const renderedItems = container.querySelectorAll('[data-testid="document-item"]');
    expect(renderedItems.length).toBeLessThan(50); // 只渲染可见的
  });
  
  it('应该避免不必要的重渲染', async () => {
    const documents = DocumentFactory.createMany(100);
    const onSelect = jest.fn();
    
    const { rerender } = render(
      <DocumentList 
        documents={documents}
        onSelect={onSelect}
      />
    );
    
    // 使用React DevTools Profiler API
    const renderCount = await measureRenderCount(() => {
      // 更新不相关的prop
      rerender(
        <DocumentList 
          documents={documents}
          onSelect={onSelect}
          className="new-class"
        />
      );
    });
    
    expect(renderCount).toBe(0); // 不应该触发子组件重渲染
  });
});
```

### 5.2 内存泄漏测试

```typescript
// memory/memoryLeak.test.tsx
describe('Memory Leak Detection', () => {
  it('组件卸载后应该清理所有订阅', async () => {
    const { unmount } = render(<KnowledgeManagement />);
    
    // 获取初始内存
    const initialMemory = (performance as any).memory.usedJSHeapSize;
    
    // 执行大量操作
    for (let i = 0; i < 100; i++) {
      const { unmount: cleanup } = render(<KnowledgeManagement />);
      cleanup();
    }
    
    // 强制垃圾回收（需要启动Chrome with --expose-gc）
    if (global.gc) global.gc();
    
    // 检查内存增长
    const finalMemory = (performance as any).memory.usedJSHeapSize;
    const memoryGrowth = finalMemory - initialMemory;
    
    // 内存增长不应超过10MB
    expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024);
  });
});
```

---

## 六、测试工具和辅助函数

### 6.1 测试工具库

```typescript
// test/utils/testUtils.tsx
import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// 自定义render函数，包含所有Provider
export const renderWithProviders = (
  ui: ReactElement,
  options?: RenderOptions
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
      mutations: { retry: false }
    }
  });
  
  const AllProviders = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
  
  return render(ui, { wrapper: AllProviders, ...options });
};

// 等待加载完成
export const waitForLoadingToFinish = () =>
  waitFor(() => {
    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
  });

// 模拟文件
export const createMockFile = (
  name: string,
  size: number,
  type: string
): File => {
  const content = new Array(size).fill('a').join('');
  return new File([content], name, { type });
};
```

### 6.2 自定义匹配器

```typescript
// test/matchers/customMatchers.ts
expect.extend({
  toBeValidDocument(received) {
    const pass = 
      received.id &&
      received.name &&
      received.content &&
      received.createdAt;
    
    return {
      pass,
      message: () =>
        pass
          ? `Expected document to be invalid`
          : `Expected valid document with id, name, content, and createdAt`
    };
  },
  
  toHaveProgress(received, expected) {
    const progressBar = received.querySelector('[role="progressbar"]');
    const currentValue = progressBar?.getAttribute('aria-valuenow');
    const pass = currentValue === String(expected);
    
    return {
      pass,
      message: () =>
        pass
          ? `Expected progress not to be ${expected}%`
          : `Expected progress to be ${expected}%, but was ${currentValue}%`
    };
  }
});
```

---

## 七、CI集成

```yaml
# .github/workflows/frontend-test.yml
name: Frontend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [16, 18, 20]
        
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Lint
        run: npm run lint
        
      - name: Type check
        run: npm run type-check
        
      - name: Unit tests
        run: npm run test:unit -- --coverage
        
      - name: Integration tests
        run: npm run test:integration
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          flags: frontend
          
      - name: E2E tests
        run: |
          npm run build
          npm run test:e2e
          
      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            coverage/
            playwright-report/
            test-results/
```

---

## 八、测试报告和监控

### 8.1 测试报告生成

```json
// package.json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:coverage:html": "jest --coverage --coverageReporters=html",
    "test:unit": "jest --testPathPattern=\\.test\\.",
    "test:integration": "jest --testPathPattern=\\.integration\\.test\\.",
    "test:e2e": "playwright test",
    "test:performance": "jest --testPathPattern=\\.perf\\.test\\.",
    "test:report": "jest --coverage --json --outputFile=test-report.json"
  }
}
```

### 8.2 测试质量指标

```typescript
// test/metrics/testQuality.ts
export const analyzeTestQuality = () => {
  const metrics = {
    coverage: {
      line: 87.3,
      branch: 82.1,
      function: 89.7
    },
    testTypes: {
      unit: 234,
      integration: 45,
      e2e: 12
    },
    performance: {
      avgDuration: 125, // ms
      slowestTest: 'DocumentUpload.e2e.test',
      p95Duration: 450
    },
    flakiness: {
      flakyTests: ['SearchBar.integration.test'],
      failureRate: 0.02
    }
  };
  
  return metrics;
};
```

---

这份前端TDD设计文档提供了完整的测试驱动开发流程和最佳实践。需要我继续创建后端TDD设计文档吗？