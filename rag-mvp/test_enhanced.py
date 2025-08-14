#!/usr/bin/env python3
"""
测试增强版RAG系统
"""

import requests
import json
import time
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:8001"

def test_health():
    """测试健康检查"""
    print("\n1. 测试健康检查...")
    response = requests.get(f"{BASE_URL}/api/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ 系统状态: {data['status']}")
        print(f"   📊 统计信息: {json.dumps(data['statistics'], indent=2)}")
        return True
    else:
        print(f"   ❌ 健康检查失败: {response.status_code}")
        return False

def test_upload():
    """测试文档上传"""
    print("\n2. 测试文档上传...")
    
    # 创建测试文件
    test_file = Path("test_enhanced.txt")
    test_content = """
    这是增强版RAG系统的测试文档。
    
    LangChain提供了强大的文档处理能力：
    1. 智能文本分割 - 支持多种分割策略
    2. 语义保持 - 保持上下文的完整性
    3. 递归分割 - 适应不同的文档结构
    
    ChromaDB作为向量数据库的优势：
    - 持久化存储：数据不会丢失
    - 高效检索：毫秒级响应
    - 元数据过滤：支持复杂查询
    - 易于集成：简单的API接口
    
    这个系统结合了两者的优势，提供了更好的文档处理和检索能力。
    """
    
    test_file.write_text(test_content)
    
    # 测试不同的分割策略
    splitter_types = ["recursive", "char", "token"]
    
    for splitter in splitter_types:
        print(f"\n   测试 {splitter} 分割器...")
        
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/plain')}
            params = {'splitter_type': splitter}
            response = requests.post(f"{BASE_URL}/api/upload", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 上传成功 ({splitter}):")
            print(f"      - 文档ID: {data['doc_id']}")
            print(f"      - 片段数: {data['chunks_count']}")
            print(f"      - 总字符: {data['total_chars']}")
            print(f"      - 处理时间: {data['processing_time']:.2f}秒")
        else:
            print(f"   ❌ 上传失败 ({splitter}): {response.status_code}")
    
    # 清理测试文件
    test_file.unlink()
    return True

def test_query():
    """测试查询功能"""
    print("\n3. 测试查询功能...")
    
    queries = [
        "LangChain的优势是什么？",
        "ChromaDB有哪些特点？",
        "文档处理能力",
        "向量数据库"
    ]
    
    for query in queries:
        print(f"\n   查询: '{query}'")
        
        payload = {
            "query": query,
            "top_k": 3
        }
        
        response = requests.post(f"{BASE_URL}/api/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 查询成功:")
            print(f"      - 结果数: {len(data['results'])}")
            print(f"      - 处理时间: {data['processing_time']:.3f}秒")
            
            if data['results']:
                print(f"      - 最相关片段:")
                for i, result in enumerate(data['results'][:2], 1):
                    content_preview = result['content'][:100] + "..."
                    print(f"        {i}. {content_preview}")
        else:
            print(f"   ❌ 查询失败: {response.status_code}")
    
    return True

def test_statistics():
    """测试统计功能"""
    print("\n4. 测试统计功能...")
    
    response = requests.get(f"{BASE_URL}/api/statistics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ 统计信息:")
        print(f"      - 文档数: {data['documents_count']}")
        print(f"      - 总片段数: {data['total_chunks']}")
        print(f"      - 搜索次数: {data['search_count']}")
        print(f"      - 平均处理时间: {data['avg_processing_time']:.2f}秒")
        print(f"      - ChromaDB信息: {json.dumps(data['chroma_db'], indent=8)}")
        return True
    else:
        print(f"   ❌ 获取统计失败: {response.status_code}")
        return False

def test_document_list():
    """测试文档列表"""
    print("\n5. 测试文档列表...")
    
    response = requests.get(f"{BASE_URL}/api/documents")
    
    if response.status_code == 200:
        documents = response.json()
        print(f"   ✅ 文档列表 (共 {len(documents)} 个):")
        
        for doc in documents[:5]:  # 只显示前5个
            print(f"      - {doc['filename']}")
            print(f"        ID: {doc['id']}")
            print(f"        片段数: {doc['chunk_count']}")
            print(f"        分割器: {doc['splitter_type']}")
            print(f"        处理时间: {doc.get('processing_time', 0):.2f}秒")
        
        return True
    else:
        print(f"   ❌ 获取文档列表失败: {response.status_code}")
        return False

def test_search_logs():
    """测试搜索日志"""
    print("\n6. 测试搜索日志...")
    
    response = requests.get(f"{BASE_URL}/api/search-logs?limit=5")
    
    if response.status_code == 200:
        logs = response.json()
        print(f"   ✅ 搜索日志 (最近 {len(logs)} 条):")
        
        for log in logs:
            print(f"      - 查询: '{log['query']}'")
            print(f"        结果数: {log['results_count']}")
            print(f"        耗时: {log['processing_time']:.3f}秒")
            print(f"        时间: {log['timestamp']}")
        
        return True
    else:
        print(f"   ❌ 获取搜索日志失败: {response.status_code}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 增强版RAG系统测试")
    print("=" * 60)
    
    # 等待服务启动
    print("\n⏳ 等待服务启动...")
    time.sleep(2)
    
    # 运行测试
    tests = [
        test_health,
        test_upload,
        test_query,
        test_statistics,
        test_document_list,
        test_search_logs
    ]
    
    success_count = 0
    for test in tests:
        try:
            if test():
                success_count += 1
        except Exception as e:
            print(f"   ❌ 测试异常: {str(e)}")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"✅ 测试完成: {success_count}/{len(tests)} 通过")
    print("=" * 60)
    
    if success_count == len(tests):
        print("\n🎉 所有测试通过！增强版系统运行正常。")
        print("\n📌 下一步:")
        print("   1. 访问 http://localhost:8001/docs 查看API文档")
        print("   2. 使用更多文档测试系统")
        print("   3. 调整切片策略优化效果")
    else:
        print("\n⚠️ 部分测试失败，请检查系统配置。")

if __name__ == "__main__":
    main()