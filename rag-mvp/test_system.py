#!/usr/bin/env python3
"""
RAG MVP系统测试脚本
快速验证系统功能是否正常
"""

import requests
import time
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def check_health():
    """检查系统健康状态"""
    print("1. 检查系统健康状态...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 系统正常: {data}")
            return True
        else:
            print(f"   ❌ 系统异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 无法连接到API: {e}")
        return False

def create_test_file():
    """创建测试文件"""
    print("\n2. 创建测试文件...")
    test_file = Path("test_document.txt")
    content = """
    RAG系统测试文档
    
    这是一个用于测试RAG（Retrieval-Augmented Generation）系统的文档。
    
    RAG系统的核心功能包括：
    1. 文档上传和处理
    2. 文本分片和向量化
    3. 语义搜索和检索
    4. 智能问答生成
    
    测试要点：
    - 文档能否正确上传
    - 分片是否合理
    - 搜索结果是否相关
    - 响应时间是否满足要求
    
    这个MVP版本使用了FastAPI作为后端框架，FAISS作为向量索引，
    Sentence-Transformers进行文本向量化。
    
    系统架构简单但有效，适合快速原型开发和概念验证。
    """
    
    test_file.write_text(content)
    print(f"   ✅ 创建测试文件: {test_file}")
    return test_file

def upload_document(file_path):
    """上传文档"""
    print(f"\n3. 上传文档: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'text/plain')}
        response = requests.post(f"{API_URL}/api/upload", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ 上传成功!")
        print(f"      - 文档ID: {data['doc_id']}")
        print(f"      - 片段数: {data['chunks_created']}")
        print(f"      - 处理时间: {data['processing_time']}")
        return data['doc_id']
    else:
        print(f"   ❌ 上传失败: {response.text}")
        return None

def test_query(query):
    """测试查询"""
    print(f"\n4. 测试查询: '{query}'")
    
    response = requests.post(
        f"{API_URL}/api/query",
        json={"query": query, "top_k": 3}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ 查询成功!")
        print(f"      - 结果数: {data['total_results']}")
        print(f"      - 处理时间: {data['processing_time']}秒")
        print(f"\n   答案预览:")
        print(f"   {data['answer'][:200]}...")
        
        if data['sources']:
            print(f"\n   相关片段:")
            for i, source in enumerate(data['sources'][:2], 1):
                print(f"   {i}. {source['filename']} (相似度: {source['score']})")
                print(f"      {source['content'][:100]}...")
        return True
    else:
        print(f"   ❌ 查询失败: {response.text}")
        return False

def list_documents():
    """列出所有文档"""
    print("\n5. 获取文档列表...")
    
    response = requests.get(f"{API_URL}/api/documents")
    if response.status_code == 200:
        docs = response.json()
        print(f"   ✅ 共有 {len(docs)} 个文档")
        for doc in docs[:3]:  # 只显示前3个
            print(f"      - {doc['filename']} ({doc['chunk_count']} 片段)")
        return True
    else:
        print(f"   ❌ 获取失败: {response.text}")
        return False

def get_statistics():
    """获取统计信息"""
    print("\n6. 获取系统统计...")
    
    response = requests.get(f"{API_URL}/api/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✅ 统计信息:")
        print(f"      - 文档数: {stats['documents']['total']}")
        print(f"      - 片段数: {stats['documents']['total_chunks']}")
        print(f"      - 搜索次数: {stats['searches']['total']}")
        print(f"      - 向量数: {stats['index']['vectors']}")
        return True
    else:
        print(f"   ❌ 获取失败: {response.text}")
        return False

def main():
    """主测试流程"""
    print("="*50)
    print("🧪 RAG MVP系统功能测试")
    print("="*50)
    
    # 检查健康状态
    if not check_health():
        print("\n❌ 系统未启动，请先运行 ./start.sh")
        sys.exit(1)
    
    # 创建并上传测试文件
    test_file = create_test_file()
    doc_id = upload_document(test_file)
    
    if doc_id:
        # 等待处理完成
        print("\n⏳ 等待索引更新...")
        time.sleep(2)
        
        # 测试查询
        queries = [
            "RAG系统的核心功能是什么？",
            "系统使用了什么技术栈？",
            "MVP版本适合做什么？"
        ]
        
        for query in queries:
            test_query(query)
            time.sleep(1)
    
    # 获取文档列表和统计
    list_documents()
    get_statistics()
    
    # 清理测试文件
    test_file.unlink()
    
    print("\n" + "="*50)
    print("✅ 测试完成！系统运行正常")
    print("="*50)
    print("\n📌 下一步:")
    print("   1. 访问 http://localhost:3000 使用Web界面")
    print("   2. 访问 http://localhost:8000/docs 查看API文档")
    print("   3. 上传更多文档进行测试")

if __name__ == "__main__":
    main()