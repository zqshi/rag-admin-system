#!/usr/bin/env python3
"""
诊断RAG系统中的重复问题
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path
import chromadb
from chromadb.config import Settings

def diagnose_chroma_duplicates():
    """检查ChromaDB中的重复片段"""
    print("🔍 检查ChromaDB中的重复问题...")
    
    # 连接ChromaDB
    chroma_path = Path("chroma_db")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection("rag_documents")
    
    # 获取所有文档
    result = collection.get(include=['documents', 'metadatas'])
    
    print(f"📊 总片段数: {len(result['ids'])}")
    
    # 检查重复内容
    content_counter = Counter(result['documents'])
    duplicates = {content: count for content, count in content_counter.items() if count > 1}
    
    if duplicates:
        print(f"❌ 发现 {len(duplicates)} 个重复内容:")
        for i, (content, count) in enumerate(list(duplicates.items())[:3]):
            print(f"  {i+1}. 重复{count}次: {content[:100]}...")
    else:
        print("✅ 未发现重复内容")
    
    # 检查重复ID
    id_counter = Counter(result['ids'])
    duplicate_ids = {id_: count for id_, count in id_counter.items() if count > 1}
    
    if duplicate_ids:
        print(f"❌ 发现 {len(duplicate_ids)} 个重复ID:")
        for id_, count in list(duplicate_ids.items())[:5]:
            print(f"  - {id_}: {count}次")
    else:
        print("✅ 未发现重复ID")
    
    # 检查文档来源分布
    sources = [meta.get('source', 'Unknown') for meta in result['metadatas']]
    source_counter = Counter(sources)
    
    print(f"\n📁 文档来源分布:")
    for source, count in source_counter.most_common(10):
        source_name = Path(source).name if source != 'Unknown' else source
        print(f"  - {source_name}: {count} 片段")
    
    return {
        'total_chunks': len(result['ids']),
        'duplicate_contents': len(duplicates),
        'duplicate_ids': len(duplicate_ids),
        'sources': dict(source_counter)
    }

def diagnose_sqlite_data():
    """检查SQLite数据库中的数据"""
    print("\n🗄️ 检查SQLite数据库...")
    
    db_path = Path("data/rag.db")
    if not db_path.exists():
        print("❌ 数据库文件不存在")
        return {}
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 检查文档表
    cursor.execute('SELECT COUNT(*) FROM documents')
    doc_count = cursor.fetchone()[0]
    print(f"📚 文档总数: {doc_count}")
    
    # 检查文档详情
    cursor.execute('''
        SELECT filename, chunk_count, splitter_type, processing_time 
        FROM documents 
        ORDER BY upload_time DESC 
        LIMIT 5
    ''')
    docs = cursor.fetchall()
    
    print("\n📋 最近文档:")
    total_chunks = 0
    for filename, chunk_count, splitter_type, proc_time in docs:
        total_chunks += chunk_count or 0
        print(f"  - {filename}: {chunk_count}片段 ({splitter_type}, {proc_time:.2f}s)")
    
    # 检查搜索日志
    cursor.execute('SELECT COUNT(*) FROM search_logs')
    search_count = cursor.fetchone()[0]
    print(f"\n🔍 搜索记录总数: {search_count}")
    
    cursor.execute('''
        SELECT query, results_count, processing_time 
        FROM search_logs 
        ORDER BY timestamp DESC 
        LIMIT 3
    ''')
    searches = cursor.fetchall()
    
    print("\n🕐 最近搜索:")
    for query, results_count, proc_time in searches:
        print(f"  - '{query[:50]}...': {results_count}结果 ({proc_time:.3f}s)")
    
    conn.close()
    
    return {
        'document_count': doc_count,
        'total_chunks_expected': total_chunks,
        'search_count': search_count
    }

def test_query_duplicates():
    """测试查询重复问题"""
    print("\n🧪 测试查询重复问题...")
    
    import requests
    
    test_queries = [
        "什么是RAG系统？",
        "大模型测试基准",
        "文档处理方法"
    ]
    
    for query in test_queries:
        try:
            response = requests.post(
                "http://localhost:8000/api/query",
                json={"query": query, "top_k": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                answer = data.get('answer', '')
                
                print(f"\n📝 查询: '{query}'")
                print(f"   结果数: {len(results)}")
                print(f"   回答长度: {len(answer)} 字符")
                
                # 检查结果重复
                contents = [r['content'] for r in results]
                content_counter = Counter(contents)
                duplicates = [content for content, count in content_counter.items() if count > 1]
                
                if duplicates:
                    print(f"   ❌ 发现 {len(duplicates)} 个重复结果")
                    for i, dup in enumerate(duplicates[:2]):
                        print(f"      {i+1}. {dup[:80]}...")
                else:
                    print(f"   ✅ 无重复结果")
                
                # 检查回答重复
                sentences = answer.split('。')
                sentence_counter = Counter([s.strip() for s in sentences if len(s.strip()) > 10])
                answer_duplicates = [s for s, count in sentence_counter.items() if count > 1]
                
                if answer_duplicates:
                    print(f"   ❌ 回答存在重复句子: {len(answer_duplicates)} 个")
                    for i, dup in enumerate(answer_duplicates[:2]):
                        print(f"      {i+1}. {dup[:60]}...")
                else:
                    print(f"   ✅ 回答无明显重复")
                    
            else:
                print(f"   ❌ 查询失败: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 查询错误: {e}")

def main():
    print("🔍 RAG系统重复问题诊断")
    print("=" * 50)
    
    # 检查ChromaDB
    chroma_stats = diagnose_chroma_duplicates()
    
    # 检查SQLite
    sqlite_stats = diagnose_sqlite_data()
    
    # 测试查询
    test_query_duplicates()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 诊断总结:")
    print(f"  ChromaDB片段数: {chroma_stats.get('total_chunks', 0)}")
    print(f"  SQLite预期片段数: {sqlite_stats.get('total_chunks_expected', 0)}")
    print(f"  重复内容数: {chroma_stats.get('duplicate_contents', 0)}")
    print(f"  重复ID数: {chroma_stats.get('duplicate_ids', 0)}")
    
    if chroma_stats.get('duplicate_contents', 0) > 0 or chroma_stats.get('duplicate_ids', 0) > 0:
        print("\n⚠️  发现重复问题，建议清理重复数据")
    else:
        print("\n✅ 数据完整性检查通过")

if __name__ == "__main__":
    main()