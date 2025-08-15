#!/usr/bin/env python3
"""
修复RAG系统中的重复问题
"""

import sqlite3
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib
from collections import defaultdict

def clean_duplicate_documents():
    """清理重复文档"""
    print("🧹 清理重复文档...")
    
    # 连接数据库
    db_path = Path("data/rag.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 找出重复文档（相同文件名）
    cursor.execute('''
        SELECT filename, COUNT(*) as count, GROUP_CONCAT(id) as ids
        FROM documents 
        GROUP BY filename 
        HAVING count > 1
        ORDER BY count DESC
    ''')
    
    duplicates = cursor.fetchall()
    
    if not duplicates:
        print("✅ 未发现重复文档")
        return
    
    print(f"❌ 发现 {len(duplicates)} 个重复文档:")
    
    for filename, count, ids_str in duplicates:
        ids = ids_str.split(',')
        print(f"  - {filename}: {count} 个副本")
        
        # 保留最新的，删除其他的
        cursor.execute('''
            SELECT id, upload_time FROM documents 
            WHERE filename = ? 
            ORDER BY upload_time DESC
        ''', (filename,))
        
        docs = cursor.fetchall()
        keep_id = docs[0][0]  # 保留最新的
        delete_ids = [doc[0] for doc in docs[1:]]  # 删除其他的
        
        print(f"    保留: {keep_id}")
        print(f"    删除: {delete_ids}")
        
        # 删除重复的文档记录
        for delete_id in delete_ids:
            cursor.execute('DELETE FROM documents WHERE id = ?', (delete_id,))
            print(f"    已删除文档记录: {delete_id}")
    
    conn.commit()
    conn.close()
    print("✅ 数据库清理完成")

def clean_duplicate_chunks():
    """清理ChromaDB中的重复片段"""
    print("\n🧹 清理ChromaDB中的重复片段...")
    
    # 连接ChromaDB
    chroma_path = Path("chroma_db")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection("rag_documents")
    
    # 获取所有数据
    result = collection.get(include=['documents', 'metadatas'])
    
    print(f"📊 当前片段数: {len(result['ids'])}")
    
    # 找出重复内容
    content_to_ids = defaultdict(list)
    for i, content in enumerate(result['documents']):
        content_hash = hashlib.md5(content.encode()).hexdigest()
        content_to_ids[content_hash].append((result['ids'][i], content))
    
    # 找出需要删除的重复项
    duplicates_to_delete = []
    unique_kept = 0
    
    for content_hash, items in content_to_ids.items():
        if len(items) > 1:
            # 保留第一个，删除其他的
            keep_id = items[0][0]
            delete_ids = [item[0] for item in items[1:]]
            duplicates_to_delete.extend(delete_ids)
            
            print(f"内容哈希 {content_hash[:8]}...")
            print(f"  保留: {keep_id}")
            print(f"  删除: {delete_ids} (共{len(delete_ids)}个)")
        
        unique_kept += 1
    
    # 执行删除
    if duplicates_to_delete:
        print(f"\n🗑️ 删除 {len(duplicates_to_delete)} 个重复片段...")
        collection.delete(ids=duplicates_to_delete)
        print("✅ ChromaDB清理完成")
        
        # 验证结果
        result_after = collection.get(include=['documents'])
        print(f"📊 清理后片段数: {len(result_after['ids'])}")
        print(f"📉 删除了 {len(result['ids']) - len(result_after['ids'])} 个重复片段")
    else:
        print("✅ 未发现重复片段")

def update_document_stats():
    """更新文档统计信息"""
    print("\n📊 更新文档统计信息...")
    
    # 连接数据库
    db_path = Path("data/rag.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 连接ChromaDB
    chroma_path = Path("chroma_db")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection("rag_documents")
    result = collection.get(include=['metadatas'])
    
    # 统计每个文档的实际片段数
    doc_chunks = defaultdict(int)
    for metadata in result['metadatas']:
        doc_id = metadata.get('doc_id')
        if doc_id:
            doc_chunks[doc_id] += 1
    
    # 更新数据库中的统计信息
    for doc_id, actual_chunk_count in doc_chunks.items():
        cursor.execute('''
            UPDATE documents 
            SET chunk_count = ? 
            WHERE id = ?
        ''', (actual_chunk_count, doc_id))
        print(f"  更新文档 {doc_id}: {actual_chunk_count} 片段")
    
    conn.commit()
    conn.close()
    print("✅ 统计信息更新完成")

def main():
    print("🔧 修复RAG系统重复问题")
    print("=" * 50)
    
    try:
        # 1. 清理重复文档记录
        clean_duplicate_documents()
        
        # 2. 清理重复片段
        clean_duplicate_chunks()
        
        # 3. 更新统计信息
        update_document_stats()
        
        print("\n" + "=" * 50)
        print("✅ 重复问题修复完成！")
        print("\n建议重启API服务以生效:")
        print("  1. 停止当前服务 (Ctrl+C)")
        print("  2. 重新运行: python api/main.py")
        
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()