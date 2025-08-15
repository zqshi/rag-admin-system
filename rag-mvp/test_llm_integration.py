#!/usr/bin/env python3
"""
测试LLM集成是否正常工作
"""

import requests
import json

def test_llm_integration():
    """测试LLM集成状态"""
    print("🧪 测试LLM集成状态...")
    
    # 测试健康检查
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ API服务正常")
            print(f"   - LLM可用: {health.get('llm', {}).get('available', False)}")
            print(f"   - LLM提供商: {health.get('llm', {}).get('provider', 'None')}")
            print(f"   - LLM模型: {health.get('llm', {}).get('model', 'None')}")
            
            components = health.get('components', {})
            for comp, status in components.items():
                print(f"   - {comp}: {status}")
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 无法连接API服务: {e}")
        return False
    
    # 测试简单查询
    print("\n🔍 测试智能问答...")
    try:
        query_data = {
            "query": "什么是RAG系统？",
            "top_k": 3
        }
        
        response = requests.post(
            "http://localhost:8000/api/query",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 查询成功")
            print(f"   - 查询: {result.get('query')}")
            print(f"   - 结果数: {len(result.get('results', []))}")
            print(f"   - 处理时间: {result.get('processing_time', 0):.3f}秒")
            
            answer = result.get('answer', '')
            print(f"   - 回答类型: {'LLM智能回答' if '基于知识库' not in answer and '简单模式' not in answer else '降级模式'}")
            print(f"   - 回答预览: {answer[:100]}...")
            
            return True
        else:
            print(f"❌ 查询失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 查询测试失败: {e}")
        return False

def main():
    print("🔬 LLM集成测试工具")
    print("=" * 50)
    
    success = test_llm_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ LLM集成测试通过")
        print("\n💡 建议:")
        print("   1. 检查LLM配置是否正确")
        print("   2. 确认API密钥设置")
        print("   3. 测试更复杂的查询")
    else:
        print("❌ LLM集成测试失败")
        print("\n🔧 排查步骤:")
        print("   1. 确认API服务正在运行")
        print("   2. 检查网络连接")
        print("   3. 查看服务日志")
        print("   4. 配置离线模式")

if __name__ == "__main__":
    main()