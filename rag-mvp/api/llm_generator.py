#!/usr/bin/env python3
"""
LLM生成器 - 完成RAG的生成环节
支持多种LLM提供商：OpenAI、Claude、通义千问等
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)

class BaseLLMGenerator(ABC):
    """LLM生成器基类"""
    
    @abstractmethod
    def generate(self, query: str, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """生成回答"""
        pass

class OpenAIGenerator(BaseLLMGenerator):
    """OpenAI GPT生成器"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("OpenAI API Key未设置，请设置环境变量 OPENAI_API_KEY")
    
    def generate(self, query: str, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """使用OpenAI生成回答"""
        if not self.api_key:
            return self._fallback_response(query, contexts)
        
        try:
            import openai
            openai.api_key = self.api_key
            
            # 构建提示词
            prompt = self._build_prompt(query, contexts)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手，基于提供的文档内容回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"OpenAI生成失败: {str(e)}")
            return self._fallback_response(query, contexts, error=str(e))
    
    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """构建提示词"""
        context_text = "\n\n".join([f"【片段{i+1}】{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""
基于以下文档片段，回答用户的问题。要求：
1. 仅基于提供的文档内容回答
2. 如果文档中没有相关信息，明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的片段内容

文档内容：
{context_text}

用户问题：{query}

请提供详细的回答：
        """.strip()
        
        return prompt
    
    def _fallback_response(self, query: str, contexts: List[str], error: str = None) -> Dict[str, Any]:
        """降级响应"""
        if not contexts:
            answer = "抱歉，没有找到相关的文档内容来回答您的问题。"
        else:
            answer = f"基于检索到的 {len(contexts)} 个文档片段：\n\n"
            for i, context in enumerate(contexts[:3], 1):
                preview = context[:300] + "..." if len(context) > 300 else context
                answer += f"{i}. {preview}\n\n"
        
        return {
            "answer": answer,
            "model": "fallback",
            "tokens_used": 0,
            "success": False,
            "error": error
        }

class ClaudeGenerator(BaseLLMGenerator):
    """Claude生成器"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229", base_url: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        self.model = model
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        
        if not self.api_key:
            logger.warning("Claude API Key未设置，请设置环境变量 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN")
    
    def generate(self, query: str, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """使用Claude生成回答"""
        if not self.api_key:
            return self._fallback_response(query, contexts)
        
        try:
            import anthropic
            
            # 创建客户端，支持自定义base_url
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            client = anthropic.Anthropic(**client_kwargs)
            
            # 构建提示词
            prompt = self._build_prompt(query, contexts)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text.strip()
            
            return {
                "answer": answer,
                "model": self.model,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Claude生成失败: {str(e)}")
            return self._fallback_response(query, contexts, error=str(e))
    
    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """构建提示词"""
        context_text = "\n\n".join([f"文档片段{i+1}：\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""
你是一个专业的AI助手。请基于以下提供的文档内容回答用户问题：

<documents>
{context_text}
</documents>

<question>
{query}
</question>

请按以下要求回答：
1. 仅基于提供的文档内容进行回答
2. 回答要准确、详细、有逻辑性
3. 如果文档中没有足够信息，请明确指出
4. 可以适当引用文档中的关键信息

请提供您的回答：
        """.strip()
        
        return prompt
    
    def _fallback_response(self, query: str, contexts: List[str], error: str = None) -> Dict[str, Any]:
        """降级响应"""
        if not contexts:
            answer = "抱歉，没有找到相关的文档内容来回答您的问题。"
        else:
            answer = f"基于检索到的文档内容：\n\n"
            for i, context in enumerate(contexts[:3], 1):
                preview = context[:300] + "..." if len(context) > 300 else context
                answer += f"**片段{i}：**\n{preview}\n\n"
        
        return {
            "answer": answer,
            "model": "fallback",
            "tokens_used": 0,
            "success": False,
            "error": error
        }

class LocalLLMGenerator(BaseLLMGenerator):
    """本地LLM生成器（使用Ollama）"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, query: str, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """使用本地LLM生成回答"""
        try:
            import requests
            
            prompt = self._build_prompt(query, contexts)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "num_predict": kwargs.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    "answer": answer,
                    "model": f"ollama/{self.model}",
                    "tokens_used": len(answer.split()),  # 近似计算
                    "success": True
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"本地LLM生成失败: {str(e)}")
            return self._fallback_response(query, contexts, error=str(e))
    
    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        """构建提示词"""
        context_text = "\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Based on the following document excerpts, please answer the user's question.

Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based only on the information in the documents above:
"""
        return prompt
    
    def _fallback_response(self, query: str, contexts: List[str], error: str = None) -> Dict[str, Any]:
        """降级响应"""
        answer = f"本地LLM服务不可用，基于检索结果的简单回答：\n\n"
        for i, context in enumerate(contexts[:3], 1):
            preview = context[:200] + "..." if len(context) > 200 else context
            answer += f"{i}. {preview}\n\n"
        
        return {
            "answer": answer,
            "model": "fallback",
            "tokens_used": 0,
            "success": False,
            "error": error
        }

class RAGGenerator:
    """RAG生成器管理类"""
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        初始化RAG生成器
        
        Args:
            provider: 提供商 (openai, claude, local)
            **kwargs: 传递给具体生成器的参数
        """
        self.provider = provider
        
        if provider == "openai":
            self.generator = OpenAIGenerator(**kwargs)
        elif provider == "claude":
            self.generator = ClaudeGenerator(**kwargs)
        elif provider == "local":
            self.generator = LocalLLMGenerator(**kwargs)
        else:
            raise ValueError(f"不支持的provider: {provider}")
        
        logger.info(f"RAG生成器初始化完成: {provider}")
    
    def generate_answer(self, query: str, retrieved_results: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        生成RAG答案
        
        Args:
            query: 用户问题
            retrieved_results: 检索结果
            **kwargs: 生成参数
            
        Returns:
            生成结果字典
        """
        start_time = time.time()
        
        # 提取文本内容
        contexts = [result.get('content', '') for result in retrieved_results]
        contexts = [ctx for ctx in contexts if ctx.strip()]  # 过滤空内容
        
        if not contexts:
            return {
                "answer": "抱歉，没有找到相关的文档内容来回答您的问题。请尝试换个问题或上传相关文档。",
                "sources": [],
                "generation_time": time.time() - start_time,
                "success": False,
                "model": "none"
            }
        
        # 调用生成器
        result = self.generator.generate(query, contexts, **kwargs)
        
        # 添加时间信息
        result["generation_time"] = time.time() - start_time
        
        # 添加来源信息
        sources = []
        for retrieved_result in retrieved_results:
            if 'metadata' in retrieved_result:
                source = retrieved_result['metadata'].get('source', 'Unknown')
                if source not in sources:
                    sources.append(source.split('/')[-1])  # 只保留文件名
        
        result["sources"] = sources[:5]  # 最多5个来源
        
        logger.info(f"RAG生成完成: query='{query[:50]}...', time={result['generation_time']:.3f}s")
        
        return result

# 测试函数
if __name__ == "__main__":
    # 测试RAG生成器
    generator = RAGGenerator(provider="openai")  # 或 "claude", "local"
    
    # 模拟检索结果
    mock_results = [
        {
            "content": "LangChain是一个用于开发语言模型应用程序的框架。它提供了丰富的工具来处理文档、实现RAG系统等。",
            "metadata": {"source": "langchain_intro.pdf"}
        },
        {
            "content": "ChromaDB是一个开源的向量数据库，专门用于存储和检索向量化的文档。它支持持久化存储和高效搜索。",
            "metadata": {"source": "chromadb_guide.md"}
        }
    ]
    
    # 生成答案
    result = generator.generate_answer(
        query="什么是LangChain和ChromaDB？",
        retrieved_results=mock_results
    )
    
    print(f"生成结果：{json.dumps(result, indent=2, ensure_ascii=False)}")