#!/usr/bin/env python3
"""
FAQ抽取器 - 使用LLM从文档中自动抽取FAQ
"""

import os
import json
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import asyncio

# 添加父目录到路径，以便导入llm_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_generator import RAGGenerator

logger = logging.getLogger(__name__)

class FAQExtractor:
    """FAQ自动抽取器"""
    
    def __init__(self, llm_provider: str = "claude"):
        """
        初始化FAQ抽取器
        
        Args:
            llm_provider: LLM提供商 (openai, claude, local, mock)
        """
        self.llm_provider = llm_provider
        self.llm_generator = None
        self._init_llm()
    
    def _init_llm(self):
        """初始化LLM"""
        if self.llm_provider == "mock":
            logger.info("使用Mock模式")
            return
            
        try:
            # 使用现有的RAGGenerator来初始化LLM
            if self.llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OpenAI API Key未设置，使用Mock模式")
                    self.llm_provider = "mock"
                    return
                self.llm_generator = RAGGenerator(provider="openai")
                
            elif self.llm_provider == "claude":
                api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
                if not api_key:
                    logger.warning("Claude API Key未设置，使用Mock模式")
                    self.llm_provider = "mock"
                    return
                self.llm_generator = RAGGenerator(provider="claude")
                
            elif self.llm_provider == "local":
                self.llm_generator = RAGGenerator(provider="local", model="qwen2:1.5b")
                
            logger.info(f"LLM初始化成功: {self.llm_provider}")
            
        except Exception as e:
            logger.error(f"LLM初始化失败: {str(e)}, 降级为Mock模式")
            self.llm_provider = "mock"
            self.llm_generator = None
    
    async def extract_faqs_from_chunks(self, 
                                     chunks: List[Dict], 
                                     doc_id: str,
                                     max_faqs: int = 10) -> List[Dict[str, Any]]:
        """
        从文档切片中抽取FAQ
        
        Args:
            chunks: 文档切片列表
            doc_id: 文档ID
            max_faqs: 最大FAQ数量
            
        Returns:
            List[Dict]: FAQ列表
        """
        try:
            # 将切片内容合并
            combined_content = self._combine_chunks(chunks)
            
            # 分段处理，避免内容过长
            content_segments = self._split_content(combined_content, max_length=4000)
            
            all_faqs = []
            for i, segment in enumerate(content_segments):
                logger.info(f"处理第{i+1}/{len(content_segments)}段内容...")
                
                # 调用LLM生成FAQ
                segment_faqs = await self._generate_faqs_for_segment(
                    segment, 
                    doc_id,
                    max_faqs_per_segment=max(2, max_faqs // len(content_segments))
                )
                
                all_faqs.extend(segment_faqs)
                
                # 添加延迟避免API限制
                time.sleep(0.5)
            
            # 去重和排序
            unique_faqs = self._deduplicate_faqs(all_faqs)
            
            # 限制数量
            final_faqs = unique_faqs[:max_faqs]
            
            logger.info(f"从文档{doc_id}中成功抽取{len(final_faqs)}个FAQ")
            return final_faqs
            
        except Exception as e:
            logger.error(f"FAQ抽取失败: {str(e)}")
            raise
    
    def _combine_chunks(self, chunks: List[Dict]) -> str:
        """合并切片内容"""
        # 按chunk_index排序
        sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
        
        # 合并内容
        content_parts = []
        for chunk in sorted_chunks:
            content = chunk.get('content', '').strip()
            if content:
                content_parts.append(content)
        
        return '\n\n'.join(content_parts)
    
    def _split_content(self, content: str, max_length: int = 4000) -> List[str]:
        """将长内容分段处理"""
        if len(content) <= max_length:
            return [content]
        
        segments = []
        sentences = content.split('。')
        
        current_segment = ""
        for sentence in sentences:
            if len(current_segment + sentence) < max_length:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    async def _generate_faqs_for_segment(self, 
                                       content: str, 
                                       doc_id: str,
                                       max_faqs_per_segment: int = 3) -> List[Dict[str, Any]]:
        """为内容段生成FAQ"""
        
        if self.llm_provider == "mock" or not self.llm_generator:
            return self._generate_faqs_mock(content, doc_id, max_faqs_per_segment)
        else:
            return await self._generate_faqs_with_llm(content, doc_id, max_faqs_per_segment)
    
    async def _generate_faqs_with_llm(self, content: str, doc_id: str, max_faqs: int) -> List[Dict]:
        """使用LLM生成FAQ"""
        try:
            # 构建专门用于FAQ生成的提示词
            prompt = self._build_faq_prompt(content, max_faqs)
            
            # 模拟检索结果格式（仅包含内容）
            mock_retrieval_results = [{
                "content": content,
                "metadata": {"source": f"document_{doc_id}"}
            }]
            
            # 调用LLM生成器
            result = self.llm_generator.generate_answer(
                query=prompt,
                retrieved_results=mock_retrieval_results,
                max_tokens=1500,
                temperature=0.3
            )
            
            if result["success"]:
                answer = result["answer"]
                logger.info(f"LLM原始输出: {answer[:500]}...")
                faqs = self._parse_llm_response(answer, doc_id)
                logger.info(f"LLM成功生成{len(faqs)}个FAQ")
                return faqs
            else:
                logger.warning(f"LLM生成失败: {result.get('error', 'unknown error')}")
                return self._generate_faqs_mock(content, doc_id, max_faqs)
                
        except Exception as e:
            logger.error(f"LLM FAQ生成失败: {str(e)}")
            return self._generate_faqs_mock(content, doc_id, max_faqs)
    
    def _generate_faqs_mock(self, content: str, doc_id: str, max_faqs: int) -> List[Dict]:
        """Mock模式生成FAQ（用于测试）"""
        logger.info("使用Mock模式生成FAQ")
        
        # 简单的模拟FAQ生成
        faqs = []
        
        # 基于内容长度和关键词生成模拟FAQ
        keywords = self._extract_keywords(content)
        
        mock_questions = [
            f"什么是{keywords[0] if keywords else '主要概念'}？",
            f"如何理解{keywords[1] if len(keywords) > 1 else '相关内容'}？",
            f"{keywords[2] if len(keywords) > 2 else '文档中'}的具体应用是什么？"
        ]
        
        for i, question in enumerate(mock_questions[:max_faqs]):
            faq_id = hashlib.md5(f"{doc_id}_{question}_{time.time()}".encode()).hexdigest()[:12]
            
            # 简单的答案生成
            answer = f"根据文档内容，{question.replace('？', '')}主要涉及文档中提到的相关概念和应用。"
            if len(content) > 100:
                answer += f"具体内容包括：{content[:100]}..."
            
            faqs.append({
                "faq_id": faq_id,
                "question": question,
                "answer": answer,
                "doc_id": doc_id,
                "created_time": datetime.now().isoformat(),
                "status": "active",
                "quality_score": 0.7 + (i * 0.1),  # 模拟质量评分
                "extracted_by": "llm_mock",
                "category": "general",
                "tags": keywords[:3],
                "metadata": {
                    "extraction_method": "mock",
                    "content_length": len(content),
                    "segment_index": i
                }
            })
        
        return faqs
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词（简单实现）"""
        # 简单的关键词提取
        import re
        
        # 去除标点符号，提取中文词汇
        words = re.findall(r'[\u4e00-\u9fff]+', content)
        
        # 过滤常见词汇
        stop_words = {'的', '是', '在', '有', '和', '了', '这', '那', '与', '及', '等', '之', '为', '可以', '能够', '如果', '但是', '因为', '所以'}
        
        keywords = []
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                keywords.append(word)
        
        # 返回前10个关键词
        return list(set(keywords))[:10]
    
    def _build_faq_prompt(self, content: str, max_faqs: int) -> str:
        """构建FAQ生成提示词"""
        if self.llm_provider == "local":
            # 为本地模型优化的提示词，使用中文
            return f"""根据以下文档内容，生成{max_faqs}个FAQ问答对。

要求：
- 生成用户可能会问的常见问题
- 基于文档内容提供准确答案
- 严格使用JSON数组格式
- 答案要简洁且信息丰富

输出格式：必须返回有效的JSON数组，例如：
[{{"question":"问题","answer":"答案"}}]

文档内容：
{content}

请直接输出JSON数组："""
        else:
            # 原始的中文提示词（适用于API模型）
            return f"""
请基于以下文档内容，生成{max_faqs}个高质量的FAQ（问答对）。

任务要求：
1. 问题应该是用户可能会问的常见问题，具有实用性
2. 答案要基于文档内容，准确且详细
3. 涵盖文档的核心概念、关键信息和实用知识点
4. 问题要具体明确，避免过于宽泛
5. 答案要包含足够的细节，对用户有实际帮助
6. 请严格按照JSON格式返回，不要包含其他解释性文字

输出格式（必须严格遵守）：
```json
[
    {{
        "question": "具体的问题内容",
        "answer": "基于文档的详细答案"
    }},
    {{
        "question": "另一个问题",
        "answer": "对应的详细答案"
    }}
]
```

文档内容：
{content}

请直接输出JSON格式的FAQ列表：
"""
    
    def _parse_llm_response(self, response: str, doc_id: str) -> List[Dict]:
        """解析LLM返回的FAQ数据"""
        try:
            logger.info(f"开始解析响应: {response[:500]}...")
            
            # 多种方式尝试提取JSON
            json_str = None
            
            # 方法1: 寻找```json包装
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 方法2: 寻找[...]格式的JSON数组
                array_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if array_match:
                    json_str = array_match.group(0)
                else:
                    # 方法3: 寻找部分JSON的开始
                    start_match = re.search(r'\[\s*\{', response)
                    if start_match:
                        start_pos = start_match.start()
                        json_str = response[start_pos:]
                        # 尝试找到结束位置
                        brace_count = 0
                        bracket_count = 0
                        end_pos = len(json_str)
                        for i, char in enumerate(json_str):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                            elif char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0 and brace_count == 0:
                                    end_pos = i + 1
                                    break
                        json_str = json_str[:end_pos]
                    else:
                        # 方法4: 尝试整个响应
                        json_str = response.strip()
            
            # 清理JSON字符串
            if json_str:
                json_str = json_str.strip()
                # 移除可能的markdown标记
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                # 移除可能的前后缀文字
                if not json_str.startswith('['):
                    start_bracket = json_str.find('[')
                    if start_bracket != -1:
                        json_str = json_str[start_bracket:]
                        
                logger.debug(f"提取的JSON: {json_str[:300]}...")
            
            faqs_data = json.loads(json_str)
            
            # 处理不同的返回格式
            if isinstance(faqs_data, dict):
                # 如果返回的是单个对象，转换为数组
                if 'question' in faqs_data and 'answer' in faqs_data:
                    faqs_data = [faqs_data]
                    logger.info("将单个FAQ对象转换为数组")
                else:
                    logger.warning(f"LLM返回的对象格式不正确: {faqs_data}")
                    return []
            elif not isinstance(faqs_data, list):
                logger.warning(f"LLM返回的不是数组或对象格式: {type(faqs_data)}")
                return []
            
            # 转换为标准格式
            result_faqs = []
            for i, faq in enumerate(faqs_data):
                if isinstance(faq, dict) and 'question' in faq and 'answer' in faq:
                    # 验证问题和答案不为空
                    question = str(faq['question']).strip()
                    answer = str(faq['answer']).strip()
                    
                    if not question or not answer or len(question) < 5:
                        logger.warning(f"跳过无效的FAQ项: {faq}")
                        continue
                    
                    faq_id = hashlib.md5(f"{doc_id}_{question}_{time.time()}_{i}".encode()).hexdigest()[:12]
                    
                    # 基于问题长度和答案长度计算质量评分
                    quality_score = self._calculate_faq_quality(question, answer)
                    
                    result_faqs.append({
                        "faq_id": faq_id,
                        "question": question,
                        "answer": answer,
                        "doc_id": doc_id,
                        "created_time": datetime.now().isoformat(),
                        "status": "active",
                        "quality_score": quality_score,
                        "extracted_by": f"llm_{self.llm_provider}",
                        "category": faq.get('category', 'general'),
                        "tags": faq.get('tags', []),
                        "metadata": {
                            "extraction_method": self.llm_provider,
                            "content_preview": response[:200] + "..." if len(response) > 200 else response
                        }
                    })
            
            logger.info(f"成功解析{len(result_faqs)}个FAQ")
            return result_faqs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            logger.debug(f"尝试解析的内容: {json_str[:500] if json_str else 'None'}")
            
            # 容错处理：尝试修复JSON
            return self._try_fix_json_and_parse(response, doc_id)
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            logger.info(f"开始解析响应: {response[:500]}...")
            return []
    
    def _try_fix_json_and_parse(self, response: str, doc_id: str) -> List[Dict]:
        """尝试修复损坏的JSON并解析"""
        try:
            # 尝试手动解析简单格式
            import re
            
            # 1. 尝试处理多个JSON对象的情况
            json_objects = re.findall(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_objects:
                result_faqs = []
                for i, json_obj in enumerate(json_objects):
                    try:
                        faq_data = json.loads(json_obj)
                        if 'question' in faq_data and 'answer' in faq_data:
                            question = str(faq_data['question']).strip()
                            answer = str(faq_data['answer']).strip()
                            
                            if len(question) >= 5 and len(answer) >= 10:
                                faq_id = hashlib.md5(f"{doc_id}_{question}_{time.time()}_{i}".encode()).hexdigest()[:12]
                                quality_score = self._calculate_faq_quality(question, answer)
                                
                                result_faqs.append({
                                    "faq_id": faq_id,
                                    "question": question,
                                    "answer": answer,
                                    "doc_id": doc_id,
                                    "created_time": datetime.now().isoformat(),
                                    "status": "active",
                                    "quality_score": quality_score,
                                    "extracted_by": f"llm_{self.llm_provider}_multi_obj",
                                    "category": "general",
                                    "tags": [],
                                    "metadata": {
                                        "extraction_method": f"{self.llm_provider}_multi_object_fixed",
                                        "object_index": i
                                    }
                                })
                    except Exception as e:
                        logger.debug(f"解析JSON对象{i}失败: {str(e)}")
                        continue
                
                if result_faqs:
                    logger.info(f"通过多对象模式解析到{len(result_faqs)}个FAQ")
                    return result_faqs
            
            # 2. 查找问答对模式
            patterns = [
                r'["\']question["\']\s*:\s*["\']([^"\'\n]+)["\']\s*,\s*["\']answer["\']\s*:\s*["\']([^"\'\n]+)["\']',
                r'question["\']?\s*:\s*["\']([^"\'\n]+)["\']\s*,\s*answer["\']?\s*:\s*["\']([^"\'\n]+)["\']',
                r'[Qq]uestion\s*:\s*([^\n]+)\n[Aa]nswer\s*:\s*([^\n]+)',
                r'问题\s*[:\uff1a]\s*([^\n]+)\n答案\s*[:\uff1a]\s*([^\n]+)',
            ]
            
            result_faqs = []
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
                if matches:
                    logger.info(f"通过模式匹配找到{len(matches)}个FAQ")
                    
                    for i, (question, answer) in enumerate(matches):
                        question = question.strip().strip('"\'')
                        answer = answer.strip().strip('"\'')
                        
                        if len(question) >= 5 and len(answer) >= 10:
                            faq_id = hashlib.md5(f"{doc_id}_{question}_{time.time()}_{i}".encode()).hexdigest()[:12]
                            quality_score = self._calculate_faq_quality(question, answer)
                            
                            result_faqs.append({
                                "faq_id": faq_id,
                                "question": question,
                                "answer": answer,
                                "doc_id": doc_id,
                                "created_time": datetime.now().isoformat(),
                                "status": "active",
                                "quality_score": quality_score,
                                "extracted_by": f"llm_{self.llm_provider}_fixed",
                                "category": "general",
                                "tags": [],
                                "metadata": {
                                    "extraction_method": f"{self.llm_provider}_pattern_fixed",
                                    "pattern_used": pattern
                                }
                            })
                    
                    if result_faqs:
                        return result_faqs
            
            logger.warning("无法修复JSON或提取FAQ")
            return []
            
        except Exception as e:
            logger.error(f"JSON修复失败: {str(e)}")
            return []
    
    def _calculate_faq_quality(self, question: str, answer: str) -> float:
        """计算FAQ质量评分"""
        score = 0.5  # 基础分
        
        # 问题质量评估
        if len(question) >= 10 and len(question) <= 100:  # 合适的问题长度
            score += 0.1
        if '?' in question or '？' in question:  # 包含问号
            score += 0.1
        if any(word in question for word in ['什么', '如何', '为什么', '怎么', '哪些', 'what', 'how', 'why']):
            score += 0.1
            
        # 答案质量评估
        if len(answer) >= 20:  # 答案有一定长度
            score += 0.1
        if len(answer) >= 50:  # 答案详细
            score += 0.1
            
        return min(1.0, score)  # 最高1.0分
    
    def _deduplicate_faqs(self, faqs: List[Dict]) -> List[Dict]:
        """去重FAQ"""
        seen_questions = set()
        unique_faqs = []
        
        for faq in faqs:
            question_key = faq['question'].lower().strip()
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_faqs.append(faq)
        
        # 按质量评分排序
        return sorted(unique_faqs, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    def estimate_extraction_time(self, chunks_count: int) -> int:
        """估算抽取时间（秒）"""
        # 根据切片数量估算时间
        base_time = 10  # 基础时间
        chunk_time = chunks_count * 0.5  # 每个切片0.5秒
        
        if self.llm_provider == "mock":
            return int(base_time + chunk_time * 0.1)  # Mock模式更快
        elif self.llm_provider == "local":
            return int(base_time + chunk_time * 2)  # 本地模型较慢
        else:
            return int(base_time + chunk_time)  # API调用标准时间