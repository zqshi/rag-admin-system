"""
增强版文档处理器 - 使用LangChain和ChromaDB
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# LangChain相关
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ChromaDB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """增强版文档处理器"""
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "paraphrase-MiniLM-L6-v2",
                 collection_name: str = "rag_documents"):
        """
        初始化文档处理器
        
        Args:
            chroma_db_path: ChromaDB数据库路径
            embedding_model: 向量模型名称
            collection_name: 集合名称
        """
        self.chroma_db_path = chroma_db_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # 初始化ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 初始化向量函数 - 支持离线模式
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            logger.info(f"成功加载嵌入模型: {embedding_model}")
        except Exception as e:
            logger.warning(f"加载嵌入模型失败: {e}")
            logger.info("降级使用默认嵌入函数")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # 获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"使用现有集合: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG系统文档集合"}
            )
            logger.info(f"创建新集合: {collection_name}")
        
        # 初始化LangChain嵌入模型 - 支持离线模式
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"成功加载LangChain嵌入模型: {embedding_model}")
        except Exception as e:
            logger.warning(f"加载LangChain嵌入模型失败: {e}")
            logger.info("跳过LangChain嵌入模型初始化")
            self.embeddings = None
        
        # 初始化文本分割器
        self._init_text_splitters()
    
    def _init_text_splitters(self):
        """初始化各种文本分割器"""
        # 递归字符分割器（推荐）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )
        
        # 基于Token的分割器 - 支持离线模式
        try:
            self.token_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=50,
                tokens_per_chunk=100,
                model_name=self.embedding_model
            )
            logger.info("Token分割器初始化成功")
        except Exception as e:
            logger.warning(f"Token分割器初始化失败: {e}")
            # 降级使用字符分割器
            self.token_splitter = CharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator="\n"
            )
            logger.info("降级使用字符分割器代替Token分割器")
        
        # 字符分割器
        self.char_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档对象列表
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                # 通用加载器
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            raise
    
    def split_documents(self, 
                       documents: List[Document], 
                       splitter_type: str = "recursive") -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 文档列表
            splitter_type: 分割器类型 (recursive, token, char)
            
        Returns:
            分割后的文档片段列表
        """
        if splitter_type == "recursive":
            splitter = self.recursive_splitter
        elif splitter_type == "token":
            splitter = self.token_splitter
        elif splitter_type == "char":
            splitter = self.char_splitter
        else:
            splitter = self.recursive_splitter
        
        chunks = splitter.split_documents(documents)
        logger.info(f"文档分割完成: {len(documents)} 页 -> {len(chunks)} 片段")
        return chunks
    
    def _document_exists(self, doc_id: str) -> bool:
        """检查文档是否已存在"""
        try:
            result = self.collection.get(
                where={"doc_id": doc_id},
                limit=1
            )
            return len(result['ids']) > 0
        except Exception as e:
            logger.warning(f"检查文档存在性时出错: {e}")
            return False
    
    def process_and_store(self, 
                         file_path: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         splitter_type: str = "recursive") -> Dict[str, Any]:
        """
        处理并存储文档到向量数据库
        
        Args:
            file_path: 文档路径
            metadata: 元数据
            splitter_type: 分割器类型
            
        Returns:
            处理结果
        """
        # 生成文档ID
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        
        # 检查是否已存在相同文档ID
        if self._document_exists(doc_id):
            logger.warning(f"文档已存在: {doc_id}")
            raise ValueError(f"文档已存在，请勿重复上传: {Path(file_path).name}")
        
        # 加载文档
        documents = self.load_document(file_path)
        
        # 分割文档
        chunks = self.split_documents(documents, splitter_type)
        
        # 准备数据
        texts = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            chunk_metadata = {
                "source": file_path,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_length": len(chunk.page_content),
                **(metadata or {}),
                **(chunk.metadata or {})
            }
            
            texts.append(chunk.page_content)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        # 存储到ChromaDB
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功存储 {len(chunks)} 个文档片段到ChromaDB")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "total_chars": sum(len(text) for text in texts)
            }
            
        except Exception as e:
            logger.error(f"存储文档失败: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            where: 过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 查询更多结果以便去重
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results * 2, 20),
                where=where
            )
            
            # 格式化结果并去重
            formatted_results = []
            seen_contents = set()
            
            for i in range(len(results['ids'][0])):
                content = results['documents'][0][i]
                content_hash = hash(content.strip())
                
                # 跳过重复内容
                if content_hash in seen_contents:
                    continue
                
                seen_contents.add(content_hash)
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': content,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
                
                # 达到所需数量就停止
                if len(formatted_results) >= n_results:
                    break
            
            logger.info(f"搜索完成: 查询='{query[:50]}...', 结果数={len(formatted_results)} (去重后)")
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功
        """
        try:
            # 获取所有相关的chunk IDs
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"删除文档: {doc_id}, 片段数: {len(results['ids'])}")
                return True
            else:
                logger.warning(f"未找到文档: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 获取集合信息
            count = self.collection.count()
            
            # 获取所有文档的元数据
            all_docs = self.collection.get()
            unique_docs = set()
            
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    if 'doc_id' in metadata:
                        unique_docs.add(metadata['doc_id'])
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "db_path": self.chroma_db_path
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {
                "error": str(e),
                "total_chunks": 0,
                "unique_documents": 0
            }
    
    def update_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        更新文档片段的元数据
        
        Args:
            chunk_id: 片段ID
            metadata: 新的元数据
            
        Returns:
            是否成功
        """
        try:
            # 获取现有数据
            result = self.collection.get(ids=[chunk_id])
            
            if result['ids']:
                # 更新元数据
                current_metadata = result['metadatas'][0]
                current_metadata.update(metadata)
                
                # 重新添加（ChromaDB会自动更新）
                self.collection.update(
                    ids=[chunk_id],
                    metadatas=[current_metadata]
                )
                
                logger.info(f"更新元数据成功: {chunk_id}")
                return True
            else:
                logger.warning(f"未找到片段: {chunk_id}")
                return False
                
        except Exception as e:
            logger.error(f"更新元数据失败: {str(e)}")
            raise


# 测试函数
if __name__ == "__main__":
    # 创建处理器实例
    processor = EnhancedDocumentProcessor(
        chroma_db_path="./test_chroma_db",
        collection_name="test_collection"
    )
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"数据库统计: {stats}")
    
    # 测试搜索
    if stats['total_chunks'] > 0:
        results = processor.search("测试查询", n_results=3)
        print(f"搜索结果: {len(results)} 条")