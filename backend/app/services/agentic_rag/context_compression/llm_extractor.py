"""
L3层: 基于LLM的精细上下文压缩器

使用大语言模型从文档中提取与查询最相关的片段
适用于金融领域的专业术语理解和复杂语义提取
"""

import time
import re
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import RegexParser

from .base_compressor import BaseCompressor, Document, CompressionResult
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)


class FinancialContextCompressor(BaseCompressor):
    """
    金融领域的上下文压缩器

    特点:
    - 使用LLM进行语义理解
    - 精确提取相关句子/段落
    - 保留上下文连贯性
    - 适用于金融专业术语
    """

    def __init__(
        self,
        llm,
        compression_rate: float = 0.5,
        max_length: int = 2000,
        keep_structure: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        初始化LLM压缩器

        Args:
            llm: LangChain LLM实例
            compression_rate: 目标压缩率（0-1之间）
            max_length: 单个文档最大保留长度
            keep_structure: 是否保持文档结构
            config: 其他配置
        """
        super().__init__(config)

        self.llm = llm
        self.compression_rate = compression_rate
        self.max_length = max_length
        self.keep_structure = keep_structure

        # 构建提取链
        self.extraction_chain = self._build_extraction_chain()

        logger.info(
            f"FinancialContextCompressor初始化: "
            f"compression_rate={compression_rate}, max_length={max_length}"
        )

    def _build_extraction_chain(self) -> LLMChain:
        """构建LLM提取链"""
        prompt_template = """
你是一个专业的金融文档分析师。你的任务是从文档中提取与用户查询最相关的信息。

**用户查询**: {query}

**文档内容**:
```
{context}
```

**提取要求**:
1. 只保留直接回答查询所需的关键信息
2. 删除无关的背景描述、修饰性语句
3. 保留具体的数字、数据、时间、公司名称等关键实体
4. 保持原有的文档结构（段落分隔）
5. 如果是表格数据，保留相关行
6. 提取的内容总长度不超过{max_length}字符

**输出格式**:
请直接输出提取的内容，不要添加任何解释或标注。

提取的相关信息:
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "context", "max_length"]
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    async def compress(
        self,
        query: str,
        documents: List[Document],
        compression_rate: float = None,
        max_length: int = None,
        **kwargs
    ) -> CompressionResult:
        """
        使用LLM压缩文档

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            compression_rate: 目标压缩率（None则使用默认值）
            max_length: 单个文档最大长度（None则使用默认值）
            **kwargs: 其他参数

        Returns:
            CompressionResult: 压缩结果
        """
        start_time = time.time()

        # 参数设置
        compression_rate = compression_rate or self.compression_rate
        max_length = max_length or self.max_length

        try:
            compressed_docs = []
            total_tokens_saved = 0

            # 计算目标文档数量
            target_count = max(1, int(len(documents) * compression_rate))

            logger.info(
                f"LLM压缩: {len(documents)}个文档 → 目标{target_count}个 "
                f"(rate={compression_rate:.2%})"
            )

            # 对前target_count个文档进行LLM提取
            for i, doc in enumerate(documents[:target_count]):
                try:
                    # 调用LLM提取
                    extracted_content = await self._extract_relevant_content(
                        query=query,
                        document=doc,
                        max_length=max_length
                    )

                    # 如果提取成功且内容不为空
                    if extracted_content and len(extracted_content.strip()) > 50:
                        # 计算节省的tokens
                        tokens_saved = self._estimate_tokens(doc.page_content) - \
                                       self._estimate_tokens(extracted_content)
                        total_tokens_saved += max(0, tokens_saved)

                        # 创建压缩后的文档
                        compressed_doc = Document(
                            page_content=extracted_content.strip(),
                            metadata={
                                **doc.metadata,
                                "compressed": True,
                                "original_length": len(doc.page_content),
                                "compressed_length": len(extracted_content),
                                "compression_method": "llm_extraction"
                            }
                        )
                        compressed_docs.append(compressed_doc)

                        logger.debug(
                            f"文档{i+1}: {len(doc.page_content)} → "
                            f"{len(extracted_content)} 字符 "
                            f"({(len(extracted_content)/len(doc.page_content)):.1%})"
                        )

                except Exception as e:
                    # 如果单个文档提取失败，保留原文
                    logger.warning(f"文档{i+1}提取失败，保留原文: {e}")
                    compressed_docs.append(doc)

            # 计算统计信息
            compression_time = time.time() - start_time

            result = CompressionResult(
                compressed_docs=compressed_docs,
                original_count=len(documents),
                compressed_count=len(compressed_docs),
                compression_ratio=len(compressed_docs) / len(documents) if documents else 0,
                tokens_saved=total_tokens_saved,
                compression_time=compression_time,
                metadata={
                    "method": "llm_extraction",
                    "compression_rate": compression_rate,
                    "max_length": max_length,
                    "avg_compression_per_doc": total_tokens_saved / len(compressed_docs) if compressed_docs else 0
                }
            )

            # 更新统计
            self._update_stats(len(documents), total_tokens_saved, compression_time)

            logger.info(
                f"FinancialContextCompressor: {len(documents)} → {len(compressed_docs)} "
                f"(tokens_saved={total_tokens_saved}, time={compression_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"FinancialContextCompressor压缩失败: {e}")
            # 返回原始文档
            return CompressionResult(
                compressed_docs=documents,
                original_count=len(documents),
                compressed_count=len(documents),
                compression_ratio=1.0,
                tokens_saved=0,
                compression_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _extract_relevant_content(
        self,
        query: str,
        document: Document,
        max_length: int
    ) -> str:
        """
        提取文档中的相关内容

        Args:
            query: 用户查询
            document: 文档
            max_length: 最大长度

        Returns:
            提取的内容
        """
        # 如果文档本身就很短，直接返回
        if len(document.page_content) <= max_length * 0.8:
            return document.page_content

        try:
            # 调用LLM
            result = await self.extraction_chain.arun(
                query=query,
                context=document.page_content,
                max_length=max_length
            )

            # 清理结果
            extracted = result.strip()

            # 如果LLM返回过长，截断
            if len(extracted) > max_length:
                extracted = extracted[:max_length] + "..."

            return extracted

        except Exception as e:
            logger.warning(f"LLM提取失败: {e}")
            # 降级方案：返回前N个字符
            return document.page_content[:max_length] + "..."

    async def extract_key_sentences(
        self,
        query: str,
        document: Document,
        top_k: int = 5
    ) -> List[str]:
        """
        提取文档中的关键句子

        Args:
            query: 用户查询
            document: 文档
            top_k: 返回的句子数量

        Returns:
            关键句子列表
        """
        # 分割文档为句子
        sentences = self._split_sentences(document.page_content)

        if len(sentences) <= top_k:
            return sentences

        # 使用LLM对句子评分
        prompt = PromptTemplate(
            template=(
                "用户查询: {query}\n\n"
                "句子: {sentence}\n\n"
                "该句子是否与查询相关？请用0-1之间的分数表示相关性。\n"
                "只输出分数，不要输出其他内容。"
            ),
            input_variables=["query", "sentence"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        scored_sentences = []
        for sentence in sentences:
            try:
                score_text = await chain.arun(query=query, sentence=sentence)
                score = float(re.findall(r'[\d.]+', score_text)[0]) if re.findall(r'[\d.]+', score_text) else 0
                scored_sentences.append((sentence, score))
            except:
                scored_sentences.append((sentence, 0))

        # 排序并返回top_k
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored_sentences[:top_k]]

    def _split_sentences(self, text: str) -> List[str]:
        """
        分割文本为句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 简单的句子分割（按句号、问号、感叹号）
        sentences = re.split(r'[。！？.!?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _update_stats(self, doc_count: int, tokens_saved: int, compression_time: float):
        """更新统计信息"""
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_documents_processed"] += doc_count
        self.compression_stats["total_tokens_saved"] += tokens_saved
        self.compression_stats["total_time"] += compression_time


def get_financial_compressor(
    llm,
    compression_rate: float = 0.5,
    max_length: int = 2000
) -> FinancialContextCompressor:
    """
    获取金融压缩器实例

    Args:
        llm: LangChain LLM实例
        compression_rate: 压缩率
        max_length: 最大长度

    Returns:
        FinancialContextCompressor实例
    """
    return FinancialContextCompressor(
        llm=llm,
        compression_rate=compression_rate,
        max_length=max_length
    )
