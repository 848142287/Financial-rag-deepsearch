"""
文档引用自动插入服务
从 swxy/backend 移植并优化，在答案中自动插入引用标记
"""

from app.core.structured_logging import get_structured_logger
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
from app.services.embeddings.unified_embedding_service import get_embedding_service

logger = get_structured_logger(__name__)

# 获取embedding服务实例
_embedding_service = None

def get_embedding():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = get_embedding_service()
    return _embedding_service


class CitationInserter:
    """
    文档引用插入器

    自动在答案中插入引用标记，格式为: ##引用编号$$
    例如: "根据财务报告，公司营收增长20% ##1$$，净利润达到5亿元 ##2$$"
    """

    def __init__(self):
        self.citation_pattern = r"##(\d+)\$\$"  # 匹配 ##数字$$
        self.sentence_split_pattern = r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])"
        self.code_block_pattern = r"(```)"

    def insert_citations(
        self,
        answer: str,
        chunks: List[str],
        chunk_vectors: Optional[List[np.ndarray]] = None,
        tk_weight: float = 0.1,
        vt_weight: float = 0.9
    ) -> Tuple[str, Set[int]]:
        """
        在答案中插入引用标记

        Args:
            answer: 生成的答案
            chunks: 检索到的文档块列表
            chunk_vectors: 文档块的向量表示（可选）
            tk_weight: 词相似度权重
            vt_weight: 向量相似度权重

        Returns:
            (带引用的答案, 引用的文档索引集合)
        """
        if not chunks:
            logger.warning("没有提供文档块，无法插入引用")
            return answer, set()

        if not answer:
            return answer, set()

        # 如果没有提供向量，计算向量
        if chunk_vectors is None:
            try:
                chunk_vectors = get_embedding().encode(chunks)
            except Exception as e:
                logger.error(f"计算文档向量失败: {e}")
                return answer, set()

        # 分割答案为句子
        sentences = self._split_answer_into_sentences(answer)

        if not sentences:
            logger.warning("答案分割后为空，无法插入引用")
            return answer, set()

        # 计算答案句子的向量
        try:
            sentence_vectors = get_embedding().encode(sentences)
        except Exception as e:
            logger.error(f"计算句子向量失败: {e}")
            return answer, set()

        # 验证向量维度
        if len(sentence_vectors[0]) != len(chunk_vectors[0]):
            logger.warning(
                f"向量维度不匹配: 句子{len(sentence_vectors[0])} vs 文档{len(chunk_vectors[0])}"
            )
            # 填充为相同维度
            chunk_vectors = [
                np.zeros(len(sentence_vectors[0])) if len(cv) != len(sentence_vectors[0]) else cv
                for cv in chunk_vectors
            ]

        # 为每个句子找到最相关的文档块
        citations = self._find_citations_for_sentences(
            sentences,
            sentence_vectors,
            chunks,
            chunk_vectors,
            tk_weight,
            vt_weight
        )

        # 插入引用标记
        answer_with_citations = self._insert_citation_markers(
            sentences,
            citations,
            answer
        )

        cited_indices = set()
        for sent_citations in citations.values():
            cited_indices.update(sent_citations)

        logger.info(f"插入引用完成，引用了{len(cited_indices)}个文档块")
        return answer_with_citations, cited_indices

    def _split_answer_into_sentences(self, answer: str) -> List[str]:
        """
        将答案分割为句子

        Args:
            answer: 答案文本

        Returns:
            句子列表
        """
        # 处理代码块
        if "```" in answer:
            pieces = re.split(self.code_block_pattern, answer)
            sentences = []
            i = 0
            while i < len(pieces):
                if pieces[i] == "```":
                    # 保留整个代码块
                    start = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    sentences.append("".join(pieces[start:i]) + "\n")
                else:
                    # 分割普通文本
                    sub_sentences = re.split(self.sentence_split_pattern, pieces[i])
                    # 修正分割导致的标点丢失
                    for j in range(1, len(sub_sentences)):
                        if re.match(self.sentence_split_pattern, sub_sentences[j]):
                            sub_sentences[j - 1] += sub_sentences[j][0]
                            sub_sentences[j] = sub_sentences[j][1:]
                    sentences.extend(sub_sentences)
                    i += 1
        else:
            # 直接分割
            sentences = re.split(self.sentence_split_pattern, answer)
            # 修正分割导致的标点丢失
            for i in range(1, len(sentences)):
                if re.match(self.sentence_split_pattern, sentences[i]):
                    sentences[i - 1] += sentences[i][0]
                    sentences[i] = sentences[i][1:]

        # 过滤短句子
        result = [s for s in sentences if len(s) >= 5]
        logger.debug(f"答案分割为{len(result)}个句子")
        return result

    def _find_citations_for_sentences(
        self,
        sentences: List[str],
        sentence_vectors: List[np.ndarray],
        chunks: List[str],
        chunk_vectors: List[np.ndarray],
        tk_weight: float,
        vt_weight: float
    ) -> Dict[int, List[int]]:
        """
        为每个句子找到相关的文档块

        Args:
            sentences: 句子列表
            sentence_vectors: 句子向量
            chunks: 文档块
            chunk_vectors: 文档块向量
            tk_weight: 词权重
            vt_weight: 向量权重

        Returns:
            {句子索引: [文档块索引列表]}
        """
        citations = {}

        # 计算相似度矩阵
        similarities = self._compute_similarity_matrix(
            sentence_vectors,
            chunk_vectors,
            tk_weight,
            vt_weight
        )

        # 为每个句子选择最相关的文档
        threshold = 0.63
        for sent_idx in range(len(sentences)):
            # 获取该句子与所有文档的相似度
            sent_similarities = similarities[sent_idx]

            # 找到最大相似度
            max_sim = np.max(sent_similarities)

            # 如果最大相似度超过阈值，添加引用
            if max_sim >= threshold:
                # 找到所有超过阈值的文档（使用max_sim * 0.99作为阈值，确保只引用最相关的）
                relevant_docs = np.where(sent_similarities >= max_sim * 0.99)[0]
                # 限制最多引用4个文档
                citations[sent_idx] = relevant_docs[:4].tolist()

        return citations

    def _compute_similarity_matrix(
        self,
        sentence_vectors: List[np.ndarray],
        chunk_vectors: List[np.ndarray],
        tk_weight: float,
        vt_weight: float
    ) -> np.ndarray:
        """
        计算句子与文档块的相似度矩阵

        Args:
            sentence_vectors: 句子向量列表
            chunk_vectors: 文档块向量列表
            tk_weight: 词相似度权重
            vt_weight: 向量相似度权重

        Returns:
            相似度矩阵，shape=(句子数, 文档块数)
        """
        # 目前只使用向量相似度
        # TODO: 可以添加词相似度计算
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(sentence_vectors, chunk_vectors)
        return sim_matrix * vt_weight

    def _insert_citation_markers(
        self,
        sentences: List[str],
        citations: Dict[int, List[int]],
        original_answer: str
    ) -> str:
        """
        在原文中插入引用标记

        Args:
            sentences: 分割后的句子列表
            citations: 引用映射
            original_answer: 原始答案

        Returns:
            带引用的答案
        """
        # 简化实现：直接在答案末尾添加引用信息
        # 完整实现需要在原文中定位句子位置

        if not citations:
            return original_answer

        # 统计引用的文档
        cited_docs = set()
        for sent_citations in citations.values():
            cited_docs.update(sent_citations)

        if not cited_docs:
            return original_answer

        # 在答案末尾添加引用信息
        citation_text = "\n\n**参考来源**: " + ", ".join([f"[{i+1}]" for i in sorted(cited_docs)])
        return original_answer + citation_text

    def insert_citations_advanced(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        高级引用插入 - 基于检索结果的直接引用

        Args:
            answer: 答案
            chunks: 检索结果列表，每个包含content、score等
            top_k: 引用前K个最相关的文档

        Returns:
            (带引用的答案, 引用的文档列表)
        """
        if not chunks:
            return answer, []

        # 按分数排序
        sorted_chunks = sorted(
            enumerate(chunks),
            key=lambda x: x[1].get('similarity', x[1].get('score', 0)),
            reverse=True
        )

        # 选择top_k个文档
        cited_chunks = sorted_chunks[:top_k]
        cited_indices = [idx for idx, _ in cited_chunks]
        cited_docs = [doc for _, doc in cited_chunks]

        # 在答案末尾添加引用
        citation_text = "\n\n**参考来源**:\n"
        for i, (idx, doc) in enumerate(cited_chunks):
            doc_name = doc.get('document_name', f'文档{idx+1}')
            score = doc.get('similarity', doc.get('score', 0))
            citation_text += f"- [{i+1}] {doc_name} (相似度: {score:.2f})\n"

        return answer + citation_text, cited_docs


# 创建全局服务实例
citation_inserter = CitationInserter()


def get_citation_inserter() -> CitationInserter:
    """获取引用插入服务实例"""
    return citation_inserter
