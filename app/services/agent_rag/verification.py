"""
验证引擎
验证生成答案的准确性和可靠性
"""

import asyncio
from typing import List, Dict, Any, Optional
import re
import logging

from ..llm_service import llm_service

logger = logging.getLogger(__name__)


class VerificationEngine:
    """验证引擎"""

    def __init__(self):
        self.verification_rules = {
            'factual_consistency': 0.3,
            'source_citation': 0.25,
            'logical_coherence': 0.25,
            'completeness': 0.2
        }

    async def verify(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        验证答案

        Args:
            query: 原始查询
            answer: 生成的答案
            retrieved_docs: 检索到的文档

        Returns:
            验证结果
        """
        verification_result = {
            'is_valid': True,
            'confidence': 0.0,
            'scores': {},
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        try:
            # 执行各项验证
            consistency_score = await self._verify_factual_consistency(
                query, answer, retrieved_docs
            )
            citation_score = self._verify_source_citation(answer, retrieved_docs)
            coherence_score = self._verify_logical_coherence(query, answer)
            completeness_score = self._verify_completeness(query, answer)

            # 记录各项得分
            verification_result['scores'] = {
                'factual_consistency': consistency_score,
                'source_citation': citation_score,
                'logical_coherence': coherence_score,
                'completeness': completeness_score
            }

            # 计算综合置信度
            confidence = (
                consistency_score * self.verification_rules['factual_consistency'] +
                citation_score * self.verification_rules['source_citation'] +
                coherence_score * self.verification_rules['logical_coherence'] +
                completeness_score * self.verification_rules['completeness']
            )
            verification_result['confidence'] = confidence

            # 判断是否通过验证
            verification_result['is_valid'] = confidence >= 0.7

            # 生成错误和警告
            if consistency_score < 0.6:
                verification_result['errors'].append('答案与检索文档存在事实不一致')
            if citation_score < 0.5:
                verification_result['warnings'].append('答案缺乏充分的引用来源')
            if coherence_score < 0.6:
                verification_result['errors'].append('答案逻辑不够连贯')
            if completeness_score < 0.6:
                verification_result['warnings'].append('答案可能不够完整')

            # 生成改进建议
            verification_result['suggestions'] = self._generate_suggestions(
                verification_result['scores']
            )

            logger.info(f"答案验证完成，置信度: {confidence:.3f}")
            return verification_result

        except Exception as e:
            logger.error(f"答案验证失败: {e}")
            verification_result['is_valid'] = False
            verification_result['errors'].append(f'验证过程出错: {e}')
            return verification_result

    async def _verify_factual_consistency(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """验证事实一致性"""
        try:
            # 构建验证prompt
            docs_text = '\n'.join([doc.get('content', '')[:300] for doc in retrieved_docs[:5]])

            prompt = f"""
请验证以下答案是否与提供的文档内容一致。

查询: {query}
答案: {answer}

参考文档:
{docs_text}

请评估答案与文档的一致性，返回0-1之间的分数。
0表示完全不一致，1表示完全一致。
"""

            response = await llm_service.generate_response(prompt)

            # 提取分数
            score = self._extract_score_from_response(response)
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"事实一致性验证失败: {e}")
            return 0.5

    def _verify_source_citation(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """验证来源引用"""
        if not retrieved_docs:
            return 0.0

        # 检查是否有引用标记
        citation_patterns = [
            r'\[来源\d+\]',
            r'\[文档\d+\]',
            r'according to',
            r'based on',
            r'根据'
        ]

        has_citation = any(re.search(pattern, answer, re.IGNORECASE) for pattern in citation_patterns)

        if has_citation:
            # 检查引用数量
            citations = len(re.findall(r'\[\d+\]', answer))
            citation_score = min(citations / 3.0, 1.0)  # 3个引用为满分
        else:
            # 没有显式引用，检查是否隐式引用
            doc_words = set()
            for doc in retrieved_docs[:3]:
                doc_words.update(doc.get('content', '').lower().split()[:50])

            answer_words = set(answer.lower().split())
            overlap = len(doc_words.intersection(answer_words))
            citation_score = min(overlap / 20.0, 0.5)  # 最多0.5分

        return citation_score

    def _verify_logical_coherence(self, query: str, answer: str) -> float:
        """验证逻辑连贯性"""
        # 简单的逻辑检查
        coherence_indicators = [
            '因此', '所以', '综上所述', '首先', '其次', '最后',
            '另外', '此外', '然而', '但是', '不过'
        ]

        # 检查是否有连接词
        has_connectors = any(indicator in answer for indicator in coherence_indicators)

        # 检查句子结构
        sentences = re.split(r'[。！？]', answer)
        avg_sentence_length = np.mean([len(s) for s in sentences if s])

        # 综合评分
        coherence_score = 0.0
        if has_connectors:
            coherence_score += 0.4
        if 10 <= avg_sentence_length <= 50:
            coherence_score += 0.3
        if len(sentences) >= 2:
            coherence_score += 0.3

        return coherence_score

    def _verify_completeness(self, query: str, answer: str) -> float:
        """验证答案完整性"""
        # 检查答案长度
        min_length = 50
        max_length = 1000
        length_score = 0.0

        if min_length <= len(answer) <= max_length:
            length_score = 1.0
        elif len(answer) < min_length:
            length_score = len(answer) / min_length
        else:
            length_score = max_length / len(answer)

        # 检查是否包含查询关键词
        query_words = set(query.split())
        answer_words = set(answer.split())
        keyword_coverage = len(query_words.intersection(answer_words)) / max(len(query_words), 1)

        # 综合评分
        completeness_score = 0.6 * length_score + 0.4 * keyword_coverage
        return completeness_score

    def _extract_score_from_response(self, response: str) -> float:
        """从响应中提取分数"""
        # 查找数字
        numbers = re.findall(r'0\.\d+|\d+', response)
        for num in numbers:
            try:
                score = float(num)
                if 0 <= score <= 1:
                    return score
            except:
                continue

        # 如果没有找到分数，返回默认值
        return 0.7

    def _generate_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if scores.get('factual_consistency', 0) < 0.7:
            suggestions.append('请确保答案与检索文档内容一致')
        if scores.get('source_citation', 0) < 0.6:
            suggestions.append('请增加具体的来源引用')
        if scores.get('logical_coherence', 0) < 0.6:
            suggestions.append('请改善答案的逻辑结构和连贯性')
        if scores.get('completeness', 0) < 0.6:
            suggestions.append('请提供更完整的答案')

        return suggestions