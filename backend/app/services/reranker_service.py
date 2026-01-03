"""
重排序服务 - 使用LLM对检索结果进行精细化重排
提升Top-K结果的准确性
"""

from app.core.structured_logging import get_structured_logger
import json
from typing import List, Dict, Any

from app.services.llm.unified_llm_service import LLMService

logger = get_structured_logger(__name__)


class RerankerService:
    """重排序服务"""

    def __init__(self):
        self.llm_service = LLMService()

    async def rerank_results(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        top_k: int = 10,
        method: str = "llm"
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 用户查询
            initial_results: 初始检索结果
            top_k: 返回的Top-K数量
            method: 重排序方法 (llm/score_fusion)

        Returns:
            重排序后的结果列表
        """
        if not initial_results:
            return []

        if len(initial_results) <= top_k:
            logger.info(f"结果数量({len(initial_results)}) <= top_k({top_k})，无需重排序")
            return initial_results

        try:
            if method == "llm":
                return await self._llm_rerank(query, initial_results, top_k)
            elif method == "score_fusion":
                return self._score_fusion_rerank(initial_results, top_k)
            else:
                logger.warning(f"未知的重排序方法: {method}，使用LLM方法")
                return await self._llm_rerank(query, initial_results, top_k)

        except Exception as e:
            logger.error(f"重排序失败: {e}，返回原始结果")
            return initial_results[:top_k]

    async def _llm_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        使用LLM进行重排序

        策略：
        1. 将查询和文档片段发送给LLM
        2. 让LLM为每个文档打分（0-100）
        3. 按分数重新排序
        """
        # 限制重排序的数量（避免成本过高）
        max_rerank = 15
        candidates = results[:max_rerank]

        # 构建重排序提示词
        rerank_prompt = self._build_rerank_prompt(query, candidates)

        try:
            # 调用LLM
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0.1,  # 低温度保证稳定性
                max_tokens=500
            )

            content = response.get("content", "")

            # 解析LLM返回的分数
            scores = self._parse_rerank_scores(content)

            if not scores:
                logger.warning("LLM重排序分数解析失败，使用原始排序")
                return results[:top_k]

            # 将分数添加到结果中
            for i, result in enumerate(candidates):
                if i < len(scores):
                    result['rerank_score'] = scores[i]
                    result['rerank_method'] = 'llm'
                else:
                    result['rerank_score'] = 0.0
                    result['rerank_method'] = 'llm'

            # 按重排序分数排序
            reranked = sorted(
                candidates,
                key=lambda x: x.get('rerank_score', 0),
                reverse=True
            )

            logger.info(f"LLM重排序完成，Top-{top_k}分数: {[r.get('rerank_score', 0) for r in reranked[:top_k]]}")

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return results[:top_k]

    def _build_rerank_prompt(self, query: str, candidates: List[Dict]) -> str:
        """构建重排序提示词"""

        prompt = f"""请评估以下文档片段与用户问题的相关度。

用户问题：{query}

文档片段：
"""

        for i, doc in enumerate(candidates, 1):
            content = doc.get('content', '')[:300]  # 限制长度
            title = doc.get('title', doc.get('filename', '未知'))
            prompt += f"\n{i}. 标题：{title}\n   内容：{content}\n"

        prompt += """
评估标准：
- 100分：完全相关，直接回答问题
- 75-99分：高度相关，包含重要信息
- 50-74分：中等相关，部分有用
- 25-49分：低度相关，少量有用
- 0-24分：不相关或基本无用

请仅返回JSON格式的评分，格式如：
{"scores": [85, 92, 45, 78, 60, ...]}

确保：
1. 分数数量与文档数量一致（共%d份文档）
2. 每个分数在0-100之间
3. 仅返回JSON，不要其他文字""" % len(candidates)

        return prompt

    def _parse_rerank_scores(self, content: str) -> List[float]:
        """解析LLM返回的重排序分数"""
        try:
            # 尝试直接解析JSON
            content = content.strip()

            # 移除可能的markdown标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
            elif isinstance(data, list):
                scores = data
            else:
                logger.warning(f"无法识别的返回格式: {type(data)}")
                return []

            # 验证分数
            validated_scores = []
            for score in scores:
                try:
                    s = float(score)
                    if 0 <= s <= 100:
                        validated_scores.append(s)
                    else:
                        logger.warning(f"分数超出范围[0,100]: {s}")
                        validated_scores.append(max(0, min(100, s)))
                except (ValueError, TypeError):
                    logger.warning(f"无效分数: {score}")
                    validated_scores.append(50.0)  # 默认分数

            return validated_scores

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 内容: {content[:200]}")
            return []
        except Exception as e:
            logger.error(f"解析分数失败: {e}")
            return []

    def _score_fusion_rerank(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        基于分数融合的重排序

        策略：
        1. 结合原始相似度分数和其他特征
        2. 计算综合分数
        3. 按综合分数排序
        """
        for result in results:
            # 获取原始分数
            original_score = result.get('score', 0)
            similarity_percentage = result.get('similarity_percentage', 0)

            # 转换为0-1范围
            if similarity_percentage > 1:
                normalized_score = similarity_percentage / 100
            else:
                normalized_score = similarity_percentage

            # 综合特征（可以根据需要添加更多特征）
            # 特征1：原始相似度（权重0.7）
            # 特征2：内容长度适中文档更有价值（权重0.2）
            # 特征3：标题包含关键词（权重0.1）

            content_length_score = self._calculate_length_score(result.get('content', ''))
            title_score = self._calculate_title_score(result.get('title', ''))

            fusion_score = (
                normalized_score * 0.7 +
                content_length_score * 0.2 +
                title_score * 0.1
            )

            result['rerank_score'] = fusion_score
            result['rerank_method'] = 'score_fusion'

        # 按融合分数排序
        reranked = sorted(
            results,
            key=lambda x: x.get('rerank_score', 0),
            reverse=True
        )

        logger.info(f"分数融合重排序完成，Top-{top_k}分数: {[r.get('rerank_score', 0) for r in reranked[:top_k]]}")

        return reranked[:top_k]

    def _calculate_length_score(self, content: str) -> float:
        """
        计算内容长度分数

        策略：适中的长度分数最高
        - 太短（<100字符）：信息不足
        - 适中（100-800字符）：信息密度好
        - 太长（>800字符）：可能冗余
        """
        length = len(content)

        if length < 100:
            return length / 100  # 0-1
        elif length <= 800:
            return 1.0  # 最优
        else:
            return max(0.5, 1.0 - (length - 800) / 2000)  # 逐渐降低

    def _calculate_title_score(self, title: str) -> float:
        """
        计算标题质量分数

        策略：
        - 有标题 > 无标题
        - 标题长度适中 > 太短或太长
        """
        if not title or title == 'N/A':
            return 0.3

        length = len(title)

        if 5 <= length <= 50:
            return 1.0
        elif length < 5:
            return 0.6
        else:
            return max(0.7, 1.0 - (length - 50) / 100)

    async def rerank_with_filtering(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        min_score: float = 0.5,
        diversity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        带过滤和多样性的重排序

        Args:
            query: 用户查询
            results: 检索结果
            top_k: 返回数量
            min_score: 最低分数阈值
            diversity: 是否增加多样性（MMR算法）

        Returns:
            重排序并过滤后的结果
        """
        # 先进行基础重排序
        reranked = await self.rerank_results(query, results, top_k * 2, method="score_fusion")

        # 过滤低分结果
        filtered = [r for r in reranked if r.get('rerank_score', 0) >= min_score]

        if not filtered:
            logger.warning(f"所有结果分数都低于阈值{min_score}，返回Top-{top_k}")
            return reranked[:top_k]

        # 如果需要多样性，使用MMR
        if diversity and len(filtered) > 1:
            filtered = self._apply_mmr(filtered, top_k, lambda_param=0.5)

        return filtered[:top_k]

    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        应用MMR（Maximal Marginal Relevance）算法增加多样性

        Args:
            results: 重排序后的结果
            top_k: 返回数量
            lambda_param: 平衡相关性和多样性的参数
                          - 1.0: 只考虑相关性
                          - 0.5: 平衡相关性和多样性
                          - 0.0: 只考虑多样性

        Returns:
            增加多样性后的结果
        """
        if not results:
            return []

        selected = []
        remaining = results.copy()

        # 选择第一个（最相关的）
        if remaining:
            selected.append(remaining.pop(0))

        # 迭代选择剩余的
        while len(selected) < min(top_k, len(results)) and remaining:
            best_idx = 0
            best_score = -1

            for i, candidate in enumerate(remaining):
                # 计算相关性分数
                relevance = candidate.get('rerank_score', 0)

                # 计算与已选择结果的最大相似度（简化版：基于内容长度和标题）
                max_similarity = 0
                for sel in selected:
                    # 简化的相似度计算
                    sim = self._compute_simple_similarity(candidate, sel)
                    max_similarity = max(max_similarity, sim)

                # MMR分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # 移动到已选择
            selected.append(remaining.pop(best_idx))

        return selected

    def _compute_simple_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """
        简化的文档相似度计算
        在实际应用中，可以使用embeddings计算cosine similarity
        """
        # 基于标题的简单相似度
        title1 = doc1.get('title', '').lower()
        title2 = doc2.get('title', '').lower()

        if not title1 or not title2:
            return 0.0

        # 简单的Jaccard相似度
        words1 = set(title1.split())
        words2 = set(title2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# 全局实例
_reranker_service = None


def get_reranker_service() -> RerankerService:
    """获取重排序服务实例"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
