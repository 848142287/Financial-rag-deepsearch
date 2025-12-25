"""
增强版答案生成服务
提供结构化、高质量的答案生成
"""

import logging
from typing import Dict, Any, List
import json

from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class EnhancedAnswerService:
    """增强版答案生成服务"""
    
    def __init__(self):
        self.llm_service = LLMService()
    
    async def generate_structured_answer(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]],
        normalize_scores: bool = True,
        score_stats: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        生成结构化答案
        
        Args:
            query: 用户问题
            search_results: 检索结果列表
            normalize_scores: 是否归一化相似度分数
            score_stats: 分数统计信息 {min, max, avg}
            
        Returns:
            结构化答案字典
        """
        if not search_results:
            return {
                "summary": "抱歉，没有找到相关文档。",
                "main_points": [],
                "detailed_explanation": "",
                "examples": [],
                "sources": [],
                "confidence": 0
            }
        
        # 归一化分数
        normalized_results = self._normalize_scores(
            search_results, 
            normalize_scores, 
            score_stats
        )
        
        # 构建上下文
        context = self._build_context(query, normalized_results)
        
        # 生成结构化答案
        structured_answer = await self._generate_structured_answer_with_llm(
            query, context, normalized_results
        )
        
        # 添加元数据
        structured_answer["sources"] = self._format_sources(normalized_results)
        structured_answer["confidence"] = self._calculate_confidence(normalized_results)
        structured_answer["total_sources"] = len(normalized_results)
        
        return structured_answer
    
    def _normalize_scores(
        self, 
        results: List[Dict], 
        enable: bool = True,
        stats: Dict[str, float] = None
    ) -> List[Dict]:
        """归一化相似度分数到0-1范围"""
        if not enable or not results:
            return results
        
        # 计算统计信息
        if stats is None:
            scores = [r.get('score', 0) for r in results]
            stats = {
                'min': min(scores),
                'max': max(scores),
                'avg': sum(scores) / len(scores)
            }
        
        # 归一化
        normalized = []
        score_range = stats['max'] - stats['min']
        
        for result in results:
            result_copy = result.copy()
            original_score = result.get('score', 0)
            
            if score_range > 0:
                normalized_score = (original_score - stats['min']) / score_range
            else:
                normalized_score = 1.0
            
            result_copy['original_score'] = original_score
            result_copy['normalized_score'] = round(normalized_score, 4)
            result_copy['similarity_percentage'] = round(normalized_score * 100, 2)
            
            normalized.append(result_copy)
        
        return normalized
    
    def _build_context(self, query: str, results: List[Dict]) -> str:
        """构建LLM上下文"""
        context_parts = []
        
        for i, result in enumerate(results[:5], 1):  # 只使用前5个结果
            score = result.get('similarity_percentage', result.get('score', 0))
            content = result.get('content', '')[:800]  # 限制长度
            
            context_part = f"""
【文档片段{i}】(相关度: {score}%)
标题: {result.get('title', 'N/A')}
内容: {content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _generate_structured_answer_with_llm(
        self, 
        query: str, 
        context: str, 
        results: List[Dict]
    ) -> Dict[str, Any]:
        """使用LLM生成结构化答案"""
        
        system_prompt = """你是一个专业的金融研究助手。基于提供的文档内容，生成结构化的答案。

你的回答必须包含以下部分：
1. summary: 一句话简洁总结答案（不超过50字）
2. main_points: 3-5个关键要点（每个要点不超过100字）
3. detailed_explanation: 详细解释（300-500字）
4. examples: 相关示例（如果有）

要求：
- 严格基于提供的文档内容
- 答案要准确、专业、有条理
- 如果文档中没有相关信息，明确说明
- 使用中文回答
- 以JSON格式返回"""

        user_prompt = f"""基于以下文档内容回答问题：

问题: {query}

文档内容:
{context}

请以JSON格式返回结构化答案，格式如下：
{{
    "summary": "一句话总结",
    "main_points": ["要点1", "要点2", "要点3"],
    "detailed_explanation": "详细解释...",
    "examples": ["示例1", "示例2"]
}}"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                use_qwen=False
            )
            
            content = response.get("content", "").strip()
            
            # 解析JSON
            try:
                # 提取JSON部分
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed_answer = json.loads(content)
                
                # 确保所有字段都存在
                return {
                    "summary": parsed_answer.get("summary", "无法生成摘要"),
                    "main_points": parsed_answer.get("main_points", []),
                    "detailed_explanation": parsed_answer.get("detailed_explanation", ""),
                    "examples": parsed_answer.get("examples", [])
                }
            except json.JSONDecodeError as e:
                logger.error(f"解析LLM JSON响应失败: {e}, 原始内容: {content}")
                # 降级到简单格式
                return {
                    "summary": content[:100] if len(content) > 100 else content,
                    "main_points": [content],
                    "detailed_explanation": content,
                    "examples": []
                }
        
        except Exception as e:
            logger.error(f"LLM生成答案失败: {e}")
            # 降级：从检索结果中提取信息
            return self._fallback_answer(query, results)
    
    def _fallback_answer(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """降级答案生成（不使用LLM）"""
        top_result = results[0]
        content = top_result.get('content', '')
        
        return {
            "summary": f"找到相关文档: {top_result.get('title', 'N/A')}",
            "main_points": [
                f"最相关文档: {top_result.get('title', 'N/A')}",
                f"相似度: {top_result.get('similarity_percentage', top_result.get('score', 0))}%",
                f"共找到 {len(results)} 个相关片段"
            ],
            "detailed_explanation": content[:500],
            "examples": []
        }
    
    def _format_sources(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """格式化来源信息"""
        sources = []
        
        for result in results[:10]:  # 最多返回10个来源
            source = {
                "document_id": result.get('id', 'N/A'),
                "title": result.get('title', 'N/A'),
                "content_preview": result.get('content', '')[:200],
                "original_score": result.get('original_score', result.get('score', 0)),
                "normalized_score": result.get('normalized_score', 0),
                "similarity_percentage": result.get('similarity_percentage', 0)
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """计算整体置信度"""
        if not results:
            return 0.0
        
        # 基于归一化分数计算
        normalized_scores = [
            r.get('normalized_score', r.get('score', 0)) 
            for r in results
        ]
        
        if not normalized_scores:
            return 0.0
        
        # 使用Top-3分数的平均值作为置信度
        top_scores = sorted(normalized_scores, reverse=True)[:3]
        confidence = sum(top_scores) / len(top_scores) * 100
        
        return round(confidence, 2)
