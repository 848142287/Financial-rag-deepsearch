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
        """使用LLM生成结构化答案 - 优化版提示词"""

        # 优化的系统提示词 - 更明确和结构化
        system_prompt = """你是一个资深的金融领域研究专家和知识分析师，具备以下核心能力：

## 专业领域
- 金融数据分析与解读
- 研究报告深度分析
- 实体关系推理与知识图谱应用
- 多源信息综合与提炼

## 回答原则
1. **准确性优先**: 仅基于提供的文档内容回答，不编造信息
2. **结构清晰**: 使用层次化结构组织信息
3. **可追溯性**: 明确标注信息来源
4. **深度分析**: 不仅陈述事实，更要分析关联和影响
5. **简明扼要**: 在保证完整性的前提下避免冗余

## 输出格式要求
你必须以JSON格式返回，包含以下字段：

1. **summary** (string): 核心答案一句话总结（不超过80字）
   - 直接回答问题的核心
   - 包含关键数字或结论

2. **main_points** (array): 3-5个关键要点
   - 每个要点独立完整
   - 按重要性排序
   - 每点不超过120字

3. **detailed_explanation** (string): 深度分析
   - 背景说明（如有必要）
   - 详细的数据分析
   - 实体间的关联分析
   - 趋势或影响说明
   - 建议或启示（如适用）
   - 400-600字

4. **examples** (array): 具体案例或数据
   - 从文档中提取的实际例子
   - 关键数据点
   - 对比分析

## 质量标准
- 如果文档信息不足，明确说明缺失的部分
- 使用专业但易懂的语言
- 保持客观中立的立场
- 利用知识图谱中的实体关系进行深度推理"""

        # 优化的用户提示词 - 更好的上下文组织
        user_prompt = f"""# 查询任务

## 用户问题
{query}

## 参考文档资料
以下是与问题相关的文档片段（按相关度排序）：

{context}

## 分析要求
1. 仔细阅读所有文档片段
2. 识别关键实体、数据和关系
3. 综合多个文档的信息进行分析
4. 提炼核心答案并提供深度见解
5. 如果信息矛盾，说明不同来源

## 输出要求
请以严格的JSON格式返回答案，确保：
- JSON格式正确，可被解析
- 所有必需字段都存在
- 内容基于提供的文档
- 使用中文回答

JSON格式示例：
{{
    "summary": "核心答案的精炼总结",
    "main_points": [
        "关键要点1，包含具体数据",
        "关键要点2，说明重要关联",
        "关键要点3，提供深度洞察"
    ],
    "detailed_explanation": "详细的分析说明，包括背景、数据分析、实体关系、影响评估等",
    "examples": [
        "具体案例或数据点1",
        "具体案例或数据点2"
    ]
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
