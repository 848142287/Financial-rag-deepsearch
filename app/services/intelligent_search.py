"""
智能搜索服务
基于文档元数据的语义搜索和问答功能
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from ..models.document import Document
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentSearchService:
    """智能搜索服务"""

    def __init__(self, db: Session):
        self.db = db
        self.concept_expansions = {
            'MACD': ['MACD', '平滑异同移动平均线', '指数平滑移动平均线', '趋势指标', '技术分析'],
            '量化': ['量化', '量化投资', '程序化交易', '算法交易', '数学模型', '统计套利', '数量化'],
            '择时': ['择时', '时机选择', '入场时机', '出场时机', '买卖点', '市场时机'],
            '多因子': ['多因子', '多因子模型', '因子投资', '多因子选股', '因子分析', '因子选股'],
            '风险管理': ['风险', '风险控制', '止损', '资金管理', '仓位管理', '风险控制'],
            '资产配置': ['资产配置', '投资组合', '组合优化', '大类资产', '投资组合'],
            '技术分析': ['技术分析', '技术指标', '图表分析', '趋势分析', '技术指标'],
            '基本面分析': ['基本面', '价值投资', '估值分析', '财务分析', '基本面分析'],
            '市场情绪': ['市场情绪', '情绪指标', '情绪分析', '投资者情绪', '市场情绪'],
            '行业配置': ['行业配置', '行业轮动', '行业选择', '行业配置', '行业轮动']
        }

        self.brokers = [
            '国泰君安', '国信证券', '华泰证券', '招商证券', '中信证券', '海通证券',
            '申万宏源', '中金公司', '广发证券', '安信证券', '光大证券', '东方证券',
            '兴业证券', '华安证券', '民生证券', '平安证券', '国金证券', '华西证券',
            '中泰证券', '长江证券', '海通期货', '国盛证券', '中银证券', '华西证券',
            '国盛证券', '中银证券'
        ]

        self.technical_indicators = [
            'MACD', 'RSI', 'KDJ', 'BOLL', '布林带', '均线', 'MA', 'EMA', 'SMA', 'WMA',
            'KELLY', 'K线', '成交量', '量价', '换手率', '市盈率', 'PE', 'PB', '市净率',
            '动量指标', '超买超卖', '支撑阻力', '趋势指标', 'EMS', 'TRIX', 'RSRS'
        ]

    def intelligent_search(self, query: str, limit: int = 10, search_type: str = "semantic") -> Dict[str, Any]:
        """
        智能搜索主入口

        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            search_type: 搜索类型 (semantic, keyword, hybrid)

        Returns:
            搜索结果
        """
        try:
            logger.info(f"智能搜索: '{query}', 类型: {search_type}")

            # 分析查询意图
            intent_analysis = self._analyze_query_intent(query)

            # 执行搜索
            if search_type == "semantic":
                results = self._semantic_search(query, intent_analysis, limit)
            elif search_type == "keyword":
                results = self._keyword_search(query, limit)
            else:  # hybrid
                results = self._hybrid_search(query, intent_analysis, limit)

            # 生成搜索摘要
            summary = self._generate_search_summary(query, results, intent_analysis)

            return {
                "query": query,
                "search_type": search_type,
                "intent_analysis": intent_analysis,
                "results_count": len(results),
                "results": results,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            return {
                "query": query,
                "search_type": search_type,
                "intent_analysis": {},
                "results_count": 0,
                "results": [],
                "summary": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """分析查询意图"""
        intent = {
            "query_type": "general",
            "entities": [],
            "concepts": [],
            "brokers": [],
            "indicators": [],
            "time_period": None,
            "question_type": None
        }

        query_lower = query.lower()

        # 识别查询类型
        if any(word in query_lower for word in ['如何', '怎么', '方法', '步骤']):
            intent["query_type"] = "how_to"
            intent["question_type"] = "method"
        elif any(word in query_lower for word in ['什么', '定义', '解释', '意思']):
            intent["query_type"] = "what_is"
            intent["question_type"] = "definition"
        elif any(word in query_lower for word in ['为什么', '原因', '原理', '作用']):
            intent["query_type"] = "why"
            intent["question_type"] = "explanation"
        elif any(word in query_lower for word in ['哪个', '哪家', '谁', '推荐']):
            intent["query_type"] = "which"
            intent["question_type"] = "comparison"
        elif any(word in query_lower for word in ['比较', '对比', '区别', '优缺点']):
            intent["query_type"] = "comparison"
            intent["question_type"] = "comparison"

        # 识别券商
        for broker in self.brokers:
            if broker in query:
                intent["brokers"].append(broker)

        # 识别技术指标
        for indicator in self.technical_indicators:
            if indicator.lower() in query_lower or indicator in query:
                intent["indicators"].append(indicator)

        # 识别概念
        for concept, expansions in self.concept_expansions.items():
            if concept.lower() in query_lower or any(exp.lower() in query_lower for exp in expansions):
                intent["concepts"].append(concept)

        # 识别时间周期
        time_patterns = [
            r'(\d{4})年', r'(\d{4})-(\d{1,2})', r'(\d{4})\d{2}\d{2}',
            r'最近(\d+)', r'过去(\d+)', r'(\d+)年以来'
        ]

        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                intent["time_period"] = match.group(0)
                break

        return intent

    def _semantic_search(self, query: str, intent: Dict, limit: int) -> List[Dict]:
        """语义搜索"""
        documents = self.db.query(Document).all()
        scored_docs = []

        for doc in documents:
            score = self._calculate_semantic_score(doc, query, intent)
            if score > 0:
                scored_docs.append((doc, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 格式化结果
        results = []
        for doc, score in scored_docs[:limit]:
            result = self._format_search_result(doc, score, "semantic")
            results.append(result)

        return results

    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """关键词搜索"""
        documents = self.db.query(Document).all()
        scored_docs = []

        query_keywords = query.lower().split()

        for doc in documents:
            score = self._calculate_keyword_score(doc, query_keywords)
            if score > 0:
                scored_docs.append((doc, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 格式化结果
        results = []
        for doc, score in scored_docs[:limit]:
            result = self._format_search_result(doc, score, "keyword")
            results.append(result)

        return results

    def _hybrid_search(self, query: str, intent: Dict, limit: int) -> List[Dict]:
        """混合搜索（语义+关键词）"""
        # 获取语义搜索结果
        semantic_results = self._semantic_search(query, intent, limit * 2)

        # 获取关键词搜索结果
        keyword_results = self._keyword_search(query, limit * 2)

        # 合并和去重
        combined_results = {}

        # 添加语义搜索结果（权重0.6）
        for result in semantic_results:
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["score"] = result["score"] * 0.6
            else:
                combined_results[doc_id]["score"] += result["score"] * 0.6

        # 添加关键词搜索结果（权重0.4）
        for result in keyword_results:
            doc_id = result["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["score"] = result["score"] * 0.4
                combined_results[doc_id]["search_type"] = "keyword"
            else:
                combined_results[doc_id]["score"] += result["score"] * 0.4
                combined_results[doc_id]["search_type"] = "hybrid"

        # 按分数排序并限制数量
        final_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)

        return final_results[:limit]

    def _calculate_semantic_score(self, doc: Document, query: str, intent: Dict) -> float:
        """计算语义搜索分数"""
        score = 0.0
        query_lower = query.lower()

        # 基础文本匹配
        searchable_text = f"{doc.title} {doc.file_name} {doc.description or ''}".lower()

        # 概念匹配
        for concept in intent.get("concepts", []):
            if concept.lower() in searchable_text:
                score += 8.0
            # 概念扩展匹配
            expansions = self.concept_expansions.get(concept, [])
            for exp in expansions:
                if exp.lower() in searchable_text:
                    score += 4.0

        # 券商匹配
        for broker in intent.get("brokers", []):
            if broker.lower() in searchable_text:
                score += 10.0

        # 技术指标匹配
        for indicator in intent.get("indicators", []):
            if indicator.lower() in searchable_text or indicator in searchable_text:
                score += 6.0

        # 标题权重
        if any(word in doc.title.lower() for word in query_lower.split()):
            score += 5.0

        # 部分匹配
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in searchable_text)
        if matches > 0:
            score += matches * 1.5

        return score

    def _calculate_keyword_score(self, doc: Document, keywords: List[str]) -> float:
        """计算关键词搜索分数"""
        score = 0.0
        searchable_text = f"{doc.title} {doc.file_name} {doc.description or ''}".lower()

        for keyword in keywords:
            if keyword in searchable_text:
                # 标题完全匹配
                if keyword in doc.title.lower():
                    score += 10.0
                # 文件名匹配
                elif keyword in doc.file_name.lower():
                    score += 5.0
                # 描述匹配
                elif keyword in (doc.description or "").lower():
                    score += 3.0
                # 其他匹配
                else:
                    score += 1.0

        return score

    def _format_search_result(self, doc: Document, score: float, search_type: str) -> Dict[str, Any]:
        """格式化搜索结果"""
        # 提取文档关键信息
        metadata = self._extract_document_metadata(doc)

        return {
            "id": doc.id,
            "title": doc.title,
            "file_name": doc.filename,
            "description": getattr(doc, 'description', ''),
            "score": round(score, 2),
            "search_type": search_type,
            "category": getattr(doc, 'category', ''),
            "tags": getattr(doc, 'tags', []),
            "file_size": doc.file_size,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "metadata": metadata,
            "preview": self._generate_preview(doc)
        }

    def _extract_document_metadata(self, doc: Document) -> Dict[str, Any]:
        """提取文档元数据"""
        full_text = f"{doc.title} {doc.filename} {doc.description or ''}"

        # 识别券商
        broker = next((b for b in self.brokers if b in full_text), None)

        # 识别技术指标
        indicators = [i for i in self.technical_indicators if i.lower() in full_text.lower() or i in full_text]

        # 识别概念
        concepts = []
        for concept, expansions in self.concept_expansions.items():
            if concept.lower() in full_text.lower() or any(exp.lower() in full_text.lower() for exp in expansions):
                concepts.append(concept)

        # 提取日期
        date_match = re.search(r'(\d{4})\D*(\d{1,2})?\D*(\d{1,2})?', full_text)
        date = date_match.group(0) if date_match else None

        return {
            "broker": broker,
            "indicators": indicators[:5],  # 限制数量
            "concepts": concepts[:5],      # 限制数量
            "date": date,
            "file_type": doc.file_type,
            "content_type": doc.content_type
        }

    def _generate_preview(self, doc: Document) -> str:
        """生成文档预览"""
        preview_parts = []

        if doc.title:
            preview_parts.append(f"标题: {doc.title}")

        if doc.filename:
            preview_parts.append(f"文件: {doc.filename}")

        if doc.description:
            preview_parts.append(f"描述: {doc.description}")

        # 添加关键信息
        metadata = self._extract_document_metadata(doc)
        if metadata.get("broker"):
            preview_parts.append(f"券商: {metadata['broker']}")

        if metadata.get("concepts"):
            preview_parts.append(f"概念: {', '.join(metadata['concepts'][:3])}")

        if metadata.get("indicators"):
            preview_parts.append(f"技术指标: {', '.join(metadata['indicators'][:3])}")

        return " | ".join(preview_parts)

    def _generate_search_summary(self, query: str, results: List[Dict], intent: Dict) -> Dict[str, Any]:
        """生成搜索摘要"""
        summary = {
            "query_intent": intent.get("query_type", "general"),
            "found_entities": {
                "brokers": [],
                "concepts": [],
                "indicators": []
            },
            "result_statistics": {
                "total_results": len(results),
                "avg_score": 0.0,
                "score_distribution": {"high": 0, "medium": 0, "low": 0}
            },
            "recommendations": []
        }

        if not results:
            return summary

        # 统计找到的实体
        all_brokers = set()
        all_concepts = set()
        all_indicators = set()

        for result in results:
            metadata = result.get("metadata", {})
            if metadata.get("broker"):
                all_brokers.add(metadata["broker"])
            all_concepts.update(metadata.get("concepts", []))
            all_indicators.update(metadata.get("indicators", []))

        summary["found_entities"]["brokers"] = list(all_brokers)[:5]
        summary["found_entities"]["concepts"] = list(all_concepts)[:5]
        summary["found_entities"]["indicators"] = list(all_indicators)[:5]

        # 计算分数统计
        scores = [r["score"] for r in results]
        summary["result_statistics"]["avg_score"] = round(sum(scores) / len(scores), 2)

        for score in scores:
            if score >= 15:
                summary["result_statistics"]["score_distribution"]["high"] += 1
            elif score >= 8:
                summary["result_statistics"]["score_distribution"]["medium"] += 1
            else:
                summary["result_statistics"]["score_distribution"]["low"] += 1

        # 生成建议
        if len(results) < 5:
            summary["recommendations"].append("结果较少，建议使用更通用的关键词")
        elif len(results) > 50:
            summary["recommendations"].append("结果较多，建议添加更多筛选条件")

        if intent.get("concepts"):
            summary["recommendations"].append(f"检测到投资概念，您可以进一步探索相关的{', '.join(intent['concepts'])}")

        return summary

    def get_search_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """获取搜索建议"""
        suggestions = []
        query_lower = query.lower()

        # 概念建议
        for concept, expansions in self.concept_expansions.items():
            if concept.lower().startswith(query_lower):
                suggestions.append(concept)
            for exp in expansions:
                if exp.lower().startswith(query_lower) and len(suggestions) < limit:
                    suggestions.append(exp)

        # 券商建议
        for broker in self.brokers:
            if broker.startswith(query) and len(suggestions) < limit:
                suggestions.append(broker)

        # 技术指标建议
        for indicator in self.technical_indicators:
            if indicator.lower().startswith(query_lower) and len(suggestions) < limit:
                suggestions.append(indicator)

        return suggestions[:limit]