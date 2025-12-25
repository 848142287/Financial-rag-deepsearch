"""
查询理解模块
实现意图分类、实体识别、查询扩展等功能
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import logging

from ..knowledge_base.entity_extractor import financial_entity_extractor
from ..llm_service import llm_service
from ..financial_llm_service import financial_llm_service, FinancialTaskType, ModelType
from .agent_engine import QueryType

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """查询意图"""
    primary_intent: QueryType
    confidence: float
    secondary_intents: List[QueryType]
    time_range: Optional[Dict[str, str]] = None
    comparison_entities: Optional[List[str]] = None
    analysis_type: Optional[str] = None


@dataclass
class QueryUnderstanding:
    """查询理解结果"""
    original_query: str
    clean_query: str
    intent: QueryIntent
    entities: List[Dict[str, Any]]
    keywords: List[str]
    query_complexity: str  # simple, moderate, complex
    required_capabilities: List[str]
    expanded_queries: List[str]
    metadata: Dict[str, Any]


class QueryUnderstanding:
    """查询理解器"""

    def __init__(self):
        # 意图识别关键词
        self.intent_keywords = {
            QueryType.FACTUAL: ['是什么', '哪个', '谁', '哪里', '何时', '定义'],
            QueryType.ANALYTICAL: ['分析', '评估', '影响', '趋势', '原因', '为什么'],
            QueryType.COMPARATIVE: ['比较', '对比', '差异', '优劣', '哪个好', 'vs'],
            QueryType.TEMPORAL: ['最近', '历史', '趋势', '变化', '预测', '未来'],
            QueryType.CAUSAL: ['导致', '影响', '原因', '因为', '所以', '结果'],
            QueryType.AGGREGATE: ['总计', '平均', '统计', '汇总', '总数', '占比']
        }

        # 时间表达式
        self.time_patterns = [
            r'(\d{4}年?)',
            r'(\d{1,2}月)',
            r'最近(\d+)(年|季度|月)',
            r'(今年|去年|前年)',
            r'(本季度|上季度)',
            r'(Q[1-4])\s*(\d{4})?',
            r'(上半年|下半年)'
        ]

        # 金融分析类型
        self.analysis_types = {
            '财务分析': ['财务', '财报', '利润', '收入', '成本'],
            '市场分析': ['市场', '股价', '市值', '涨跌幅', '成交量'],
            '行业分析': ['行业', '竞争', '市场地位', '份额'],
            '风险分析': ['风险', '波动', '违约', '评级'],
            '投资分析': ['投资', '回报', '收益', '估值']
        }

    async def understand(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryUnderstanding:
        """
        理解查询

        Args:
            query: 原始查询
            context: 查询上下文

        Returns:
            查询理解结果
        """
        logger.info(f"开始查询理解: {query}")

        try:
            # 清理查询
            clean_query = self._clean_query(query)

            # 使用金融模型进行增强意图识别
            intent = await self._enhanced_identify_intent(clean_query)

            # 使用金融模型进行实体抽取
            entities = await self._enhanced_entity_extraction(clean_query)

            # 使用金融模型进行关键词提取
            keywords = await self._enhanced_keyword_extraction(clean_query)

            # 查询复杂度评估
            complexity = self._assess_complexity(clean_query, intent, entities)

            # 能力需求分析
            required_capabilities = self._analyze_capabilities(intent, entities, complexity)

            # 查询扩展
            expanded_queries = await self._enhanced_query_expansion(clean_query, intent, entities)

            # 构建理解结果
            understanding = QueryUnderstanding(
                original_query=query,
                clean_query=clean_query,
                intent=intent,
                entities=entities,
                keywords=keywords,
                query_complexity=complexity,
                required_capabilities=required_capabilities,
                expanded_queries=expanded_queries,
                metadata={
                    'context': context,
                    'processed_at': datetime.utcnow().isoformat(),
                    'financial_model_used': True
                }
            )

            logger.info(f"查询理解完成: 意图={intent.primary_intent.value}, 复杂度={complexity}")

            return understanding

        except Exception as e:
            logger.error(f"查询理解失败: {e}")
            # 返回默认理解结果
            return QueryUnderstanding(
                original_query=query,
                clean_query=query,
                intent=QueryIntent(
                    primary_intent=QueryType.FACTUAL,
                    confidence=0.5,
                    secondary_intents=[]
                ),
                entities=[],
                keywords=self._extract_keywords(query),
                query_complexity='simple',
                required_capabilities=['basic_retrieval'],
                expanded_queries=[],
                metadata={'error': str(e)}
            )

    def _clean_query(self, query: str) -> str:
        """清理查询文本"""
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query.strip())

        # 去除特殊字符
        query = re.sub(r'[^\w\s\u4e00-\u9fff.,?!:;()（）。，？！：；]', ' ', query)

        # 统一标点符号
        replacements = {
            '？': '?',
            '！': '!',
            '：': ':',
            '；': ';',
            '，': ',',
            '。': '.'
        }
        for old, new in replacements.items():
            query = query.replace(old, new)

        return query.strip()

    async def _identify_intent(self, query: str) -> QueryIntent:
        """识别查询意图"""
        intent_scores = {}

        # 基于关键词的意图识别
        for intent_type, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            intent_scores[intent_type] = score / len(keywords)

        # 选择主要意图
        if not any(intent_scores.values()):
            # 使用LLM进行意图识别
            primary_intent = await self._llm_intent_classification(query)
            confidence = 0.7
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[primary_intent] * 2, 1.0)

        # 识别次要意图
        secondary_intents = [
            intent for intent, score in intent_scores.items()
            if intent != primary_intent and score > 0.1
        ]

        # 时间范围识别
        time_range = self._extract_time_range(query)

        # 比较实体识别
        comparison_entities = self._extract_comparison_entities(query)

        # 分析类型识别
        analysis_type = self._identify_analysis_type(query)

        return QueryIntent(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents[:2],  # 最多2个次要意图
            time_range=time_range,
            comparison_entities=comparison_entities,
            analysis_type=analysis_type
        )

    async def _llm_intent_classification(self, query: str) -> QueryType:
        """使用LLM进行意图分类"""
        prompt = f"""
请分析以下查询的主要意图类型。

查询: {query}

意图类型:
1. factual - 事实查询（是什么、哪个、定义等）
2. analytical - 分析查询（分析、评估、趋势等）
3. comparative - 比较查询（比较、对比、差异等）
4. temporal - 时间序列查询（历史、趋势、预测等）
5. causal - 因果查询（原因、影响、导致等）
6. aggregate - 聚合查询（总计、平均、统计等）

请只返回对应的类型名称（如: factual）。
"""

        try:
            response = await llm_service.generate_response(prompt)
            response = response.strip().lower()

            # 映射响应到枚举
            intent_mapping = {
                'factual': QueryType.FACTUAL,
                'analytical': QueryType.ANALYTICAL,
                'comparative': QueryType.COMPARATIVE,
                'temporal': QueryType.TEMPORAL,
                'causal': QueryType.CAUSAL,
                'aggregate': QueryType.AGGREGATE
            }

            return intent_mapping.get(response, QueryType.FACTUAL)

        except Exception as e:
            logger.error(f"LLM意图分类失败: {e}")
            return QueryType.FACTUAL

    def _extract_time_range(self, query: str) -> Optional[Dict[str, str]]:
        """提取时间范围"""
        time_info = {}

        for pattern in self.time_patterns:
            matches = re.findall(pattern, query)
            if matches:
                if '年' in pattern:
                    time_info['year'] = matches[0]
                elif '月' in pattern:
                    time_info['month'] = matches[0]
                elif '最近' in pattern:
                    time_info['recent'] = f"{matches[0][0]}{matches[0][1]}"

        return time_info if time_info else None

    def _extract_comparison_entities(self, query: str) -> List[str]:
        """提取比较实体"""
        comparison_indicators = ['vs', '对比', '比较', '和', '与', '及']
        entities = []

        # 简单实现：基于分隔符
        for indicator in comparison_indicators:
            if indicator in query:
                parts = query.split(indicator)
                if len(parts) >= 2:
                    # 提取实体名称
                    for part in parts:
                        words = part.strip().split()
                        if words:
                            # 通常实体是第一个或最后一个词
                            entities.append(words[0])
                            if len(words) > 1:
                                entities.append(words[-1])

        return list(set(entities))  # 去重

    def _identify_analysis_type(self, query: str) -> Optional[str]:
        """识别分析类型"""
        for analysis_type, keywords in self.analysis_types.items():
            if any(keyword in query for keyword in keywords):
                return analysis_type
        return None

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 移除停用词
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was',
            'were', 'be', 'have', 'had', 'do', 'does', 'did', 'will', 'would'
        }

        words = []
        for word in query.split():
            word = word.strip('.,?!:;()（）。，？！：；')
            if (len(word) > 1 and word.lower() not in stop_words and
                not word.isdigit() and not re.match(r'^[^\w\u4e00-\u9fff]+$', word)):
                words.append(word)

        return words

    def _assess_complexity(
        self,
        query: str,
        intent: QueryIntent,
        entities: List
    ) -> str:
        """评估查询复杂度"""
        complexity_score = 0

        # 基于查询长度
        if len(query) > 50:
            complexity_score += 1
        if len(query) > 100:
            complexity_score += 1

        # 基于意图类型
        if intent.primary_intent in [QueryType.ANALYTICAL, QueryType.COMPARATIVE, QueryType.CAUSAL]:
            complexity_score += 2

        # 基于实体数量
        if len(entities) > 3:
            complexity_score += 1
        if len(entities) > 5:
            complexity_score += 1

        # 基于时间范围
        if intent.time_range:
            complexity_score += 1

        # 基于比较实体
        if intent.comparison_entities:
            complexity_score += len(intent.comparison_entities)

        # 分类
        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 5:
            return 'moderate'
        else:
            return 'complex'

    def _analyze_capabilities(
        self,
        intent: QueryIntent,
        entities: List,
        complexity: str
    ) -> List[str]:
        """分析所需能力"""
        capabilities = ['basic_retrieval']

        # 基于意图
        if intent.primary_intent in [QueryType.ANALYTICAL, QueryType.CAUSAL]:
            capabilities.append('reasoning')
        if intent.primary_intent == QueryType.COMPARATIVE:
            capabilities.append('comparison')
        if intent.primary_intent == QueryType.TEMPORAL:
            capabilities.append('temporal_analysis')
        if intent.primary_intent == QueryType.AGGREGATE:
            capabilities.append('aggregation')

        # 基于复杂度
        if complexity == 'complex':
            capabilities.extend(['multi_hop', 'synthesis'])
        elif complexity == 'moderate':
            capabilities.append('synthesis')

        # 基于实体
        if entities:
            capabilities.append('entity_linking')

        return list(set(capabilities))

    async def _expand_query(
        self,
        query: str,
        intent: QueryIntent,
        entities: List
    ) -> List[str]:
        """扩展查询"""
        expansions = []

        # 基于实体扩展
        if entities:
            entity_names = [e.text for e in entities if e.type in ['COMPANY', 'PERSON']]
            if entity_names:
                expansions.append(f"{query} {' '.join(entity_names)}")

        # 基于意图扩展
        if intent.primary_intent == QueryType.ANALYTICAL:
            expansions.append(f"{query} 分析报告")
            expansions.append(f"{query} 评估")
        elif intent.primary_intent == QueryType.COMPARATIVE:
            expansions.append(f"{query} 对比分析")
        elif intent.primary_intent == QueryType.TEMPORAL:
            expansions.append(f"{query} 历史数据")
            expansions.append(f"{query} 趋势分析")

        # 基于分析类型扩展
        if intent.analysis_type:
            expansions.append(f"{query} {intent.analysis_type}")

        # 使用LLM生成扩展
        try:
            llm_expansions = await self._llm_query_expansion(query, intent)
            expansions.extend(llm_expansions)
        except Exception as e:
            logger.error(f"LLM查询扩展失败: {e}")

        return list(set(expansions))[:5]  # 最多5个扩展查询

    async def _enhanced_identify_intent(self, query: str) -> QueryIntent:
        """使用金融模型进行增强意图识别"""
        try:
            # 首先使用传统方法进行意图识别
            intent = await self._identify_intent(query)

            # 使用金融模型进行情感分析，增强意图理解
            sentiment_result = await financial_llm_service.analyze_sentiment(
                query, ModelType.FINBERT
            )

            # 根据情感调整意图置信度
            if sentiment_result.confidence > 0.7:
                if sentiment_result.result.sentiment in ["positive", "negative"]:
                    # 情感明确的查询可能与分析相关
                    if intent.primary_intent == QueryType.FACTUAL:
                        intent.secondary_intents.append(QueryType.ANALYTICAL)
                        intent.confidence = max(intent.confidence, 0.8)

            return intent

        except Exception as e:
            logger.error(f"增强意图识别失败: {e}")
            # 回退到传统方法
            return await self._identify_intent(query)

    async def _enhanced_entity_extraction(self, query: str) -> List[Dict[str, Any]]:
        """使用金融模型进行增强实体抽取"""
        try:
            # 使用金融模型进行实体识别
            entity_result = await financial_llm_service.extract_entities(query)

            # 转换为标准格式
            entities = []
            for entity in entity_result.result:
                entities.append({
                    'text': entity.entity,
                    'type': entity.entity_type,
                    'confidence': entity.confidence,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'metadata': entity.metadata
                })

            # 如果金融模型没有识别到实体，使用传统方法
            if not entities:
                entities = await financial_entity_extractor.extract_entities(query)

            return entities

        except Exception as e:
            logger.error(f"增强实体抽取失败: {e}")
            # 回退到传统方法
            return await financial_entity_extractor.extract_entities(query)

    async def _enhanced_keyword_extraction(self, query: str) -> List[str]:
        """使用金融模型进行增强关键词提取"""
        try:
            # 使用金融模型提取关键词
            keyword_result = await financial_llm_service.extract_keywords(query, top_k=10)

            # 提取关键词文本
            financial_keywords = [kw[0] for kw in keyword_result.result]

            # 结合传统关键词提取方法
            traditional_keywords = self._extract_keywords(query)

            # 合并并去重，优先金融关键词
            all_keywords = financial_keywords + [kw for kw in traditional_keywords if kw not in financial_keywords]

            return all_keywords[:15]  # 最多返回15个关键词

        except Exception as e:
            logger.error(f"增强关键词提取失败: {e}")
            # 回退到传统方法
            return self._extract_keywords(query)

    async def _enhanced_query_expansion(
        self,
        query: str,
        intent: QueryIntent,
        entities: List
    ) -> List[str]:
        """使用金融模型进行增强查询扩展"""
        try:
            # 获取传统扩展查询
            expansions = await self._expand_query(query, intent, entities)

            # 使用金融模型生成语义扩展
            if entities:
                entity_texts = [e.get('text', '') for e in entities[:3]]  # 最多3个实体
                entity_context = ' '.join(entity_texts)

                # 基于实体的扩展
                entity_expansions = [
                    f"{query} {entity_text}" for entity_text in entity_texts
                ]
                expansions.extend(entity_expansions)

            # 基于金融领域知识的专业扩展
            financial_expansions = self._generate_financial_expansions(query, intent)
            expansions.extend(financial_expansions)

            # 去重并限制数量
            unique_expansions = list(set(expansions))
            return unique_expansions[:8]  # 最多8个扩展查询

        except Exception as e:
            logger.error(f"增强查询扩展失败: {e}")
            # 回退到传统方法
            return await self._expand_query(query, intent, entities)

    def _generate_financial_expansions(self, query: str, intent: QueryIntent) -> List[str]:
        """生成金融领域专业扩展"""
        expansions = []

        # 金融术语映射
        term_mappings = {
            '股价': ['股票价格', '收盘价', '开盘价', '股价走势'],
            '财报': ['财务报告', '年报', '季报', '财务报表'],
            '业绩': ['经营业绩', '盈利能力', '营收表现'],
            '风险': ['风险因素', '风险控制', '风险管理'],
            '投资': ['投资机会', '投资价值', '投资回报'],
            '分析': ['深度分析', '专业分析', '技术分析', '基本面分析']
        }

        # 检查查询中的金融术语并生成扩展
        for term, alternatives in term_mappings.items():
            if term in query:
                for alt in alternatives[:2]:  # 每个术语最多2个扩展
                    expansion = query.replace(term, alt)
                    if expansion != query:
                        expansions.append(expansion)

        # 基于意图的专业扩展
        if intent.primary_intent == QueryType.ANALYTICAL:
            expansions.extend([
                f"{query} 专业评估",
                f"{query} 深度解读",
                f"{query} 行业分析"
            ])
        elif intent.primary_intent == QueryType.TEMPORAL:
            expansions.extend([
                f"{query} 历史表现",
                f"{query} 未来趋势",
                f"{query} 同比增长"
            ])
        elif intent.primary_intent == QueryType.COMPARATIVE:
            expansions.extend([
                f"{query} 行业对比",
                f"{query} 竞争分析",
                f"{query} 基准比较"
            ])

        return expansions

    async def _llm_query_expansion(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[str]:
        """使用LLM生成查询扩展"""
        prompt = f"""
请为以下查询生成2-3个语义相似但表达方式不同的扩展查询。

原始查询: {query}
主要意图: {intent.primary_intent.value}

请生成有助于信息检索的扩展查询，每行一个。
"""

        try:
            response = await llm_service.generate_response(prompt)
            expansions = [line.strip() for line in response.split('\n') if line.strip()]
            return expansions[:3]
        except Exception as e:
            logger.error(f"LLM查询扩展失败: {e}")
            return []