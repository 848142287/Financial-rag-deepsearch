"""
查询特征提取和意图分类
"""

import re
import jieba
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import json

from .strategies import QueryFeatures

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"      # 简单查询
    MODERATE = "moderate"  # 中等复杂度
    COMPLEX = "complex"    # 复杂查询


class QueryIntent(Enum):
    """查询意图"""
    FACTUAL = "factual"           # 事实查询：什么是XX，XX的营收是多少
    ANALYTICAL = "analytical"     # 分析查询：分析XX的财务状况，比较XX和YY
    RELATIONAL = "relational"     # 关系查询：XX对YY的影响，XX与YY的关系
    TEMPORAL = "temporal"         # 时间查询：XX在2023年的表现，最近XX的变化趋势
    COMPARATIVE = "comparative"   # 比较查询：对比XX和YY的优劣势
    CAUSAL = "causal"            # 因果查询：为什么XX上涨，XX下跌的原因
    PREDICTIVE = "predictive"     # 预测查询：XX未来的走势
    DEFINITIONAL = "definitional"  # 定义查询：什么是XX
    PROCEDURAL = "procedural"     # 流程查询：如何投资XX
    RESEARCH = "research"         # 研究查询：关于XX的研究报告


@dataclass
class FinancialEntity:
    """金融实体"""
    name: str
    type: str  # company, stock, index, concept, person
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """查询分析器"""

    def __init__(self):
        # 初始化jieba分词
        try:
            jieba.initialize()
            # 加载金融词典
            self._load_financial_dictionary()
        except Exception as e:
            logger.warning(f"Failed to initialize jieba: {e}")

        # 金融实体类型
        self.entity_types = {
            'company': '公司',
            'stock': '股票',
            'index': '指数',
            'concept': '概念',
            'person': '人物',
            'industry': '行业',
            'product': '产品'
        }

        # 关系词汇
        self.relation_words = {
            # 影响关系
            '影响', '作用于', '推动', '拉动', '促进', '压制', '制约', '限制', '支持',
            '助力', '拖累', '驱动', '传导', '波及',
            # 比较关系
            '对比', '比较', '相比', '优于', '劣于', '高于', '低于', '超过', '不如',
            '相似', '不同', '差异', '区别',
            # 因果关系
            '因为', '由于', '导致', '引起', '造成', '结果', '原因', '所以',
            # 时间关系
            '之前', '之后', '期间', '同时', '当时', '近期', '远期', '现在',
            '历史', '未来', '预测', '预期',
            # 属性关系
            '具有', '包含', '包括', '属于', '特征', '属性', '性质'
        }

        # 时间指示词
        self.time_indicators = {
            # 年份
            '2023年', '2024年', '2025年', '2022年', '2021年', '2020年',
            '今年', '去年', '前年', '明年', '后年',
            # 季度
            '第一季度', '第二季度', '第三季度', '第四季度',
            'Q1', 'Q2', 'Q3', 'Q4',
            '一季度', '二季度', '三季度', '四季度',
            '上半年', '下半年',
            # 月份
            '1月', '2月', '3月', '4月', '5月', '6月',
            '7月', '8月', '9月', '10月', '11月', '12月',
            # 时间范围
            '最近', '近期', '过去', '未来', '当前', '现在',
            '本周', '上周', '下周', '本月', '上月', '下月',
            '近一周', '近一月', '近一年', '近三年',
            # 趋势
            '趋势', '走势', '变化', '波动', '增长', '下降', '上涨', '下跌',
            '同比', '环比'
        }

        # 复杂度指示词
        self.complexity_indicators = {
            # 数量词
            '多个', '各种', '所有', '全部', '部分', '一些',
            # 比较级
            '更', '最', '较', '非常', '特别', '尤其', '主要', '重要',
            # 条件词
            '如果', '假设', '假如', '尽管', '虽然', '但是', '然而',
            # 逻辑词
            '并且', '而且', '或者', '不仅', '既', '同时', '此外', '另外',
            # 分析词
            '分析', '研究', '调查', '评估', '判断', '预测', '推算',
            # 量化词
            '程度', '比例', '占比', '份额', '百分比', '倍数'
        }

        # 通用停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还是', '为', '着', '这样', '因为', '所以'
        }

    def _load_financial_dictionary(self):
        """加载金融词典"""
        # 这里可以从文件或数据库加载专业词典
        financial_terms = [
            # 股票相关
            'A股', 'H股', '美股', '港股', 'B股', '创业板', '科创板', '新三板',
            '上证指数', '深证成指', '创业板指', '科创50', '沪深300',
            '市盈率', '市净率', 'ROE', 'ROA', 'EPS', '净利润', '营业收入',
            '市值', '股价', '涨停', '跌停', '成交量', '换手率',
            # 银行业
            '存款', '贷款', '利率', '利息', '理财', '基金', '保险',
            '不良贷款率', '拨备覆盖率', '资本充足率', '流动性',
            # 宏观经济
            'GDP', 'CPI', 'PPI', '通胀', '通缩', '货币政策', '财政政策',
            '加息', '降息', '降准', 'QE', '经济增速',
            # 公司财务
            '资产负债表', '利润表', '现金流量表', '财务报表', '年报', '季报',
            '营收', '利润', '净利润', '毛利率', '净利率', '资产收益率',
            # 行业概念
            '新能源', '半导体', '人工智能', '5G', '大数据', '云计算', '区块链',
            '医药', '消费', '制造', '金融', '地产', '汽车', '能源',
            # 投资概念
            '投资', '融资', '上市', 'IPO', '并购', '重组', '分红', '配股'
        ]

        for term in financial_terms:
            jieba.add_word(term, freq=1000)

    def analyze_query(self, query: str) -> QueryFeatures:
        """分析查询特征"""
        try:
            # 1. 预处理
            cleaned_query = self._preprocess_query(query)

            # 2. 基础特征提取
            query_length = len(cleaned_query.split())
            words = jieba.lcut(cleaned_query)

            # 3. 实体提取
            entities = self._extract_entities(cleaned_query, words)

            # 4. 关系词识别
            relation_words = self._extract_relation_words(words, cleaned_query)

            # 5. 时间指示词识别
            time_indicators = self._extract_time_indicators(words, cleaned_query)

            # 6. 复杂度指示词识别
            complexity_indicators = self._extract_complexity_indicators(words, cleaned_query)

            # 7. 查询意图识别
            intent, question_type = self._classify_intent(cleaned_query, words, entities)

            # 8. 领域识别
            domain = self._classify_domain(words, entities)

            # 9. 计算复杂度分数
            complexity_score = self._calculate_complexity_score(
                query_length, len(entities), len(relation_words),
                len(time_indicators), len(complexity_indicators), intent
            )

            # 10. 答案粒度评估
            answer_granularity = self._estimate_answer_granularity(intent, query_length, complexity_score)

            return QueryFeatures(
                query=query,
                entity_count=len(entities),
                relation_complexity=self._calculate_relation_complexity(relation_words, entities),
                time_sensitivity=self._calculate_time_sensitivity(time_indicators),
                answer_granularity=answer_granularity,
                query_length=query_length,
                question_type=question_type,
                intent=intent.value if intent else "",
                domain=domain,
                complexity_score=complexity_score,
                extracted_entities=[e.name for e in entities],
                relation_words=relation_words,
                time_indicators=time_indicators,
                complexity_indicators=complexity_indicators,
                metadata={
                    "words": words,
                    "entities": [e.__dict__ for e in entities]
                }
            )

        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            # 返回默认特征
            return QueryFeatures(query=query)

    def _preprocess_query(self, query: str) -> str:
        """预处理查询"""
        # 移除标点符号
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        # 转换为小写
        cleaned = cleaned.lower()
        # 移除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_entities(self, query: str, words: List[str]) -> List[FinancialEntity]:
        """提取金融实体"""
        entities = []

        # 1. 基于规则的实体识别
        entities.extend(self._extract_entities_by_rules(query, words))

        # 2. 基于词典的实体识别
        entities.extend(self._extract_entities_by_dictionary(words))

        # 3. 基于模式的实体识别
        entities.extend(self._extract_entities_by_patterns(query))

        # 4. 去重和排序
        entities = self._deduplicate_entities(entities)

        return entities

    def _extract_entities_by_rules(self, query: str, words: List[str]) -> List[FinancialEntity]:
        """基于规则提取实体"""
        entities = []

        # 股票代码模式 (如: 000001.SZ, 600519.SH)
        stock_pattern = r'\d{6}\.(SH|SZ|BJ|HK)'
        for match in re.finditer(stock_pattern, query):
            entities.append(FinancialEntity(
                name=match.group(),
                type='stock',
                confidence=0.9
            ))

        # 公司名称模式 (通常以股份、集团等结尾)
        company_suffixes = ['股份', '集团', '公司', '有限公司', '股份公司', '集团股份']
        for suffix in company_suffixes:
            pattern = f'[^，。！？\s]*{suffix}'
            matches = re.findall(pattern, query)
            for match in matches:
                name = match.strip()
                if len(name) > 2:  # 过滤太短的匹配
                    entities.append(FinancialEntity(
                        name=name,
                        type='company',
                        confidence=0.7
                    ))

        return entities

    def _extract_entities_by_dictionary(self, words: List[str]) -> List[FinancialEntity]:
        """基于词典提取实体"""
        entities = []

        # 预定义的金融实体字典
        entity_dict = {
            # 股票
            '贵州茅台': {'type': 'stock', 'name': '贵州茅台', 'aliases': ['茅台']},
            '中国平安': {'type': 'company', 'name': '中国平安', 'aliases': ['平安保险']},
            '招商银行': {'type': 'stock', 'name': '招商银行', 'aliases': ['招行']},
            '比亚迪': {'type': 'stock', 'name': '比亚迪', 'aliases': ['BYD']},
            '宁德时代': {'type': 'stock', 'name': '宁德时代', 'aliases': ['CATL']},
            # 指数
            '上证指数': {'type': 'index', 'name': '上证指数', 'aliases': ['上证综指']},
            '深证成指': {'type': 'index', 'name': '深证成指', 'aliases': ['深成指']},
            '创业板指': {'type': 'index', 'name': '创业板指', 'aliases': ['创业板']},
            # 概念
            '新能源': {'type': 'concept', 'name': '新能源', 'aliases': ['新能源车']},
            '人工智能': {'type': 'concept', 'name': '人工智能', 'aliases': ['AI']},
            '半导体': {'type': 'concept', 'name': '半导体', 'aliases': ['芯片']},
            '医药': {'type': 'concept', 'name': '医药', 'aliases': ['生物医药']},
            # 行业
            '银行业': {'type': 'industry', 'name': '银行业', 'aliases': ['银行']},
            '保险业': {'type': 'industry', 'name': '保险业', 'aliases': ['保险']},
            '房地产': {'type': 'industry', 'name': '房地产', 'aliases': ['地产']},
            '汽车': {'type': 'industry', 'name': '汽车', 'aliases': ['汽车业']},
            '制造业': {'type': 'industry', 'name': '制造业', 'aliases': ['制造']},
            '消费': {'type': 'concept', 'name': '消费', 'aliases': ['消费品']},
            '科技': {'type': 'concept', 'name': '科技', 'aliases': ['科技股']}
        }

        # 尝试完整匹配
        for word in words:
            if word in entity_dict:
                entity_info = entity_dict[word]
                entities.append(FinancialEntity(
                    name=entity_info['name'],
                    type=entity_info['type'],
                    confidence=0.8,
                    aliases=entity_info['aliases']
                ))

        # 尝试部分匹配（处理多字词组）
        for i in range(len(words)):
            for j in range(i + 2, min(i + 6, len(words) + 1)):  # 最多匹配5个词
                phrase = ''.join(words[i:j])
                if phrase in entity_dict:
                    entity_info = entity_dict[phrase]
                    entities.append(FinancialEntity(
                        name=entity_info['name'],
                        type=entity_info['type'],
                        confidence=0.9,
                        aliases=entity_info['aliases']
                    ))

        return entities

    def _extract_entities_by_patterns(self, query: str) -> List[FinancialEntity]:
        """基于模式提取实体"""
        entities = []

        # 数字+单位模式 (如: 500亿, 3.2%)
        number_pattern = r'\d+\.?\d*\s*(亿|万|元|%|倍|倍|年|月|日|季|度)'
        for match in re.finditer(number_pattern, query):
            entities.append(FinancialEntity(
                name=match.group(),
                type='concept',
                confidence=0.6
            ))

        return entities

    def _deduplicate_entities(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """去重实体"""
        seen_names = set()
        unique_entities = []

        for entity in entities:
            # 检查名称或别名是否已存在
            if entity.name not in seen_names and not any(alias in seen_names for alias in entity.aliases):
                seen_names.add(entity.name)
                seen_names.update(entity.aliases)
                unique_entities.append(entity)

        # 按置信度排序
        unique_entities.sort(key=lambda x: x.confidence, reverse=True)

        return unique_entities

    def _extract_relation_words(self, words: List[str], query: str) -> List[str]:
        """提取关系词"""
        relation_words = []

        # 从分词结果中查找
        for word in words:
            if word in self.relation_words:
                relation_words.append(word)

        # 从原文中查找多字关系词
        for relation_word in self.relation_words:
            if len(relation_word) > 1 and relation_word in query:
                if relation_word not in relation_words:
                    relation_words.append(relation_word)

        return list(set(relation_words))

    def _extract_time_indicators(self, words: List[str], query: str) -> List[str]:
        """提取时间指示词"""
        time_indicators = []

        # 从分词结果中查找
        for word in words:
            if word in self.time_indicators:
                time_indicators.append(word)

        # 从原文中查找多字时间词
        for time_word in self.time_indicators:
            if len(time_word) > 1 and time_word in query:
                if time_word not in time_indicators:
                    time_indicators.append(time_word)

        return list(set(time_indicators))

    def _extract_complexity_indicators(self, words: List[str], query: str) -> List[str]:
        """提取复杂度指示词"""
        complexity_indicators = []

        for word in words:
            if word in self.complexity_indicators:
                complexity_indicators.append(word)

        return list(set(complexity_indicators))

    def _classify_intent(self, query: str, words: List[str], entities: List[FinancialEntity]) -> Tuple[Optional[QueryIntent], str]:
        """分类查询意图"""
        # 疑问词检测
        question_words = ['什么', '什么是', '如何', '怎么', '为什么', '哪个', '哪', '吗', '呢']
        has_question = any(qw in query for qw in question_words)

        # 简单事实查询意图
        if len(entities) <= 1 and len(relation_words) == 0 and len(time_indicators) == 0:
            if has_question:
                return QueryIntent.DEFINITIONAL, "definition"
            else:
                return QueryIntent.FACTUAL, "factual"

        # 关系查询意图
        if len(relation_words) > 0 or len(entities) >= 2:
            if len(time_indicators) > 0:
                return QueryIntent.TEMPORAL, "temporal"
            else:
                return QueryIntent.RELATIONAL, "relational"

        # 分析查询意图
        analysis_words = ['分析', '评估', '判断', '研究', '调查', '评价']
        if any(aw in words for aw in analysis_words):
            return QueryIntent.ANALYTICAL, "analytical"

        # 比较查询意图
        comparison_words = ['对比', '比较', '相比', 'vs', 'versus', '优于', '劣于', '差异']
        if any(cw in words for cw in comparison_words):
            return QueryIntent.COMPARATIVE, "comparative"

        # 因果查询意图
        causal_words = ['为什么', '原因', '由于', '导致', '造成', '结果']
        if any(cw in words for cw in causal_words):
            return QueryIntent.CAUSAL, "causal"

        # 预测查询意图
        predictive_words = ['预测', '预期', '未来', '前景', '走势', '趋势']
        if any(pw in words for pw in predictive_words):
            return QueryIntent.PREDICTIVE, "predictive"

        # 时间查询意图
        if len(time_indicators) > 0:
            return QueryIntent.TEMPORAL, "temporal"

        # 研究查询意图
        research_words = ['研究', '报告', '调查', '分析报告', '研究报告', '白皮书']
        if any(rw in words for rw in research_words):
            return QueryIntent.RESEARCH, "research"

        # 默认意图
        return QueryIntent.FACTUAL, "factual"

    def _classify_domain(self, words: List[str], entities: List[FinancialEntity]) -> str:
        """分类查询领域"""
        # 基于实体类型判断
        entity_types = [e.type for e in entities]

        if 'stock' in entity_types:
            return 'stock_market'
        elif 'industry' in entity_types:
            return 'industry_analysis'
        elif 'concept' in entity_types:
            return 'concept_research'
        elif 'person' in entity_types:
            return 'executive_analysis'
        elif 'company' in entity_types:
            return 'company_analysis'

        # 基于关键词判断
        if any(word in ['财报', '业绩', '营收', '利润', '财务'] for word in words):
            return 'financial_analysis'
        elif any(word in ['投资', '融资', '估值', '定价'] for word in words):
            return 'investment_analysis'
        elif any(word in ['宏观', '经济', '政策', '监管'] for word in words):
            return 'macro_economic'

        return 'general'

    def _calculate_complexity_score(self, query_length: int, entity_count: int,
                                    relation_complexity: int, time_sensitivity: int,
                                    complexity_indicator_count: int, intent: Optional[QueryIntent]) -> float:
        """计算复杂度分数"""
        score = 0.0

        # 查询长度因子 (0-0.3)
        if query_length <= 10:
            score += 0.1
        elif query_length <= 20:
            score += 0.2
        else:
            score += 0.3

        # 实体数量因子 (0-0.3)
        if entity_count == 0:
            score += 0.0
        elif entity_count == 1:
            score += 0.1
        elif entity_count <= 3:
            score += 0.2
        else:
            score += 0.3

        # 关系复杂度因子 (0-0.3)
        score += min(0.3, relation_complexity * 0.1)

        # 时间敏感度因子 (0-0.2)
        score += min(0.2, time_sensitivity * 0.05)

        # 复杂度指示词因子 (0-0.2)
        score += min(0.2, complexity_indicator_count * 0.05)

        # 意图因子 (0-0.2)
        if intent:
            intent_weights = {
                QueryIntent.FACTUAL: 0.1,
                QueryIntent.DEFINITIONAL: 0.15,
                QueryIntent.RELATIONAL: 0.3,
                QueryIntent.ANALYTICAL: 0.35,
                QueryIntent.COMPARATIVE: 0.3,
                QueryIntent.CAUSAL: 0.4,
                QueryIntent.PREDICTIVE: 0.4,
                QueryIntent.RESEARCH: 0.35,
                QueryIntent.PROCEDURAL: 0.25
            }
            score += intent_weights.get(intent, 0.2)

        return max(0.0, min(1.0, score))

    def _calculate_relation_complexity(self, relation_words: List[str], entities: List[FinancialEntity]) -> int:
        """计算关系复杂度"""
        if not relation_words:
            return 0

        complexity = len(relation_words)

        # 实体数量增加复杂度
        if len(entities) > 2:
            complexity += 1

        # 多重关系词增加复杂度
        complex_relations = ['影响', '传导', '作用于', '促进', '制约']
        complex_count = sum(1 for word in relation_words if word in complex_relations)
        complexity += complex_count * 0.5

        return int(complexity)

    def _calculate_time_sensitivity(self, time_indicators: List[str]) -> int:
        """计算时间敏感度"""
        if not time_indicators:
            return 0

        sensitivity = len(time_indicators)

        # 范围时间词增加敏感度
        range_time_words = ['趋势', '变化', '历史', '未来', '预测']
        range_count = sum(1 for word in time_indicators if word in range_time_words)
        sensitivity += range_count * 0.5

        return int(sensitivity)

    def _estimate_answer_granularity(self, intent: Optional[QueryIntent], query_length: int, complexity_score: float) -> int:
        """评估答案粒度"""
        base_granularity = 3  # 默认中等粒度

        if intent:
            if intent == QueryIntent.FACTUAL:
                base_granularity = 1  # 简短答案
            elif intent == QueryIntent.DEFINITIONAL:
                base_granularity = 2  # 中等答案
            elif intent in [QueryIntent.ANALYTICAL, QueryIntent.RESEARCH]:
                base_granularity = 5  # 详细答案
            elif intent == QueryIntent.PREDICTIVE:
                base_granularity = 4  # 预测性答案

        # 根据查询长度调整
        if query_length > 20:
            base_granularity += 1
        elif query_length > 30:
            base_granularity += 1

        # 根据复杂度调整
        if complexity_score > 0.7:
            base_granularity += 1
        elif complexity_score < 0.3:
            base_granularity = max(1, base_granularity - 1)

        return max(1, min(5, base_granularity))


# 全局查询分析器实例
query_analyzer = QueryAnalyzer()