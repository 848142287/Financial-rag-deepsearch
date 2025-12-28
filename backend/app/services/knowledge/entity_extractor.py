"""
实体抽取器
从金融文档中抽取关键实体，包括公司、人物、财务指标、时间等
"""

import asyncio
import re
from typing import List, Dict, Any, Tuple, Set, Optional
import jieba
import jieba.posseg as pseg
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体数据类"""
    text: str
    type: str
    confidence: float
    start: int
    end: int
    context: str
    metadata: Dict[str, Any] = None


class FinancialEntityExtractor:
    """金融领域实体抽取器"""

    def __init__(self):
        # 初始化jieba分词
        jieba.initialize()

        # 金融实体类型定义
        self.entity_types = {
            'COMPANY': '公司',
            'PERSON': '人物',
            'STOCK': '股票',
            'BOND': '债券',
            'FUND': '基金',
            'FINANCIAL_INDICATOR': '财务指标',
            'FINANCIAL_TERM': '金融术语',
            'TIME': '时间',
            'MONEY': '金额',
            'PERCENTAGE': '百分比',
            'LOCATION': '地点',
            'INDUSTRY': '行业',
            'PRODUCT': '产品'
        }

        # 加载自定义词典
        self._load_financial_dict()

    def _load_financial_dict(self):
        """加载金融专业词典"""
        # 金融术语
        financial_terms = [
            '市盈率', '市净率', 'ROE', 'ROA', '净利润率', '毛利率',
            '资产负债率', '流动比率', '速动比率', '现金流量', '营收',
            '净利润', 'EBITDA', '息税前利润', '每股收益', 'EPS',
            '市值', '股价', '分红', '配股', '增发', '回购',
            '期货', '期权', '掉期', '远期', '互换', '资产证券化',
            'IPO', 'M&A', '并购', '重组', '借壳上市', '私有化'
        ]

        # 公司后缀
        company_suffixes = [
            '有限公司', '股份有限公司', '集团', '控股', '科技',
            '银行', '保险', '证券', '基金', '投资', '信托'
        ]

        # 添加到jieba词典
        for term in financial_terms + company_suffixes:
            jieba.add_word(term)

    async def extract_entities(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        use_llm: bool = False
    ) -> List[Entity]:
        """
        抽取实体

        Args:
            text: 输入文本
            chunk_id: 文本块ID
            use_llm: 是否使用LLM增强抽取

        Returns:
            实体列表
        """
        entities = []

        # 基础规则抽取
        entities.extend(self._extract_by_rules(text, chunk_id))

        # 正则表达式抽取
        entities.extend(self._extract_by_regex(text, chunk_id))

        # NER模型抽取（如果可用）
        entities.extend(self._extract_by_ner(text, chunk_id))

        # LLM增强抽取（可选）
        if use_llm:
            llm_entities = await self._extract_by_llm(text, chunk_id)
            entities.extend(llm_entities)

        # 去重和合并
        entities = self._merge_entities(entities)

        return entities

    def _extract_by_rules(self, text: str, chunk_id: str) -> List[Entity]:
        """基于规则的实体抽取"""
        entities = []

        # 分词
        words = pseg.cut(text)

        # 词性标注过滤
        financial_pos = {
            'nz': '其他专名',
            'ns': '地名',
            'nt': '机构名',
            'nr': '人名'
        }

        for word, flag in words:
            if len(word) < 2:  # 过滤太短的词
                continue

            # 公司名称识别
            if any(suffix in word for suffix in ['有限公司', '股份有限公司', '集团', '银行', '保险', '证券']):
                entities.append(Entity(
                    text=word,
                    type='COMPANY',
                    confidence=0.9,
                    start=text.find(word),
                    end=text.find(word) + len(word),
                    context=self._get_context(text, word),
                    metadata={'source': 'rule', 'pos': flag, 'chunk_id': chunk_id}
                ))

            # 人名识别
            elif flag == 'nr' and len(word) >= 2:
                entities.append(Entity(
                    text=word,
                    type='PERSON',
                    confidence=0.7,
                    start=text.find(word),
                    end=text.find(word) + len(word),
                    context=self._get_context(text, word),
                    metadata={'source': 'rule', 'pos': flag, 'chunk_id': chunk_id}
                ))

            # 金融术语
            elif word in self._get_financial_terms():
                entity_type = self._classify_financial_term(word)
                entities.append(Entity(
                    text=word,
                    type=entity_type,
                    confidence=0.8,
                    start=text.find(word),
                    end=text.find(word) + len(word),
                    context=self._get_context(text, word),
                    metadata={'source': 'rule', 'pos': flag, 'chunk_id': chunk_id}
                ))

        return entities

    def _extract_by_regex(self, text: str, chunk_id: str) -> List[Entity]:
        """正则表达式实体抽取"""
        entities = []

        patterns = [
            # 金额 (e.g., 100万元, $1.2亿)
            (r'(\d+\.?\d*)\s*(万|亿|千|百)?\s*(元|美元|USD|CNY|人民币)',
             'MONEY', 0.9),

            # 百分比
            (r'(\d+\.?\d*)\s*%', 'PERCENTAGE', 0.95),

            # 股票代码 (e.g., 000001.SZ, 600000.SH)
            (r'(\d{6})\.(SH|SZ|BJ)', 'STOCK', 0.95),

            # 时间
            (r'(\d{4}年|\d{1,2}月|\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})',
             'TIME', 0.9),

            # 财务指标 (e.g., ROE: 15%, 毛利率: 20%)
            (r'(ROE|ROA|市盈率|市净率|毛利率|净利率)[：:]\s*(\d+\.?\d*)\s*%?',
             'FINANCIAL_INDICATOR', 0.9),

            # 公司名称 (简单模式)
            (r'([A-Za-z\u4e00-\u9fa5]+(?:集团|有限公司|股份有限公司|科技|控股))',
             'COMPANY', 0.7)
        ]

        for pattern, entity_type, confidence in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group(1) if match.groups() else match.group(0)

                entities.append(Entity(
                    text=entity_text,
                    type=entity_type,
                    confidence=confidence,
                    start=match.start(),
                    end=match.end(),
                    context=self._get_context(text, entity_text),
                    metadata={'source': 'regex', 'chunk_id': chunk_id}
                ))

        return entities

    def _extract_by_ner(self, text: str, chunk_id: str) -> List[Entity]:
        """使用NER模型抽取实体"""
        try:
            # 尝试使用spaCy进行NER
            import spacy

            # 加载中文模型，如果不存在则使用英文模型
            try:
                nlp = spacy.load('zh_core_web_sm')
            except OSError:
                try:
                    nlp = spacy.load('en_core_web_sm')
                except OSError:
                    # 如果没有预训练模型，返回空列表
                    logger.warning("No spaCy model available for NER")
                    return []

            # 处理文本
            doc = nlp(text)

            entities = []
            for ent in doc.ents:
                # 映射spaCy的实体标签到我们的类型
                entity_type = self._map_spacy_label(ent.label_)

                if entity_type:
                    entity = Entity(
                        text=ent.text,
                        type=entity_type,
                        confidence=0.8,  # spaCy的置信度默认值
                        chunk_id=chunk_id,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    )
                    entities.append(entity)

            return entities

        except ImportError:
            # 如果spaCy不可用，尝试使用简单的规则匹配
            return self._extract_by_rules(text, chunk_id)
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def _extract_by_rules(self, text: str, chunk_id: str) -> List[Entity]:
        """使用规则抽取实体（备用方案）"""
        entities = []

        # 定义金融实体模式
        patterns = {
            'COMPANY': [
                r'[\u4e00-\u9fff]+银行',
                r'[\u4e00-\u9fff]+保险',
                r'[\u4e00-\u9fff]+证券',
                r'[\u4e00-\u9fff]+集团',
                r'[\u4e00-\u9fff]+有限公司',
                r'[\u4e00-\u9fff]+股份'
            ],
            'STOCK': [
                r'\d{6}',
                r'[A-Z]+\d{4}',
            ],
            'PERSON': [
                r'[\u4e00-\u9fff]{2,4}(?:董事长|总裁|CEO|总经理|行长)'
            ]
        }

        import re

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        confidence=0.6,  # 规则匹配的置信度较低
                        chunk_id=chunk_id,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)

        return entities

    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """映射spaCy实体标签到我们的类型"""
        mapping = {
            'ORG': 'COMPANY',
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',  # 地缘政治实体
            'MONEY': 'FINANCIAL_METRIC',
            'DATE': 'TIME',
            'CARDINAL': 'NUMBER'
        }
        return mapping.get(spacy_label)

    async def _extract_by_llm(self, text: str, chunk_id: str) -> List[Entity]:
        """使用LLM进行增强实体抽取"""
        try:
            from app.services.llm_service import LLMService
            llm_service = LLMService()

            # 构建实体抽取的prompt
            prompt = f"""
你是一个金融领域的实体识别专家。请从以下文本中抽取出所有相关的金融实体。

实体类型包括：
1. COMPANY - 公司/机构名称
2. PERSON - 人物姓名
3. STOCK - 股票代码
4. BOND - 债券
5. FUND - 基金
6. FINANCIAL_INDICATOR - 财务指标（如市盈率、ROE等）
7. FINANCIAL_TERM - 金融术语（如IPO、M&A等）
8. MONEY - 金额/数值
9. PERCENTAGE - 百分比
10. LOCATION - 地点/地区
11. INDUSTRY - 行业名称
12. TIME - 时间相关

请以JSON格式返回，格式如下：
{{
    "entities": [
        {{
            "text": "实体文本",
            "type": "实体类型",
            "start": 起始位置,
            "end": 结束位置,
            "confidence": 置信度(0-1),
            "description": "实体描述"
        }}
    ]
}}

文本内容：
{text[:3000]}  # 限制文本长度
"""

            # 调用LLM
            response = await llm_service.simple_chat(
                prompt=prompt,
                temperature=0.1
            )

            # 解析LLM返回的JSON
            import json
            import re

            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else:
                # 如果无法解析，返回空列表
                logger.warning(f"Failed to parse LLM response for entity extraction: {response[:100]}...")
                return []

            entities = []
            for entity_info in result.get("entities", []):
                try:
                    entity = Entity(
                        text=entity_info.get("text", ""),
                        type=entity_info.get("type", "UNKNOWN"),
                        start=entity_info.get("start", 0),
                        end=entity_info.get("end", 0),
                        confidence=float(entity_info.get("confidence", 0.8)),
                        chunk_id=chunk_id,
                        properties={
                            "source": "llm",
                            "description": entity_info.get("description", ""),
                            "extraction_method": "llm_enhanced"
                        }
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.error(f"Error parsing entity: {e}")

            logger.info(f"LLM extracted {len(entities)} entities from chunk {chunk_id}")
            return entities

        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {str(e)}")
            return []

    def _get_context(self, text: str, entity: str, window: int = 50) -> str:
        """获取实体的上下文"""
        start = max(0, text.find(entity) - window)
        end = min(len(text), text.find(entity) + len(entity) + window)
        return text[start:end]

    def _get_financial_terms(self) -> Set[str]:
        """获取金融术语集合"""
        return {
            '市盈率', '市净率', 'ROE', 'ROA', '净利润率', '毛利率',
            '资产负债率', '流动比率', '速动比率', '现金流量', '营收',
            '净利润', 'EBITDA', '息税前利润', '每股收益', 'EPS',
            '市值', '股价', '分红', '配股', '增发', '回购',
            '期货', '期权', '掉期', '远期', '互换', '资产证券化',
            'IPO', 'M&A', '并购', '重组', '借壳上市', '私有化'
        }

    def _classify_financial_term(self, term: str) -> str:
        """分类金融术语"""
        if term in ['市盈率', '市净率', 'ROE', 'ROA', '净利润率', '毛利率']:
            return 'FINANCIAL_INDICATOR'
        elif term in ['期货', '期权', '掉期', '远期', '互换']:
            return 'PRODUCT'
        elif term in ['IPO', 'M&A', '并购', '重组']:
            return 'FINANCIAL_TERM'
        else:
            return 'FINANCIAL_TERM'

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重叠的实体"""
        if not entities:
            return []

        # 按位置排序
        entities.sort(key=lambda x: (x.start, x.end))

        merged = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]

            # 检查重叠
            if current.start < last.end:
                # 重叠，选择置信度更高的
                if current.confidence > last.confidence:
                    merged[-1] = current
                # 或者合并类型相同的实体
                elif current.type == last.type and current.confidence == last.confidence:
                    # 选择更长的实体
                    if len(current.text) > len(last.text):
                        merged[-1] = current
            else:
                merged.append(current)

        return merged

    async def extract_relations(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        抽取实体间的关系

        Args:
            entities: 实体列表
            text: 原始文本

        Returns:
            关系列表
        """
        relations = []

        # 简单的规则关系抽取
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relation = self._extract_rule_relation(entity1, entity2, text)
                if relation:
                    relations.append(relation)

        return relations

    def _extract_rule_relation(
        self,
        entity1: Entity,
        entity2: Entity,
        text: str
    ) -> Optional[Dict[str, Any]]:
        """基于规则抽取两个实体间的关系"""
        # 获取两个实体之间的文本
        start = min(entity1.end, entity2.end)
        end = max(entity1.start, entity2.start)
        between_text = text[start:end].strip()

        # 关系规则
        relation_patterns = {
            'INVESTS_IN': ['投资', '参股', '持股'],
            'ACQUIRES': ['收购', '并购', '收购了'],
            'SUBSIDIARY_OF': ['子公司', '隶属于', '旗下'],
            'CEO_OF': ['CEO', '首席执行官', '总经理'],
            'LOCATED_IN': ['位于', '在', '总部在'],
            'BELONGS_TO': ['属于', '归...所有', '旗下']
        }

        for relation_type, keywords in relation_patterns.items():
            if any(keyword in between_text for keyword in keywords):
                return {
                    'type': relation_type,
                    'source': entity1.text,
                    'target': entity2.text,
                    'confidence': 0.7,
                    'evidence': between_text
                }

        return None


# 全局实体抽取器实例
financial_entity_extractor = FinancialEntityExtractor()