"""
金融实体关系抽取器
支持金融实体类型和跨模态关联
"""

import asyncio
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class FinancialEntityType(Enum):
    """金融实体类型"""
    # 公司实体
    COMPANY = "COMPANY"                    # 公司
    SUBSIDIARY = "SUBSIDIARY"              # 子公司
    JOINT_VENTURE = "JOINT_VENTURE"        # 合资公司

    # 财务指标
    REVENUE = "REVENUE"                    # 营收
    PROFIT = "PROFIT"                      # 利润
    NET_PROFIT = "NET_PROFIT"              # 净利润
    GROSS_PROFIT = "GROSS_PROFIT"          # 毛利润
    ASSETS = "ASSETS"                      # 资产
    LIABILITIES = "LIABILITIES"            # 负债
    EQUITY = "EQUITY"                      # 股本
    CASH_FLOW = "CASH_FLOW"                # 现金流
    EPS = "EPS"                           # 每股收益
    PE_RATIO = "PE_RATIO"                 # 市盈率
    MARKET_CAP = "MARKET_CAP"             # 市值

    # 时间和周期
    FISCAL_YEAR = "FISCAL_YEAR"           # 财年
    QUARTER = "QUARTER"                   # 季度
    MONTH = "MONTH"                       # 月份
    REPORTING_PERIOD = "REPORTING_PERIOD" # 报告期

    # 金融产品
    STOCK = "STOCK"                       # 股票
    BOND = "BOND"                         # 债券
    FUND = "FUND"                         # 基金
    DERIVATIVE = "DERIVATIVE"             # 衍生品
    OPTION = "OPTION"                     # 期权
    FUTURES = "FUTURES"                   # 期货

    # 地理和市场
    MARKET = "MARKET"                     # 市场
    EXCHANGE = "EXCHANGE"                 # 交易所
    REGION = "REGION"                     # 地区
    COUNTRY = "COUNTRY"                   # 国家

    # 业务和行业
    INDUSTRY = "INDUSTRY"                 # 行业
    SECTOR = "SECTOR"                     # 板块
    BUSINESS_LINE = "BUSINESS_LINE"       # 业务线
    PRODUCT = "PRODUCT"                   # 产品

    # 人物和职位
    EXECUTIVE = "EXECUTIVE"               # 高管
    CEO = "CEO"                          # 首席执行官
    CFO = "CFO"                          # 首席财务官
    BOARD_MEMBER = "BOARD_MEMBER"         # 董事会成员
    ANALYST = "ANALYST"                   # 分析师

    # 事件和概念
    MERGER = "MERGER"                     # 并购
    ACQUISITION = "ACQUISITION"           # 收购
    IPO = "IPO"                          # 首次公开募股
    DIVIDEND = "DIVIDEND"                 # 分红
    STOCK_SPLIT = "STOCK_SPLIT"           # 拆股
    BUYBACK = "BUYBACK"                   # 回购


class RelationType(Enum):
    """关系类型"""
    # 公司关系
    SUBSIDIARY_OF = "SUBSIDIARY_OF"       # 子公司关系
    ACQUIRED = "ACQUIRED"                 # 收购关系
    MERGED_WITH = "MERGED_WITH"           # 合并关系
    COMPETITOR = "COMPETITOR"             # 竞争关系
    PARTNER = "PARTNER"                   # 合作关系

    # 财务关系
    HAS_REVENUE = "HAS_REVENUE"           # 拥有营收
    HAS_PROFIT = "HAS_PROFIT"             # 拥有利润
    HAS_ASSETS = "HAS_ASSETS"             # 拥有资产
    HAS_MARKET_CAP = "HAS_MARKET_CAP"     # 拥有市值
    TRADING_AT = "TRADING_AT"             # 交易价格
    LISTED_ON = "LISTED_ON"               # 上市地点

    # 时间关系
    REPORTED_IN = "REPORTED_IN"           # 报告期
    OCCURRED_IN = "OCCURRED_IN"           # 发生时间
    ANNOUNCED_ON = "ANNOUNCED_ON"         # 公告时间

    # 业务关系
    OPERATES_IN = "OPERATES_IN"           # 运营地区
    BELONGS_TO = "BELONGS_TO"             # 属于行业
    PROVIDES = "PROVIDES"                 # 提供产品
    WORKS_FOR = "WORKS_FOR"               # 工作关系

    # 金融关系
    HOLDS = "HOLDS"                       # 持有
    INVESTED_IN = "INVESTED_IN"           # 投资
    ISSUED = "ISSUED"                     # 发行


@dataclass
class FinancialEntity:
    """金融实体"""
    entity_id: str
    text: str                           # 原始文本
    entity_type: FinancialEntityType    # 实体类型
    confidence: float                  # 置信度
    start_pos: int                     # 开始位置
    end_pos: int                       # 结束位置
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "text"          # 来源类型：text/table/image


@dataclass
class FinancialRelation:
    """金融关系"""
    relation_id: str
    subject_id: str                    # 主体实体ID
    object_id: str                     # 客体实体ID
    relation_type: RelationType        # 关系类型
    confidence: float                  # 置信度
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "text"          # 来源类型


@dataclass
class CrossModalAssociation:
    """跨模态关联"""
    association_id: str
    entity_id: str                     # 实体ID
    associated_content: Dict[str, Any] # 关联内容
    content_type: str                  # 内容类型：table/image/chart
    confidence: float                  # 关联置信度
    spatial_info: Optional[Dict[str, Any]] = None  # 空间信息


class FinancialEntityExtractor:
    """金融实体抽取器"""

    def __init__(self):
        # 实体识别模式
        self.entity_patterns = self._init_entity_patterns()

        # 关系抽取模式
        self.relation_patterns = self._init_relation_patterns()

        # 金融词典
        self.financial_dictionary = self._load_financial_dictionary()

        # 公司后缀列表
        self.company_suffixes = [
            "有限公司", "股份有限公司", "集团", "控股", "公司", "企业",
            "Co., Ltd", "Corporation", "Inc", "LLC", "Group", "Holdings"
        ]

        # 财政指标关键词
        self.financial_indicators = {
            "营收": FinancialEntityType.REVENUE,
            "收入": FinancialEntityType.REVENUE,
            "利润": FinancialEntityType.PROFIT,
            "净利润": FinancialEntityType.NET_PROFIT,
            "毛利润": FinancialEntityType.GROSS_PROFIT,
            "资产": FinancialEntityType.ASSETS,
            "负债": FinancialEntityType.LIABILITIES,
            "股本": FinancialEntityType.EQUITY,
            "现金流": FinancialEntityType.CASH_FLOW,
            "每股收益": FinancialEntityType.EPS,
            "市盈率": FinancialEntityType.PE_RATIO,
            "市值": FinancialEntityType.MARKET_CAP
        }

    async def extract_entities(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[FinancialEntity]:
        """
        抽取金融实体

        Args:
            text: 输入文本
            context: 可选的上下文信息

        Returns:
            抽取到的实体列表
        """
        try:
            entities = await self._extract_entities(text, "text")
            return entities
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []

    async def extract_entities_and_relations(self, text: str,
                                           context: Optional[Dict[str, Any]] = None,
                                           content_type: str = "text") -> Tuple[List[FinancialEntity], List[FinancialRelation]]:
        """抽取实体和关系"""
        try:
            # 抽取实体
            entities = await self._extract_entities(text, content_type)

            # 抽取关系
            relations = await self._extract_relations(text, entities, content_type)

            # 如果有表格或图像上下文，进行跨模态关联
            if context and content_type != "text":
                await self._extract_cross_modal_associations(entities, context)

            # 后处理：实体消歧和关系验证
            entities = await self._post_process_entities(entities)
            relations = await self._post_process_relations(relations, entities)

            return entities, relations

        except Exception as e:
            logger.error(f"实体关系抽取失败: {e}")
            return [], []

    async def _extract_entities(self, text: str, content_type: str) -> List[FinancialEntity]:
        """抽取实体"""
        entities = []

        # 1. 基于规则的抽取
        rule_entities = await self._extract_entities_by_rules(text, content_type)
        entities.extend(rule_entities)

        # 2. 基于词典的抽取
        dict_entities = await self._extract_entities_by_dict(text, content_type)
        entities.extend(dict_entities)

        # 3. 基于模型的抽取（如果可用）
        # model_entities = await self._extract_entities_by_model(text)
        # entities.extend(model_entities)

        # 去重
        entities = self._deduplicate_entities(entities)

        return entities

    async def _extract_entities_by_rules(self, text: str, content_type: str) -> List[FinancialEntity]:
        """基于规则的实体抽取"""
        entities = []

        # 抽取公司
        companies = await self._extract_companies(text, content_type)
        entities.extend(companies)

        # 抽取财务指标
        indicators = await self._extract_financial_indicators(text, content_type)
        entities.extend(indicators)

        # 抽取时间
        times = await self._extract_time_entities(text, content_type)
        entities.extend(times)

        # 抽取金融产品
        products = await self._extract_financial_products(text, content_type)
        entities.extend(products)

        return entities

    async def _extract_companies(self, text: str, content_type: str) -> List[FinancialEntity]:
        """抽取公司实体"""
        entities = []

        # 公司名称模式
        company_pattern = r'([A-Za-z\u4e00-\u9fff][^(（\n]*?(?:' + '|'.join(self.company_suffixes) + ')[^(（\n]*)'

        matches = re.finditer(company_pattern, text)
        for match in matches:
            entity = FinancialEntity(
                entity_id=str(uuid.uuid4()),
                text=match.group(1).strip(),
                entity_type=FinancialEntityType.COMPANY,
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end(),
                source_type=content_type
            )
            entities.append(entity)

        # 抽取知名公司（通过词典）
        for company_name in self.financial_dictionary.get("companies", []):
            if company_name in text:
                start = text.find(company_name)
                entity = FinancialEntity(
                    entity_id=str(uuid.uuid4()),
                    text=company_name,
                    entity_type=FinancialEntityType.COMPANY,
                    confidence=0.95,
                    start_pos=start,
                    end_pos=start + len(company_name),
                    metadata={"known_company": True},
                    source_type=content_type
                )
                entities.append(entity)

        return entities

    async def _extract_financial_indicators(self, text: str, content_type: str) -> List[FinancialEntity]:
        """抽取财务指标"""
        entities = []

        # 财务指标 + 数值模式
        indicator_pattern = r'([^，。\n]*(?:' + '|'.join(re.escape(k) for k in self.financial_indicators.keys()) + ')[^，。\n]*?[\\d,，.]+[万亿千百元%]*)'

        matches = re.finditer(indicator_pattern, text)
        for match in matches:
            matched_text = match.group(1)

            # 确定指标类型
            entity_type = None
            for keyword, et in self.financial_indicators.items():
                if keyword in matched_text:
                    entity_type = et
                    break

            if entity_type:
                entity = FinancialEntity(
                    entity_id=str(uuid.uuid4()),
                    text=matched_text,
                    entity_type=entity_type,
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    source_type=content_type
                )
                entities.append(entity)

        return entities

    async def _extract_time_entities(self, text: str, content_type: str) -> List[FinancialEntity]:
        """抽取时间实体"""
        entities = []

        # 年份模式
        year_pattern = r'(\d{4})年'
        matches = re.finditer(year_pattern, text)
        for match in matches:
            entity = FinancialEntity(
                entity_id=str(uuid.uuid4()),
                text=match.group(1) + "年",
                entity_type=FinancialEntityType.FISCAL_YEAR,
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end(),
                source_type=content_type
            )
            entities.append(entity)

        # 季度模式
        quarter_pattern = r'(第[一二三四]季度|Q[1-4])'
        matches = re.finditer(quarter_pattern, text)
        for match in matches:
            entity = FinancialEntity(
                entity_id=str(uuid.uuid4()),
                text=match.group(1),
                entity_type=FinancialEntityType.QUARTER,
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end(),
                source_type=content_type
            )
            entities.append(entity)

        return entities

    async def _extract_financial_products(self, text: str, content_type: str) -> List[FinancialEntity]:
        """抽取金融产品"""
        entities = []

        # 股票代码模式
        stock_pattern = r'([0-9]{6}\.(?:SH|SZ))'
        matches = re.finditer(stock_pattern, text)
        for match in matches:
            entity = FinancialEntity(
                entity_id=str(uuid.uuid4()),
                text=match.group(1),
                entity_type=FinancialEntityType.STOCK,
                confidence=0.95,
                start_pos=match.start(),
                end_pos=match.end(),
                metadata={"stock_code": True},
                source_type=content_type
            )
            entities.append(entity)

        return entities

    async def _extract_entities_by_dict(self, text: str, content_type: str) -> List[FinancialEntity]:
        """基于词典的实体抽取"""
        entities = []

        for category, terms in self.financial_dictionary.items():
            for term in terms:
                if term in text:
                    start = text.find(term)
                    # 确定实体类型
                    entity_type = self._map_category_to_entity_type(category)
                    if entity_type:
                        entity = FinancialEntity(
                            entity_id=str(uuid.uuid4()),
                            text=term,
                            entity_type=entity_type,
                            confidence=0.9,
                            start_pos=start,
                            end_pos=start + len(term),
                            metadata={"dict_source": category},
                            source_type=content_type
                        )
                        entities.append(entity)

        return entities

    async def _extract_relations(self, text: str, entities: List[FinancialEntity],
                               content_type: str) -> List[FinancialRelation]:
        """抽取关系"""
        relations = []

        # 基于规则的关系抽取
        rule_relations = await self._extract_relations_by_rules(text, entities, content_type)
        relations.extend(rule_relations)

        # 基于依存解析的关系抽取（如果可用）
        # dependency_relations = await self._extract_relations_by_dependency(text, entities)
        # relations.extend(dependency_relations)

        return relations

    async def _extract_relations_by_rules(self, text: str, entities: List[FinancialEntity],
                                        content_type: str) -> List[FinancialRelation]:
        """基于规则的关系抽取"""
        relations = []

        # 创建实体映射
        entity_map = {e.text: e for e in entities}

        # 公司-财务指标关系
        for company_entity in [e for e in entities if e.entity_type == FinancialEntityType.COMPANY]:
            for indicator_entity in [e for e in entities if e.entity_type in [
                FinancialEntityType.REVENUE, FinancialEntityType.PROFIT,
                FinancialEntityType.ASSETS, FinancialEntityType.MARKET_CAP
            ]]:
                # 检查是否在同一句话中
                company_pos = company_entity.start_pos
                indicator_pos = indicator_entity.start_pos

                # 寻找包含两个实体的句子
                sentence_start = max(0, min(company_pos, indicator_pos) - 100)
                sentence_end = min(len(text), max(company_pos, indicator_pos) + 100)
                sentence = text[sentence_start:sentence_end]

                if company_entity.text in sentence and indicator_entity.text in sentence:
                    # 确定关系类型
                    relation_type = self._determine_company_indicator_relation(
                        company_entity.text, indicator_entity.text, sentence
                    )

                    if relation_type:
                        relation = FinancialRelation(
                            relation_id=str(uuid.uuid4()),
                            subject_id=company_entity.entity_id,
                            object_id=indicator_entity.entity_id,
                            relation_type=relation_type,
                            confidence=0.75,
                            metadata={"source_sentence": sentence},
                            source_type=content_type
                        )
                        relations.append(relation)

        return relations

    async def _extract_cross_modal_associations(self, entities: List[FinancialEntity],
                                              context: Dict[str, Any]):
        """抽取跨模态关联"""
        if context.get("content_type") == "table":
            await self._associate_with_table(entities, context)
        elif context.get("content_type") == "image":
            await self._associate_with_image(entities, context)
        elif context.get("content_type") == "chart":
            await self._associate_with_chart(entities, context)

    async def _associate_with_table(self, entities: List[FinancialEntity],
                                  table_context: Dict[str, Any]):
        """与表格关联"""
        # 查找实体在表格中的位置
        for entity in entities:
            # 简化实现：检查实体是否在表格文本中
            table_text = table_context.get("text", "")
            if entity.text in table_text:
                association = CrossModalAssociation(
                    association_id=str(uuid.uuid4()),
                    entity_id=entity.entity_id,
                    associated_content=table_context,
                    content_type="table",
                    confidence=0.8
                )
                # 存储关联信息（这里简化处理）

    async def _associate_with_image(self, entities: List[FinancialEntity],
                                  image_context: Dict[str, Any]):
        """与图像关联"""
        # 基于OCR文本和图像描述进行关联
        ocr_text = image_context.get("ocr_text", "")
        image_description = image_context.get("description", "")

        for entity in entities:
            if entity.text in ocr_text or entity.text in image_description:
                association = CrossModalAssociation(
                    association_id=str(uuid.uuid4()),
                    entity_id=entity.entity_id,
                    associated_content=image_context,
                    content_type="image",
                    confidence=0.7
                )

    def _init_entity_patterns(self) -> Dict[str, str]:
        """初始化实体识别模式"""
        return {
            "company": r'[A-Za-z\u4e00-\u9fff][^(（\n]*?(?:有限公司|股份有限公司|集团|控股)[^(（\n]*',
            "financial_number": r'[\d,，.]+[万亿千百元%]',
            "stock_code": r'[0-9]{6}\.(?:SH|SZ)',
            "year": r'\d{4}年',
            "quarter": r'第[一二三四]季度|Q[1-4]'
        }

    def _init_relation_patterns(self) -> Dict[str, str]:
        """初始化关系抽取模式"""
        return {
            "has_revenue": r'(.+?)营收[为是]?([\d,，.]+)',
            "has_profit": r'(.+?)利润[为是]?([\d,，.]+)',
            "acquired": r'(.+?)收购(.+?)',
            "subsidiary": r'(.+?)是(.+?)的子公司'
        }

    def _load_financial_dictionary(self) -> Dict[str, List[str]]:
        """加载金融词典"""
        # 简化实现，实际应该从文件或数据库加载
        return {
            "companies": [
                "腾讯控股", "阿里巴巴", "百度", "京东", "美团", "小米",
                "Apple", "Microsoft", "Google", "Amazon", "Facebook"
            ],
            "exchanges": [
                "上海证券交易所", "深圳证券交易所", "香港交易所", "纳斯达克", "纽交所"
            ],
            "industries": [
                "互联网", "金融", "房地产", "制造业", "零售", "科技"
            ]
        }

    def _map_category_to_entity_type(self, category: str) -> Optional[FinancialEntityType]:
        """映射词典类别到实体类型"""
        mapping = {
            "companies": FinancialEntityType.COMPANY,
            "exchanges": FinancialEntityType.EXCHANGE,
            "industries": FinancialEntityType.INDUSTRY
        }
        return mapping.get(category)

    def _determine_company_indicator_relation(self, company: str, indicator: str,
                                            sentence: str) -> Optional[RelationType]:
        """确定公司-指标关系类型"""
        if "营收" in indicator or "收入" in indicator:
            return RelationType.HAS_REVENUE
        elif "利润" in indicator:
            return RelationType.HAS_PROFIT
        elif "资产" in indicator:
            return RelationType.HAS_ASSETS
        elif "市值" in indicator:
            return RelationType.HAS_MARKET_CAP
        return None

    def _deduplicate_entities(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """去重实体"""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity.text, entity.entity_type, entity.start_pos)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)

        return deduplicated

    async def _post_process_entities(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """后处理实体"""
        # 实体标准化
        for entity in entities:
            entity.text = entity.text.strip()

            # 公司名称标准化
            if entity.entity_type == FinancialEntityType.COMPANY:
                entity.text = self._normalize_company_name(entity.text)

        return entities

    async def _post_process_relations(self, relations: List[FinancialRelation],
                                    entities: List[FinancialEntity]) -> List[FinancialRelation]:
        """后处理关系"""
        entity_ids = {e.entity_id for e in entities}

        # 过滤无效关系
        valid_relations = []
        for relation in relations:
            if relation.subject_id in entity_ids and relation.object_id in entity_ids:
                valid_relations.append(relation)

        return valid_relations

    def _normalize_company_name(self, name: str) -> str:
        """标准化公司名称"""
        # 移除多余空格和特殊字符
        name = re.sub(r'\s+', '', name)
        return name


# 全局实例
financial_entity_extractor = FinancialEntityExtractor()