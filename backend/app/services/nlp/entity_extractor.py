"""
实体抽取器包装类
为文档处理流水线提供实体抽取功能
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from app.services.knowledge.entity_extractor import FinancialEntityExtractor

logger = logging.getLogger(__name__)


class EntityExtractor:
    """实体抽取器包装类"""

    def __init__(self):
        self.extractor = FinancialEntityExtractor()

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        抽取文本中的实体

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        try:
            # 使用同步方式调用异步服务（在Celery任务中）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                entities = loop.run_until_complete(
                    self.extractor.extract_entities(text)
                )
                # 转换为字典格式
                return [self._entity_to_dict(entity) for entity in entities]
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        抽取实体间的关系

        Args:
            text: 输入文本
            entities: 实体列表

        Returns:
            关系列表
        """
        try:
            # 简单的关系抽取逻辑
            relations = []

            # 公司-职位关系
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j and entity1['type'] == 'COMPANY' and entity2['type'] == 'PERSON':
                        # 检查文本中是否同时出现且位置相近
                        if self._are_entities_nearby(text, entity1, entity2, window=50):
                            relations.append({
                                'source': entity1['text'],
                                'target': entity2['text'],
                                'relation': 'HAS_EMPLOYEE',
                                'confidence': 0.7,
                                'source_type': 'COMPANY',
                                'target_type': 'PERSON'
                            })

                    elif i != j and entity1['type'] == 'PERSON' and entity2['type'] == 'COMPANY':
                        # 检查文本中是否同时出现且位置相近
                        if self._are_entities_nearby(text, entity1, entity2, window=50):
                            relations.append({
                                'source': entity1['text'],
                                'target': entity2['text'],
                                'relation': 'WORKS_FOR',
                                'confidence': 0.7,
                                'source_type': 'PERSON',
                                'target_type': 'COMPANY'
                            })

            return relations

        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []

    async def extract_entities_async(self, text: str) -> List[Dict[str, Any]]:
        """
        异步抽取文本中的实体

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        entities = await self.extractor.extract_entities(text)
        return [self._entity_to_dict(entity) for entity in entities]

    async def extract_relations_async(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        异步抽取实体间的关系

        Args:
            text: 输入文本
            entities: 实体列表

        Returns:
            关系列表
        """
        return self.extract_relations(text, entities)

    def _entity_to_dict(self, entity) -> Dict[str, Any]:
        """将Entity对象转换为字典"""
        return {
            'text': entity.text,
            'type': entity.type,
            'confidence': entity.confidence,
            'start': entity.start,
            'end': entity.end,
            'context': entity.context,
            'metadata': entity.metadata or {},
            'id': f"{entity.type}_{entity.text}_{entity.start}"
        }

    def _are_entities_nearby(self, text: str, entity1: Dict[str, Any], entity2: Dict[str, Any], window: int = 50) -> bool:
        """检查两个实体在文本中的距离是否在指定窗口内"""
        distance = abs(entity1['start'] - entity2['start'])
        return distance <= window