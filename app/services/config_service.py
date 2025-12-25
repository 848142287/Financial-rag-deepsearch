"""
配置管理服务
负责动态管理和调整同步配置
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.synchronization import SyncConfiguration

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""

    def __init__(self, db: Session):
        self.db = db
        self._config_cache = {}
        self._last_cache_update = {}

    async def get_configuration(
        self, name: str, use_cache: bool = True
    ) -> Optional[Dict]:
        """
        获取配置

        Args:
            name: 配置名称
            use_cache: 是否使用缓存

        Returns:
            Optional[Dict]: 配置内容
        """
        try:
            # 检查缓存
            if use_cache and name in self._config_cache:
                # 检查缓存是否过期（5分钟）
                last_update = self._last_cache_update.get(name, datetime.min)
                if (datetime.utcnow() - last_update).seconds < 300:
                    return self._config_cache[name]

            # 从数据库获取
            config = self.db.query(SyncConfiguration).filter(
                and_(
                    SyncConfiguration.name == name,
                    SyncConfiguration.is_active == True
                )
            ).first()

            if not config:
                logger.warning(f"Configuration '{name}' not found")
                return None

            # 转换为字典
            config_dict = {
                "id": config.id,
                "name": config.name,
                "description": config.description,
                "enable_vision_model": config.enable_vision_model,
                "enable_entity_filter": config.enable_entity_filter,
                "enable_incremental_sync": config.enable_incremental_sync,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "vector_dimension": config.vector_dimension,
                "embedding_model": config.embedding_model,
                "entity_types": config.entity_types,
                "extraction_rules": config.extraction_rules,
                "sync_batch_size": config.sync_batch_size,
                "retry_attempts": config.retry_attempts,
                "timeout_seconds": config.timeout_seconds,
                "applicable_document_types": config.applicable_document_types,
                "created_at": config.created_at.isoformat() if config.created_at else None,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }

            # 更新缓存
            self._config_cache[name] = config_dict
            self._last_cache_update[name] = datetime.utcnow()

            return config_dict

        except Exception as e:
            logger.error(f"Error getting configuration '{name}': {str(e)}")
            return None

    async def create_configuration(
        self,
        name: str,
        description: Optional[str] = None,
        config_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        创建配置

        Args:
            name: 配置名称
            description: 配置描述
            config_data: 配置数据

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            # 检查名称是否已存在
            existing_config = self.db.query(SyncConfiguration).filter(
                SyncConfiguration.name == name
            ).first()

            if existing_config:
                return False, f"Configuration '{name}' already exists"

            # 使用默认值或提供的数据
            config = SyncConfiguration(
                name=name,
                description=description or f"Configuration for {name}",
                enable_vision_model=config_data.get("enable_vision_model", False),
                enable_entity_filter=config_data.get("enable_entity_filter", True),
                enable_incremental_sync=config_data.get("enable_incremental_sync", True),
                chunk_size=config_data.get("chunk_size", 1000),
                chunk_overlap=config_data.get("chunk_overlap", 200),
                vector_dimension=config_data.get("vector_dimension", 1536),
                embedding_model=config_data.get("embedding_model", "text-embedding-ada-002"),
                entity_types=config_data.get("entity_types", [
                    "company", "person", "financial_product", "concept", "numeric_entity"
                ]),
                extraction_rules=config_data.get("extraction_rules", {
                    "cross_modal_association": True,
                    "entity_disambiguation": True,
                    "company_normalization": True
                }),
                sync_batch_size=config_data.get("sync_batch_size", 100),
                retry_attempts=config_data.get("retry_attempts", 3),
                timeout_seconds=config_data.get("timeout_seconds", 300),
                applicable_document_types=config_data.get("applicable_document_types", [
                    "pdf", "docx", "txt", "csv", "xlsx", "md"
                ])
            )

            self.db.add(config)
            self.db.commit()

            # 清除缓存
            if name in self._config_cache:
                del self._config_cache[name]
                if name in self._last_cache_update:
                    del self._last_cache_update[name]

            logger.info(f"Created configuration '{name}'")
            return True, f"Configuration '{name}' created successfully"

        except Exception as e:
            logger.error(f"Error creating configuration '{name}': {str(e)}")
            return False, f"Failed to create configuration: {str(e)}"

    async def update_configuration(
        self, name: str, updates: Dict
    ) -> Tuple[bool, str]:
        """
        更新配置

        Args:
            name: 配置名称
            updates: 更新内容

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            config = self.db.query(SyncConfiguration).filter(
                SyncConfiguration.name == name
            ).first()

            if not config:
                return False, f"Configuration '{name}' not found"

            # 更新字段
            for field, value in updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
                elif field in ["entity_types", "extraction_rules", "applicable_document_types"]:
                    # JSON字段特殊处理
                    setattr(config, field, value)

            config.updated_at = datetime.utcnow()
            self.db.commit()

            # 清除缓存
            if name in self._config_cache:
                del self._config_cache[name]
                if name in self._last_cache_update:
                    del self._last_cache_update[name]

            logger.info(f"Updated configuration '{name}'")
            return True, f"Configuration '{name}' updated successfully"

        except Exception as e:
            logger.error(f"Error updating configuration '{name}': {str(e)}")
            return False, f"Failed to update configuration: {str(e)}"

    async def delete_configuration(self, name: str) -> Tuple[bool, str]:
        """
        删除配置

        Args:
            name: 配置名称

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            config = self.db.query(SyncConfiguration).filter(
                SyncConfiguration.name == name
            ).first()

            if not config:
                return False, f"Configuration '{name}' not found"

            # 软删除：标记为非活跃
            config.is_active = False
            config.updated_at = datetime.utcnow()
            self.db.commit()

            # 清除缓存
            if name in self._config_cache:
                del self._config_cache[name]
                if name in self._last_cache_update:
                    del self._last_cache_update[name]

            logger.info(f"Deactivated configuration '{name}'")
            return True, f"Configuration '{name}' deactivated successfully"

        except Exception as e:
            logger.error(f"Error deleting configuration '{name}': {str(e)}")
            return False, f"Failed to delete configuration: {str(e)}"

    async def list_configurations(
        self, active_only: bool = True
    ) -> List[Dict]:
        """
        列出配置

        Args:
            active_only: 是否只列活跃配置

        Returns:
            List[Dict]: 配置列表
        """
        try:
            query = self.db.query(SyncConfiguration)

            if active_only:
                query = query.filter(SyncConfiguration.is_active == True)

            configs = query.all()

            result = []
            for config in configs:
                result.append({
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "enable_vision_model": config.enable_vision_model,
                    "enable_entity_filter": config.enable_entity_filter,
                    "enable_incremental_sync": config.enable_incremental_sync,
                    "chunk_size": config.chunk_size,
                    "vector_dimension": config.vector_dimension,
                    "embedding_model": config.embedding_model,
                    "created_at": config.created_at.isoformat() if config.created_at else None,
                    "updated_at": config.updated_at.isoformat() if config.updated_at else None,
                    "is_active": config.is_active
                })

            return result

        except Exception as e:
            logger.error(f"Error listing configurations: {str(e)}")
            return []

    async def get_configuration_for_document_type(
        self, document_type: str
    ) -> Optional[Dict]:
        """
        获取适用于特定文档类型的配置

        Args:
            document_type: 文档类型

        Returns:
            Optional[Dict]: 配置内容
        """
        try:
            # 查找适用于该文档类型的配置
            configs = self.db.query(SyncConfiguration).filter(
                and_(
                    SyncConfiguration.is_active == True,
                    SyncConfiguration.applicable_document_types.contains([document_type])
                )
            ).all()

            if not configs:
                # 返回默认配置
                return await self.get_configuration("default")

            # 返回第一个匹配的配置（可以优化为更复杂的选择逻辑）
            config = configs[0]
            return await self.get_configuration(config.name)

        except Exception as e:
            logger.error(f"Error getting configuration for document type '{document_type}': {str(e)}")
            return None


class PresetConfigurations:
    """预设配置"""

    @staticmethod
    def get_stock_report_config() -> Dict:
        """个股研报配置"""
        return {
            "enable_vision_model": True,
            "enable_entity_filter": True,
            "enable_incremental_sync": True,
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "vector_dimension": 1536,
            "embedding_model": "text-embedding-ada-002",
            "entity_types": [
                "company", "person", "financial_product", "concept", "numeric_entity",
                "stock_symbol", "financial_metric"
            ],
            "extraction_rules": {
                "cross_modal_association": True,
                "entity_disambiguation": True,
                "company_normalization": True,
                "stock_price_extraction": True,
                "financial_ratio_extraction": True
            },
            "sync_batch_size": 50,
            "retry_attempts": 3,
            "timeout_seconds": 600,
            "applicable_document_types": ["pdf", "docx"]
        }

    @staticmethod
    def get_industry_report_config() -> Dict:
        """行业研报配置"""
        return {
            "enable_vision_model": True,
            "enable_entity_filter": True,
            "enable_incremental_sync": True,
            "chunk_size": 2000,
            "chunk_overlap": 400,
            "vector_dimension": 1536,
            "embedding_model": "text-embedding-ada-002",
            "entity_types": [
                "company", "person", "financial_product", "concept", "numeric_entity",
                "industry", "market_segment", "technology"
            ],
            "extraction_rules": {
                "cross_modal_association": True,
                "entity_disambiguation": True,
                "company_normalization": True,
                "trend_analysis": True,
                "market_share_analysis": True
            },
            "sync_batch_size": 80,
            "retry_attempts": 3,
            "timeout_seconds": 900,
            "applicable_document_types": ["pdf", "docx", "xlsx"]
        }

    @staticmethod
    def get_macro_report_config() -> Dict:
        """宏观研报配置"""
        return {
            "enable_vision_model": True,
            "enable_entity_filter": True,
            "enable_incremental_sync": True,
            "chunk_size": 1800,
            "chunk_overlap": 350,
            "vector_dimension": 1536,
            "embedding_model": "text-embedding-ada-002",
            "entity_types": [
                "company", "person", "financial_product", "concept", "numeric_entity",
                "country", "region", "economic_indicator", "policy"
            ],
            "extraction_rules": {
                "cross_modal_association": True,
                "entity_disambiguation": True,
                "economic_indicator_extraction": True,
                "policy_impact_analysis": True
            },
            "sync_batch_size": 60,
            "retry_attempts": 3,
            "timeout_seconds": 1200,
            "applicable_document_types": ["pdf", "docx", "csv", "xlsx"]
        }


class ConfigService:
    """配置服务主类"""

    def __init__(self, db: Session):
        self.db = db
        self.config_manager = ConfigManager(db)
        self.preset_configs = PresetConfigurations()

    async def initialize_default_configurations(self):
        """初始化默认配置"""
        try:
            default_configs = [
                ("default", "默认同步配置", self.preset_configs.get_stock_report_config()),
                ("stock_report", "个股研报配置", self.preset_configs.get_stock_report_config()),
                ("industry_report", "行业研报配置", self.preset_configs.get_industry_report_config()),
                ("macro_report", "宏观研报配置", self.preset_configs.get_macro_report_config()),
                ("lightweight", "轻量级配置", {
                    "enable_vision_model": False,
                    "enable_entity_filter": False,
                    "enable_incremental_sync": True,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "vector_dimension": 768,
                    "embedding_model": "text-embedding-ada-002",
                    "entity_types": ["company", "person"],
                    "extraction_rules": {},
                    "sync_batch_size": 200,
                    "retry_attempts": 2,
                    "timeout_seconds": 180,
                    "applicable_document_types": ["txt", "md"]
                }),
                ("high_performance", "高性能配置", {
                    "enable_vision_model": True,
                    "enable_entity_filter": True,
                    "enable_incremental_sync": True,
                    "chunk_size": 2500,
                    "chunk_overlap": 500,
                    "vector_dimension": 3072,
                    "embedding_model": "text-embedding-3-large",
                    "entity_types": [
                        "company", "person", "financial_product", "concept",
                        "numeric_entity", "stock_symbol", "financial_metric",
                        "industry", "market_segment", "technology", "country",
                        "region", "economic_indicator", "policy"
                    ],
                    "extraction_rules": {
                        "cross_modal_association": True,
                        "entity_disambiguation": True,
                        "company_normalization": True,
                        "stock_price_extraction": True,
                        "financial_ratio_extraction": True,
                        "trend_analysis": True,
                        "market_share_analysis": True,
                        "economic_indicator_extraction": True,
                        "policy_impact_analysis": True
                    },
                    "sync_batch_size": 30,
                    "retry_attempts": 5,
                    "timeout_seconds": 1800,
                    "applicable_document_types": ["pdf", "docx", "xlsx", "csv"]
                })
            ]

            created_count = 0
            for name, description, config_data in default_configs:
                success, _ = await self.config_manager.create_configuration(
                    name, description, config_data
                )
                if success:
                    created_count += 1
                else:
                    logger.info(f"Configuration '{name}' already exists, skipping")

            logger.info(f"Initialized {created_count} default configurations")
            return True

        except Exception as e:
            logger.error(f"Error initializing default configurations: {str(e)}")
            return False

    async def get_appropriate_config(
        self, document_type: str, report_type: Optional[str] = None
    ) -> Dict:
        """
        获取适合的配置

        Args:
            document_type: 文档类型
            report_type: 报告类型（可选）

        Returns:
            Dict: 配置内容
        """
        try:
            # 优先根据报告类型选择配置
            if report_type:
                config_map = {
                    "stock": "stock_report",
                    "industry": "industry_report",
                    "macro": "macro_report",
                    "individual": "stock_report",
                    "sector": "industry_report",
                    "economic": "macro_report"
                }

                config_name = config_map.get(report_type.lower())
                if config_name:
                    config = await self.config_manager.get_configuration(config_name)
                    if config:
                        return config

            # 根据文档类型选择配置
            config = await self.config_manager.get_configuration_for_document_type(
                document_type
            )
            if config:
                return config

            # 返回默认配置
            default_config = await self.config_manager.get_configuration("default")
            if default_config:
                return default_config

            # 返回硬编码的默认配置
            logger.warning("No configuration found, using hardcoded defaults")
            return self.preset_configs.get_stock_report_config()

        except Exception as e:
            logger.error(f"Error getting appropriate config: {str(e)}")
            return self.preset_configs.get_stock_report_config()

    async def validate_configuration(self, config_data: Dict) -> Tuple[bool, List[str]]:
        """
        验证配置数据

        Args:
            config_data: 配置数据

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []

        # 验证必需字段
        required_fields = ["chunk_size", "vector_dimension", "embedding_model"]
        for field in required_fields:
            if field not in config_data or config_data[field] is None:
                errors.append(f"Missing required field: {field}")

        # 验证数值范围
        if "chunk_size" in config_data:
            chunk_size = config_data["chunk_size"]
            if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 10000:
                errors.append("chunk_size must be an integer between 100 and 10000")

        if "vector_dimension" in config_data:
            vector_dim = config_data["vector_dimension"]
            if not isinstance(vector_dim, int) or vector_dim < 128 or vector_dim > 32768:
                errors.append("vector_dimension must be an integer between 128 and 32768")

        if "sync_batch_size" in config_data:
            batch_size = config_data["sync_batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
                errors.append("sync_batch_size must be an integer between 1 and 1000")

        # 验证列表字段
        if "entity_types" in config_data:
            entity_types = config_data["entity_types"]
            if not isinstance(entity_types, list):
                errors.append("entity_types must be a list")
            elif len(entity_types) == 0:
                errors.append("entity_types cannot be empty")

        # 验证布尔字段
        boolean_fields = [
            "enable_vision_model", "enable_entity_filter", "enable_incremental_sync"
        ]
        for field in boolean_fields:
            if field in config_data and not isinstance(config_data[field], bool):
                errors.append(f"{field} must be a boolean value")

        return len(errors) == 0, errors

    async def export_configuration(self, name: str) -> Optional[str]:
        """
        导出配置为JSON字符串

        Args:
            name: 配置名称

        Returns:
            Optional[str]: JSON字符串
        """
        try:
            config = await self.config_manager.get_configuration(name)
            if not config:
                return None

            # 移除数据库字段
            export_data = {
                k: v for k, v in config.items()
                if k not in ["id", "created_at", "updated_at"]
            }

            return json.dumps(export_data, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error exporting configuration '{name}': {str(e)}")
            return None

    async def import_configuration(
        self, name: str, json_data: str, overwrite: bool = False
    ) -> Tuple[bool, str]:
        """
        从JSON字符串导入配置

        Args:
            name: 配置名称
            json_data: JSON数据
            overwrite: 是否覆盖现有配置

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            config_data = json.loads(json_data)

            # 验证配置
            is_valid, errors = await self.validate_configuration(config_data)
            if not is_valid:
                return False, f"Invalid configuration: {'; '.join(errors)}"

            # 检查是否已存在
            existing = await self.config_manager.get_configuration(name, use_cache=False)
            if existing and not overwrite:
                return False, f"Configuration '{name}' already exists. Use overwrite=True to replace it."

            # 创建或更新配置
            if existing and overwrite:
                success, message = await self.config_manager.update_configuration(name, config_data)
            else:
                # 添加描述
                if "description" not in config_data:
                    config_data["description"] = f"Imported configuration: {name}"

                success, message = await self.config_manager.create_configuration(
                    name, config_data.get("description"), config_data
                )

            return success, message

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON data: {str(e)}"
        except Exception as e:
            logger.error(f"Error importing configuration '{name}': {str(e)}")
            return False, f"Failed to import configuration: {str(e)}"