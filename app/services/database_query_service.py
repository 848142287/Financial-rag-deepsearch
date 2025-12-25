"""
数据库查询服务
负责通过自然语言接口从数据库提取内容并转换为知识库格式
"""

import io
import requests
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
import logging
from pathlib import Path

from sqlalchemy.orm import Session
from app.models.document import Document, DocumentStatus
from app.core.database import get_db
from app.services.consolidated_document_service import ConsolidatedDocumentService as DocumentProcessor

logger = logging.getLogger(__name__)


class DatabaseQueryService:
    """数据库查询服务"""

    def __init__(self, db: Session):
        self.db = db
        self.api_base_url = "http://192.168.36.14:6856/n2sql/get/result"
        self.supported_databases = [
            "mysql", "postgresql", "oracle", "sqlserver", "sqlite"
        ]
        self.supported_scenarios = [
            "financial_analysis", "risk_assessment", "portfolio_management",
            "market_research", "compliance_reporting", "custom_query"
        ]

    async def query_database_to_knowledge(
        self,
        natural_language_query: str,
        database_name: str,
        database_type: str,
        scenario: str,
        user_id: int,
        config: Optional[Dict] = None
    ) -> Tuple[bool, str, Optional[int]]:
        """
        通过自然语言查询数据库并将结果添加到知识库

        Args:
            natural_language_query: 自然语言查询
            database_name: 数据库名称
            database_type: 数据库类型
            scenario: 查询场景
            user_id: 用户ID
            config: 额外配置

        Returns:
            Tuple[bool, str, Optional[int]]: (是否成功, 消息, 文档ID)
        """
        try:
            # 验证输入参数
            if not self._validate_inputs(natural_language_query, database_type, scenario):
                return False, "Invalid input parameters", None

            # 调用自然语言转SQL接口
            csv_content = await self._call_n2sql_api(
                natural_language_query,
                database_name,
                database_type,
                scenario
            )

            if not csv_content:
                return False, "Failed to get data from database", None

            # 将CSV内容保存为文件
            csv_file_path = await self._save_csv_file(
                csv_content,
                natural_language_query,
                database_name,
                scenario
            )

            if not csv_file_path:
                return False, "Failed to save CSV file", None

            # 创建文档记录
            document_id = await self._create_document_record(
                csv_file_path,
                natural_language_query,
                database_name,
                database_type,
                scenario,
                user_id
            )

            if not document_id:
                return False, "Failed to create document record", None

            # 使用现有的文档处理器处理CSV
            processor = DocumentProcessor(self.db)
            success, message = await processor.process_document(document_id, config)

            if not success:
                return False, f"Failed to process document: {message}", None

            logger.info(f"Successfully added database query result to knowledge base, document_id: {document_id}")
            return True, f"Successfully added query result to knowledge base (document_id: {document_id})", document_id

        except Exception as e:
            logger.error(f"Error in query_database_to_knowledge: {str(e)}")
            return False, f"Error: {str(e)}", None

    def _validate_inputs(
        self,
        natural_language_query: str,
        database_type: str,
        scenario: str
    ) -> bool:
        """验证输入参数"""
        if not natural_language_query or len(natural_language_query.strip()) < 5:
            logger.warning("Natural language query too short")
            return False

        if database_type not in self.supported_databases:
            logger.warning(f"Unsupported database type: {database_type}")
            return False

        if scenario not in self.supported_scenarios:
            logger.warning(f"Unsupported scenario: {scenario}")
            return False

        return True

    async def _call_n2sql_api(
        self,
        natural_language_query: str,
        database_name: str,
        database_type: str,
        scenario: str
    ) -> Optional[str]:
        """
        调用自然语言转SQL API

        Args:
            natural_language_query: 自然语言查询
            database_name: 数据库名称
            database_type: 数据库类型
            scenario: 查询场景

        Returns:
            Optional[str]: CSV内容
        """
        try:
            # 构建请求参数
            params = {
                "query": natural_language_query.strip(),
                "database": database_name,
                "db_type": database_type,
                "scenario": scenario,
                "format": "csv"
            }

            # 设置请求头
            headers = {
                "Accept": "text/csv",
                "User-Agent": "Financial-RAG-System/1.0"
            }

            # 发送GET请求
            logger.info(f"Calling N2SQL API: {self.api_base_url} with params: {params}")

            response = requests.get(
                self.api_base_url,
                params=params,
                headers=headers,
                timeout=300  # 5分钟超时
            )

            if response.status_code == 200:
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')
                if 'text/csv' in content_type or 'text/plain' in content_type:
                    return response.text
                else:
                    logger.warning(f"Unexpected content type: {content_type}")
                    return response.text  # 尝试解析为文本
            else:
                logger.error(f"N2SQL API returned status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("N2SQL API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling N2SQL API: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in _call_n2sql_api: {str(e)}")
            return None

    async def _save_csv_file(
        self,
        csv_content: str,
        query: str,
        database_name: str,
        scenario: str
    ) -> Optional[str]:
        """
        保存CSV内容到文件

        Args:
            csv_content: CSV内容
            query: 原始查询
            database_name: 数据库名称
            scenario: 场景

        Returns:
            Optional[str]: 文件路径
        """
        try:
            # 创建文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 清理查询字符串，移除特殊字符
            clean_query = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)[:50]
            filename = f"db_query_{database_name}_{scenario}_{timestamp}_{clean_query}.csv"

            # 确保目录存在
            upload_dir = Path("uploads/database_queries")
            upload_dir.mkdir(parents=True, exist_ok=True)

            # 保存文件
            file_path = upload_dir / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            logger.info(f"CSV file saved to: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error saving CSV file: {str(e)}")
            return None

    async def _create_document_record(
        self,
        file_path: str,
        query: str,
        database_name: str,
        database_type: str,
        scenario: str,
        user_id: int
    ) -> Optional[int]:
        """
        创建文档记录

        Args:
            file_path: 文件路径
            query: 查询语句
            database_name: 数据库名称
            database_type: 数据库类型
            scenario: 场景
            user_id: 用户ID

        Returns:
            Optional[int]: 文档ID
        """
        try:
            # 创建文档记录
            document = Document(
                filename=Path(file_path).name,
                file_path=file_path,
                file_type="csv",
                file_size=Path(file_path).stat().st_size,
                status=DocumentStatus.PENDING,
                user_id=user_id,
                doc_metadata={
                    "source_type": "database_query",
                    "original_query": query,
                    "database_name": database_name,
                    "database_type": database_type,
                    "scenario": scenario,
                    "created_at": datetime.utcnow().isoformat(),
                    "api_endpoint": self.api_base_url
                }
            )

            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)

            logger.info(f"Created document record with ID: {document.id}")
            return document.id

        except Exception as e:
            logger.error(f"Error creating document record: {str(e)}")
            self.db.rollback()
            return None

    async def get_supported_databases(self) -> List[str]:
        """获取支持的数据库类型"""
        return self.supported_databases.copy()

    async def get_supported_scenarios(self) -> List[str]:
        """获取支持的场景"""
        return self.supported_scenarios.copy()

    async def validate_database_connection(
        self,
        database_name: str,
        database_type: str
    ) -> Tuple[bool, str]:
        """
        验证数据库连接

        Args:
            database_name: 数据库名称
            database_type: 数据库类型

        Returns:
            Tuple[bool, str]: (是否连接成功, 消息)
        """
        try:
            # 使用简单的测试查询
            test_query = "SELECT 1 as test"

            csv_content = await self._call_n2sql_api(
                test_query,
                database_name,
                database_type,
                "custom_query"
            )

            if csv_content:
                return True, "Database connection successful"
            else:
                return False, "Failed to connect to database"

        except Exception as e:
            return False, f"Connection error: {str(e)}"