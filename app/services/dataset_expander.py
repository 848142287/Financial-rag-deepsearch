"""
数据集扩展管理器
负责扩大和丰富文档数据集，支持多源数据采集和自动化处理
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from urllib.parse import urljoin, urlparse
import pandas as pd
from bs4 import BeautifulSoup
import feedparser
import re

from .data_balancer import DocumentMetadata, DocumentCategory, DocumentSource, data_balancer

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型"""
    RSS_FEED = "rss_feed"
    WEB_CRAWLER = "web_crawler"
    API_ENDPOINT = "api_endpoint"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"


class ProcessingStatus(Enum):
    """处理状态"""
    PENDING = "待处理"
    DOWNLOADING = "下载中"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "失败"
    SKIPPED = "跳过"


@dataclass
class DataSource:
    """数据源配置"""
    source_id: str
    name: str
    type: DataSourceType
    url: Optional[str] = None
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    category: DocumentCategory = DocumentCategory.NEWS_ANALYSIS
    source: DocumentSource = DocumentSource.OTHER
    update_frequency: int = 3600  # 更新频率（秒）
    enabled: bool = True
    last_updated: Optional[str] = None
    custom_parser: Optional[str] = None


@dataclass
class DocumentTask:
    """文档处理任务"""
    task_id: str
    source_id: str
    url: str
    title: str
    category: DocumentCategory
    source: DocumentSource
    status: ProcessingStatus
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetExpander:
    """数据集扩展器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据集扩展器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_sources: Dict[str, DataSource] = {}
        self.tasks: Dict[str, DocumentTask] = {}
        self.download_dir = self.config.get("download_dir", "/tmp/dataset_expander")
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 5)
        self.session: Optional[aiohttp.ClientSession] = None

        # 确保下载目录存在
        os.makedirs(self.download_dir, exist_ok=True)

        # 初始化数据源
        self._initialize_default_sources()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")

        return {
            "download_dir": "/tmp/dataset_expander",
            "max_concurrent_tasks": 5,
            "user_agent": "Financial-RAG-Dataset-Expander/1.0",
            "timeout": 30,
            "retry_attempts": 3
        }

    def _initialize_default_sources(self):
        """初始化默认数据源"""
        default_sources = [
            # 金融新闻RSS源
            DataSource(
                source_id="reuters_financial",
                name="路透社金融新闻",
                type=DataSourceType.RSS_FEED,
                url="https://www.reuters.com/rssFeed/businessNews",
                category=DocumentCategory.NEWS_ANALYSIS,
                source=DocumentSource.REUTERS,
                update_frequency=1800  # 30分钟
            ),
            DataSource(
                source_id="bloomberg_markets",
                name="彭博市场新闻",
                type=DataSourceType.RSS_FEED,
                url="https://feeds.bloomberg.com/markets/news.rss",
                category=DocumentCategory.MARKET_DATA,
                source=DocumentSource.BLOOMBERG,
                update_frequency=1800
            ),
            # 监管机构数据源
            DataSource(
                source_id="sec_filings",
                name="美国证监会文件",
                type=DataSourceType.API_ENDPOINT,
                url="https://www.sec.gov/Archives/edgar/data/",
                category=DocumentCategory.REGULATORY_FILING,
                source=DocumentSource.SEC,
                update_frequency=3600
            ),
            DataSource(
                source_id="csrc_announcements",
                name="中国证监会公告",
                type=DataSourceType.WEB_CRAWLER,
                url="http://www.csrc.gov.cn/pub/newsite/zjhxwfc/",
                category=DocumentCategory.POLICY_DOCUMENT,
                source=DocumentSource.CSRC,
                update_frequency=3600
            ),
            # 交易所数据
            DataSource(
                source_id="sse_disclosure",
                name="上交所披露",
                type=DataSourceType.WEB_CRAWLER,
                url="http://www.sse.com.cn/disclosure/",
                category=DocumentCategory.ANNUAL_REPORT,
                source=DocumentSource.SSE,
                update_frequency=1800
            ),
            DataSource(
                source_id="szse_disclosure",
                name="深交所披露",
                type=DataSourceType.WEB_CRAWLER,
                url="http://www.szse.cn/disclosure/",
                category=DocumentCategory.QUARTERLY_REPORT,
                source=DocumentSource.SZSE,
                update_frequency=1800
            )
        ]

        for source in default_sources:
            self.data_sources[source.source_id] = source

        logger.info(f"已初始化{len(default_sources)}个默认数据源")

    async def start_session(self):
        """启动HTTP会话"""
        if not self.session:
            headers = {
                "User-Agent": self.config.get("user_agent", "Financial-RAG-Dataset-Expander/1.0")
            }
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            self.session = None

    async def add_data_source(self, source: DataSource) -> bool:
        """
        添加数据源

        Args:
            source: 数据源配置

        Returns:
            是否添加成功
        """
        try:
            self.data_sources[source.source_id] = source
            logger.info(f"数据源添加成功: {source.name}")
            return True
        except Exception as e:
            logger.error(f"添加数据源失败: {str(e)}")
            return False

    async def remove_data_source(self, source_id: str) -> bool:
        """
        移除数据源

        Args:
            source_id: 数据源ID

        Returns:
            是否移除成功
        """
        try:
            if source_id in self.data_sources:
                del self.data_sources[source_id]
                logger.info(f"数据源移除成功: {source_id}")
                return True
            else:
                logger.warning(f"数据源不存在: {source_id}")
                return False
        except Exception as e:
            logger.error(f"移除数据源失败: {str(e)}")
            return False

    async def collect_from_source(self, source_id: str) -> List[DocumentTask]:
        """
        从指定数据源收集文档

        Args:
            source_id: 数据源ID

        Returns:
            创建的任务列表
        """
        if source_id not in self.data_sources:
            logger.error(f"数据源不存在: {source_id}")
            return []

        source = self.data_sources[source_id]
        if not source.enabled:
            logger.info(f"数据源已禁用: {source_id}")
            return []

        await self.start_session()

        try:
            if source.type == DataSourceType.RSS_FEED:
                return await self._collect_from_rss(source)
            elif source.type == DataSourceType.WEB_CRAWLER:
                return await self._collect_from_web(source)
            elif source.type == DataSourceType.API_ENDPOINT:
                return await self._collect_from_api(source)
            else:
                logger.warning(f"不支持的数据源类型: {source.type}")
                return []

        except Exception as e:
            logger.error(f"从数据源收集失败 {source_id}: {str(e)}")
            return []

    async def _collect_from_rss(self, source: DataSource) -> List[DocumentTask]:
        """从RSS源收集文档"""
        tasks = []

        try:
            logger.info(f"开始从RSS源收集: {source.name}")

            feed = feedparser.parse(source.url)
            if feed.bozo:
                logger.warning(f"RSS解析警告: {source.url}, {feed.bozo_exception}")

            for entry in feed.entries[:20]:  # 限制最新20条
                try:
                    # 生成任务ID
                    task_id = hashlib.md5(f"{source.source_id}_{entry.link}".encode()).hexdigest()

                    # 检查是否已存在
                    if task_id in self.tasks:
                        continue

                    # 创建任务
                    task = DocumentTask(
                        task_id=task_id,
                        source_id=source.source_id,
                        url=entry.link,
                        title=entry.get('title', '未知标题'),
                        category=source.category,
                        source=source.source,
                        status=ProcessingStatus.PENDING,
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                        metadata={
                            'published': entry.get('published'),
                            'summary': entry.get('summary', ''),
                            'author': entry.get('author', '')
                        }
                    )

                    self.tasks[task_id] = task
                    tasks.append(task)

                except Exception as e:
                    logger.warning(f"处理RSS条目失败: {str(e)}")
                    continue

            logger.info(f"从RSS源创建{len(tasks)}个任务: {source.name}")
            return tasks

        except Exception as e:
            logger.error(f"RSS收集失败: {source.url}, {str(e)}")
            return []

    async def _collect_from_web(self, source: DataSource) -> List[DocumentTask]:
        """从网页爬取收集文档"""
        tasks = []

        try:
            logger.info(f"开始从网页爬取: {source.name}")

            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.error(f"网页访问失败: {source.url}, 状态码: {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # 查找文档链接
                links = self._extract_document_links(soup, source.url)

                for link in links[:30]:  # 限制30个链接
                    try:
                        # 生成任务ID
                        task_id = hashlib.md5(f"{source.source_id}_{link}".encode()).hexdigest()

                        # 检查是否已存在
                        if task_id in self.tasks:
                            continue

                        # 提取标题
                        title = self._extract_title_from_url(link)

                        # 创建任务
                        task = DocumentTask(
                            task_id=task_id,
                            source_id=source.source_id,
                            url=link,
                            title=title,
                            category=source.category,
                            source=source.source,
                            status=ProcessingStatus.PENDING,
                            created_at=datetime.now().isoformat(),
                            updated_at=datetime.now().isoformat()
                        )

                        self.tasks[task_id] = task
                        tasks.append(task)

                    except Exception as e:
                        logger.warning(f"处理网页链接失败: {link}, {str(e)}")
                        continue

            logger.info(f"从网页爬取创建{len(tasks)}个任务: {source.name}")
            return tasks

        except Exception as e:
            logger.error(f"网页爬取失败: {source.url}, {str(e)}")
            return []

    async def _collect_from_api(self, source: DataSource) -> List[DocumentTask]:
        """从API端点收集文档"""
        tasks = []

        try:
            logger.info(f"开始从API收集: {source.name}")

            headers = source.headers or {}
            if source.api_key:
                headers['Authorization'] = f'Bearer {source.api_key}'

            params = source.params or {}
            params.update({
                'limit': 50,
                'sort_by': 'date',
                'order': 'desc'
            })

            async with self.session.get(
                source.url,
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    logger.error(f"API访问失败: {source.url}, 状态码: {response.status}")
                    return []

                data = await response.json()

                # 处理API响应
                documents = self._parse_api_response(data, source.type)

                for doc in documents:
                    try:
                        # 生成任务ID
                        task_id = hashlib.md5(f"{source.source_id}_{doc['url']}".encode()).hexdigest()

                        # 检查是否已存在
                        if task_id in self.tasks:
                            continue

                        # 创建任务
                        task = DocumentTask(
                            task_id=task_id,
                            source_id=source.source_id,
                            url=doc['url'],
                            title=doc.get('title', '未知标题'),
                            category=source.category,
                            source=source.source,
                            status=ProcessingStatus.PENDING,
                            created_at=datetime.now().isoformat(),
                            updated_at=datetime.now().isoformat(),
                            metadata=doc.get('metadata', {})
                        )

                        self.tasks[task_id] = task
                        tasks.append(task)

                    except Exception as e:
                        logger.warning(f"处理API文档失败: {str(e)}")
                        continue

            logger.info(f"从API创建{len(tasks)}个任务: {source.name}")
            return tasks

        except Exception as e:
            logger.error(f"API收集失败: {source.url}, {str(e)}")
            return []

    def _extract_document_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """从网页中提取文档链接"""
        links = []

        try:
            # 查找PDF、DOC、DOCX等文档链接
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)

                # 检查文件扩展名
                if any(absolute_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                    links.append(absolute_url)

            # 去重
            links = list(set(links))
            return links

        except Exception as e:
            logger.error(f"提取文档链接失败: {str(e)}")
            return []

    def _extract_title_from_url(self, url: str) -> str:
        """从URL提取标题"""
        try:
            # 从URL路径中提取文件名
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)

            # 移除扩展名
            name_without_ext = os.path.splitext(filename)[0]

            # 清理和格式化标题
            title = re.sub(r'[-_]', ' ', name_without_ext)
            title = re.sub(r'\s+', ' ', title).strip()

            return title if title else "未知文档"

        except Exception:
            return "未知文档"

    def _parse_api_response(self, data: Dict[str, Any], api_type: str) -> List[Dict[str, Any]]:
        """解析API响应"""
        documents = []

        try:
            if api_type == "sec_filings":
                # SEC文件API响应格式
                filings = data.get('filings', [])
                for filing in filings:
                    documents.append({
                        'url': filing.get('filingUrl', ''),
                        'title': filing.get('description', ''),
                        'metadata': {
                            'company': filing.get('companyName', ''),
                            'filing_type': filing.get('type', ''),
                            'filing_date': filing.get('filedAt', '')
                        }
                    })

            else:
                # 通用API响应格式
                items = data.get('items', data.get('results', []))
                for item in items:
                    documents.append({
                        'url': item.get('url', item.get('link', '')),
                        'title': item.get('title', item.get('name', '')),
                        'metadata': {
                            'description': item.get('description', ''),
                            'date': item.get('date', item.get('published', ''))
                        }
                    })

            return documents

        except Exception as e:
            logger.error(f"解析API响应失败: {str(e)}")
            return []

    async def process_task(self, task_id: str) -> bool:
        """
        处理单个任务

        Args:
            task_id: 任务ID

        Returns:
            是否处理成功
        """
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return False

        task = self.tasks[task_id]

        try:
            # 更新任务状态
            task.status = ProcessingStatus.DOWNLOADING
            task.updated_at = datetime.now().isoformat()

            # 下载文件
            file_path = await self._download_document(task.url, task_id)
            if not file_path:
                task.status = ProcessingStatus.FAILED
                task.error_message = "文件下载失败"
                return False

            task.file_path = file_path
            task.status = ProcessingStatus.PROCESSING

            # 处理文档并添加到数据平衡器
            success = await self._process_and_add_document(task, file_path)

            if success:
                task.status = ProcessingStatus.COMPLETED
                logger.info(f"任务处理成功: {task_id}")
                return True
            else:
                task.status = ProcessingStatus.FAILED
                task.error_message = "文档处理失败"
                return False

        except Exception as e:
            logger.error(f"任务处理失败 {task_id}: {str(e)}")
            task.status = ProcessingStatus.FAILED
            task.error_message = str(e)
            return False

    async def _download_document(self, url: str, task_id: str) -> Optional[str]:
        """下载文档"""
        try:
            await self.start_session()

            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"下载失败: {url}, 状态码: {response.status}")
                    return None

                # 获取文件扩展名
                content_type = response.headers.get('content-type', '')
                extension = self._get_extension_from_content_type(content_type)

                # 生成文件路径
                filename = f"{task_id}{extension}"
                file_path = os.path.join(self.download_dir, filename)

                # 保存文件
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

                logger.info(f"文件下载成功: {file_path}")
                return file_path

        except Exception as e:
            logger.error(f"下载文档失败: {url}, {str(e)}")
            return None

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """根据内容类型获取文件扩展名"""
        type_mapping = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'text/plain': '.txt',
            'text/html': '.html'
        }

        return type_mapping.get(content_type.lower(), '.bin')

    async def _process_and_add_document(self, task: DocumentTask, file_path: str) -> bool:
        """处理文档并添加到数据平衡器"""
        try:
            # 获取文件信息
            file_size = os.path.getsize(file_path)

            # 计算质量分数（简单实现，可以后续优化）
            quality_score = await self._calculate_document_quality(file_path, task)

            # 创建文档元数据
            metadata = DocumentMetadata(
                document_id=task.task_id,
                category=task.category,
                source=task.source,
                file_path=file_path,
                file_size=file_size,
                page_count=await self._get_page_count(file_path),
                upload_time=task.created_at,
                quality_score=quality_score,
                tags=self._extract_tags(task),
                language="zh"
            )

            # 添加到数据平衡器
            success = await data_balancer.add_document(metadata)

            if success:
                logger.info(f"文档添加到平衡器成功: {task.task_id}")
            else:
                logger.warning(f"文档添加到平衡器失败: {task.task_id}")

            return success

        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            return False

    async def _calculate_document_quality(self, file_path: str, task: DocumentTask) -> float:
        """计算文档质量分数"""
        try:
            # 基础分数
            base_score = 0.7

            # 根据文件大小调整
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:  # 1MB
                base_score += 0.1
            elif file_size < 10 * 1024:  # 10KB
                base_score -= 0.2

            # 根据标题质量调整
            title = task.title.lower()
            if any(keyword in title for keyword in ['年报', '季报', '报告', '分析']):
                base_score += 0.1

            # 根据来源调整
            if task.source in [DocumentSource.SEC, DocumentSource.CSRC, DocumentSource.SSE, DocumentSource.SZSE]:
                base_score += 0.1

            # 根据元数据调整
            if task.metadata:
                if task.metadata.get('published') or task.metadata.get('filing_date'):
                    base_score += 0.05

            return min(1.0, max(0.0, base_score))

        except Exception:
            return 0.5

    async def _get_page_count(self, file_path: str) -> int:
        """获取文档页数"""
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == '.pdf':
                from PyPDF2 import PdfReader
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    return len(reader.pages)
            else:
                return 1  # 非PDF文档默认1页
        except Exception:
            return 1

    def _extract_tags(self, task: DocumentTask) -> List[str]:
        """提取文档标签"""
        tags = []

        try:
            # 从类别添加标签
            tags.append(task.category.value)

            # 从来源添加标签
            tags.append(task.source.value)

            # 从标题提取关键词
            title = task.title.lower()
            keywords = ['年报', '季报', '半年报', '招股说明书', '重组', '并购', '投资', '分析']
            for keyword in keywords:
                if keyword in title:
                    tags.append(keyword)

        except Exception as e:
            logger.warning(f"提取标签失败: {str(e)}")

        return tags

    async def process_pending_tasks(self) -> Dict[str, int]:
        """批量处理待处理任务"""
        await self.start_session()

        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        try:
            # 获取待处理任务
            pending_tasks = [
                task for task in self.tasks.values()
                if task.status == ProcessingStatus.PENDING
            ]

            results['total'] = len(pending_tasks)
            logger.info(f"开始批量处理{len(pending_tasks)}个待处理任务")

            # 限制并发数
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

            async def process_with_semaphore(task):
                async with semaphore:
                    return await self.process_task(task.task_id)

            # 并发处理
            tasks_list = [process_with_semaphore(task) for task in pending_tasks]
            results_list = await asyncio.gather(*tasks_list, return_exceptions=True)

            # 统计结果
            for result in results_list:
                if isinstance(result, Exception):
                    results['failed'] += 1
                elif result:
                    results['success'] += 1
                else:
                    results['failed'] += 1

            logger.info(f"批量处理完成: 成功{results['success']}, 失败{results['failed']}")

        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")

        finally:
            await self.close_session()

        return results

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            total_tasks = len(self.tasks)
            status_counts = {}
            source_counts = {}
            category_counts = {}

            for task in self.tasks.values():
                # 状态统计
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

                # 来源统计
                source = task.source.value
                source_counts[source] = source_counts.get(source, 0) + 1

                # 类别统计
                category = task.category.value
                category_counts[category] = category_counts.get(category, 0) + 1

            return {
                'total_tasks': total_tasks,
                'total_sources': len(self.data_sources),
                'enabled_sources': sum(1 for s in self.data_sources.values() if s.enabled),
                'status_distribution': status_counts,
                'source_distribution': source_counts,
                'category_distribution': category_counts,
                'download_dir': self.download_dir
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {'error': str(e)}

    async def export_tasks(self, output_path: str):
        """导出任务数据"""
        try:
            tasks_data = []
            for task in self.tasks.values():
                task_dict = {
                    'task_id': task.task_id,
                    'source_id': task.source_id,
                    'url': task.url,
                    'title': task.title,
                    'category': task.category.value,
                    'source': task.source.value,
                    'status': task.status.value,
                    'created_at': task.created_at,
                    'updated_at': task.updated_at,
                    'error_message': task.error_message,
                    'file_path': task.file_path,
                    'metadata': task.metadata
                }
                tasks_data.append(task_dict)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)

            logger.info(f"任务数据已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出任务数据失败: {str(e)}")


# 全局数据集扩展器实例
dataset_expander = DatasetExpander()