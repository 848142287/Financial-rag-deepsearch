"""
联网搜索服务
使用通义千问的联网搜索功能进行网络检索
"""

import asyncio
from app.core.structured_logging import get_structured_logger
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import re

logger = get_structured_logger(__name__)


class SearchStatus(Enum):
    """搜索状态"""
    PLANNING = "planning"
    SEARCHING = "searching"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SearchPlan:
    """搜索计划"""
    query: str
    search_queries: List[str]  # 拆分的搜索查询
    search_intent: str  # 搜索意图
    required_info: List[str]  # 需要收集的信息
    sources_to_check: List[str]  # 需要查看的来源类型


@dataclass
class WebSearchResult:
    """联网搜索结果"""
    status: SearchStatus
    query: str
    plan: SearchPlan
    search_results: List[Dict[str, Any]]  # 搜索到的网页内容
    markdown_report: str  # 生成的Markdown报告
    sources: List[Dict[str, Any]]  # 来源列表
    metadata: Dict[str, Any]  # 元数据
    timestamp: datetime


class WebSearchService:
    """联网搜索服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化联网搜索服务

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.api_key = self.config.get('qwen_api_key', 'sk-5233a3a4b1a24426b6846a432794bbe2')
        self.base_url = self.config.get(
            'qwen_base_url',
            'https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        # 初始化ChatTongqi模型
        try:
            from langchain_community.chat_models import ChatTongyi
            self.qwen_model = ChatTongyi(
                model='qwen-max',
                api_key=self.api_key,
                base_url=self.base_url,
                model_kwargs={
                    "enable_search": True,  # 开启联网搜索
                }
            )
            logger.info("ChatTongyi initialized with web search enabled")
        except ImportError:
            logger.warning("langchain_community not available, web search will be limited")
            self.qwen_model = None
        except Exception as e:
            logger.error(f"Failed to initialize ChatTongyi: {e}")
            self.qwen_model = None

    async def web_search(
        self,
        query: str,
        max_results: int = 10,
        enable_plan: bool = True
    ) -> WebSearchResult:
        """
        执行联网搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            enable_plan: 是否生成搜索计划

        Returns:
            联网搜索结果
        """
        timestamp = datetime.now()

        try:
            # 步骤1: 生成搜索计划
            if enable_plan:
                plan = await self._generate_search_plan(query)
                logger.info(f"Generated search plan: {plan.search_queries}")
            else:
                plan = SearchPlan(
                    query=query,
                    search_queries=[query],
                    search_intent="general",
                    required_info=[],
                    sources_to_check=[]
                )

            # 步骤2: 执行网页搜索
            search_results = await self._execute_web_search(
                plan.search_queries,
                max_results
            )

            # 步骤3: 生成Markdown报告
            markdown_report = await self._generate_markdown_report(
                query,
                plan,
                search_results
            )

            # 步骤4: 提取来源信息
            sources = self._extract_sources(search_results)

            # 构建结果
            result = WebSearchResult(
                status=SearchStatus.COMPLETED,
                query=query,
                plan=plan,
                search_results=search_results,
                markdown_report=markdown_report,
                sources=sources,
                metadata={
                    'total_results': len(search_results),
                    'search_queries_used': plan.search_queries,
                    'generation_time': datetime.now().isoformat()
                },
                timestamp=timestamp
            )

            # 步骤5: 保存报告文件
            await self._save_report(result)

            return result

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return WebSearchResult(
                status=SearchStatus.FAILED,
                query=query,
                plan=SearchPlan(query, [query], "general", [], []),
                search_results=[],
                markdown_report=f"搜索失败: {str(e)}",
                sources=[],
                metadata={'error': str(e)},
                timestamp=timestamp
            )

    async def _generate_search_plan(self, query: str) -> SearchPlan:
        """生成搜索计划"""
        planning_prompt = f"""请为以下查询生成搜索计划。

用户查询：{query}

请分析：
1. 用户的搜索意图是什么？
2. 需要拆分成哪些具体的搜索查询？
3. 需要收集哪些关键信息？
4. 应该查看哪些类型的来源？

请以JSON格式返回：
{{
    "search_intent": "搜索意图描述",
    "search_queries": ["查询1", "查询2", "查询3"],
    "required_info": ["信息1", "信息2"],
    "sources_to_check": ["官方来源", "新闻报道", "研究报告"]
}}"""

        try:
            if self.qwen_model:
                response = await asyncio.to_thread(
                    self.qwen_model.invoke,
                    planning_prompt
                )
                content = response.content

                # 解析JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group())

                    return SearchPlan(
                        query=query,
                        search_queries=plan_data.get('search_queries', [query]),
                        search_intent=plan_data.get('search_intent', 'general'),
                        required_info=plan_data.get('required_info', []),
                        sources_to_check=plan_data.get('sources_to_check', [])
                    )

        except Exception as e:
            logger.warning(f"Failed to generate search plan: {e}")

        # 降级方案：简单拆分
        return SearchPlan(
            query=query,
            search_queries=self._simple_query_split(query),
            search_intent="general",
            required_info=[],
            sources_to_check=[]
        )

    def _simple_query_split(self, query: str) -> List[str]:
        """简单的查询拆分"""
        queries = [query]

        # 如果查询较长，尝试拆分
        if len(query) > 50:
            # 按标点符号拆分
            parts = re.split(r'[，,；;、]', query)
            if len(parts) > 1:
                queries = [p.strip() for p in parts if p.strip()]

        return queries[:3]  # 最多3个查询

    async def _execute_web_search(
        self,
        search_queries: List[str],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """执行网页搜索"""
        all_results = []

        for search_query in search_queries:
            try:
                # 使用通义千问的联网搜索功能
                search_prompt = f"""请搜索以下信息：{search_query}

请提供：
1. 最相关的网页内容摘要
2. 信息来源和发布时间
3. 关键数据和事实

请以结构化的方式返回，包括标题、摘要、来源、时间等信息。"""

                if self.qwen_model:
                    response = await asyncio.to_thread(
                        self.qwen_model.invoke,
                        search_prompt
                    )

                    # 解析搜索结果
                    result = {
                        'query': search_query,
                        'content': response.content,
                        'model_response': response.content
                    }

                    all_results.append(result)

                # 避免请求过快
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Search failed for query '{search_query}': {e}")
                continue

        return all_results[:max_results]

    async def _generate_markdown_report(
        self,
        query: str,
        plan: SearchPlan,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """生成Markdown报告"""
        if not search_results:
            return f"# {query}\n\n未找到相关搜索结果。"

        # 使用LLM生成结构化报告
        report_prompt = f"""请基于以下搜索结果生成一份结构化的Markdown报告。

用户查询：{query}

搜索结果：
{json.dumps(search_results, ensure_ascii=False, indent=2)}

请生成一份包含以下部分的报告：
1. 概述
2. 主要发现
3. 详细分析
4. 数据和事实
5. 来源列表

使用Markdown格式，确保内容清晰、结构完整。"""

        try:
            if self.qwen_model:
                response = await asyncio.to_thread(
                    self.qwen_model.invoke,
                    report_prompt
                )

                markdown_content = response.content

                # 格式化增强
                markdown_content = self._enhance_markdown(
                    markdown_content,
                    query,
                    plan
                )

                return markdown_content

        except Exception as e:
            logger.error(f"Failed to generate markdown report: {e}")

        # 降级方案：简单拼接
        return self._generate_simple_report(query, search_results)

    def _enhance_markdown(
        self,
        markdown: str,
        query: str,
        plan: SearchPlan
    ) -> str:
        """增强Markdown格式"""
        # 添加标题
        enhanced = f"# {query}\n\n"
        enhanced += f"**搜索时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        enhanced += f"**搜索意图**: {plan.search_intent}\n\n"
        enhanced += "---\n\n"
        enhanced += markdown

        return enhanced

    def _generate_simple_report(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """生成简单报告"""
        report = f"# {query}\n\n"
        report += f"**搜索时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"

        for i, result in enumerate(search_results, 1):
            report += f"## 结果 {i}\n\n"
            report += f"**查询**: {result.get('query', 'N/A')}\n\n"
            report += f"{result.get('content', 'No content')}\n\n"
            report += "---\n\n"

        return report

    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从搜索结果中提取来源信息"""
        sources = []

        for result in search_results:
            content = result.get('content', '')
            model_response = result.get('model_response', '')

            # 尝试提取URL和来源
            urls = re.findall(r'https?://[^\s\)]+', content)
            if not urls:
                urls = re.findall(r'https?://[^\s\)]+', model_response)

            for url in urls:
                sources.append({
                    'url': url,
                    'type': self._guess_source_type(url)
                })

        return sources

    def _guess_source_type(self, url: str) -> str:
        """猜测来源类型"""
        if 'gov.cn' in url:
            return 'government'
        elif '.edu.' in url:
            return 'education'
        elif 'news' in url or 'sina' in url or '163' in url:
            return 'news'
        elif 'wiki' in url:
            return 'wiki'
        else:
            return 'other'

    async def _save_report(self, result: WebSearchResult):
        """保存报告文件"""
        try:
            from pathlib import Path

            # 创建保存目录
            save_dir = Path("reports/web_search")
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', result.query)[:50]
            filename = f"{timestamp_str}_{safe_query}.md"

            file_path = save_dir / filename

            # 保存Markdown文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result.markdown_report)

            logger.info(f"Saved web search report to: {file_path}")

            # 保存元数据JSON
            metadata_path = save_dir / f"{filename}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': result.query,
                    'status': result.status.value,
                    'sources': result.sources,
                    'metadata': result.metadata,
                    'timestamp': result.timestamp.isoformat()
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved metadata to: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")


# 使用示例
async def main():
    """测试联网搜索功能"""
    config = {
        'qwen_api_key': 'sk-5233a3a4b1a24426b6846a432794bbe2',
        'qwen_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    }

    service = WebSearchService(config)

    result = await service.web_search(
        query="2024年中国GDP增长率预测",
        max_results=5
    )

    print(result.markdown_report)


if __name__ == "__main__":
    asyncio.run(main())
