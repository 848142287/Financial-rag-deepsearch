#!/usr/bin/env python3
"""
数据初始化脚本
用于初始化数据库和导入基础数据
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import get_db
from app.core.db_init import db_initializer
from app.models.document import Document, DocumentChunk
from app.models.conversation import Conversation, Message
from app.models.admin import SystemConfig

class DataInitializer:
    """数据初始化器"""

    def __init__(self):
        self.initialized = False

    async def initialize_all(self) -> bool:
        """初始化所有数据"""
        try:
            logger.info("开始初始化系统数据...")

            # 1. 初始化数据库结构
            await self.init_database()

            # 2. 插入系统配置
            await self.insert_system_configs()

            # 3. 插入示例文档
            await self.insert_sample_documents()

            # 4. 插入示例对话
            await self.insert_sample_conversations()

            self.initialized = True
            logger.info("数据初始化完成")
            return True

        except Exception as e:
            logger.error(f"数据初始化失败: {e}")
            return False

    async def init_database(self):
        """初始化数据库"""
        logger.info("初始化数据库结构...")
        success = await db_initializer.init_database()
        if not success:
            raise Exception("数据库结构初始化失败")
        logger.info("数据库结构初始化完成")

    async def insert_system_configs(self):
        """插入系统配置"""
        logger.info("插入系统配置...")

        configs = {
            # 基础配置
            "system_name": "金融研报智能系统",
            "system_version": "1.0.0",

            # 文件处理配置
            "max_file_size_mb": "50",
            "supported_file_types": json.dumps(["pdf", "docx", "xlsx", "txt", "md"]),
            "chunk_size": "512",
            "chunk_overlap": "50",
            "max_chunks_per_document": "1000",

            # AI模型配置
            "embedding_model": "qwen2.5-vl-embedding",
            "llm_model": "deepseek-chat",
            "rerank_model": "qwen3-rerank",
            "multimodal_model": "qwen-vl-plus",

            # 检索配置
            "default_top_k": "10",
            "similarity_threshold": "0.7",
            "max_concurrent_tasks": "10",

            # 缓存配置
            "cache_ttl_seconds": "3600",
            "enable_vector_cache": "true",
            "enable_graph_cache": "true",

            # 性能配置
            "batch_size": "100",
            "prefetch_enabled": "true",
            "parallel_processing": "true"
        }

        async with get_db() as db:
            for config_key, config_value in configs.items():
                # 检查是否已存在
                existing = await db.execute(
                    "SELECT id FROM system_configs WHERE config_key = :config_key",
                    {"config_key": config_key}
                )
                if not existing.first():
                    config = SystemConfig(
                        config_key=config_key,
                        config_value=config_value,
                        description=f"System configuration for {config_key}"
                    )
                    db.add(config)

            await db.commit()
            logger.info(f"插入 {len(configs)} 个系统配置")

    async def insert_sample_documents(self):
        """插入示例文档"""
        logger.info("插入示例文档...")

        sample_docs = [
            {
                "title": "2024年中国银行业发展报告",
                "filename": "bank_report_2024.pdf",
                "file_type": "pdf",
                "file_size": 2048576,
                "status": "completed",
                "content": """
2024年中国银行业发展报告

一、行业发展概况
2024年，中国银行业整体运行平稳，资产质量持续改善。截至年末，银行业金融机构总资产达到350万亿元，同比增长8.5%。

二、主要银行表现
1. 工商银行：资产规模最大，净利润超过3000亿元
2. 建设银行：房贷业务领先，资产质量优良
3. 农业银行：服务三农，县域业务优势明显
4. 中国银行：国际化程度最高，海外业务占比超过25%

三、风险控制
全行业不良贷款率控制在1.8%以内，拨备覆盖率达到190%以上。

四、数字化转型
银行业加速数字化转型，手机银行用户数超过10亿，线上业务占比超过70%。
                """.strip(),
                "metadata": {
                    "source": "中国银行业协会",
                    "year": 2024,
                    "category": "银行业",
                    "language": "zh-CN"
                }
            },
            {
                "title": "2024年保险业投资策略报告",
                "filename": "insurance_strategy_2024.pdf",
                "file_type": "pdf",
                "file_size": 1536789,
                "status": "completed",
                "content": """
2024年保险业投资策略报告

一、宏观经济环境
2024年全球经济复苏态势分化，国内经济稳中向好，为保险资金运用提供了良好的市场环境。

二、投资建议
1. 增配权益类资产，建议占比15-20%
2. 重点关注科技创新、新能源、高端制造等领域
3. 稳健配置固定收益资产，防范信用风险

三、风险提示
1. 关注利率波动对固定收益资产的影响
2. 警惕部分行业的信用风险
3. 加强流动性管理

四、创新投资
探索投资ESG相关项目，支持绿色发展和可持续发展。
                """.strip(),
                "metadata": {
                    "source": "中国保险资产管理业协会",
                    "year": 2024,
                    "category": "保险业",
                    "language": "zh-CN"
                }
            },
            {
                "title": "证券市场2024年度分析",
                "filename": "securities_analysis_2024.xlsx",
                "file_type": "xlsx",
                "file_size": 892456,
                "status": "completed",
                "content": """
证券市场2024年度分析

一、市场表现
- 上证综指：全年上涨12.5%
- 深证成指：全年上涨18.3%
- 创业板指：全年上涨25.6%

二、行业表现排名
1. 新能源汽车：涨幅45.2%
2. 人工智能：涨幅38.7%
3. 半导体：涨幅32.1%
4. 医药生物：涨幅28.9%
5. 消费升级：涨幅22.4%

三、投资者结构
- 机构投资者占比提升至45%
- 外资持股比例达到5.2%
- 个人投资者数量突破2亿

四、2025年展望
预计2025年市场将呈现结构性机会，重点关注科技创新和消费升级相关板块。
                """.strip(),
                "metadata": {
                    "source": "中国证券业协会",
                    "year": 2024,
                    "category": "证券业",
                    "language": "zh-CN"
                }
            }
        ]

        async with get_db() as db:
            for doc_data in sample_docs:
                # 检查文档是否已存在
                existing = await db.execute(
                    "SELECT id FROM documents WHERE filename = :filename",
                    {"filename": doc_data["filename"]}
                )
                if not existing.first():
                    # 创建文档
                    document = Document(
                        title=doc_data["title"],
                        filename=doc_data["filename"],
                        file_type=doc_data["file_type"],
                        file_size=doc_data["file_size"],
                        status=doc_data["status"],
                        metadata=json.dumps(doc_data["metadata"]),
                        created_at=datetime.utcnow()
                    )
                    db.add(document)
                    await db.flush()  # 获取document.id

                    # 创建文档块
                    chunks = self._create_chunks(doc_data["content"])
                    for i, chunk_content in enumerate(chunks):
                        chunk = DocumentChunk(
                            document_id=document.id,
                            chunk_index=i,
                            content=chunk_content,
                            start_char=i * 500,
                            end_char=(i + 1) * 500,
                            word_count=len(chunk_content.split()),
                            created_at=datetime.utcnow()
                        )
                        db.add(chunk)

                    logger.info(f"插入示例文档: {doc_data['title']}")

            await db.commit()
            logger.info(f"插入 {len(sample_docs)} 个示例文档")

    def _create_chunks(self, content: str, chunk_size: int = 500) -> list:
        """将内容分块"""
        words = content.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)

        return chunks

    async def insert_sample_conversations(self):
        """插入示例对话"""
        logger.info("插入示例对话...")

        sample_conversations = [
            {
                "title": "银行业发展咨询",
                "messages": [
                    {
                        "role": "user",
                        "content": "2024年中国银行业的发展情况如何？"
                    },
                    {
                        "role": "assistant",
                        "content": "根据2024年中国银行业发展报告，银行业整体运行平稳，资产质量持续改善。截至年末，银行业金融机构总资产达到350万亿元，同比增长8.5%。主要银行表现良好，工商银行资产规模最大，净利润超过3000亿元；建设银行在房贷业务方面领先；农业银行服务三农优势明显；中国银行国际化程度最高。"
                    },
                    {
                        "role": "user",
                        "content": "银行业的数字化转型情况怎么样？"
                    },
                    {
                        "role": "assistant",
                        "content": "银行业在数字化转型方面取得了显著进展。手机银行用户数已超过10亿，线上业务占比超过70%。银行业加速数字化转型，包括智能客服、移动支付、线上贷款、数字货币等多个方面，极大地提升了服务效率和客户体验。"
                    }
                ]
            },
            {
                "title": "保险投资策略咨询",
                "messages": [
                    {
                        "role": "user",
                        "content": "2024年保险业的投资策略有什么建议？"
                    },
                    {
                        "role": "assistant",
                        "content": "根据2024年保险业投资策略报告，主要建议包括：1. 增配权益类资产，建议占比15-20%；2. 重点关注科技创新、新能源、高端制造等领域；3. 稳健配置固定收益资产，防范信用风险。同时，需要关注利率波动对固定收益资产的影响，警惕部分行业的信用风险，并加强流动性管理。"
                    }
                ]
            }
        ]

        async with get_db() as db:
            for conv_data in sample_conversations:
                # 创建对话
                conversation = Conversation(
                    title=conv_data["title"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(conversation)
                await db.flush()

                # 添加消息
                for msg_data in conv_data["messages"]:
                    message = Message(
                        conversation_id=conversation.id,
                        role=msg_data["role"],
                        content=msg_data["content"],
                        created_at=datetime.utcnow()
                    )
                    db.add(message)

                logger.info(f"插入示例对话: {conv_data['title']}")

            await db.commit()
            logger.info(f"插入 {len(sample_conversations)} 个示例对话")


async def main():
    """主函数"""
    initializer = DataInitializer()

    try:
        success = await initializer.initialize_all()
        if success:
            print("✅ 数据初始化成功")
        else:
            print("❌ 数据初始化失败")
            return 1
    except Exception as e:
        print(f"❌ 初始化过程中发生错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))