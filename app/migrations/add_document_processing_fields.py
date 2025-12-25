"""
数据库迁移脚本：添加文档处理自动化触发机制所需的字段

此脚本添加以下字段到documents表：
- task_id: Celery任务ID
- processing_mode: 处理模式
- error_message: 错误信息
- processing_result: 处理结果
"""

import asyncio
import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from sqlalchemy import text
from app.core.database import get_db

async def add_document_processing_fields():
    """添加文档处理相关字段到documents表"""

    queries = [
        # 添加task_id字段
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS task_id VARCHAR(255)
        """,

        # 为task_id添加索引
        """
        CREATE INDEX IF NOT EXISTS idx_documents_task_id
        ON documents(task_id)
        """,

        # 添加processing_mode字段
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS processing_mode VARCHAR(50)
        """,

        # 添加error_message字段
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS error_message TEXT
        """,

        # 添加processing_result字段
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS processing_result JSON
        """,

        # 添加batch_id字段（用于批量处理）
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS batch_id VARCHAR(255)
        """,

        # 为batch_id添加索引
        """
        CREATE INDEX IF NOT EXISTS idx_documents_batch_id
        ON documents(batch_id)
        """
    ]

    async for db in get_db():
        try:
            for query in queries:
                await db.execute(text(query))
                print(f"执行成功: {query.strip().split()[0]} {query.strip().split()[1] if len(query.strip().split()) > 1 else ''}")

            await db.commit()
            print("数据库迁移完成！")

        except Exception as e:
            await db.rollback()
            print(f"数据库迁移失败: {e}")
            raise
        finally:
            await db.close()

if __name__ == "__main__":
    asyncio.run(add_document_processing_fields())