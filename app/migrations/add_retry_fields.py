"""
数据库迁移脚本：添加文档重试机制所需字段

此脚本添加以下字段到documents表：
- retry_count: 重试次数
- next_retry_at: 下次重试时间
"""

import sys
from pathlib import Path

# 添加backend目录到Python路径
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from sqlalchemy import text
from app.core.database import engine

def add_retry_fields():
    """添加重试相关字段到documents表"""

    # 先检查列是否存在
    check_queries = [
        "SELECT COUNT(*) as count FROM information_schema.columns WHERE table_name='documents' AND column_name='retry_count'",
        "SELECT COUNT(*) as count FROM information_schema.columns WHERE table_name='documents' AND column_name='next_retry_at'"
    ]

    # 要执行的ALTER语句
    alter_queries = [
        ("ALTER TABLE documents ADD COLUMN retry_count INT DEFAULT 0", "retry_count"),
        ("ALTER TABLE documents ADD COLUMN next_retry_at TIMESTAMP NULL", "next_retry_at")
    ]

    # 要创建的索引
    index_queries = [
        "CREATE INDEX idx_documents_retry_count ON documents(retry_count)",
        "CREATE INDEX idx_documents_next_retry_at ON documents(next_retry_at)"
    ]

    with engine.connect() as connection:
        try:
            # 检查列是否存在
            results = []
            for check_query in check_queries:
                result = connection.execute(text(check_query)).fetchone()
                results.append(result[0] > 0)

            # 添加不存在的列
            for i, (alter_query, column_name) in enumerate(alter_queries):
                if not results[i]:
                    connection.execute(text(alter_query))
                    print(f"成功添加列: {column_name}")
                else:
                    print(f"列已存在，跳过: {column_name}")

            # 创建索引（如果不存在）
            for index_query in index_queries:
                try:
                    connection.execute(text(index_query))
                    print("成功创建索引")
                except Exception as e:
                    if "Duplicate key name" in str(e) or "already exists" in str(e):
                        print("索引已存在，跳过")
                    else:
                        raise

            connection.commit()
            print("数据库迁移完成！")

        except Exception as e:
            connection.rollback()
            print(f"数据库迁移失败: {e}")
            raise

if __name__ == "__main__":
    add_retry_fields()