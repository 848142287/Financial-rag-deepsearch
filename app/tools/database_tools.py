"""
数据库验证和修复工具
整合所有数据库相关的维护功能
"""

import pymysql
import json
import os
from typing import Dict, List, Any


class DatabaseTools:
    """数据库工具类"""

    def __init__(self):
        self.conn = None
        # 直接使用环境变量或默认值
        self.mysql_config = {
            'host': os.getenv('MYSQL_HOST', 'mysql'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER', 'rag_user'),
            'password': os.getenv('MYSQL_PASSWORD', 'rag_pass'),
            'database': os.getenv('MYSQL_DATABASE', 'financial_rag'),
        }

    def connect(self):
        """连接数据库"""
        self.conn = pymysql.connect(
            **self.mysql_config,
            cursorclass=pymysql.cursors.DictCursor
        )
        return self.conn

    def check_data_integrity(self) -> Dict[str, Any]:
        """检查数据完整性"""
        with self.conn.cursor() as cursor:
            # 文档总数
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()['count']

            # 分块总数
            cursor.execute("SELECT COUNT(*) as count FROM document_chunks")
            chunk_count = cursor.fetchone()['count']

            # 实体总数
            cursor.execute("SELECT COUNT(*) as count FROM entities")
            entity_count = cursor.fetchone()['count']

            # 状态分布
            cursor.execute("SELECT status, COUNT(*) as count FROM documents GROUP BY status")
            status_distribution = {row['status']: row['count'] for row in cursor.fetchall()}

            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "total_entities": entity_count,
                "status_distribution": status_distribution
            }

    def verify_sync_status(self) -> Dict[str, Any]:
        """验证数据同步状态"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    d.id,
                    d.filename,
                    d.status,
                    d.chunks_count,
                    (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) as actual_chunks
                FROM documents d
                WHERE d.chunks_count != (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id)
                LIMIT 100
            """)
            inconsistencies = cursor.fetchall()

            return {
                "sync_status": "ok" if len(inconsistencies) == 0 else "issues_found",
                "inconsistencies": inconsistencies,
                "count": len(inconsistencies)
            }

    def get_missing_documents(self, start_id: int, end_id: int) -> List[Dict]:
        """获取缺失的文档"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, filename, chunks_count, entities_count
                FROM documents
                WHERE id >= %s AND id <= %s
                ORDER BY id
            """, (start_id, end_id))
            return cursor.fetchall()

    def export_documents_for_backup(self, doc_ids: List[int]) -> List[Dict]:
        """导出文档用于备份"""
        results = []

        with self.conn.cursor() as cursor:
            for doc_id in doc_ids:
                cursor.execute("""
                    SELECT id, filename, file_path, status, chunks_count, entities_count
                    FROM documents
                    WHERE id = %s
                """, (doc_id,))
                doc_info = cursor.fetchone()

                if not doc_info:
                    continue

                cursor.execute("""
                    SELECT content, chunk_index, metadata
                    FROM document_chunks
                    WHERE document_id = %s
                    ORDER BY chunk_index
                """, (doc_id,))
                chunks = cursor.fetchall()

                results.append({
                    "document_id": doc_info['id'],
                    "filename": doc_info['filename'],
                    "chunks": chunks
                })

        return results

    def fix_metadata_issues(self) -> Dict[str, Any]:
        """修复元数据问题"""
        fixed_count = 0

        with self.conn.cursor() as cursor:
            # 更新chunks_count
            cursor.execute("""
                UPDATE documents d
                SET chunks_count = (
                    SELECT COUNT(*)
                    FROM document_chunks
                    WHERE document_id = d.id
                )
                WHERE d.chunks_count != (
                    SELECT COUNT(*)
                    FROM document_chunks
                    WHERE document_id = d.id
                )
            """)
            fixed_count = cursor.rowcount
            self.conn.commit()

        return {
            "fixed_count": fixed_count,
            "status": "success"
        }

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
