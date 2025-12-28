"""
文档导出工具
支持将数据库中的文档导出为JSON文件
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pymysql


class DocumentExporter:
    """文档导出器"""

    def __init__(self, output_dir: str = "/data/exports"):
        self.output_dir = output_dir
        self.conn = None
        self.exported_count = 0
        self.failed_count = 0
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
        return self

    def export_document(self, doc_id: int) -> Dict[str, Any]:
        """导出单个文档"""
        with self.conn.cursor() as cursor:
            # 获取文档信息
            cursor.execute("""
                SELECT id, filename, file_path, status, chunks_count, entities_count
                FROM documents
                WHERE id = %s
            """, (doc_id,))
            doc_info = cursor.fetchone()

            if not doc_info:
                return None

            # 获取文档内容
            cursor.execute("""
                SELECT content, chunk_index, metadata
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index
            """, (doc_id,))
            chunks = cursor.fetchall()

            # 格式化内容
            parsed_content = []
            for chunk in chunks:
                content_item = {
                    "text": chunk.get('content', ''),
                    "type": "text"
                }

                metadata = chunk.get('metadata', {})
                if isinstance(metadata, dict):
                    if metadata.get('type'):
                        content_item["type"] = metadata['type']
                    if metadata.get('section_title'):
                        content_item["section_title"] = metadata['section_title']

                parsed_content.append(content_item)

            return {
                "document_id": doc_info['id'],
                "filename": doc_info['filename'],
                "parsed_content": {
                    "content": parsed_content
                }
            }

    def export_documents(self, doc_ids: List[int], save_to_disk: bool = False) -> List[Dict]:
        """批量导出文档"""
        results = []

        for i, doc_id in enumerate(doc_ids, 1):
            try:
                doc_data = self.export_document(doc_id)

                if doc_data:
                    results.append(doc_data)
                    self.exported_count += 1

                    # 保存到磁盘
                    if save_to_disk:
                        self._save_document(doc_data)

                if i % 100 == 0:
                    print(f"已处理: {i}/{len(doc_ids)}")

            except Exception as e:
                print(f"导出文档 {doc_id} 失败: {str(e)}")
                self.failed_count += 1

        return results

    def _save_document(self, doc_data: Dict[str, Any]):
        """保存文档到磁盘"""
        os.makedirs(self.output_dir, exist_ok=True)

        doc_id = doc_data['document_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存processed文件
        processed_file = os.path.join(
            self.output_dir,
            f"document_{doc_id}_processed_{timestamp}.json"
        )

        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)

        # 保存raw文件
        raw_content = {
            "document_id": doc_id,
            "filename": doc_data.get('filename', ''),
            "raw_content": "\n".join([
                item.get('text', '')
                for item in doc_data.get('parsed_content', {}).get('content', [])
            ])
        }

        raw_file = os.path.join(
            self.output_dir,
            f"document_{doc_id}_raw_{timestamp}.json"
        )

        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(raw_content, f, ensure_ascii=False, indent=2)

    def export_range(self, start_id: int, end_id: int, save_to_disk: bool = False) -> List[Dict]:
        """导出ID范围内的文档"""
        doc_ids = list(range(start_id, end_id + 1))
        return self.export_documents(doc_ids, save_to_disk)

    def export_missing_documents(self, save_to_disk: bool = True) -> Dict[str, Any]:
        """导出所有文档（用于补全本地文件）"""
        # 获取所有文档ID
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT id FROM documents ORDER BY id")
            all_doc_ids = [row['id'] for row in cursor.fetchall()]

        results = self.export_documents(all_doc_ids, save_to_disk)

        return {
            "total_documents": len(all_doc_ids),
            "exported": self.exported_count,
            "failed": self.failed_count,
            "results": results
        }

    def get_export_summary(self) -> Dict[str, Any]:
        """获取导出摘要"""
        return {
            "exported_count": self.exported_count,
            "failed_count": self.failed_count,
            "success_rate": self.exported_count / max(self.exported_count + self.failed_count, 1)
        }

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
