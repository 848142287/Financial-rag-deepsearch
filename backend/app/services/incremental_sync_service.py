"""
增量同步服务
优化数据同步策略，减少不必要的重复同步
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import pymysql

try:
    from pymilvus import connections, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class IncrementalSyncService:
    """增量同步服务"""

    def __init__(self):
        self.mysql_config = {
            'host': 'mysql',
            'port': 3306,
            'user': 'rag_user',
            'password': 'rag_pass',
            'database': 'financial_rag'
        }
        self.milvus_config = {
            'host': 'milvus',
            'port': '19530',
            'collection_name': 'document_embeddings'
        }
        self.neo4j_config = {
            'uri': 'bolt://neo4j:7687',
            'auth': ('neo4j', 'neo4j123')
        }

    def get_last_sync_timestamp(self) -> Dict[str, Optional[datetime]]:
        """获取最后同步时间戳"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            # 创建同步记录表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sync_type VARCHAR(50) NOT NULL,
                    last_sync_time TIMESTAMP NOT NULL,
                    last_synced_id VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_sync_type (sync_type),
                    INDEX idx_last_sync_time (last_sync_time)
                )
            """)

            # 获取最后同步时间
            cursor.execute("SELECT sync_type, last_sync_time, last_synced_id FROM sync_records ORDER BY last_sync_time DESC")
            records = cursor.fetchall()

            last_sync = {}
            for record in records:
                last_sync[record[0]] = {
                    'timestamp': record[1],
                    'last_synced_id': record[2]
                }

            conn.close()
            return last_sync

        except Exception as e:
            logger.error(f"获取最后同步时间戳失败: {e}")
            return {}

    def update_sync_timestamp(self, sync_type: str, last_synced_id: str = None):
        """更新同步时间戳"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            # 插入或更新同步记录
            cursor.execute("""
                INSERT INTO sync_records (sync_type, last_sync_time, last_synced_id, status)
                VALUES (%s, %s, %s, 'completed')
                ON DUPLICATE KEY UPDATE
                last_sync_time = VALUES(last_sync_time),
                last_synced_id = VALUES(last_synced_id),
                status = 'completed',
                updated_at = CURRENT_TIMESTAMP
            """, (sync_type, datetime.now(), last_synced_id))

            conn.commit()
            conn.close()
            logger.info(f"更新同步时间戳: {sync_type}")

        except Exception as e:
            logger.error(f"更新同步时间戳失败: {e}")

    def get_new_documents(self, since: datetime = None) -> List[int]:
        """获取需要同步的新文档ID"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            if since:
                cursor.execute("""
                    SELECT id FROM documents
                    WHERE status = 'completed'
                    AND updated_at > %s
                    ORDER BY updated_at ASC
                """, (since,))
            else:
                cursor.execute("""
                    SELECT id FROM documents
                    WHERE status = 'completed'
                    ORDER BY updated_at ASC
                """)

            document_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            return document_ids

        except Exception as e:
            logger.error(f"获取新文档列表失败: {e}")
            return []

    def get_modified_vectors(self, since: datetime = None) -> List[Dict]:
        """获取需要同步的修改向量"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            if since:
                cursor.execute("""
                    SELECT id, document_id, chunk_id, content, embedding, created_at, updated_at
                    FROM vector_storage
                    WHERE updated_at > %s
                    ORDER BY updated_at ASC
                """, (since,))
            else:
                cursor.execute("""
                    SELECT id, document_id, chunk_id, content, embedding, created_at, updated_at
                    FROM vector_storage
                    ORDER BY updated_at ASC
                """)

            vectors = []
            for row in cursor.fetchall():
                vectors.append({
                    'id': row[0],
                    'document_id': row[1],
                    'chunk_id': row[2],
                    'content': row[3],
                    'embedding': row[4],
                    'created_at': row[5],
                    'updated_at': row[6]
                })

            conn.close()
            return vectors

        except Exception as e:
            logger.error(f"获取修改向量失败: {e}")
            return []

    def get_modified_entities(self, since: datetime = None) -> List[Dict]:
        """获取需要同步的修改实体"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            if since:
                cursor.execute("""
                    SELECT node_id, document_id, node_type, node_name, properties, confidence, created_at, updated_at
                    FROM knowledge_graph_nodes
                    WHERE updated_at > %s
                    ORDER BY updated_at ASC
                """, (since,))
            else:
                cursor.execute("""
                    SELECT node_id, document_id, node_type, node_name, properties, confidence, created_at, updated_at
                    FROM knowledge_graph_nodes
                    ORDER BY updated_at ASC
                """)

            entities = []
            for row in cursor.fetchall():
                entities.append({
                    'node_id': row[0],
                    'document_id': row[1],
                    'node_type': row[2],
                    'node_name': row[3],
                    'properties': row[4],
                    'confidence': row[5],
                    'created_at': row[6],
                    'updated_at': row[7]
                })

            conn.close()
            return entities

        except Exception as e:
            logger.error(f"获取修改实体失败: {e}")
            return []

    def incremental_sync_vectors(self, vectors: List[Dict]) -> Dict[str, Any]:
        """增量同步向量到Milvus"""
        if not MILVUS_AVAILABLE or not vectors:
            return {'status': 'skipped', 'reason': 'Milvus不可用或没有数据'}

        try:
            connections.connect('default', host=self.milvus_config['host'], port=self.milvus_config['port'])
            collection = Collection(self.milvus_config['collection_name'])
            collection.load()

            synced_count = 0
            failed_count = 0
            batch_size = 100

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]

                try:
                    # 准备数据
                    ids = []
                    document_ids = []
                    chunk_ids = []
                    contents = []
                    embeddings = []
                    metadatas = []
                    created_ats = []

                    for vector in batch:
                        # 解析嵌入向量
                        try:
                            embedding_data = json.loads(vector['embedding']) if vector['embedding'] else []
                            if len(embedding_data) < 1024:
                                embedding_data = embedding_data + [0.0] * (1024 - len(embedding_data))
                            elif len(embedding_data) > 1024:
                                embedding_data = embedding_data[:1024]
                        except:
                            embedding_data = [0.0] * 1024

                        ids.append(int(time.time() * 1000000) + len(ids))
                        document_ids.append(int(vector['document_id']) if str(vector['document_id']).isdigit() else hash(str(vector['document_id'])) % 2147483647)
                        chunk_ids.append(int(vector['chunk_id']) if vector['chunk_id'] else 0)
                        contents.append(vector['content'][:65535] if vector['content'] else "")
                        embeddings.append(embedding_data)
                        metadatas.append(json.dumps({
                            'incremental_sync': True,
                            'sync_timestamp': datetime.now().isoformat(),
                            'mysql_id': vector['id']
                        }))
                        created_ats.append(int(vector['created_at'].timestamp() * 1000) if vector['created_at'] else int(time.time() * 1000))

                    # 插入到Milvus
                    data = [ids, document_ids, chunk_ids, contents, embeddings, metadatas, created_ats]
                    collection.insert(data)

                    synced_count += len(batch)

                except Exception as e:
                    logger.error(f"批量同步向量失败: {e}")
                    failed_count += len(batch)
                    continue

            # 刷新数据
            collection.flush()

            result = {
                'status': 'completed',
                'synced_count': synced_count,
                'failed_count': failed_count,
                'total_count': len(vectors),
                'milvus_total': collection.num_entities
            }

            # 更新同步时间戳
            last_vector_id = vectors[-1]['id'] if vectors else None
            self.update_sync_timestamp('vectors', str(last_vector_id))

            return result

        except Exception as e:
            logger.error(f"增量向量同步失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    def incremental_sync_entities(self, entities: List[Dict]) -> Dict[str, Any]:
        """增量同步实体到Neo4j"""
        if not NEO4J_AVAILABLE or not entities:
            return {'status': 'skipped', 'reason': 'Neo4j不可用或没有数据'}

        try:
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=self.neo4j_config['auth']
            )

            synced_count = 0
            failed_count = 0

            with driver.session() as session:
                for entity in entities:
                    try:
                        node_id = str(entity['node_id'])
                        document_id = str(entity['document_id'])
                        node_type = str(entity['node_type'])
                        node_name = str(entity['node_name'])[:100]
                        confidence = float(entity['confidence']) if entity['confidence'] else 0.8

                        # 解析properties但只使用简单字段
                        properties_data = {}
                        try:
                            if entity['properties']:
                                prop_dict = json.loads(entity['properties'])
                                for key, value in prop_dict.items():
                                    if isinstance(value, (str, int, float)):
                                        properties_data[str(key)[:50]] = str(value)[:200]
                        except:
                            properties_data = {"source": "incremental_sync"}

                        # 创建或更新节点
                        query = """
                        MERGE (n:Entity {node_id: $node_id})
                        SET n.document_id = $document_id,
                            n.node_type = $node_type,
                            n.node_name = $node_name,
                            n.confidence = $confidence,
                            n.incremental_synced = true,
                            n.sync_timestamp = datetime()
                        """

                        session.run(query,
                            node_id=node_id,
                            document_id=document_id,
                            node_type=node_type,
                            node_name=node_name,
                            confidence=confidence
                        )

                        synced_count += 1

                    except Exception as e:
                        logger.error(f"同步实体失败 (ID: {entity['node_id']}): {e}")
                        failed_count += 1
                        continue

            # 获取最终统计
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN COUNT(n) as count")
                total_nodes = result.single()[0]

            driver.close()

            result = {
                'status': 'completed',
                'synced_count': synced_count,
                'failed_count': failed_count,
                'total_count': len(entities),
                'neo4j_total': total_nodes
            }

            # 更新同步时间戳
            last_entity_id = entities[-1]['node_id'] if entities else None
            self.update_sync_timestamp('entities', str(last_entity_id))

            return result

        except Exception as e:
            logger.error(f"增量实体同步失败: {e}")
            return {'status': 'failed', 'error': str(e)}

    def run_incremental_sync(self) -> Dict[str, Any]:
        """运行增量同步"""
        try:
            logger.info("开始增量同步")

            # 获取最后同步时间
            last_sync = self.get_last_sync_timestamp()

            # 获取各类型的最后同步时间
            vectors_last_sync = last_sync.get('vectors', {}).get('timestamp')
            entities_last_sync = last_sync.get('entities', {}).get('timestamp')

            # 设置默认时间（如果之前没有同步记录）
            if not vectors_last_sync:
                vectors_last_sync = datetime.now() - timedelta(hours=24)
            if not entities_last_sync:
                entities_last_sync = datetime.now() - timedelta(hours=24)

            # 获取需要同步的数据
            vectors_to_sync = self.get_modified_vectors(vectors_last_sync)
            entities_to_sync = self.get_modified_entities(entities_last_sync)

            logger.info(f"发现 {len(vectors_to_sync)} 个修改向量，{len(entities_to_sync)} 个修改实体")

            # 执行同步
            results = {
                'timestamp': datetime.now().isoformat(),
                'vectors': self.incremental_sync_vectors(vectors_to_sync),
                'entities': self.incremental_sync_entities(entities_to_sync)
            }

            # 计算总体状态
            total_synced = results['vectors'].get('synced_count', 0) + results['entities'].get('synced_count', 0)
            total_failed = results['vectors'].get('failed_count', 0) + results['entities'].get('failed_count', 0)

            if total_synced > 0 and total_failed == 0:
                results['overall_status'] = 'success'
            elif total_synced > 0 and total_failed > 0:
                results['overall_status'] = 'partial_success'
            elif total_synced == 0 and total_failed == 0:
                results['overall_status'] = 'no_changes'
            else:
                results['overall_status'] = 'failed'

            logger.info(f"增量同步完成: 总同步 {total_synced}，失败 {total_failed}")

            return results

        except Exception as e:
            logger.error(f"增量同步执行失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'failed',
                'error': str(e)
            }