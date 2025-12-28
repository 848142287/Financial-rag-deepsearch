"""
数据同步监控任务
定期检查MySQL、Milvus、Neo4j之间的数据同步状态
"""

import logging
import json
import pymysql
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

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

from app.core.database import SessionLocal
from app.models.document import Document

logger = logging.getLogger(__name__)

class DataSyncMonitor:
    """数据同步监控器"""

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

    def get_mysql_stats(self) -> Dict[str, int]:
        """获取MySQL数据统计"""
        try:
            conn = pymysql.connect(**self.mysql_config)
            cursor = conn.cursor()

            stats = {}

            # 文档数量
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats['documents'] = cursor.fetchone()[0]

            # 向量数量
            cursor.execute("SELECT COUNT(*) FROM vector_storage")
            stats['vectors'] = cursor.fetchone()[0]

            # 实体数量
            cursor.execute("SELECT COUNT(*) FROM knowledge_graph_nodes")
            stats['entities'] = cursor.fetchone()[0]

            # 关系数量
            cursor.execute("SELECT COUNT(*) FROM knowledge_graph_relations")
            stats['relations'] = cursor.fetchone()[0]

            # 最近处理时间
            cursor.execute("SELECT MAX(updated_at) FROM documents WHERE status = 'completed'")
            result = cursor.fetchone()
            stats['last_processed'] = result[0].isoformat() if result and result[0] else None

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"获取MySQL统计失败: {e}")
            return {}

    def get_milvus_stats(self) -> Dict[str, int]:
        """获取Milvus数据统计"""
        if not MILVUS_AVAILABLE:
            logger.warning("Milvus不可用")
            return {}

        try:
            connections.connect('default', host=self.milvus_config['host'], port=self.milvus_config['port'])

            collection_name = self.milvus_config['collection_name']
            if not utility.has_collection(collection_name):
                return {'vectors': 0}

            collection = Collection(collection_name)
            collection.load()

            return {'vectors': collection.num_entities}

        except Exception as e:
            logger.error(f"获取Milvus统计失败: {e}")
            return {}

    def get_neo4j_stats(self) -> Dict[str, int]:
        """获取Neo4j数据统计"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j不可用")
            return {}

        try:
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=self.neo4j_config['auth']
            )

            with driver.session() as session:
                # 节点数量
                result = session.run("MATCH (n) RETURN COUNT(n) as count")
                nodes = result.single()[0]

                # 关系数量
                result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count")
                relations = result.single()[0]

                driver.close()
                return {'entities': nodes, 'relations': relations}

        except Exception as e:
            logger.error(f"获取Neo4j统计失败: {e}")
            return {}

    def calculate_sync_status(self) -> Dict[str, Any]:
        """计算同步状态"""
        mysql_stats = self.get_mysql_stats()
        milvus_stats = self.get_milvus_stats()
        neo4j_stats = self.get_neo4j_stats()

        # 计算同步率
        sync_status = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {
                'mysql': mysql_stats,
                'milvus': milvus_stats,
                'neo4j': neo4j_stats
            }
        }

        # 向量同步率
        if mysql_stats.get('vectors', 0) > 0:
            milvus_vectors = milvus_stats.get('vectors', 0)
            vector_sync_rate = (milvus_vectors / mysql_stats['vectors']) * 100
            sync_status['vector_sync_rate'] = round(vector_sync_rate, 2)
            sync_status['vector_sync_status'] = 'good' if vector_sync_rate >= 95 else 'warning' if vector_sync_rate >= 80 else 'critical'
        else:
            sync_status['vector_sync_rate'] = 100
            sync_status['vector_sync_status'] = 'good'

        # 实体同步率
        if mysql_stats.get('entities', 0) > 0:
            neo4j_entities = neo4j_stats.get('entities', 0)
            entity_sync_rate = (neo4j_entities / mysql_stats['entities']) * 100
            sync_status['entity_sync_rate'] = round(entity_sync_rate, 2)
            sync_status['entity_sync_status'] = 'good' if entity_sync_rate >= 95 else 'warning' if entity_sync_rate >= 80 else 'critical'
        else:
            sync_status['entity_sync_rate'] = 100
            sync_status['entity_sync_status'] = 'good'

        # 关系同步率
        if mysql_stats.get('relations', 0) > 0:
            neo4j_relations = neo4j_stats.get('relations', 0)
            relation_sync_rate = (neo4j_relations / mysql_stats['relations']) * 100
            sync_status['relation_sync_rate'] = round(relation_sync_rate, 2)
            sync_status['relation_sync_status'] = 'good' if relation_sync_rate >= 95 else 'warning' if relation_sync_rate >= 80 else 'critical'
        else:
            sync_status['relation_sync_rate'] = 100
            sync_status['relation_sync_status'] = 'good'

        # 整体状态
        overall_status = 'good'
        if any(status == 'critical' for status in [sync_status.get('vector_sync_status'), sync_status.get('entity_sync_status'), sync_status.get('relation_sync_status')]):
            overall_status = 'critical'
        elif any(status == 'warning' for status in [sync_status.get('vector_sync_status'), sync_status.get('entity_sync_status'), sync_status.get('relation_sync_status')]):
            overall_status = 'warning'

        sync_status['overall_status'] = overall_status

        return sync_status

    def check_processing_health(self) -> Dict[str, Any]:
        """检查处理健康状态"""
        try:
            session = SessionLocal()

            # 检查最近处理的文档
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_completed = session.query(Document).filter(
                Document.status == 'completed',
                Document.processed_at >= one_hour_ago
            ).count()

            # 检查失败文档
            failed_documents = session.query(Document).filter(
                Document.status == 'PROCESSING_FAILED'
            ).count()

            # 检查处理中文档
            processing_documents = session.query(Document).filter(
                Document.status == 'processing'
            ).count()

            session.close()

            health_status = {
                'timestamp': datetime.now().isoformat(),
                'recent_completed': recent_completed,
                'failed_documents': failed_documents,
                'processing_documents': processing_documents,
                'health_status': 'good'
            }

            # 评估健康状态
            if failed_documents > 10:
                health_status['health_status'] = 'critical'
            elif failed_documents > 5 or processing_documents > 100:
                health_status['health_status'] = 'warning'

            return health_status

        except Exception as e:
            logger.error(f"检查处理健康状态失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'health_status': 'error',
                'error': str(e)
            }

    def generate_alerts(self, sync_status: Dict[str, Any], health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成告警"""
        alerts = []

        # 同步状态告警
        if sync_status.get('overall_status') == 'critical':
            alerts.append({
                'type': 'critical',
                'message': '数据同步状态严重落后',
                'details': {
                    'vector_sync_rate': sync_status.get('vector_sync_rate', 0),
                    'entity_sync_rate': sync_status.get('entity_sync_rate', 0),
                    'relation_sync_rate': sync_status.get('relation_sync_rate', 0)
                },
                'timestamp': datetime.now().isoformat()
            })
        elif sync_status.get('overall_status') == 'warning':
            alerts.append({
                'type': 'warning',
                'message': '数据同步状态需要关注',
                'details': {
                    'vector_sync_rate': sync_status.get('vector_sync_rate', 0),
                    'entity_sync_rate': sync_status.get('entity_sync_rate', 0),
                    'relation_sync_rate': sync_status.get('relation_sync_rate', 0)
                },
                'timestamp': datetime.now().isoformat()
            })

        # 健康状态告警
        if health_status.get('health_status') == 'critical':
            alerts.append({
                'type': 'critical',
                'message': '文档处理健康状态异常',
                'details': {
                    'failed_documents': health_status.get('failed_documents', 0),
                    'processing_documents': health_status.get('processing_documents', 0)
                },
                'timestamp': datetime.now().isoformat()
            })
        elif health_status.get('health_status') == 'warning':
            alerts.append({
                'type': 'warning',
                'message': '文档处理健康状态需要关注',
                'details': {
                    'failed_documents': health_status.get('failed_documents', 0),
                    'processing_documents': health_status.get('processing_documents', 0)
                },
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def run_monitoring_check(self) -> Dict[str, Any]:
        """运行监控检查"""
        try:
            logger.info("开始数据同步监控检查")

            sync_status = self.calculate_sync_status()
            health_status = self.check_processing_health()
            alerts = self.generate_alerts(sync_status, health_status)

            monitoring_result = {
                'timestamp': datetime.now().isoformat(),
                'sync_status': sync_status,
                'health_status': health_status,
                'alerts': alerts,
                'summary': {
                    'overall_status': sync_status.get('overall_status', 'unknown'),
                    'alert_count': len(alerts),
                    'data_integrity_score': min(
                        sync_status.get('vector_sync_rate', 0),
                        sync_status.get('entity_sync_rate', 0),
                        sync_status.get('relation_sync_rate', 0)
                    )
                }
            }

            # 记录结果
            if alerts:
                for alert in alerts:
                    logger.warning(f"监控告警: {alert['message']} - {alert['details']}")

            logger.info(f"监控检查完成 - 状态: {sync_status.get('overall_status')}, 告警数: {len(alerts)}")

            return monitoring_result

        except Exception as e:
            logger.error(f"监控检查失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

def run_sync_monitoring():
    """运行同步监控的主函数"""
    monitor = DataSyncMonitor()
    result = monitor.run_monitoring_check()
    return result

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    result = run_sync_monitoring()
    print(json.dumps(result, ensure_ascii=False, indent=2))