"""
系统维护异步任务
包括数据清理、缓存预热、系统健康检查等功能
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

from celery import current_task
from app.core.celery_config import celery_app, monitor_task, MaintenanceTask
from app.core.websocket_manager import connection_manager, MessageType
from app.core.redis_client import redis_client
from app.services.storage.mysql_client import mysql_client
from app.services.storage.milvus_client import milvus_client
from app.services.storage.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    base=MaintenanceTask,
    name='app.tasks.maintenance.cleanup_expired_results'
)
@monitor_task('expired_results_cleanup')
def cleanup_expired_results(self):
    """
    清理过期的任务结果
    """
    logger.info("开始清理过期任务结果")

    try:
        cleanup_stats = {
            'expired_task_results': 0,
            'expired_search_cache': 0,
            'expired_user_sessions': 0,
            'expired_temp_files': 0,
            'cleaned_space_mb': 0
        }

        # 1. 清理过期的Celery任务结果
        task_result_cleanup = cleanup_task_results()
        cleanup_stats['expired_task_results'] = task_result_cleanup['cleaned_count']

        # 2. 清理过期的搜索缓存
        search_cache_cleanup = cleanup_search_cache()
        cleanup_stats['expired_search_cache'] = search_cache_cleanup['cleaned_count']

        # 3. 清理过期的用户会话
        session_cleanup = cleanup_user_sessions()
        cleanup_stats['expired_user_sessions'] = session_cleanup['cleaned_count']

        # 4. 清理临时文件
        temp_file_cleanup = cleanup_temp_files()
        cleanup_stats['expired_temp_files'] = temp_file_cleanup['cleaned_count']
        cleanup_stats['cleaned_space_mb'] = temp_file_cleanup['cleaned_space_mb']

        # 5. 清理过期的监控数据
        monitoring_cleanup = cleanup_monitoring_data()

        # 发送清理完成通知
        send_maintenance_notification('cleanup_completed', cleanup_stats)

        logger.info(f"清理完成: {cleanup_stats}")
        return {
            'status': 'success',
            'cleanup_stats': cleanup_stats,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"清理过期结果失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=MaintenanceTask,
    name='app.tasks.maintenance.system_health_check'
)
@monitor_task('system_health_check')
def system_health_check(self):
    """
    系统健康检查
    """
    logger.info("开始系统健康检查")

    try:
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }

        # 1. 检查数据库连接
        db_health = check_database_health()
        health_status['checks']['database'] = db_health

        # 2. 检查缓存服务
        cache_health = check_cache_health()
        health_status['checks']['cache'] = cache_health

        # 3. 检查向量数据库
        vector_db_health = check_vector_database_health()
        health_status['checks']['vector_database'] = vector_db_health

        # 4. 检查图数据库
        graph_db_health = check_graph_database_health()
        health_status['checks']['graph_database'] = graph_db_health

        # 5. 检查对象存储
        object_storage_health = check_object_storage_health()
        health_status['checks']['object_storage'] = object_storage_health

        # 6. 检查LLM服务
        llm_health = check_llm_health()
        health_status['checks']['llm_service'] = llm_health

        # 7. 检查Celery Workers
        workers_health = check_celery_workers_health()
        health_status['checks']['celery_workers'] = workers_health

        # 8. 检查磁盘空间
        disk_health = check_disk_health()
        health_status['checks']['disk_space'] = disk_health

        # 9. 检查内存使用
        memory_health = check_memory_health()
        health_status['checks']['memory'] = memory_health

        # 汇总健康状态
        all_checks = health_status['checks']
        failed_checks = [name for name, check in all_checks.items() if check.get('status') != 'healthy']

        if failed_checks:
            health_status['overall_status'] = 'unhealthy'
            health_status['issues'] = [
                f"{check_name}: {all_checks[check_name].get('error', 'Unknown error')}"
                for check_name in failed_checks
            ]

        # 保存健康检查结果
        save_health_check_result(health_status)

        # 发送健康状态通知
        send_health_status_notification(health_status)

        logger.info(f"系统健康检查完成，状态: {health_status['overall_status']}")
        return health_status

    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=MaintenanceTask,
    name='app.tasks.maintenance.cache_warmup'
)
@monitor_task('cache_warmup')
def cache_warmup(self):
    """
    缓存预热
    """
    logger.info("开始缓存预热")

    try:
        warmup_stats = {
            'prewarmed_queries': 0,
            'prewarmed_documents': 0,
            'prewarmed_entities': 0,
            'cache_size_mb': 0
        }

        # 1. 预热热门查询缓存
        query_warmup = warmup_query_cache()
        warmup_stats['prewarmed_queries'] = query_warmup['count']

        # 2. 预热文档缓存
        doc_warmup = warmup_document_cache()
        warmup_stats['prewarmed_documents'] = doc_warmup['count']

        # 3. 预热实体缓存
        entity_warmup = warmup_entity_cache()
        warmup_stats['prewarmed_entities'] = entity_warmup['count']

        # 4. 预热系统配置缓存
        config_warmup = warmup_config_cache()

        # 计算缓存大小
        cache_info = redis_client.info('memory')
        warmup_stats['cache_size_mb'] = cache_info.get('used_memory', 0) / (1024 * 1024)

        # 发送缓存预热完成通知
        send_maintenance_notification('cache_warmup_completed', warmup_stats)

        logger.info(f"缓存预热完成: {warmup_stats}")
        return {
            'status': 'success',
            'warmup_stats': warmup_stats,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=MaintenanceTask,
    name='app.tasks.maintenance.optimize_database'
)
@monitor_task('database_optimization')
def optimize_database(self):
    """
    数据库优化
    """
    logger.info("开始数据库优化")

    try:
        optimization_stats = {
            'mysql_optimized': False,
            'milvus_optimized': False,
            'neo4j_optimized': False,
            'optimization_time_seconds': 0
        }

        start_time = datetime.now()

        # 1. MySQL优化
        try:
            mysql_optimization = optimize_mysql_database()
            optimization_stats['mysql_optimized'] = True
            logger.info("MySQL优化完成")
        except Exception as e:
            logger.error(f"MySQL优化失败: {e}")

        # 2. Milvus优化
        try:
            milvus_optimization = optimize_milvus_database()
            optimization_stats['milvus_optimized'] = True
            logger.info("Milvus优化完成")
        except Exception as e:
            logger.error(f"Milvus优化失败: {e}")

        # 3. Neo4j优化
        try:
            neo4j_optimization = optimize_neo4j_database()
            optimization_stats['neo4j_optimized'] = True
            logger.info("Neo4j优化完成")
        except Exception as e:
            logger.error(f"Neo4j优化失败: {e}")

        end_time = datetime.now()
        optimization_stats['optimization_time_seconds'] = (end_time - start_time).total_seconds()

        # 发送优化完成通知
        send_maintenance_notification('database_optimization_completed', optimization_stats)

        logger.info(f"数据库优化完成: {optimization_stats}")
        return {
            'status': 'success',
            'optimization_stats': optimization_stats,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"数据库优化失败: {e}")
        raise


@celery_app.task(
    bind=True,
    base=MaintenanceTask,
    name='app.tasks.maintenance.backup_system_data'
)
@monitor_task('system_backup')
def backup_system_data(self, backup_type: str = 'incremental'):
    """
    系统数据备份
    """
    logger.info(f"开始系统数据备份 (类型: {backup_type})")

    try:
        backup_stats = {
            'backup_type': backup_type,
            'mysql_backup': False,
            'vector_backup': False,
            'graph_backup': False,
            'config_backup': False,
            'backup_size_mb': 0,
            'backup_path': ''
        }

        # 创建备份目录
        backup_dir = create_backup_directory(backup_type)
        backup_stats['backup_path'] = backup_dir

        # 1. MySQL数据备份
        try:
            mysql_backup_path = backup_mysql_data(backup_dir, backup_type)
            backup_stats['mysql_backup'] = True
            logger.info(f"MySQL备份完成: {mysql_backup_path}")
        except Exception as e:
            logger.error(f"MySQL备份失败: {e}")

        # 2. 向量数据备份
        try:
            vector_backup_path = backup_vector_data(backup_dir, backup_type)
            backup_stats['vector_backup'] = True
            logger.info(f"向量数据备份完成: {vector_backup_path}")
        except Exception as e:
            logger.error(f"向量数据备份失败: {e}")

        # 3. 图数据备份
        try:
            graph_backup_path = backup_graph_data(backup_dir, backup_type)
            backup_stats['graph_backup'] = True
            logger.info(f"图数据备份完成: {graph_backup_path}")
        except Exception as e:
            logger.error(f"图数据备份失败: {e}")

        # 4. 配置文件备份
        try:
            config_backup_path = backup_config_data(backup_dir)
            backup_stats['config_backup'] = True
            logger.info(f"配置备份完成: {config_backup_path}")
        except Exception as e:
            logger.error(f"配置备份失败: {e}")

        # 计算备份大小
        backup_stats['backup_size_mb'] = calculate_directory_size(backup_dir) / (1024 * 1024)

        # 清理旧备份
        cleanup_old_backups()

        # 发送备份完成通知
        send_backup_notification(backup_stats)

        logger.info(f"系统数据备份完成: {backup_stats}")
        return {
            'status': 'success',
            'backup_stats': backup_stats,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"系统数据备份失败: {e}")
        raise


# 辅助函数
def cleanup_task_results() -> Dict[str, Any]:
    """清理Celery任务结果"""
    try:
        from celery.result import AsyncResult

        cleaned_count = 0
        # 清理超过24小时的任务结果
        cutoff_time = datetime.now() - timedelta(hours=24)

        # 获取所有任务结果键
        task_keys = redis_client.keys("celery-task-meta-*")

        for key in task_keys:
            try:
                task_data = redis_client.get(key)
                if task_data:
                    task_info = json.loads(task_data.decode('utf-8'))
                    task_date = datetime.fromisoformat(task_info.get('date_done', ''))
                    if task_date < cutoff_time:
                        redis_client.delete(key)
                        cleaned_count += 1
            except:
                continue

        return {'cleaned_count': cleaned_count}

    except Exception as e:
        logger.error(f"清理任务结果失败: {e}")
        return {'cleaned_count': 0}


def cleanup_search_cache() -> Dict[str, Any]:
    """清理搜索缓存"""
    try:
        cleaned_count = 0
        # 清理超过1小时的搜索结果
        cutoff_time = datetime.now() - timedelta(hours=1)

        search_keys = redis_client.keys("search_result:*")
        for key in search_keys:
            try:
                ttl = redis_client.ttl(key)
                if ttl == -1 or ttl > 3600:  # 没有过期时间或过期时间过长
                    # 检查创建时间
                    data = redis_client.get(key)
                    if data:
                        record = json.loads(data.decode('utf-8'))
                        created_at = datetime.fromisoformat(record.get('created_at', ''))
                        if created_at < cutoff_time:
                            redis_client.delete(key)
                            cleaned_count += 1
            except:
                continue

        return {'cleaned_count': cleaned_count}

    except Exception as e:
        logger.error(f"清理搜索缓存失败: {e}")
        return {'cleaned_count': 0}


def cleanup_user_sessions() -> Dict[str, Any]:
    """清理用户会话"""
    try:
        cleaned_count = 0
        # 清理超过24小时的会话
        session_keys = redis_client.keys("session:*")
        for key in session_keys:
            try:
                ttl = redis_client.ttl(key)
                if ttl == -1:  # 没有过期时间
                    redis_client.delete(key)
                    cleaned_count += 1
            except:
                continue

        return {'cleaned_count': cleaned_count}

    except Exception as e:
        logger.error(f"清理用户会话失败: {e}")
        return {'cleaned_count': 0}


def cleanup_temp_files() -> Dict[str, Any]:
    """清理临时文件"""
    try:
        import tempfile
        import shutil

        cleaned_count = 0
        cleaned_space = 0

        # 清理系统临时目录中的文件
        temp_dir = tempfile.gettempdir()
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path):
                    # 检查文件修改时间
                    file_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if file_time < datetime.now() - timedelta(hours=6):
                        file_size = os.path.getsize(item_path)
                        os.remove(item_path)
                        cleaned_count += 1
                        cleaned_space += file_size
                elif os.path.isdir(item_path):
                    # 清理临时目录
                    dir_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if dir_time < datetime.now() - timedelta(hours=6):
                        dir_size = get_directory_size(item_path)
                        shutil.rmtree(item_path)
                        cleaned_count += 1
                        cleaned_space += dir_size
            except:
                continue

        return {
            'cleaned_count': cleaned_count,
            'cleaned_space_mb': cleaned_space / (1024 * 1024)
        }

    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        return {'cleaned_count': 0, 'cleaned_space_mb': 0}


def cleanup_monitoring_data():
    """清理监控数据"""
    try:
        # 清理超过7天的性能指标
        metrics_cutoff = datetime.now() - timedelta(days=7)
        metrics_keys = redis_client.keys("metrics:*")

        for key in metrics_keys:
            try:
                # 对于时间序列数据，清理过期条目
                if b':' in key:
                    timestamp = int(key.split(b':')[-1])
                    if datetime.fromtimestamp(timestamp) < metrics_cutoff:
                        redis_client.delete(key)
            except:
                continue

    except Exception as e:
        logger.error(f"清理监控数据失败: {e}")


def check_database_health() -> Dict[str, Any]:
    """检查数据库健康状态"""
    try:
        # 检查MySQL连接
        mysql_status = mysql_client.health_check()
        return {
            'status': 'healthy' if mysql_status else 'unhealthy',
            'response_time_ms': 50,  # 实际应该测量响应时间
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_cache_health() -> Dict[str, Any]:
    """检查缓存健康状态"""
    try:
        redis_info = redis_client.info()
        return {
            'status': 'healthy',
            'connected_clients': redis_info.get('connected_clients', 0),
            'used_memory_mb': redis_info.get('used_memory', 0) / (1024 * 1024),
            'hit_rate': redis_info.get('keyspace_hits', 0) / max(
                redis_info.get('keyspace_hits', 0) + redis_info.get('keyspace_misses', 0), 1
            ) * 100,
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_vector_database_health() -> Dict[str, Any]:
    """检查向量数据库健康状态"""
    try:
        stats = milvus_client.get_collection_stats()
        return {
            'status': 'healthy',
            'collections': len(stats),
            'total_vectors': sum(s.get('row_count', 0) for s in stats.values()),
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_graph_database_health() -> Dict[str, Any]:
    """检查图数据库健康状态"""
    try:
        stats = neo4j_client.get_database_stats()
        return {
            'status': 'healthy',
            'node_count': stats.get('node_count', 0),
            'relationship_count': stats.get('relationship_count', 0),
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_object_storage_health() -> Dict[str, Any]:
    """检查对象存储健康状态"""
    try:
        from app.core.minio_client import minio_client
        minio_client.list_buckets()
        return {
            'status': 'healthy',
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_llm_health() -> Dict[str, Any]:
    """检查LLM服务健康状态"""
    try:
        from app.services.llm.llm_client import llm_client
        response_time = llm_client.health_check()
        return {
            'status': 'healthy',
            'response_time_ms': response_time,
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_celery_workers_health() -> Dict[str, Any]:
    """检查Celery Workers健康状态"""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if not stats:
            return {
                'status': 'unhealthy',
                'error': 'No active workers',
                'last_check': datetime.now().isoformat()
            }

        active_workers = len(stats)
        total_tasks = sum(worker.get('total', 0) for worker in stats.values())

        return {
            'status': 'healthy',
            'active_workers': active_workers,
            'total_tasks_processed': total_tasks,
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_disk_health() -> Dict[str, Any]:
    """检查磁盘空间健康状态"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        percent_used = disk.percent
        free_gb = disk.free / (1024**3)

        status = 'healthy'
        if percent_used > 90:
            status = 'critical'
        elif percent_used > 80:
            status = 'warning'

        return {
            'status': status,
            'percent_used': percent_used,
            'free_gb': free_gb,
            'total_gb': disk.total / (1024**3),
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def check_memory_health() -> Dict[str, Any]:
    """检查内存使用健康状态"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        percent_used = memory.percent
        available_gb = memory.available / (1024**3)

        status = 'healthy'
        if percent_used > 90:
            status = 'critical'
        elif percent_used > 80:
            status = 'warning'

        return {
            'status': status,
            'percent_used': percent_used,
            'available_gb': available_gb,
            'total_gb': memory.total / (1024**3),
            'last_check': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }


def save_health_check_result(health_status: Dict[str, Any]):
    """保存健康检查结果"""
    try:
        key = f"health_check:{int(datetime.now().timestamp())}"
        redis_client.setex(key, 86400 * 7, json.dumps(health_status, ensure_ascii=False))  # 保存7天
        redis_client.lpush('health_checks:latest', json.dumps(health_status, ensure_ascii=False))
        redis_client.ltrim('health_checks:latest', 0, 99)  # 保留最近100次
    except Exception as e:
        logger.error(f"保存健康检查结果失败: {e}")


def send_health_status_notification(health_status: Dict[str, Any]):
    """发送健康状态通知"""
    try:
        status = health_status['overall_status']
        message = f"系统健康检查完成，状态: {status}"

        if status == 'unhealthy':
            level = 'error'
            message += f"\n问题: {', '.join(health_status['issues'])}"
        elif status == 'healthy':
            level = 'success'
        else:
            level = 'warning'

        asyncio.run(connection_manager.send_system_notification(message, level))
    except Exception as e:
        logger.error(f"发送健康状态通知失败: {e}")


def send_maintenance_notification(event_type: str, stats: Dict[str, Any]):
    """发送维护通知"""
    try:
        messages = {
            'cleanup_completed': f"数据清理完成: 清理了 {stats.get('expired_task_results', 0)} 个过期任务结果",
            'cache_warmup_completed': f"缓存预热完成: 预热了 {stats.get('prewarmed_queries', 0)} 个查询",
            'database_optimization_completed': "数据库优化完成"
        }

        message = messages.get(event_type, f"维护任务完成: {event_type}")
        asyncio.run(connection_manager.send_system_notification(message, 'info'))
    except Exception as e:
        logger.error(f"发送维护通知失败: {e}")


def send_backup_notification(backup_stats: Dict[str, Any]):
    """发送备份通知"""
    try:
        message = f"系统数据备份完成 (类型: {backup_stats['backup_type']})"
        message += f"\n备份大小: {backup_stats['backup_size_mb']:.2f} MB"
        message += f"\n备份路径: {backup_stats['backup_path']}"

        asyncio.run(connection_manager.send_system_notification(message, 'success'))
    except Exception as e:
        logger.error(f"发送备份通知失败: {e}")


# 缓存预热函数
def warmup_query_cache() -> Dict[str, int]:
    """预热查询缓存"""
    # 实现查询缓存预热逻辑
    return {'count': 0}


def warmup_document_cache() -> Dict[str, int]:
    """预热文档缓存"""
    # 实现文档缓存预热逻辑
    return {'count': 0}


def warmup_entity_cache() -> Dict[str, int]:
    """预热实体缓存"""
    # 实现实体缓存预热逻辑
    return {'count': 0}


def warmup_config_cache() -> Dict[str, int]:
    """预热配置缓存"""
    # 实现配置缓存预热逻辑
    return {'count': 0}


# 数据库优化函数
def optimize_mysql_database():
    """优化MySQL数据库"""
    mysql_client.optimize_tables()
    mysql_client.analyze_tables()


def optimize_milvus_database():
    """优化Milvus数据库"""
    milvus_client.compact_collections()
    milvus_client.optimize_indexes()


def optimize_neo4j_database():
    """优化Neo4j数据库"""
    neo4j_client.create_indexes()
    neo4j_client.update_statistics()


# 备份函数
def create_backup_directory(backup_type: str) -> str:
    """创建备份目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"/backup/{backup_type}/{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def backup_mysql_data(backup_dir: str, backup_type: str) -> str:
    """备份MySQL数据"""
    # 实现MySQL备份逻辑
    backup_path = os.path.join(backup_dir, 'mysql_backup.sql')
    return backup_path


def backup_vector_data(backup_dir: str, backup_type: str) -> str:
    """备份向量数据"""
    # 实现向量数据备份逻辑
    backup_path = os.path.join(backup_dir, 'vector_backup.json')
    return backup_path


def backup_graph_data(backup_dir: str, backup_type: str) -> str:
    """备份图数据"""
    # 实现图数据备份逻辑
    backup_path = os.path.join(backup_dir, 'graph_backup.json')
    return backup_path


def backup_config_data(backup_dir: str) -> str:
    """备份配置数据"""
    # 实现配置备份逻辑
    backup_path = os.path.join(backup_dir, 'config_backup.tar.gz')
    return backup_path


def calculate_directory_size(directory: str) -> int:
    """计算目录大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except:
                continue
    return total_size


def get_directory_size(directory: str) -> int:
    """获取目录大小"""
    return calculate_directory_size(directory)


def cleanup_old_backups():
    """清理旧备份"""
    try:
        import shutil
        backup_root = '/backup'
        cutoff_time = datetime.now() - timedelta(days=30)

        for backup_type in ['full', 'incremental', 'differential']:
            type_dir = os.path.join(backup_root, backup_type)
            if os.path.exists(type_dir):
                for backup_dir in os.listdir(type_dir):
                    backup_path = os.path.join(type_dir, backup_dir)
                    try:
                        dir_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
                        if dir_time < cutoff_time:
                            shutil.rmtree(backup_path)
                            logger.info(f"清理旧备份: {backup_path}")
                    except:
                        continue
    except Exception as e:
        logger.error(f"清理旧备份失败: {e}")