#!/usr/bin/env python3
"""
Celery Worker启动脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

from app.core.celery_config import celery_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def start_worker(queue: str = None, concurrency: int = 4, loglevel: str = 'INFO'):
    """
    启动Celery Worker
    """
    logger.info(f"启动Celery Worker - 队列: {queue or 'default'}, 并发数: {concurrency}")

    worker_args = [
        'worker',
        f'--loglevel={loglevel}',
        f'--concurrency={concurrency}',
        '--without-gossip',
        '--without-mingle',
        '--without-heartbeat',
        '--max-tasks-per-child=1000',
        '--time-limit=300',
        '--soft-time-limit=280'
    ]

    if queue:
        worker_args.extend(['--queues', queue])

    celery_app.start(worker_args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='启动Celery Worker')
    parser.add_argument('--queue', type=str, help='指定队列名称')
    parser.add_argument('--concurrency', type=int, default=4, help='并发进程数')
    parser.add_argument('--loglevel', type=str, default='INFO', help='日志级别')

    args = parser.parse_args()

    try:
        start_worker(args.queue, args.concurrency, args.loglevel)
    except KeyboardInterrupt:
        logger.info("Celery Worker已停止")
    except Exception as e:
        logger.error(f"启动Celery Worker失败: {e}")
        sys.exit(1)