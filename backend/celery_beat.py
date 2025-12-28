#!/usr/bin/env python3
"""
Celery Beat调度器启动脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.celery_config import celery_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def start_beat_scheduler(loglevel: str = 'INFO'):
    """
    启动Celery Beat调度器
    """
    logger.info("启动Celery Beat调度器")

    beat_args = [
        'beat',
        f'--loglevel={loglevel}',
        '--pidfile=/tmp/celerybeat.pid',
        '--schedule=/tmp/celerybeat-schedule'
    ]

    celery_app.start(beat_args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='启动Celery Beat调度器')
    parser.add_argument('--loglevel', type=str, default='INFO', help='日志级别')

    args = parser.parse_args()

    try:
        start_beat_scheduler(args.loglevel)
    except KeyboardInterrupt:
        logger.info("Celery Beat调度器已停止")
    except Exception as e:
        logger.error(f"启动Celery Beat调度器失败: {e}")
        sys.exit(1)