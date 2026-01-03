"""
文档处理监控服务
监控处理进度、失败率和性能指标
"""

import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy import select, text, func
from app.core.database import async_session_maker
from app.models.document import Document

logger = logging.getLogger(__name__)

class ProcessingMonitor:
    """处理监控器"""
    
    def __init__(self):
        self.check_interval = 60  # 检查间隔：60秒
        self.running = False
        
    async def start(self):
        """启动监控"""
        self.running = True
        logger.info("🔍 处理监控服务已启动")
        
        while self.running:
            try:
                await self._check_processing_status()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"监控错误: {e}")
                await asyncio.sleep(30)
    
    def stop(self):
        """停止监控"""
        self.running = False
        logger.info("处理监控服务已停止")
    
    async def _check_processing_status(self):
        """检查处理状态"""
        async with async_session_maker() as db:
            # 1. 检查处理中的文档（可能卡住）
            stuck_result = await db.execute(
                select(Document).where(
                    Document.status == "processing",
                    Document.updated_at < datetime.now() - timedelta(minutes=10)
                )
            )
            stuck_docs = stuck_result.scalars().all()
            
            if stuck_docs:
                logger.warning(f"⚠️  发现 {len(stuck_docs)} 个卡住的文档")
                for doc in stuck_docs:
                    await self._handle_stuck_document(doc, db)
            
            # 2. 统计处理状态
            stats_result = await db.execute(
                select(Document.status, func.count(Document.id))
                .group_by(Document.status)
            )
            stats = dict(stats_result.all())
            
            total = sum(stats.values())
            completed = stats.get('completed', 0)
            processing = stats.get('processing', 0)
            failed = stats.get('failed', 0)
            
            logger.info(
                f"📊 处理统计: 总数={total}, "
                f"已完成={completed}({completed/total*100 if total > 0 else 0:.1f}%), "
                f"处理中={processing}, 失败={failed}"
            )
            
            # 3. 检查失败率
            if total > 0:
                failure_rate = failed / total
                if failure_rate > 0.1:  # 失败率超过10%
                    logger.error(f"🚨 失败率过高: {failure_rate:.1%}")
                    await self._alert_high_failure_rate(failure_rate, failed, total)
    
    async def _handle_stuck_document(self, document: Document, db):
        """处理卡住的文档"""
        waiting_time = datetime.now() - document.updated_at
        
        logger.warning(
            f"文档 {document.id} ({document.title}) 已等待 {waiting_time.seconds}秒"
        )
        
        # 超过30分钟标记为失败
        if waiting_time > timedelta(minutes=30):
            logger.error(f"文档 {document.id} 超时，标记为失败")
            document.status = "failed"
            document.error_message = f"处理超时 ({waiting_time.seconds}秒)"
            await db.commit()
    
    async def _alert_high_failure_rate(self, failure_rate: float, failed: int, total: int):
        """告警高失败率"""
        # TODO: 发送到监控系统或邮件
        alert_msg = (
            f"🚨 文档处理失败率告警\n"
            f"时间: {datetime.now().isoformat()}\n"
            f"失败率: {failure_rate:.1%}\n"
            f"失败数: {failed}/{total}"
        )
        logger.error(alert_msg)
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """获取处理指标"""
        async with async_session_maker() as db:
            # 基本统计
            result = await db.execute(
                select(Document.status, func.count(Document.id))
                .group_by(Document.status)
            )
            stats = dict(result.all())
            
            # 平均处理时间（已完成的文档）
            time_result = await db.execute(
                text("""
                    SELECT AVG(
                        TIMESTAMPDIFF(SECOND, created_at, processed_at)
                    ) as avg_time
                    FROM documents
                    WHERE status = 'completed'
                    AND processed_at IS NOT NULL
                """)
            )
            avg_time = time_result.scalar() or 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_documents': sum(stats.values()),
                'status_breakdown': stats,
                'average_processing_time_seconds': avg_time,
                'completion_rate': stats.get('completed', 0) / sum(stats.values()) if sum(stats.values()) > 0 else 0
            }

# 全局监控实例
monitor = ProcessingMonitor()

async def start_monitor():
    """启动监控服务"""
    await monitor.start()

async def stop_monitor():
    """停止监控服务"""
    monitor.stop()
