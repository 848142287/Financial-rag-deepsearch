"""
告警服务
处理数据完整性监控告警
"""

from app.core.structured_logging import get_structured_logger
import smtplib
from datetime import datetime
from typing import Dict, List, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass

logger = get_structured_logger(__name__)

@dataclass
class Alert:
    """告警数据类"""
    type: str  # critical, warning, info
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    source: str = "sync_monitoring"
    resolved: bool = False

class AlertService:
    """告警服务"""

    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',  # 根据实际情况配置
            'smtp_port': 587,
            'sender_email': 'alerts@financial-rag.com',  # 根据实际情况配置
            'sender_password': 'your_password',  # 根据实际情况配置
            'recipients': ['admin@financial-rag.com']  # 根据实际情况配置
        }
        self.webhook_config = {
            'slack_webhook': 'https://hooks.slack.com/...',  # 根据实际情况配置
            'teams_webhook': 'https://outlook.office.com/webhook/...'  # 根据实际情况配置
        }

    def create_alert(self, alert_type: str, title: str, message: str, details: Dict[str, Any] = None) -> Alert:
        """创建告警"""
        return Alert(
            type=alert_type,
            title=title,
            message=message,
            details=details or {},
            timestamp=datetime.now()
        )

    def format_alert_message(self, alert: Alert) -> str:
        """格式化告警消息"""
        timestamp_str = alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')

        message = f"""
🚨 {alert.type.upper()} ALERT 🚨

Title: {alert.title}
Time: {timestamp_str}
Source: {alert.source}

Message: {alert.message}

"""

        if alert.details:
            message += "Details:\n"
            for key, value in alert.details.items():
                message += f"  • {key}: {value}\n"

        return message

    def send_email_alert(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            if not self.email_config.get('smtp_server'):
                logger.warning("邮件配置不完整，跳过邮件告警")
                return False

            msg = MimeMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"[{alert.type.upper()}] {alert.title}"

            body = self.format_alert_message(alert)
            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])

            text = msg.as_string()
            server.sendmail(self.email_config['sender_email'], self.email_config['recipients'], text)
            server.quit()

            logger.info(f"邮件告警发送成功: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False

    def send_slack_alert(self, alert: Alert) -> bool:
        """发送Slack告警"""
        try:
            if not self.webhook_config.get('slack_webhook'):
                logger.warning("Slack Webhook配置不完整，跳过Slack告警")
                return False

            import requests

            # 根据告警类型选择颜色
            color = {
                'critical': '#ff0000',
                'warning': '#ff9900',
                'info': '#36a64f'
            }.get(alert.type, '#ff9900')

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.type.upper()}: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Financial RAG System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }

            # 添加详细信息
            if alert.details:
                details_text = "\n".join([f"• {key}: {value}" for key, value in alert.details.items()])
                payload["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": details_text,
                    "short": False
                })

            response = requests.post(
                self.webhook_config['slack_webhook'],
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Slack告警发送成功: {alert.title}")
                return True
            else:
                logger.error(f"Slack告警发送失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")
            return False

    def send_webhook_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            import requests

            payload = {
                "alert": {
                    "type": alert.type,
                    "title": alert.title,
                    "message": alert.message,
                    "details": alert.details,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source
                }
            }

            # 发送到Teams（如果配置了）
            if self.webhook_config.get('teams_webhook'):
                response = requests.post(
                    self.webhook_config['teams_webhook'],
                    json={
                        "@type": "MessageCard",
                        "@context": "http://schema.org/extensions",
                        "themeColor": "FF0000" if alert.type == 'critical' else "FF9900" if alert.type == 'warning' else "36A64F",
                        "sections": [
                            {
                                "activityTitle": f"{alert.type.upper()}: {alert.title}",
                                "activitySubtitle": alert.message,
                                "facts": [
                                    {"name": "Source", "value": alert.source},
                                    {"name": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                                ],
                                "markdown": True
                            }
                        ]
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    logger.info(f"Teams告警发送成功: {alert.title}")
                else:
                    logger.error(f"Teams告警发送失败: {response.status_code}")

            return True

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
            return False

    def process_alert(self, alert: Alert, channels: List[str] = None) -> bool:
        """处理告警 - 发送到指定渠道"""
        if channels is None:
            # 默认发送到所有配置的渠道
            channels = ['email', 'slack', 'webhook']

        success_count = 0
        total_channels = len(channels)

        for channel in channels:
            try:
                if channel == 'email':
                    if self.send_email_alert(alert):
                        success_count += 1
                elif channel == 'slack':
                    if self.send_slack_alert(alert):
                        success_count += 1
                elif channel == 'webhook':
                    if self.send_webhook_alert(alert):
                        success_count += 1
                else:
                    logger.warning(f"未知的告警渠道: {channel}")

            except Exception as e:
                logger.error(f"发送{channel}告警失败: {e}")

        # 记录告警
        self.log_alert(alert, channels, success_count, total_channels)

        return success_count > 0

    def log_alert(self, alert: Alert, channels: List[str], success_count: int, total_channels: int):
        """记录告警到日志"""
        log_level = {
            'critical': 'critical',
            'warning': 'warning',
            'info': 'info'
        }.get(alert.type, 'warning')

        message = f"告警处理: {alert.title} - {alert.message} (渠道: {', '.join(channels)}, 成功: {success_count}/{total_channels})"

        if log_level == 'critical':
            logger.critical(message)
        elif log_level == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

    def create_sync_alerts(self, sync_status: Dict[str, Any]) -> List[Alert]:
        """根据同步状态创建告警"""
        alerts = []

        # 向量同步告警
        vector_rate = sync_status.get('vector_sync_rate', 100)
        if vector_rate < 80:
            alerts.append(self.create_alert(
                alert_type='critical',
                title='Milvus向量同步严重落后',
                message=f'向量同步率仅为 {vector_rate:.1f}%',
                details={
                    'sync_rate': vector_rate,
                    'mysql_vectors': sync_status['data_sources']['mysql'].get('vectors', 0),
                    'milvus_vectors': sync_status['data_sources']['milvus'].get('vectors', 0),
                    'threshold': 80
                }
            ))
        elif vector_rate < 95:
            alerts.append(self.create_alert(
                alert_type='warning',
                title='Milvus向量同步需要关注',
                message=f'向量同步率为 {vector_rate:.1f}%，建议检查',
                details={
                    'sync_rate': vector_rate,
                    'mysql_vectors': sync_status['data_sources']['mysql'].get('vectors', 0),
                    'milvus_vectors': sync_status['data_sources']['milvus'].get('vectors', 0),
                    'threshold': 95
                }
            ))

        # 实体同步告警
        entity_rate = sync_status.get('entity_sync_rate', 100)
        if entity_rate < 80:
            alerts.append(self.create_alert(
                alert_type='critical',
                title='Neo4j实体同步严重落后',
                message=f'实体同步率仅为 {entity_rate:.1f}%',
                details={
                    'sync_rate': entity_rate,
                    'mysql_entities': sync_status['data_sources']['mysql'].get('entities', 0),
                    'neo4j_entities': sync_status['data_sources']['neo4j'].get('entities', 0),
                    'threshold': 80
                }
            ))
        elif entity_rate < 95:
            alerts.append(self.create_alert(
                alert_type='warning',
                title='Neo4j实体同步需要关注',
                message=f'实体同步率为 {entity_rate:.1f}%，建议检查',
                details={
                    'sync_rate': entity_rate,
                    'mysql_entities': sync_status['data_sources']['mysql'].get('entities', 0),
                    'neo4j_entities': sync_status['data_sources']['neo4j'].get('entities', 0),
                    'threshold': 95
                }
            ))

        # 关系同步告警
        relation_rate = sync_status.get('relation_sync_rate', 100)
        if relation_rate < 80:
            alerts.append(self.create_alert(
                alert_type='critical',
                title='Neo4j关系同步严重落后',
                message=f'关系同步率仅为 {relation_rate:.1f}%',
                details={
                    'sync_rate': relation_rate,
                    'mysql_relations': sync_status['data_sources']['mysql'].get('relations', 0),
                    'neo4j_relations': sync_status['data_sources']['neo4j'].get('relations', 0),
                    'threshold': 80
                }
            ))

        return alerts

    def create_health_alerts(self, health_status: Dict[str, Any]) -> List[Alert]:
        """根据健康状态创建告警"""
        alerts = []

        failed_docs = health_status.get('failed_documents', 0)
        processing_docs = health_status.get('processing_documents', 0)

        if failed_docs > 10:
            alerts.append(self.create_alert(
                alert_type='critical',
                title='文档处理失败数量过多',
                message=f'有 {failed_docs} 个文档处理失败',
                details={
                    'failed_documents': failed_docs,
                    'processing_documents': processing_docs,
                    'threshold': 10
                }
            ))
        elif failed_docs > 5:
            alerts.append(self.create_alert(
                alert_type='warning',
                title='文档处理失败数量需要关注',
                message=f'有 {failed_docs} 个文档处理失败',
                details={
                    'failed_documents': failed_docs,
                    'processing_documents': processing_docs,
                    'threshold': 5
                }
            ))

        if processing_docs > 100:
            alerts.append(self.create_alert(
                alert_type='warning',
                title='文档处理队列积压',
                message=f'有 {processing_docs} 个文档正在处理中',
                details={
                    'processing_documents': processing_docs,
                    'threshold': 100
                }
            ))

        return alerts