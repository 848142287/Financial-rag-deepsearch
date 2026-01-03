"""
数据库初始化脚本
使用SQLAlchemy代替Alembic进行数据库管理
"""

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import json
from app.core.structured_logging import get_structured_logger

from app.core.database import Base, engine

logger = get_structured_logger(__name__)


class DatabaseInitializer:
    """数据库初始化器"""

    def __init__(self):
        self.engine = engine
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    async def initialize(self):
        """初始化数据库（异步接口）"""
        return self.init_database()

    def init_database(self):
        """初始化数据库"""
        try:
            logger.info("开始初始化数据库...")

            # 创建所有表
            self.create_tables()
            logger.info("数据库表创建完成")

            # 插入初始数据
            self.insert_initial_data()
            logger.info("初始数据插入完成")

            # 创建索引
            self.create_indexes()
            logger.info("索引创建完成")

            logger.info("数据库初始化完成")
            return True

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False

    def create_tables(self):
        """创建数据库表"""
        try:
            # 导入所有模型以确保它们被注册到Base.metadata
            # 这里已经导入了document和conversation模块

            # 创建所有表
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建成功")

        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise

    def insert_initial_data(self):
        """插入初始数据"""
        try:
            with self.SessionLocal() as session:
                # 插入系统配置
                self._insert_system_configs(session)

                # 插入示例数据（可选）
                self._insert_sample_data(session)

                session.commit()
                logger.info("初始数据插入成功")

        except Exception as e:
            logger.error(f"插入初始数据失败: {e}")
            raise

    def _insert_system_configs(self, session):
        """插入系统配置"""

        configs = [
            ("max_file_size", "50", "最大文件大小(MB)"),
            ("supported_file_types", '["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "md"]', "支持的文件类型"),
            ("embedding_model", "text-embedding-v4", "嵌入模型"),
            ("llm_model", "deepseek-chat", "大语言模型"),
            ("rerank_model", "qwen3-rerank", "重排序模型"),
            ("multimodal_model", "qwen-vl-plus", "多模态模型"),
            ("max_chunks_per_document", "1000", "每个文档最大分块数"),
            ("chunk_size", "512", "文档分块大小"),
            ("chunk_overlap", "50", "文档分块重叠大小"),
            ("system_name", "金融研报智能系统", "系统名称"),
            ("system_version", "2.0.0", "系统版本"),
            ("max_concurrent_tasks", "10", "最大并发任务数"),
            ("default_top_k", "10", "默认检索返回数量"),
            ("similarity_threshold", "0.7", "相似度阈值"),
            ("enable_knowledge_graph", "true", "启用知识图谱"),
            ("enable_content_analysis", "true", "启用内容分析"),
            ("storage_layers", '["memory", "redis", "filesystem"]', "存储层配置")
        ]

        for config_key, config_value, description in configs:
            # 使用INSERT IGNORE避免重复插入
            insert_stmt = text("""
                INSERT IGNORE INTO system_configs
                (config_key, config_value, description, created_at, updated_at)
                VALUES (:config_key, :config_value, :description, NOW(), NOW())
            """)

            session.execute(insert_stmt, {
                "config_key": config_key,
                "config_value": config_value,
                "description": description
            })

    def _insert_sample_data(self, session):
        """插入示例数据（可选）"""
        # 这里可以插入一些示例数据
        # 例如：示例文档、示例对话等

    def create_indexes(self):
        """创建额外的索引"""
        try:
            with self.SessionLocal() as session:
                # 检查索引是否存在的函数
                check_index_sql = """
                    SELECT COUNT(*) FROM information_schema.statistics
                    WHERE table_schema = DATABASE()
                    AND table_name = :table_name
                    AND index_name = :index_name
                """

                # 文档表索引
                indexes = [
                    ("documents", "idx_documents_status", "CREATE INDEX idx_documents_status ON documents(status)"),
                    ("documents", "idx_documents_created_at", "CREATE INDEX idx_documents_created_at ON documents(created_at)"),
                    ("documents", "idx_documents_file_type", "CREATE INDEX idx_documents_file_type ON documents(file_type)"),
                ]

                # 文档块表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'document_chunks'"))
                    if result.fetchone():
                        indexes.extend([
                            ("document_chunks", "idx_document_chunks_document_id", "CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id)"),
                            ("document_chunks", "idx_document_chunks_chunk_index", "CREATE INDEX idx_document_chunks_chunk_index ON document_chunks(chunk_index)"),
                            ("document_chunks", "idx_document_chunks_embedding_id", "CREATE INDEX idx_document_chunks_embedding_id ON document_chunks(embedding_id)"),
                        ])
                except:
                    pass  # 表不存在，跳过

                # 对话表索引
                indexes.extend([
                    ("conversations", "idx_conversations_created_at", "CREATE INDEX idx_conversations_created_at ON conversations(created_at)"),
                    ("conversations", "idx_conversations_updated_at", "CREATE INDEX idx_conversations_updated_at ON conversations(updated_at)"),
                ])

                # 消息表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'messages'"))
                    if result.fetchone():
                        indexes.extend([
                            ("messages", "idx_messages_conversation_id", "CREATE INDEX idx_messages_conversation_id ON messages(conversation_id)"),
                            ("messages", "idx_messages_created_at", "CREATE INDEX idx_messages_created_at ON messages(created_at)"),
                            ("messages", "idx_messages_role", "CREATE INDEX idx_messages_role ON messages(role)"),
                        ])
                except:
                    pass  # 表不存在，跳过

                # 检索日志表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'retrieval_logs'"))
                    if result.fetchone():
                        indexes.extend([
                            ("retrieval_logs", "idx_retrieval_logs_created_at", "CREATE INDEX idx_retrieval_logs_created_at ON retrieval_logs(created_at)"),
                            ("retrieval_logs", "idx_retrieval_logs_retrieval_type", "CREATE INDEX idx_retrieval_logs_retrieval_type ON retrieval_logs(retrieval_type)"),
                            ("retrieval_logs", "idx_retrieval_logs_response_time", "CREATE INDEX idx_retrieval_logs_response_time ON retrieval_logs(response_time_ms)"),
                        ])
                except:
                    pass  # 表不存在，跳过

                # 任务表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'tasks'"))
                    if result.fetchone():
                        indexes.extend([
                            ("tasks", "idx_tasks_status", "CREATE INDEX idx_tasks_status ON tasks(status)"),
                            ("tasks", "idx_tasks_type", "CREATE INDEX idx_tasks_type ON tasks(task_type)"),
                            ("tasks", "idx_tasks_document", "CREATE INDEX idx_tasks_document ON tasks(document_id)"),
                            ("tasks", "idx_tasks_user", "CREATE INDEX idx_tasks_user ON tasks(user_id)"),
                            ("tasks", "idx_tasks_created", "CREATE INDEX idx_tasks_created ON tasks(created_at)"),
                        ])
                except:
                    pass

                # 章节表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'chapters'"))
                    if result.fetchone():
                        indexes.extend([
                            ("chapters", "idx_chapters_document", "CREATE INDEX idx_chapters_document ON chapters(document_id)"),
                            ("chapters", "idx_chapters_parent", "CREATE INDEX idx_chapters_parent ON chapters(parent_id)"),
                            ("chapters", "idx_chapters_level", "CREATE INDEX idx_chapters_level ON chapters(level)"),
                            ("chapters", "idx_chapters_type", "CREATE INDEX idx_chapters_type ON chapters(chapter_type)"),
                        ])
                except:
                    pass

                # 内容表索引（如果表存在）
                content_tables = ['image_contents', 'chart_contents', 'table_contents', 'formula_contents']
                for table in content_tables:
                    try:
                        result = session.execute(text(f"SHOW TABLES LIKE '{table}'"))
                        if result.fetchone():
                            table_prefix = table.replace('_contents', '')
                            indexes.extend([
                                (table, f"idx_{table_prefix}_document", f"CREATE INDEX idx_{table_prefix}_document ON {table}(document_id)"),
                                (table, f"idx_{table_prefix}_chapter", f"CREATE INDEX idx_{table_prefix}_chapter ON {table}(chapter_id)"),
                                (table, f"idx_{table_prefix}_page", f"CREATE INDEX idx_{table_prefix}_page ON {table}(page_number)"),
                            ])
                    except:
                        pass

                # 知识图谱表索引（如果表存在）
                kg_tables = ['knowledge_graph_nodes', 'knowledge_graph_relations', 'knowledge_graph_entities']
                for table in kg_tables:
                    try:
                        result = session.execute(text(f"SHOW TABLES LIKE '{table}'"))
                        if result.fetchone():
                            if table == 'knowledge_graph_nodes':
                                indexes.extend([
                                    (table, "idx_kg_nodes_document", "CREATE INDEX idx_kg_nodes_document ON knowledge_graph_nodes(document_id)"),
                                    (table, "idx_kg_nodes_type", "CREATE INDEX idx_kg_nodes_type ON knowledge_graph_nodes(node_type)"),
                                    (table, "idx_kg_nodes_name", "CREATE INDEX idx_kg_nodes_name ON knowledge_graph_nodes(node_name)"),
                                ])
                            elif table == 'knowledge_graph_relations':
                                indexes.extend([
                                    (table, "idx_kg_relations_document", "CREATE INDEX idx_kg_relations_document ON knowledge_graph_relations(document_id)"),
                                    (table, "idx_kg_relations_source", "CREATE INDEX idx_kg_relations_source ON knowledge_graph_relations(source_node_id)"),
                                    (table, "idx_kg_relations_target", "CREATE INDEX idx_kg_relations_target ON knowledge_graph_relations(target_node_id)"),
                                    (table, "idx_kg_relations_type", "CREATE INDEX idx_kg_relations_type ON knowledge_graph_relations(relation_type)"),
                                ])
                            elif table == 'knowledge_graph_entities':
                                indexes.extend([
                                    (table, "idx_kg_entities_document", "CREATE INDEX idx_kg_entities_document ON knowledge_graph_entities(document_id)"),
                                    (table, "idx_kg_entities_type", "CREATE INDEX idx_kg_entities_type ON knowledge_graph_entities(entity_type)"),
                                    (table, "idx_kg_entities_name", "CREATE INDEX idx_kg_entities_name ON knowledge_graph_entities(entity_name)"),
                                ])
                    except:
                        pass

                # 用户表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'users'"))
                    if result.fetchone():
                        indexes.extend([
                            ("users", "idx_users_username", "CREATE INDEX idx_users_username ON users(username)"),
                            ("users", "idx_users_email", "CREATE INDEX idx_users_email ON users(email)"),
                            ("users", "idx_users_role", "CREATE INDEX idx_users_role ON users(role)"),
                            ("users", "idx_users_active", "CREATE INDEX idx_users_active ON users(is_active)"),
                        ])
                except:
                    pass

                # 任务队列表索引（如果表存在）
                try:
                    result = session.execute(text("SHOW TABLES LIKE 'task_queue'"))
                    if result.fetchone():
                        indexes.extend([
                            ("task_queue", "idx_task_queue_task_id", "CREATE INDEX idx_task_queue_task_id ON task_queue(task_id)"),
                            ("task_queue", "idx_task_queue_status", "CREATE INDEX idx_task_queue_status ON task_queue(status)"),
                            ("task_queue", "idx_task_queue_task_type", "CREATE INDEX idx_task_queue_task_type ON task_queue(task_type)"),
                            ("task_queue", "idx_task_queue_created_at", "CREATE INDEX idx_task_queue_created_at ON task_queue(created_at)"),
                            ("task_queue", "idx_task_queue_priority", "CREATE INDEX idx_task_queue_priority ON task_queue(priority)"),
                        ])
                except:
                    pass  # 表不存在，跳过

                for table_name, index_name, create_sql in indexes:
                    # 检查索引是否已存在
                    count_result = session.execute(text(check_index_sql), {
                        "table_name": table_name,
                        "index_name": index_name
                    })

                    if count_result.scalar() == 0:
                        try:
                            session.execute(text(create_sql))
                            logger.info(f"创建索引 {index_name} 成功")
                        except Exception as e:
                            logger.warning(f"创建索引 {index_name} 失败: {e}")
                    else:
                        logger.info(f"索引 {index_name} 已存在，跳过创建")

                session.commit()
                logger.info("索引检查和创建完成")

        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            # 不抛出异常，允许应用继续运行

    def check_database_status(self):
        """检查数据库状态"""
        try:
            with self.SessionLocal() as session:
                # 检查表是否存在
                result = session.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                """))
                tables = [row[0] for row in result.fetchall()]

                logger.info(f"数据库中的表: {tables}")

                # 检查数据数量
                table_counts = {}
                for table in tables:
                    try:
                        count_result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_counts[table] = count_result.scalar()
                    except Exception as e:
                        logger.warning(f"无法查询表 {table} 的数据量: {e}")
                        table_counts[table] = -1

                return {
                    "tables": tables,
                    "counts": table_counts,
                    "status": "healthy" if tables else "empty"
                }

        except Exception as e:
            logger.error(f"检查数据库状态失败: {e}")
            return {"status": "error", "error": str(e)}

    def reset_database(self):
        """重置数据库（删除所有表并重新创建）"""
        try:
            logger.warning("开始重置数据库...")

            # 删除所有表
            Base.metadata.drop_all(bind=self.engine)
            logger.info("删除所有表完成")

            # 重新创建
            self.init_database()
            logger.info("数据库重置完成")

            return True

        except Exception as e:
            logger.error(f"重置数据库失败: {e}")
            return False

    def backup_database(self, backup_path: str):
        """备份数据库结构"""
        try:
            import json
            import os
            from sqlalchemy import inspect

            # 确保备份目录存在
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # 获取数据库检查器
            inspector = inspect(self.engine)

            # 获取所有表结构
            backup_data = {
                'tables': {},
                'indexes': {},
                'backup_time': datetime.now().isoformat()
            }

            # 备份表结构
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                primary_keys = inspector.get_pk_constraint(table_name)

                backup_data['tables'][table_name] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys,
                    'primary_keys': primary_keys
                }

            # 备份索引
            for table_name in inspector.get_table_names():
                indexes = inspector.get_indexes(table_name)
                if indexes:
                    backup_data['indexes'][table_name] = indexes

            # 写入备份文件
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"数据库结构备份完成: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"备份数据库失败: {e}")
            return False


# 全局数据库初始化器实例
db_initializer = DatabaseInitializer()