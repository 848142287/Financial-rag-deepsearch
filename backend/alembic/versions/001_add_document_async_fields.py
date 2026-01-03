"""
添加异步处理和双存储字段到documents表

Revision ID: 001_add_async_fields
Revises:
Create Date: 2026-01-03

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_add_async_fields'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """添加新字段"""

    # 1. 修改parsed_content为Text类型（支持大文本）
    op.execute("ALTER TABLE documents MODIFY COLUMN parsed_content TEXT")

    # 2. 添加向量化状态字段
    op.add_column('documents', sa.Column('vectorization_status', sa.String(50), server_default='pending'))
    op.add_column('documents', sa.Column('vectorization_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('vectorization_completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('vectors_count', sa.Integer(), server_default='0'))

    # 3. 添加知识图谱抽取状态字段
    op.add_column('documents', sa.Column('kg_extraction_status', sa.String(50), server_default='pending'))
    op.add_column('documents', sa.Column('kg_extraction_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('kg_extraction_completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('entities_count', sa.Integer(), server_default='0'))
    op.add_column('documents', sa.Column('relationships_count', sa.Integer(), server_default='0'))
    op.add_column('documents', sa.Column('metrics_count', sa.Integer(), server_default='0'))

    # 4. 添加总体enrichment状态字段
    op.add_column('documents', sa.Column('enrichment_status', sa.String(50), server_default='pending'))
    op.add_column('documents', sa.Column('enrichment_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('enrichment_completed_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('enrichment_error', sa.Text(), nullable=True))

    # 5. 添加MySQL双存储字段
    op.add_column('documents', sa.Column('redis_key', sa.String(500), nullable=True))
    op.add_column('documents', sa.Column('redis_ttl', sa.Integer(), server_default='2592000'))
    op.add_column('documents', sa.Column('mysql_storage_path', sa.String(1000), nullable=True))
    op.add_column('documents', sa.Column('storage_sync_status', sa.String(50), server_default='pending'))

    # 6. 添加索引
    op.create_index('ix_documents_vectorization_status', 'documents', ['vectorization_status'])
    op.create_index('ix_documents_kg_extraction_status', 'documents', ['kg_extraction_status'])
    op.create_index('ix_documents_enrichment_status', 'documents', ['enrichment_status'])
    op.create_index('ix_documents_storage_sync_status', 'documents', ['storage_sync_status'])

def downgrade():
    """回滚更改"""

    # 删除索引
    op.drop_index('ix_documents_storage_sync_status', 'documents')
    op.drop_index('ix_documents_enrichment_status', 'documents')
    op.drop_index('ix_documents_kg_extraction_status', 'documents')
    op.drop_index('ix_documents_vectorization_status', 'documents')

    # 删除字段
    op.drop_column('documents', 'storage_sync_status')
    op.drop_column('documents', 'mysql_storage_path')
    op.drop_column('documents', 'redis_ttl')
    op.drop_column('documents', 'redis_key')

    op.drop_column('documents', 'enrichment_error')
    op.drop_column('documents', 'enrichment_completed_at')
    op.drop_column('documents', 'enrichment_started_at')
    op.drop_column('documents', 'enrichment_status')

    op.drop_column('documents', 'metrics_count')
    op.drop_column('documents', 'relationships_count')
    op.drop_column('documents', 'entities_count')
    op.drop_column('documents', 'kg_extraction_completed_at')
    op.drop_column('documents', 'kg_extraction_started_at')
    op.drop_column('documents', 'kg_extraction_status')

    op.drop_column('documents', 'vectors_count')
    op.drop_column('documents', 'vectorization_completed_at')
    op.drop_column('documents', 'vectorization_started_at')
    op.drop_column('documents', 'vectorization_status')

    # 恢复parsed_content字段类型（如果需要）
    # op.execute("ALTER TABLE documents MODIFY COLUMN parsed_content JSON")
