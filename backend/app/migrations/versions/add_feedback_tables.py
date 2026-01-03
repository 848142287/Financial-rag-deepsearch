"""添加反馈数据表

Revision ID: add_feedback_tables
Revises:
Create Date: 2025-01-01

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_feedback_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """创建反馈相关表"""

    # 创建 feedback_records 表
    op.create_table(
        'feedback_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('query_id', sa.String(64), nullable=False, comment='查询ID'),
        sa.Column('query', sa.Text(), nullable=False, comment='查询文本'),
        sa.Column('retrieval_method', sa.String(50), nullable=False, comment='检索方法'),
        sa.Column('results', sa.JSON(), comment='检索结果列表'),
        sa.Column('rating', sa.Integer(), comment='用户评分(1-5)'),
        sa.Column('clicked_docs', sa.JSON(), comment='点击的文档ID列表'),
        sa.Column('relevance', sa.String(50), comment='相关性评价'),
        sa.Column('feedback_text', sa.Text(), comment='用户反馈文本'),
        sa.Column('latency', sa.Float(), comment='检索延迟(秒)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_feedback_records_query_id', 'feedback_records', ['query_id'])
    op.create_index('ix_feedback_records_created_at', 'feedback_records', ['created_at'])
    op.create_index('idx_query_method', 'feedback_records', ['query_id', 'retrieval_method'])
    op.create_index('idx_created_relevance', 'feedback_records', ['created_at', 'relevance'])

    # 创建 feedback_metrics 表
    op.create_table(
        'feedback_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('timestamp', sa.DateTime(), nullable=False, comment='统计时间'),
        sa.Column('retrieval_method', sa.String(50), nullable=False, comment='检索方法'),
        sa.Column('avg_rating', sa.Float(), comment='平均评分'),
        sa.Column('avg_latency', sa.Float(), comment='平均延迟'),
        sa.Column('relevance_rate', sa.Float(), comment='相关性率'),
        sa.Column('click_rate', sa.Float(), comment='点击率'),
        sa.Column('total_queries', sa.Integer(), comment='总查询数'),
        sa.Column('suggestions', sa.JSON(), comment='优化建议列表'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_feedback_metrics_timestamp', 'feedback_metrics', ['timestamp'])
    op.create_index('idx_method_timestamp', 'feedback_metrics', ['retrieval_method', 'timestamp'])

    # 创建 feedback_summaries 表
    op.create_table(
        'feedback_summaries',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('period_start', sa.DateTime(), nullable=False, comment='统计周期开始时间'),
        sa.Column('period_end', sa.DateTime(), nullable=False, comment='统计周期结束时间'),
        sa.Column('period_type', sa.String(20), nullable=False, comment='周期类型'),
        sa.Column('total_feedbacks', sa.Integer(), comment='总反馈数'),
        sa.Column('avg_rating', sa.Float(), comment='总体平均评分'),
        sa.Column('avg_latency', sa.Float(), comment='总体平均延迟'),
        sa.Column('method_stats', sa.JSON(), comment='各检索方法统计'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='记录创建时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_period', 'feedback_summaries', ['period_start', 'period_end', 'period_type'])

def downgrade():
    """删除反馈相关表"""

    op.drop_index('idx_period', table_name='feedback_summaries')
    op.drop_table('feedback_summaries')

    op.drop_index('idx_method_timestamp', table_name='feedback_metrics')
    op.drop_index('ix_feedback_metrics_timestamp', table_name='feedback_metrics')
    op.drop_table('feedback_metrics')

    op.drop_index('idx_created_relevance', table_name='feedback_records')
    op.drop_index('idx_query_method', table_name='feedback_records')
    op.drop_index('ix_feedback_records_created_at', table_name='feedback_records')
    op.drop_index('ix_feedback_records_query_id', table_name='feedback_records')
    op.drop_table('feedback_records')
