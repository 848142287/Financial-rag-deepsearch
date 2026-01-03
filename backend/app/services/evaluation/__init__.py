"""
评估系统模块
统一的RAGAS检索评估框架

主要组件:
- BaseEvaluator: 评估器抽象基类
- UnifiedRetrievalEvaluator: 统一的检索评估器
- RagasQualityEvaluator: RAGAS质量评估器
- OptimizedEvalWorker: 优化的评估Worker
- UnifiedEvaluationService: 统一的评估服务
- EvaluatorConfig: 配置管理

使用示例:
```python
from app.services.evaluation import (
    UnifiedEvaluationService,
    EvaluatorConfig
)

# 创建服务
config = EvaluatorConfig.from_env()
service = UnifiedEvaluationService(config.to_dict())

# 创建并启动评估任务
task = await service.create_evaluation_task(
    question_count=300,
    selected_documents=["/path/to/doc1.pdf", "/path/to/doc2.pdf"]
)
await service.start_evaluation(task.task_id, db_session, background_tasks)
```
"""

__all__ = [
    # 基础类
    "BaseEvaluator",
    "EvaluationResult",
    "RetrievalTestData",
    "MetricType",

    # 评估器
    "UnifiedRetrievalEvaluator",
    "RagasQualityEvaluator",

    # Worker
    "OptimizedEvalWorker",

    # 服务
    "UnifiedEvaluationService",

    # 配置
    "EvaluatorConfig",
    "EvaluationMode",
    "DEFAULT_CONFIG",
    "get_config",
    "merge_configs",
]

# 版本信息
__version__ = "2.0.0"
__author__ = "RAG Evaluation Team"
