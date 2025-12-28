"""
工作流工厂
预定义工作流模板和工作流创建工具
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .workflow_engine import WorkflowDefinition, WorkflowNode, NodeType, WorkflowEdge
from .agents import PREDEFINED_AGENTS
from .tools import PREDEFINED_TOOLS
from .nodes import PredefinedNodes

logger = logging.getLogger(__name__)


class WorkflowFactory:
    """工作流工厂类"""

    @staticmethod
    def create_intelligent_search_workflow() -> WorkflowDefinition:
        """创建智能搜索工作流"""
        nodes = [
            PredefinedNodes.create_query_analysis_node(),
            PredefinedNodes.create_search_strategy_node(),
            PredefinedNodes.create_retrieval_node(),
            PredefinedNodes.create_answer_generation_node(),
            PredefinedNodes.create_answer_evaluation_node(),
            PredefinedNodes.create_result_refinement_node()
        ]

        edges = [
            WorkflowEdge("query_analysis", "search_strategy"),
            WorkflowEdge("search_strategy", "retrieval"),
            WorkflowEdge("retrieval", "answer_generation"),
            WorkflowEdge("answer_generation", "answer_evaluation"),
            WorkflowEdge("answer_evaluation", "result_refinement")
        ]

        return WorkflowDefinition(
            id="intelligent_search",
            name="智能搜索工作流",
            description="基于多策略的智能文档搜索和答案生成",
            version="1.0",
            nodes=nodes,
            edges=edges,
            config={
                "max_execution_time": 300,  # 5分钟
                "enable_caching": True,
                "enable_evaluation": True
            }
        )

    @staticmethod
    def create_deep_search_workflow() -> WorkflowDefinition:
        """创建深度搜索工作流"""
        nodes = [
            # 开始节点
            WorkflowNode(
                id="start",
                name="开始",
                node_type=NodeType.START,
                description="深度搜索开始"
            ),

            # 查询分析
            PredefinedNodes.create_query_analysis_node(),

            # 深度检索（多轮）
            WorkflowNode(
                id="deep_retrieval",
                name="深度检索",
                node_type=NodeType.AGENT,
                description="执行多轮迭代检索",
                agent="deep_search_agent",
                config={
                    "max_iterations": 3,
                    "convergence_threshold": 0.8,
                    "iteration_strategies": ["vector", "graph", "keyword"]
                }
            ),

            # 结果分析
            WorkflowNode(
                id="result_analysis",
                name="结果分析",
                node_type=NodeType.AGENT,
                description="分析检索结果质量和相关性",
                agent="analysis_agent",
                config={
                    "analysis_types": ["relevance", "completeness", "diversity"]
                }
            ),

            # 智能综合
            WorkflowNode(
                id="intelligent_synthesis",
                name="智能综合",
                node_type=NodeType.AGENT,
                description="综合多轮检索结果生成深度答案",
                agent="llm_agent",
                config={
                    "synthesis_mode": "deep",
                    "include_sources": True,
                    "reasoning_steps": True
                }
            ),

            # 质量验证
            PredefinedNodes.create_quality_check_node(),

            # 结果优化
            WorkflowNode(
                id="deep_refinement",
                name="深度优化",
                node_type=NodeType.TRANSFORM,
                description="优化深度搜索结果",
                function=lambda state: {
                    "final_result": {
                        "query": state.get("query"),
                        "answer": state.get("answer"),
                        "depth_analysis": state.get("result_analysis"),
                        "synthesis_quality": state.get("evaluation_score"),
                        "iterations": state.get("deep_retrieval_result", {}).get("iterations", 1)
                    }
                }
            )
        ]

        edges = [
            WorkflowEdge("start", "query_analysis"),
            WorkflowEdge("query_analysis", "deep_retrieval"),
            WorkflowEdge("deep_retrieval", "result_analysis"),
            WorkflowEdge("result_analysis", "intelligent_synthesis"),
            WorkflowEdge("intelligent_synthesis", "quality_check"),
            # 质量检查分支
            WorkflowEdge("quality_check", "deep_refinement", condition=lambda state: state.get("quality_check_result", False)),
            WorkflowEdge("quality_check", "intelligent_synthesis", condition=lambda state: not state.get("quality_check_result", False)),
            WorkflowEdge("deep_refinement", "end")
        ]

        return WorkflowDefinition(
            id="deep_search",
            name="深度搜索工作流",
            description="多轮迭代的深度搜索和分析",
            version="1.0",
            nodes=nodes,
            edges=edges,
            config={
                "max_execution_time": 600,  # 10分钟
                "enable_iteration": True,
                "enable_convergence_check": True
            }
        )

    @staticmethod
    def create_document_processing_workflow() -> WorkflowDefinition:
        """创建文档处理工作流"""
        nodes = [
            # 开始
            WorkflowNode(
                id="start",
                name="开始",
                node_type=NodeType.START
            ),

            # 文档上传
            WorkflowNode(
                id="document_upload",
                name="文档上传",
                node_type=NodeType.TOOL,
                description="上传并验证文档",
                tool="document_upload"
            ),

            # 内容提取
            WorkflowNode(
                id="content_extraction",
                name="内容提取",
                node_type=NodeType.TOOL,
                description="提取文档内容和结构",
                tool="document_extract",
                config={
                    "extract_types": ["text", "tables", "images"]
                }
            ),

            # 实体识别
            WorkflowNode(
                id="entity_recognition",
                name="实体识别",
                node_type=NodeType.AGENT,
                description="识别文档中的金融实体",
                agent="nlp_agent",
                config={
                    "entity_types": ["person", "organization", "location", "financial_metric"]
                }
            ),

            # 关系抽取
            WorkflowNode(
                id="relation_extraction",
                name="关系抽取",
                node_type=NodeType.AGENT,
                description="抽取实体间的关系",
                agent="nlp_agent",
                config={
                    "relation_types": ["ownership", "investment", "employment", "financial"]
                }
            ),

            # 知识图谱构建
            WorkflowNode(
                id="knowledge_graph_construction",
                name="知识图谱构建",
                node_type=NodeType.AGENT,
                description="构建和更新知识图谱",
                agent="graph_agent"
            ),

            # 向量化处理
            WorkflowNode(
                id="vectorization",
                name="向量化处理",
                node_type=NodeType.AGENT,
                description="生成文档向量表示",
                agent="embedding_agent"
            ),

            # 索引更新
            WorkflowNode(
                id="index_update",
                name="索引更新",
                node_type=NodeType.TOOL,
                description="更新搜索索引",
                tool="index_update"
            )
        ]

        edges = [
            WorkflowEdge("start", "document_upload"),
            WorkflowEdge("document_upload", "content_extraction"),
            WorkflowEdge("content_extraction", "entity_recognition"),
            WorkflowEdge("entity_recognition", "relation_extraction"),
            WorkflowEdge("relation_extraction", "knowledge_graph_construction"),
            WorkflowEdge("knowledge_graph_construction", "vectorization"),
            WorkflowEdge("vectorization", "index_update")
        ]

        return WorkflowDefinition(
            id="document_processing",
            name="文档处理工作流",
            description="完整的文档处理和知识提取流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            config={
                "max_file_size": 100 * 1024 * 1024,  # 100MB
                "supported_formats": ["pdf", "docx", "xlsx", "txt"],
                "enable_nlp": True,
                "enable_knowledge_graph": True
            }
        )

    @staticmethod
    def create_evaluation_workflow() -> WorkflowDefinition:
        """创建评估工作流"""
        nodes = [
            # 开始
            WorkflowNode(
                id="start",
                name="开始",
                node_type=NodeType.START
            ),

            # 数据准备
            WorkflowNode(
                id="data_preparation",
                name="数据准备",
                node_type=NodeType.TRANSFORM,
                description="准备评估数据",
                function=lambda state: {
                    "question": state.get("question"),
                    "answer": state.get("answer"),
                    "contexts": state.get("contexts", []),
                    "ground_truth": state.get("ground_truth")
                }
            ),

            # RAGAS评估
            WorkflowNode(
                id="ragas_evaluation",
                name="RAGAS评估",
                node_type=NodeType.TOOL,
                description="使用RAGAS进行多维度评估",
                tool="ragas_evaluation",
                config={
                    "aspects": ["faithfulness", "answer_relevance", "context_relevance", "context_recall"]
                }
            ),

            # 质量分析
            WorkflowNode(
                id="quality_analysis",
                name="质量分析",
                node_type=NodeType.AGENT,
                description="分析评估结果并提供建议",
                agent="evaluation_agent"
            ),

            # 报告生成
            WorkflowNode(
                id="report_generation",
                name="报告生成",
                node_type=NodeType.TRANSFORM,
                description="生成评估报告",
                function=lambda state: {
                    "evaluation_report": {
                        "overall_score": state.get("ragas_evaluation", {}).get("overall_score", 0.0),
                        "detailed_scores": state.get("ragas_evaluation", {}),
                        "quality_analysis": state.get("quality_analysis_result"),
                        "recommendations": state.get("quality_analysis_result", {}).get("recommendations", []),
                        "evaluation_time": datetime.now().isoformat()
                    }
                }
            )
        ]

        edges = [
            WorkflowEdge("start", "data_preparation"),
            WorkflowEdge("data_preparation", "ragas_evaluation"),
            WorkflowEdge("ragas_evaluation", "quality_analysis"),
            WorkflowEdge("quality_analysis", "report_generation")
        ]

        return WorkflowDefinition(
            id="evaluation",
            name="评估工作流",
            description="RAG答案质量评估流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            config={
                "enable_detailed_analysis": True,
                "generate_recommendations": True
            }
        )

    @staticmethod
    def create_qa_workflow() -> WorkflowDefinition:
        """创建问答工作流"""
        nodes = [
            # 开始
            WorkflowNode(
                id="start",
                name="开始",
                node_type=NodeType.START
            ),

            # 问题理解
            WorkflowNode(
                id="question_understanding",
                name="问题理解",
                node_type=NodeType.AGENT,
                description="深度理解用户问题",
                agent="analysis_agent"
            ),

            # 快速搜索
            WorkflowNode(
                id="quick_search",
                name="快速搜索",
                node_type=NodeType.TOOL,
                description="执行快速关键词搜索",
                tool="search",
                config={"max_results": 5}
            ),

            # 相关性判断
            WorkflowNode(
                id="relevance_check",
                name="相关性检查",
                node_type=NodeType.CONDITION,
                description="检查快速搜索结果的相关性",
                function=lambda state: len(state.get("quick_search_result", [])) > 0 and
                                    any(result.get("score", 0) > 0.7 for result in state.get("quick_search_result", []))
            ),

            # 深度搜索（如果需要）
            WorkflowNode(
                id="deep_search",
                name="深度搜索",
                node_type=NodeType.AGENT,
                description="执行深度搜索",
                agent="search_agent",
                config={"search_type": "comprehensive"}
            ),

            # 答案生成
            WorkflowNode(
                id="answer_generation",
                name="答案生成",
                node_type=NodeType.AGENT,
                description="生成答案",
                agent="llm_agent"
            ),

            # 答案验证
            WorkflowNode(
                id="answer_verification",
                name="答案验证",
                node_type=NodeType.AGENT,
                description="验证答案的准确性和完整性",
                agent="verification_agent"
            ),

            # 结果返回
            WorkflowNode(
                id="result_return",
                name="结果返回",
                node_type=NodeType.TRANSFORM,
                description="格式化并返回最终结果",
                function=lambda state: {
                    "question": state.get("question"),
                    "answer": state.get("generated_answer"),
                    "confidence": state.get("answer_verification", {}).get("confidence", 0.0),
                    "sources": state.get("search_results", [])[:3]
                }
            )
        ]

        edges = [
            WorkflowEdge("start", "question_understanding"),
            WorkflowEdge("question_understanding", "quick_search"),
            WorkflowEdge("quick_search", "relevance_check"),
            # 相关性分支
            WorkflowEdge("relevance_check", "answer_generation", condition=lambda state: state.get("relevance_check_result", False)),
            WorkflowEdge("relevance_check", "deep_search", condition=lambda state: not state.get("relevance_check_result", False)),
            WorkflowEdge("deep_search", "answer_generation"),
            WorkflowEdge("answer_generation", "answer_verification"),
            WorkflowEdge("answer_verification", "result_return")
        ]

        return WorkflowDefinition(
            id="qa",
            name="问答工作流",
            description="智能问答处理流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            config={
                "enable_quick_search": True,
                "enable_deep_search_fallback": True,
                "max_answer_length": 2000
            }
        )

    @staticmethod
    def create_custom_workflow(config: Dict[str, Any]) -> WorkflowDefinition:
        """根据配置创建自定义工作流"""
        workflow_id = config.get("id", f"custom_{datetime.now().timestamp()}")
        name = config.get("name", "自定义工作流")
        description = config.get("description", "用户自定义的工作流")
        version = config.get("version", "1.0")

        nodes = []
        edges = []

        # 创建节点
        for node_config in config.get("nodes", []):
            node = WorkflowFactory._create_node_from_config(node_config)
            nodes.append(node)

        # 创建边
        for edge_config in config.get("edges", []):
            edge = WorkflowFactory._create_edge_from_config(edge_config)
            edges.append(edge)

        return WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            version=version,
            nodes=nodes,
            edges=edges,
            config=config.get("config", {})
        )

    @staticmethod
    def _create_node_from_config(config: Dict[str, Any]) -> WorkflowNode:
        """从配置创建节点"""
        node_id = config["id"]
        name = config.get("name", node_id)
        node_type = NodeType(config.get("type", "transform"))
        description = config.get("description", "")

        if node_type == NodeType.AGENT:
            agent_name = config.get("agent")
            if agent_name not in PREDEFINED_AGENTS:
                raise ValueError(f"未知的智能体: {agent_name}")
            return WorkflowNode(
                id=node_id,
                name=name,
                node_type=node_type,
                description=description,
                agent=agent_name,
                config=config.get("config", {})
            )
        elif node_type == NodeType.TOOL:
            tool_name = config.get("tool")
            if tool_name not in PREDEFINED_TOOLS:
                raise ValueError(f"未知的工具: {tool_name}")
            return WorkflowNode(
                id=node_id,
                name=name,
                node_type=node_type,
                description=description,
                tool=tool_name,
                config=config.get("config", {})
            )
        else:
            return WorkflowNode(
                id=node_id,
                name=name,
                node_type=node_type,
                description=description,
                config=config.get("config", {})
            )

    @staticmethod
    def _create_edge_from_config(config: Dict[str, Any]) -> WorkflowEdge:
        """从配置创建边"""
        from_node = config["from"]
        to_node = config["to"]
        description = config.get("description", "")

        return WorkflowEdge(
            from_node=from_node,
            to_node=to_node,
            description=description,
            config=config.get("config", {})
        )

    @staticmethod
    def get_predefined_workflows() -> Dict[str, WorkflowDefinition]:
        """获取所有预定义工作流"""
        return {
            "intelligent_search": WorkflowFactory.create_intelligent_search_workflow(),
            "deep_search": WorkflowFactory.create_deep_search_workflow(),
            "document_processing": WorkflowFactory.create_document_processing_workflow(),
            "evaluation": WorkflowFactory.create_evaluation_workflow(),
            "qa": WorkflowFactory.create_qa_workflow()
        }

    @staticmethod
    def list_workflow_templates() -> List[Dict[str, str]]:
        """列出所有工作流模板"""
        workflows = WorkflowFactory.get_predefined_workflows()
        return [
            {
                "id": workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version
            }
            for workflow_id, workflow in workflows.items()
        ]


def get_workflow_template(template_id: str) -> Optional[WorkflowDefinition]:
    """获取工作流模板"""
    workflows = WorkflowFactory.get_predefined_workflows()
    return workflows.get(template_id)