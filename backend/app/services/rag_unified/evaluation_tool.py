"""
RAG评估Tool - 使用LangChain 1.0+ Tool接口封装
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from app.services.evaluation.rag_evaluator import RAGEvaluator
from app.core.config import get_settings
from app.core.logging import logger

class RAGEvaluationInput(BaseModel):
    """RAG评估输入参数"""
    evaluation_type: str = Field(
        description="评估类型: context_precision, answer_relevancy, faithfulness, overall"
    )
    questions: List[str] = Field(description="评估问题列表")
    retrieved_contexts: Optional[List[List[str]]] = Field(
        default=None,
        description="检索到的上下文列表"
    )
    generated_answers: Optional[List[str]] = Field(
        default=None,
        description="生成的回答列表"
    )
    ground_truth_answers: Optional[List[str]] = Field(
        default=None,
        description="标准答案列表"
    )
    knowledge_base_id: Optional[int] = Field(default=None, description="知识库ID")

class RAGEvaluationTool(BaseTool):
    """
    RAG评估Tool

    基于LangChain Tool接口封装的RAG系统评估功能
    """
    name: str = "rag_evaluation"
    description: str = "评估RAG系统性能，包括上下文精确度、答案相关性、忠实度等指标"
    args_schema: Type[BaseModel] = RAGEvaluationInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = get_settings()
        self.rag_evaluator = None
        self._init_services()

    def _init_services(self):
        """初始化服务组件"""
        try:
            self.rag_evaluator = RAGEvaluator()
        except Exception as e:
            logger.error(f"Failed to initialize RAGEvaluator: {e}")

    def _run(
        self,
        evaluation_type: str,
        questions: List[str],
        retrieved_contexts: Optional[List[List[str]]] = None,
        generated_answers: Optional[List[str]] = None,
        ground_truth_answers: Optional[List[str]] = None,
        knowledge_base_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        同步执行RAG评估

        Args:
            evaluation_type: 评估类型
            questions: 评估问题列表
            retrieved_contexts: 检索上下文
            generated_answers: 生成回答
            ground_truth_answers: 标准答案
            knowledge_base_id: 知识库ID

        Returns:
            评估结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._arun(
                    evaluation_type=evaluation_type,
                    questions=questions,
                    retrieved_contexts=retrieved_contexts,
                    generated_answers=generated_answers,
                    ground_truth_answers=ground_truth_answers,
                    knowledge_base_id=knowledge_base_id
                )
            )
        finally:
            loop.close()

    async def _arun(
        self,
        evaluation_type: str,
        questions: List[str],
        retrieved_contexts: Optional[List[List[str]]] = None,
        generated_answers: Optional[List[str]] = None,
        ground_truth_answers: Optional[List[str]] = None,
        knowledge_base_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        异步执行RAG评估

        Args:
            evaluation_type: 评估类型
            questions: 评估问题列表
            retrieved_contexts: 检索上下文
            generated_answers: 生成回答
            ground_truth_answers: 标准答案
            knowledge_base_id: 知识库ID

        Returns:
            评估结果字典
        """
        if not self.rag_evaluator:
            return {
                "success": False,
                "error": "RAG evaluator not initialized",
                "scores": {}
            }

        try:
            # 根据评估类型执行相应评估
            if evaluation_type == "context_precision":
                result = await self._evaluate_context_precision(
                    questions, retrieved_contexts, ground_truth_answers
                )
            elif evaluation_type == "answer_relevancy":
                result = await self._evaluate_answer_relevancy(
                    questions, generated_answers
                )
            elif evaluation_type == "faithfulness":
                result = await self._evaluate_faithfulness(
                    questions, retrieved_contexts, generated_answers
                )
            elif evaluation_type == "overall":
                result = await self._evaluate_overall(
                    questions, retrieved_contexts, generated_answers, ground_truth_answers
                )
            else:
                result = {
                    "success": False,
                    "error": f"Unknown evaluation type: {evaluation_type}"
                }

            result["evaluation_type"] = evaluation_type
            result["question_count"] = len(questions)
            return result

        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "evaluation_type": evaluation_type,
                "scores": {}
            }

    async def _evaluate_context_precision(
        self,
        questions: List[str],
        retrieved_contexts: Optional[List[List[str]]],
        ground_truth_answers: Optional[List[str]]
    ) -> Dict[str, Any]:
        """评估上下文精确度"""
        if not retrieved_contexts or not ground_truth_answers:
            return {"success": False, "error": "Retrieved contexts and ground truth answers are required"}

        result = await self.rag_evaluator.evaluate_context_precision(
            questions=questions,
            retrieved_contexts=retrieved_contexts,
            ground_truth_answers=ground_truth_answers
        )
        return {
            "success": True,
            "scores": {
                "context_precision": result.get("score", 0.0),
                "precision_per_question": result.get("per_question_scores", [])
            },
            "detailed_results": result
        }

    async def _evaluate_answer_relevancy(
        self,
        questions: List[str],
        generated_answers: Optional[List[str]]
    ) -> Dict[str, Any]:
        """评估答案相关性"""
        if not generated_answers:
            return {"success": False, "error": "Generated answers are required"}

        result = await self.rag_evaluator.evaluate_answer_relevancy(
            questions=questions,
            generated_answers=generated_answers
        )
        return {
            "success": True,
            "scores": {
                "answer_relevancy": result.get("score", 0.0),
                "relevancy_per_question": result.get("per_question_scores", [])
            },
            "detailed_results": result
        }

    async def _evaluate_faithfulness(
        self,
        questions: List[str],
        retrieved_contexts: Optional[List[List[str]]],
        generated_answers: Optional[List[str]]
    ) -> Dict[str, Any]:
        """评估忠实度"""
        if not retrieved_contexts or not generated_answers:
            return {"success": False, "error": "Retrieved contexts and generated answers are required"}

        result = await self.rag_evaluator.evaluate_faithfulness(
            questions=questions,
            retrieved_contexts=retrieved_contexts,
            generated_answers=generated_answers
        )
        return {
            "success": True,
            "scores": {
                "faithfulness": result.get("score", 0.0),
                "faithfulness_per_question": result.get("per_question_scores", [])
            },
            "detailed_results": result
        }

    async def _evaluate_overall(
        self,
        questions: List[str],
        retrieved_contexts: Optional[List[List[str]]],
        generated_answers: Optional[List[str]],
        ground_truth_answers: Optional[List[str]]
    ) -> Dict[str, Any]:
        """综合评估"""
        result = await self.rag_evaluator.evaluate_overall(
            questions=questions,
            retrieved_contexts=retrieved_contexts or [],
            generated_answers=generated_answers or [],
            ground_truth_answers=ground_truth_answers or []
        )
        return {
            "success": True,
            "scores": {
                "overall_score": result.get("overall_score", 0.0),
                "context_precision": result.get("context_precision", 0.0),
                "answer_relevancy": result.get("answer_relevancy", 0.0),
                "faithfulness": result.get("faithfulness", 0.0)
            },
            "detailed_results": result
        }

    def get_tool_description(self) -> str:
        """获取工具详细描述"""
        return """
        RAG评估工具，支持多种评估指标：

        1. context_precision - 上下文精确度评估
        2. answer_relevancy - 答案相关性评估
        3. faithfulness - 忠实度评估
        4. overall - 综合性能评估

        使用RAGAS框架进行自动评估，可以量化RAG系统性能。
        支持批量评估和单个问题评估。
        适用于系统性能监控、模型调优和效果验证。
        """