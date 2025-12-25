"""
测试问题生成器
基于文档内容自动生成测试问题和黄金标准
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import create_engine, text


@dataclass
class Question:
    """测试问题"""
    id: str
    query: str
    question_type: str  # simple, medium, complex
    category: str  # factual, conceptual, relational, analytical
    difficulty: int  # 1-5
    relevant_doc_ids: List[str]
    expected_answer: str
    metadata: Dict[str, Any]


class QuestionGenerator:
    """测试问题生成器"""

    # 问题模板
    QUESTION_TEMPLATES = {
        "factual": [
            "什么是{topic}？",
            "{topic}的定义是什么？",
            "请解释{topic}。",
            "{entity}是什么？",
        ],
        "conceptual": [
            "{topic}的特点有哪些？",
            "如何理解{topic}？",
            "{topic}的核心内容是什么？",
            "请描述{topic}的主要特征。",
        ],
        "relational": [
            "{topic1}和{topic2}有什么关系？",
            "{topic1}对{topic2}有什么影响？",
            "{topic1}与{topic2}的区别是什么？",
            "分析{topic1}和{topic2}的关联性。",
        ],
        "analytical": [
            "如何分析{topic}？",
            "{topic}的发展趋势如何？",
            "评估{topic}的效果。",
            "{topic}的优势和劣势是什么？",
        ],
    }

    # 金融领域关键词
    FINANCE_KEYWORDS = {
        "topics": [
            "风格轮动", "股票投资", "债券投资", "基金管理",
            "风险控制", "资产配置", "量化投资", "技术分析",
            "基本面分析", "投资策略", "市场趋势", "行业分析",
            "估值模型", "投资组合", "风险管理", "绩效评估"
        ],
        "entities": [
            "A股", "H股", "ETF", "开放式基金", "封闭式基金",
            "平安证券", "国信证券", "基金公司", "上市公司",
            "沪深300", "上证指数", "深证成指"
        ]
    }

    def __init__(self, mysql_config: Dict[str, Any]):
        """
        初始化问题生成器

        Args:
            mysql_config: MySQL配置
        """
        self.mysql_config = mysql_config
        self.engine = create_engine(
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@"
            f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        )

    def generate_from_documents(
        self,
        num_questions: int = 2000,
        doc_ids: Optional[List[int]] = None
    ) -> List[Question]:
        """
        从文档生成测试问题

        Args:
            num_questions: 生成问题数量
            doc_ids: 文档ID列表（None表示所有文档）

        Returns:
            问题列表
        """
        # 获取文档
        documents = self._get_documents(doc_ids)

        if not documents:
            print("No documents found")
            return []

        questions = []

        # 按文档类型生成问题
        for i, doc in enumerate(documents):
            # 每个文档生成多个问题
            questions_per_doc = max(1, num_questions // len(documents))

            doc_questions = self._generate_questions_for_document(
                doc,
                questions_per_doc
            )

            questions.extend(doc_questions)

            if len(questions) >= num_questions:
                break

        # 截断到目标数量
        questions = questions[:num_questions]

        # 分配ID
        for i, question in enumerate(questions):
            question.id = f"q_{i+1:04d}"

        return questions

    def _get_documents(self, doc_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """获取文档列表"""
        try:
            with self.engine.connect() as conn:
                if doc_ids:
                    placeholders = ",".join([f":id{i}" for i in range(len(doc_ids))])
                    sql = text(f"""
                        SELECT
                            id,
                            title,
                            filename,
                            content_type
                        FROM documents
                        WHERE id IN ({placeholders})
                        AND status = 'completed'
                        ORDER BY id
                    """)
                    params = {f"id{i}": doc_id for i, doc_id in enumerate(doc_ids)}
                else:
                    sql = text("""
                        SELECT
                            id,
                            title,
                            filename,
                            content_type
                        FROM documents
                        WHERE status = 'completed'
                        ORDER BY id
                        LIMIT 905
                    """)
                    params = {}

                result = conn.execute(sql, params)

                documents = []
                for row in result:
                    documents.append({
                        "id": str(row.id),
                        "title": row.title,
                        "filename": row.filename,
                        "content_type": row.content_type
                    })

                return documents

        except Exception as e:
            print(f"Failed to get documents: {e}")
            return []

    def _generate_questions_for_document(
        self,
        document: Dict[str, Any],
        num_questions: int
    ) -> List[Question]:
        """为单个文档生成问题"""
        questions = []

        # 提取文档标题中的关键词
        title = document.get("title", "")
        keywords = self._extract_keywords(title)

        # 生成不同类型的问题
        question_types = ["factual", "conceptual", "relational", "analytical"]
        type_distribution = [0.4, 0.3, 0.2, 0.1]  # 分布比例

        for i in range(num_questions):
            # 选择问题类型
            q_type = random.choices(
                question_types,
                weights=type_distribution,
                k=1
            )[0]

            # 选择模板
            template = random.choice(self.QUESTION_TEMPLATES[q_type])

            # 填充模板
            if q_type in ["factual", "conceptual"]:
                query = template.format(topic=random.choice(keywords) if keywords else "相关概念")
            elif q_type == "relational":
                if len(keywords) >= 2:
                    topic1, topic2 = random.sample(keywords, 2)
                    query = template.format(topic1=topic1, topic2=topic2)
                else:
                    query = template.format(
                        topic1=keywords[0] if keywords else "主题1",
                        topic2="相关内容"
                    )
            else:  # analytical
                query = template.format(topic=random.choice(keywords) if keywords else "相关主题")

            # 确定难度级别
            difficulty = self._determine_difficulty(q_type, i, num_questions)

            # 生成期望答案（简化版）
            expected_answer = self._generate_expected_answer(query, q_type, document)

            question = Question(
                id="",  # 稍后分配
                query=query,
                question_type=q_type,
                category=self._map_type_to_category(q_type),
                difficulty=difficulty,
                relevant_doc_ids=[document["id"]],
                expected_answer=expected_answer,
                metadata={
                    "source_doc": document["id"],
                    "source_title": title,
                    "template": template
                }
            )

            questions.append(question)

        return questions

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简化实现：基于规则提取
        keywords = []

        # 金融术语
        finance_terms = [
            "风格轮动", "股票", "债券", "基金", "ETF",
            "风险", "收益", "投资", "资产配置", "量化",
            "市场", "行业", "指数", "策略", "分析"
        ]

        for term in finance_terms:
            if term in text:
                keywords.append(term)

        # 如果没有找到关键词，使用通用词
        if not keywords:
            keywords = ["投资", "分析", "策略"]

        return keywords

    def _determine_difficulty(self, q_type: str, index: int, total: int) -> int:
        """确定问题难度（1-5）"""
        base_difficulty = {
            "factual": 2,
            "conceptual": 3,
            "relational": 4,
            "analytical": 5
        }

        base = base_difficulty.get(q_type, 3)

        # 根据位置调整
        variation = (index % 3) - 1  # -1, 0, 1

        difficulty = max(1, min(5, base + variation))

        return difficulty

    def _map_type_to_category(self, q_type: str) -> str:
        """映射问题类型到类别"""
        mapping = {
            "factual": "简单事实查询",
            "conceptual": "概念解释",
            "relational": "关系推理",
            "analytical": "综合分析"
        }

        return mapping.get(q_type, "其他")

    def _generate_expected_answer(
        self,
        query: str,
        q_type: str,
        document: Dict[str, Any]
    ) -> str:
        """生成期望答案（简化版）"""
        # 简化实现：基于问题类型生成模板答案
        if q_type == "factual":
            return f"根据{document['title']}，{query.split('？')[0]}相关的定义和说明。"
        elif q_type == "conceptual":
            return f"从{document['title']}中可以了解到，{query.split('？')[0]}的主要特点包括..."
        elif q_type == "relational":
            return f"根据{document['title']}，两者存在一定的关联关系，具体表现在..."
        else:  # analytical
            return f"综合{document['title']}的内容，{query.split('？')[0]}需要从多个角度进行分析。"

    def save_questions(
        self,
        questions: List[Question],
        output_file: str = "evaluation_results/questions/test_questions.json"
    ):
        """
        保存问题到文件

        Args:
            questions: 问题列表
            output_file: 输出文件路径
        """
        # 转换为字典格式
        questions_dict = [
            {
                "id": q.id,
                "query": q.query,
                "question_type": q.question_type,
                "category": q.category,
                "difficulty": q.difficulty,
                "relevant_doc_ids": q.relevant_doc_ids,
                "expected_answer": q.expected_answer,
                "metadata": q.metadata
            }
            for q in questions
        ]

        # 添加元数据
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(questions),
            "distribution": self._calculate_distribution(questions),
            "questions": questions_dict
        }

        # 保存到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存 {len(questions)} 个问题到: {output_file}")

    def _calculate_distribution(self, questions: List[Question]) -> Dict[str, Any]:
        """计算问题分布统计"""
        distribution = {
            "by_type": {},
            "by_category": {},
            "by_difficulty": {},
            "by_source_doc": {}
        }

        for q in questions:
            # 按类型统计
            distribution["by_type"][q.question_type] = \
                distribution["by_type"].get(q.question_type, 0) + 1

            # 按类别统计
            distribution["by_category"][q.category] = \
                distribution["by_category"].get(q.category, 0) + 1

            # 按难度统计
            distribution["by_difficulty"][q.difficulty] = \
                distribution["by_difficulty"].get(q.difficulty, 0) + 1

            # 按来源文档统计
            source_doc = q.metadata.get("source_doc", "unknown")
            distribution["by_source_doc"][source_doc] = \
                distribution["by_source_doc"].get(source_doc, 0) + 1

        return distribution

    def load_questions(
        self,
        input_file: str = "evaluation_results/questions/test_questions.json"
    ) -> List[Question]:
        """
        从文件加载问题

        Args:
            input_file: 输入文件路径

        Returns:
            问题列表
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = []
            for q_data in data.get("questions", []):
                question = Question(
                    id=q_data["id"],
                    query=q_data["query"],
                    question_type=q_data["question_type"],
                    category=q_data["category"],
                    difficulty=q_data["difficulty"],
                    relevant_doc_ids=q_data["relevant_doc_ids"],
                    expected_answer=q_data["expected_answer"],
                    metadata=q_data.get("metadata", {})
                )
                questions.append(question)

            print(f"✅ 已加载 {len(questions)} 个问题")
            return questions

        except FileNotFoundError:
            print(f"❌ 文件不存在: {input_file}")
            return []
        except Exception as e:
            print(f"❌ 加载问题失败: {e}")
            return []


# 辅助函数
import os

def generate_test_questions(
    mysql_config: Dict[str, Any],
    num_questions: int = 2000,
    output_file: str = "evaluation_results/questions/test_questions.json"
) -> List[Question]:
    """
    生成测试问题

    Args:
        mysql_config: MySQL配置
        num_questions: 生成问题数量
        output_file: 输出文件路径

    Returns:
        问题列表
    """
    print(f"开始生成 {num_questions} 个测试问题...")

    generator = QuestionGenerator(mysql_config)

    questions = generator.generate_from_documents(
        num_questions=num_questions,
        doc_ids=None  # 使用所有文档
    )

    # 保存问题
    generator.save_questions(questions, output_file)

    # 打印统计信息
    distribution = generator._calculate_distribution(questions)

    print("\n问题分布统计:")
    print(f"  按类型: {distribution['by_type']}")
    print(f"  按类别: {distribution['by_category']}")
    print(f"  按难度: {distribution['by_difficulty']}")

    return questions
