"""
查询扩展服务
通过同义词、相关术语、重写等方式扩展查询，提高召回率
"""

import re
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class ExpansionTerm:
    """扩展术语"""
    term: str
    weight: float = 1.0  # 权重
    source: str = "general"  # 来源: general, domain, llm

class QueryExpansionService:
    """查询扩展服务"""

    def __init__(self):
        # 金融领域同义词库
        self.financial_synonyms = {
            # 投资相关
            "价值投资": ["基本面分析", "长期投资", "价值投资策略"],
            "量化交易": ["算法交易", "程序化交易", "量化投资", "统计套利"],
            "技术分析": ["图表分析", "技术指标", "K线分析"],
            "基本面分析": ["基本面", "公司分析", "财务分析"],

            # 行业相关
            "AI芯片": ["人工智能芯片", "AI处理器", "神经网络芯片", "GPU", "CPU"],
            "半导体": ["集成电路", "芯片产业"],
            "新能源": ["清洁能源", "绿色能源", "光伏", "风电", "电动车"],
            "金融科技": ["fintech", "金融创新", "数字金融"],

            # 投资策略
            "ESG": ["环境社会治理", "可持续投资", "绿色金融"],
            "成长股": ["成长型股票", "高增长股票"],
            "蓝筹股": ["大盘股", "优质股"],
        }

        # 术语变形规则
        self.transformation_rules = {
            "abbreviation": {
                "AI": "人工智能",
                "ESG": "环境社会治理",
                "K线": "蜡烛图",
                "ETF": "交易型开放式指数基金",
                "IPO": "首次公开募股",
                "ROE": "净资产收益率",
                "PE": "市盈率",
                "PB": "市净率",
            },
            "english_variant": {
                "fintech": ["金融科技", "金融技术"],
                "quant": ["量化", "量化交易"],
                "value investing": ["价值投资"],
                "growth stock": ["成长股"],
            }
        }

    def expand_query(
        self,
        query: str,
        max_expansions: int = 5,
        use_synonyms: bool = True,
        use_transformations: bool = True
    ) -> List[ExpansionTerm]:
        """
        扩展查询

        Args:
            query: 原始查询
            max_expansions: 最大扩展数量
            use_synonyms: 是否使用同义词扩展
            use_transformations: 是否使用术语变形

        Returns:
            扩展术语列表
        """
        expansions = []

        # 1. 原始查询（最高权重）
        expansions.append(ExpansionTerm(query, weight=1.0, source="original"))

        # 2. 同义词扩展
        if use_synonyms:
            for key, synonyms in self.financial_synonyms.items():
                if key in query:
                    for synonym in synonyms[:3]:  # 限制数量
                        if synonym not in query:
                            expansions.append(
                                ExpansionTerm(synonym, weight=0.8, source="synonym")
                            )

        # 3. 术语变形扩展
        if use_transformations:
            # 简写展开
            for abbr, full in self.transformation_rules["abbreviation"].items():
                if abbr in query and full not in query:
                    expansions.append(
                        ExpansionTerm(full, weight=0.9, source="abbreviation")
                    )

            # 英文变体
            for en, cn_list in self.transformation_rules["english_variant"].items():
                if en.lower() in query.lower():
                    for cn in cn_list:
                        if cn not in query:
                            expansions.append(
                                ExpansionTerm(cn, weight=0.8, source="translation")
                            )

        # 去重并按权重排序
        unique_terms = {}
        for exp in expansions:
            term_key = exp.term.lower()
            if term_key not in unique_terms or exp.weight > unique_terms[term_key].weight:
                unique_terms[term_key] = exp

        # 按权重排序并限制数量
        sorted_terms = sorted(unique_terms.values(), key=lambda x: -x.weight)
        return sorted_terms[:max_expansions]

    def expand_for_retrieval(
        self,
        query: str,
        strategy: str = "union"
    ) -> Dict[str, Any]:
        """
        为检索扩展查询

        Args:
            query: 原始查询
            strategy: 扩展策略
                - "union": 并集（返回所有扩展查询）
                - "weighted": 加权（带权重）
                - "concatenated": 拼接（合并为长查询）

        Returns:
            扩展结果
        """
        expansions = self.expand_query(query)

        if strategy == "union":
            return {
                'original': query,
                'expansions': [e.term for e in expansions[1:]],  # 排除原始
                'all_queries': [query] + [e.term for e in expansions[1:]],
                'weights': {e.term: e.weight for e in expansions}
            }

        elif strategy == "concatenated":
            # 合并为长查询
            all_terms = " ".join([e.term for e in expansions])
            return {
                'original': query,
                'expanded_query': all_terms,
                'expansions_used': len(expansions) - 1
            }

        elif strategy == "weighted":
            # 返回加权查询列表
            return {
                'original': query,
                'weighted_queries': [
                    {'query': e.term, 'weight': e.weight}
                    for e in expansions
                ]
            }

        return {
            'original': query,
            'expansions': expansions
        }

    def extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词

        Args:
            query: 查询文本

        Returns:
            关键词列表
        """
        # 移除标点和停用词
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)

        # 简单分词（按空格和中文）
        tokens = []
        current = ""

        for char in query:
            if char.isspace():
                if current:
                    tokens.append(current)
                    current = ""
            elif '\u4e00' <= char <= '\u9fff':
                # 中文字符
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char

        if current:
            tokens.append(current)

        # 过滤单字符和停用词
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '但', '等', '？', '？'}
        keywords = [t for t in tokens if len(t) > 1 and t not in stopwords]

        return keywords

# 全局单例
_expansion_service = None

def get_query_expansion_service() -> QueryExpansionService:
    """获取查询扩展服务单例"""
    global _expansion_service
    if _expansion_service is None:
        _expansion_service = QueryExpansionService()
    return _expansion_service
