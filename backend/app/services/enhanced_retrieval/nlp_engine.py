"""
NLP检索增强引擎
从 swxy/backend 移植，提供高级的查询分析和相似度计算功能
"""

from app.core.structured_logging import get_structured_logger
import re
import numpy as np
from collections import Counter

logger = get_structured_logger(__name__)

class NLPEngine:
    """
    NLP检索引擎

    功能：
    - 停用词过滤
    - 短语权重提升
    - 同义词扩展
    - 混合相似度计算
    """

    def __init__(self):
        # 中文停用词
        self.stop_words = set([
            "请问", "您", "你", "我", "他", "是", "的", "就", "有", "于", "及", "即",
            "在", "为", "最", "有", "从", "以", "了", "将", "与", "吗", "吧", "中",
            "什么", "怎么", "哪个", "哪些", "啥", "相关"
        ])

        # 字段权重配置（用于Elasticsearch）
        self.query_fields = {
            "title_tks": 10,         # 标题权重
            "important_kwd": 30,     # 关键词权重
            "question_tks": 20,      # 问题权重
            "content_ltks": 2,       # 内容权重
        }

    def remove_stop_words(self, text: str) -> str:
        """
        移除停用词

        Args:
            text: 输入文本

        Returns:
            去除停用词后的文本
        """
        # 去除无意义词
        patterns = [
            r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
            r"(^| )(what|who|how|which|where|why)('re|'s)? ",
            r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def extract_keywords(self, text: str, top_k: int = 30) -> List[str]:
        """
        提取关键词

        Args:
            text: 输入文本
            top_k: 返回前K个关键词

        Returns:
            关键词列表
        """
        # 移除停用词
        cleaned_text = self.remove_stop_words(text)

        # 分词（简化实现，使用jieba或其他分词器）
        words = re.findall(r'[\w]+', cleaned_text)

        # 过滤停用词
        keywords = [w for w in words if w.lower() not in self.stop_words and len(w) >= 2]

        # 统计词频
        word_freq = Counter(keywords)

        # 返回前K个高频词
        return [word for word, _ in word_freq.most_common(top_k)]

    def expand_synonyms(self, term: str, synonym_dict: Dict[str, List[str]] = None) -> List[str]:
        """
        同义词扩展

        Args:
            term: 输入词
            synonym_dict: 同义词字典

        Returns:
            同义词列表
        """
        if synonym_dict is None:
            synonym_dict = {}

        # 查找同义词
        synonyms = synonym_dict.get(term, [])

        # 如果是英文，使用wordnet（需要安装）
        if re.match(r'^[a-z]+$', term):
            try:
                from nltk.corpus import wordnet
                synsets = wordnet.synsets(term)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != term.lower():
                            synonyms.append(synonym)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"WordNet查询失败: {e}")

        # 去重
        return list(set(synonyms))

    def calculate_term_weights(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """
        计算术语权重

        Args:
            tokens: 词列表

        Returns:
            [(词, 权重), ...] 列表
        """
        weights = []

        for token in tokens:
            weight = 1.0

            # 长度惩罚
            if len(token) == 1:
                weight *= 0.01
            elif len(token) == 2 and re.match(r'^[a-z0-9]+$', token):
                weight *= 0.5

            # 数字提升
            if re.match(r'^[0-9,.]+$', token):
                weight *= 2.0

            weights.append((token, weight))

        return weights

    def hybrid_similarity(
        self,
        query_vector: np.ndarray,
        doc_vectors: List[np.ndarray],
        query_tokens: List[str],
        doc_tokens: List[List[str]],
        tk_weight: float = 0.3,
        vt_weight: float = 0.7
    ) -> np.ndarray:
        """
        混合相似度计算

        Args:
            query_vector: 查询向量
            doc_vectors: 文档向量列表
            query_tokens: 查询词
            doc_tokens: 文档词列表
            tk_weight: 词相似度权重
            vt_weight: 向量相似度权重

        Returns:
            相似度数组
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # 向量相似度
        vec_sim = cosine_similarity([query_vector], doc_vectors)[0]

        # 词相似度
        token_sim = np.array([
            self._token_similarity(query_tokens, dt)
            for dt in doc_tokens
        ])

        # 混合
        hybrid_sim = token_sim * tk_weight + vec_sim * vt_weight

        return hybrid_sim

    def _token_similarity(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        计算词相似度

        Args:
            query_tokens: 查询词
            doc_tokens: 文档词

        Returns:
            相似度分数
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        # 转为集合
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)

        # Jaccard相似度
        intersection = len(query_set & doc_set)
        union = len(query_set | doc_set)

        if union == 0:
            return 0.0

        return intersection / union

    def build_elasticsearch_query(
        self,
        query: str,
        min_match: float = 0.3,
        use_synonyms: bool = True
    ) -> Dict[str, Any]:
        """
        构建Elasticsearch查询

        Args:
            query: 查询文本
            min_match: 最小匹配度
            use_synonyms: 是否使用同义词

        Returns:
            ES查询字典
        """
        # 提取关键词
        keywords = self.extract_keywords(query, top_k=30)

        # 构建should子句
        should_clauses = []

        for keyword in keywords:
            # 精确匹配
            should_clauses.append({
                "match": {
                    "content_ltks": {
                        "query": keyword,
                        "boost": 2.0
                    }
                }
            })

            # 短语匹配（相邻词）
            if len(keyword) >= 2:
                should_clauses.append({
                    "match_phrase": {
                        "content_ltks": {
                            "query": keyword,
                            "boost": 3.0
                        }
                    }
                })

            # 同义词扩展
            if use_synonyms:
                synonyms = self.expand_synonyms(keyword)
                for synonym in synonyms:
                    should_clauses.append({
                        "match": {
                            "content_ltks": {
                                "query": synonym,
                                "boost": 1.0
                            }
                        }
                    })

        # 构建查询
        es_query = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": int(len(should_clauses) * min_match)
            }
        }

        return es_query

# 创建全局服务实例
nlp_engine = NLPEngine()

def get_nlp_engine() -> NLPEngine:
    """获取NLP引擎实例"""
    return nlp_engine
