"""
内容处理服务 - 内容去重、重要性排序、智能摘要
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class ContentPoint:
    """内容观点"""
    content: str
    importance_score: float = 0.0
    source_position: int = 0
    keywords: List[str] = field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'importance_score': self.importance_score,
            'source_position': self.source_position,
            'keywords': self.keywords,
            'category': self.category,
            'metadata': self.metadata
        }

class ContentProcessingService:
    """
    内容处理服务

    功能：
    1. 内容去重 - 去除重复的观点和段落
    2. 重要性排序 - 按重要性对内容排序
    3. 关键信息提取
    4. 内容摘要生成
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化内容处理服务

        Args:
            config: 配置参数
                - similarity_threshold: 相似度阈值（默认0.85）
                - min_content_length: 最小内容长度（默认50）
                - max_keywords: 最大关键词数量（默认5）
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.min_content_length = self.config.get('min_content_length', 50)
        self.max_keywords = self.config.get('max_keywords', 5)

        # 停用词
        self._stopwords = self._load_stopwords()

        # 重要性权重配置
        self._importance_weights = {
            'position': 0.2,  # 位置权重（前面的内容更重要）
            'length': 0.1,  # 长度权重（适度长度的内容更重要）
            'keyword_density': 0.3,  # 关键词密度
            'heading_proximity': 0.25,  # 标题接近度
            'special_markers': 0.15  # 特殊标记（加粗、列表等）
        }

        self.logger = logger

    def deduplicate_content(self, content_list: List[str]) -> List[str]:
        """
        去除重复内容

        Args:
            content_list: 内容列表

        Returns:
            List[str]: 去重后的内容列表
        """
        if not content_list:
            return []

        # 预处理：过滤短内容和标准化
        processed_contents = []
        for i, content in enumerate(content_list):
            if len(content.strip()) < self.min_content_length:
                continue
            processed_contents.append({
                'index': i,
                'original': content,
                'normalized': self._normalize_content(content)
            })

        # 计算相似度矩阵
        duplicates = set()
        for i in range(len(processed_contents)):
            if i in duplicates:
                continue

            for j in range(i + 1, len(processed_contents)):
                if j in duplicates:
                    continue

                similarity = self._calculate_similarity(
                    processed_contents[i]['normalized'],
                    processed_contents[j]['normalized']
                )

                if similarity >= self.similarity_threshold:
                    # 标记后者为重复
                    duplicates.add(j)
                    self.logger.debug(f"Duplicate found: {i} <-> {j} (similarity: {similarity:.2f})")

        # 构建结果，保留原始顺序
        result = [
            processed_contents[i]['original']
            for i in range(len(processed_contents))
            if i not in duplicates
        ]

        self.logger.info(f"Deduplication: {len(content_list)} -> {len(result)} items")
        return result

    def rank_by_importance(self, content_list: List[str],
                          context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        按重要性排序内容

        Args:
            content_list: 内容列表
            context: 上下文信息（包含标题、结构等）

        Returns:
            List[Tuple[str, float]]: (内容, 重要性分数) 列表，按重要性降序
        """
        if not content_list:
            return []

        scored_contents = []
        headings = context.get('headings', []) if context else []

        for i, content in enumerate(content_list):
            # 计算各项分数
            position_score = self._calculate_position_score(i, len(content_list))
            length_score = self._calculate_length_score(content)
            keyword_score = self._calculate_keyword_density_score(content)
            heading_score = self._calculate_heading_proximity_score(i, headings)
            marker_score = self._calculate_special_marker_score(content)

            # 加权总分
            total_score = (
                position_score * self._importance_weights['position'] +
                length_score * self._importance_weights['length'] +
                keyword_score * self._importance_weights['keyword_density'] +
                heading_score * self._importance_weights['heading_proximity'] +
                marker_score * self._importance_weights['special_markers']
            )

            scored_contents.append((content, total_score))

        # 按分数降序排序
        scored_contents.sort(key=lambda x: x[1], reverse=True)

        return scored_contents

    def extract_key_points(self, content: str, max_points: int = 10) -> List[ContentPoint]:
        """
        提取关键观点

        Args:
            content: 文档内容
            max_points: 最大观点数量

        Returns:
            List[ContentPoint]: 关键观点列表
        """
        # 分割内容为段落
        paragraphs = self._split_into_paragraphs(content)

        # 提取列表项
        list_items = self._extract_list_items(content)

        # 合并所有候选点
        candidates = []
        for i, para in enumerate(paragraphs):
            if len(para.strip()) >= self.min_content_length:
                candidates.append(para)

        candidates.extend(list_items)

        # 评分
        scored_candidates = self.rank_by_importance(candidates)

        # 提取前N个
        key_points = []
        for content, score in scored_candidates[:max_points]:
            point = ContentPoint(
                content=content.strip(),
                importance_score=score,
                keywords=self._extract_keywords(content)[:self.max_keywords]
            )
            key_points.append(point)

        return key_points

    def generate_summary(self, content: str, max_length: int = 500) -> str:
        """
        生成内容摘要

        Args:
            content: 原始内容
            max_length: 最大长度

        Returns:
            str: 摘要
        """
        # 提取关键观点
        key_points = self.extract_key_points(content, max_points=20)

        if not key_points:
            # 如果没有提取到观点，返回前N个字符
            return content[:max_length].strip() + '...' if len(content) > max_length else content

        # 构建摘要
        summary_parts = []
        current_length = 0

        for point in key_points:
            # 限制每个观点的长度
            point_content = point.content[:200]
            point_length = len(point_content)

            if current_length + point_length > max_length:
                break

            summary_parts.append(point_content)
            current_length += point_length

        summary = ' '.join(summary_parts)

        # 添加省略号
        if len(content) > current_length:
            summary += '...'

        return summary

    def _normalize_content(self, content: str) -> str:
        """标准化内容（用于去重比较）"""
        # 转小写
        content = content.lower()

        # 移除多余空格
        content = re.sub(r'\s+', ' ', content)

        # 移除标点符号（保留中文）
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', '', content)

        # 移除停用词
        words = content.split()
        words = [w for w in words if w not in self._stopwords]

        return ' '.join(words)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度（Jaccard）"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _calculate_position_score(self, index: int, total: int) -> float:
        """计算位置分数（前面的内容得分更高）"""
        if total == 0:
            return 0.0
        # 使用指数衰减
        return 1.0 - (index / total) ** 0.5

    def _calculate_length_score(self, content: str) -> float:
        """计算长度分数（适度长度得分高）"""
        length = len(content.strip())

        # 理想长度范围
        ideal_min = 100
        ideal_max = 500

        if length < ideal_min:
            # 太短，线性扣分
            return length / ideal_min
        elif length > ideal_max:
            # 太长，逐渐扣分
            return max(0.0, 1.0 - (length - ideal_max) / 1000)
        else:
            # 理想长度，满分
            return 1.0

    def _calculate_keyword_density_score(self, content: str) -> float:
        """计算关键词密度分数"""
        words = content.lower().split()
        if not words:
            return 0.0

        # 计算非停用词比例
        non_stopwords = [w for w in words if w not in self._stopwords]
        density = len(non_stopwords) / len(words) if words else 0

        return min(1.0, density * 1.5)

    def _calculate_heading_proximity_score(self, content_index: int,
                                          headings: List[Dict[str, Any]]) -> float:
        """计算标题接近度分数"""
        if not headings:
            return 0.5

        # 找到最近的前一个标题
        closest_distance = float('inf')
        for heading in headings:
            heading_pos = heading.get('position', 0)
            distance = abs(content_index - heading_pos)
            if distance < closest_distance:
                closest_distance = distance

        # 距离越近分数越高
        if closest_distance == 0:
            return 1.0
        elif closest_distance < 5:
            return 0.8
        elif closest_distance < 10:
            return 0.6
        elif closest_distance < 20:
            return 0.4
        else:
            return 0.2

    def _calculate_special_marker_score(self, content: str) -> float:
        """计算特殊标记分数"""
        score = 0.0

        # 检查加粗
        if '**' in content or '__' in content:
            score += 0.3

        # 检查列表标记
        if content.strip().startswith(('-', '*', '+', '•')):
            score += 0.2

        # 检查数字列表
        if re.match(r'^\d+\.', content.strip()):
            score += 0.2

        # 检查引用
        if content.strip().startswith('>'):
            score += 0.1

        # 检查包含关键词指示词
        keywords = ['重要', '关键', '核心', '主要', '注意', '总结', '结论']
        if any(kw in content for kw in keywords):
            score += 0.2

        return min(1.0, score)

    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单实现：基于词频
        words = content.lower().split()
        words = [w for w in words if w not in self._stopwords and len(w) > 2]

        # 统计词频
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in sorted_words[:self.max_keywords]]

    def _split_into_paragraphs(self, content: str) -> List[str]:
        """分割内容为段落"""
        # 按双换行符分割
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _extract_list_items(self, content: str) -> List[str]:
        """提取列表项"""
        list_items = []

        # 匹配无序列表
        unordered_pattern = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
        list_items.extend(unordered_pattern.findall(content))

        # 匹配有序列表
        ordered_pattern = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)
        list_items.extend(ordered_pattern.findall(content))

        return list_items

    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        # 中英文停用词
        stopwords = {
            # 英文停用词
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',

            # 中文停用词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '里', '就是', '吗', '什么', '还', '但是'
        }

        return stopwords

# 全局实例
content_processing_service = ContentProcessingService()
