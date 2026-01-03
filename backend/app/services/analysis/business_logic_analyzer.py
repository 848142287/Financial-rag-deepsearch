"""
业务逻辑分析增强器
增强表格和图表的业务逻辑分析
"""
import re
from dataclasses import dataclass
from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class BusinessInsight:
    """业务洞察"""
    category: str  # advantage, opportunity, risk, trend, pattern
    title: str
    description: str
    evidence: str = ""
    confidence: float = 0.8
    action_item: str = ""

@dataclass
class ComparisonResult:
    """对比结果"""
    entity_a: str
    entity_b: str
    metric: str
    difference: float
    percentage_diff: float = None
    winner: str = ""  # "A" or "B" or "tie"
    insight: str = ""

class BusinessLogicAnalyzer:
    """业务逻辑分析器"""

    def __init__(self):
        # 洞察类别模式
        self.insight_patterns = {
            'advantage': [
                r'(优势|领先|优势领域|核心竞争力)',
                r'(强于|优于|胜过|超越)',
                r'(最高|最大|最强|最好)',
            ],
            'opportunity': [
                r'(机会|机遇|行业机会|市场机会)',
                r'(增长空间|发展空间)',
                r'(有望|预期|将)',
            ],
            'risk': [
                r'(风险|挑战|威胁|风险提示)',
                r'(压力|下滑|下降|回落)',
                r'(警惕|注意|需要关注)',
            ],
            'trend': [
                r'(趋势|态势|变化|演变)',
                r'(持续|保持|稳定)',
                r'(提升|改善|向好)',
            ],
            'pattern': [
                r'(模式|规律|特征|特点)',
                r'(结构|分布|占比)',
                r'(关系|相关|关联)',
            ]
        }

        # 对比模式
        self.comparison_patterns = [
            r'(.+?)(?:是|为|达到)(.+?)(?:的)?(\d+\.?\d*)倍',
            r'(.+?)(?:超|超|超过|胜过)(.+?)([\d.]+)%',
            r'(.+?)与(.+?)相比(?:增长|下降|多|少)([\d.]+)(?:%|倍)',
            r'(.+?)(?:强于|优于)(.+?)',
        ]

        # 行动建议模式
        self.action_patterns = [
            r'(建议|推荐|建议配置|建议关注)',
            r'(重点关注|优先布局|配置)',
            r'(改进方向|优化方向|提升路径)',
            r'(投资建议|配置策略|行动建议)',
        ]

    def analyze_table_business_logic(
        self,
        table_data: Dict[str, Any],
        table_type: str
    ) -> List[BusinessInsight]:
        """
        分析表格的业务逻辑

        Args:
            table_data: 表格数据
            table_type: 表格类型

        Returns:
            洞察列表
        """
        insights = []

        title = table_data.get('title', '')
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])

        # 转换为文本进行分析
        table_text = self._table_to_text(title, headers, rows)

        # 1. 提取优势洞察
        advantage_insights = self._extract_insights_by_category(
            table_text, 'advantage', title
        )
        insights.extend(advantage_insights)

        # 2. 提取机会洞察
        opportunity_insights = self._extract_insights_by_category(
            table_text, 'opportunity', title
        )
        insights.extend(opportunity_insights)

        # 3. 提取风险洞察
        risk_insights = self._extract_insights_by_category(
            table_text, 'risk', title
        )
        insights.extend(risk_insights)

        # 4. 提取趋势洞察
        trend_insights = self._extract_insights_by_category(
            table_text, 'trend', title
        )
        insights.extend(trend_insights)

        # 5. 提取模式洞察
        pattern_insights = self._extract_insights_by_category(
            table_text, 'pattern', title
        )
        insights.extend(pattern_insights)

        # 6. 提取对比结果
        comparisons = self._extract_comparisons(table_text, title)
        for comp in comparisons:
            insight = BusinessInsight(
                category='comparison',
                title=f"{comp.entity_a} vs {comp.entity_b}",
                description=comp.insight,
                evidence=f"{comp.entity_a}与{comp.entity_b}在{comp.metric}上的对比",
                confidence=0.85
            )
            insights.append(insight)

        logger.info(f"分析表格业务逻辑: {title}, 发现{len(insights)}个洞察")
        return insights

    def _table_to_text(
        self,
        title: str,
        headers: List[str],
        rows: List[List]
    ) -> str:
        """将表格转换为文本"""
        parts = [title]

        # 添加表头
        if headers:
            parts.append(" | ".join(headers))

        # 添加数据行（只取前10行）
        for row in rows[:10]:
            row_text = " | ".join([str(v) if v else "" for v in row])
            parts.append(row_text)

        return "\n".join(parts)

    def _extract_insights_by_category(
        self,
        text: str,
        category: str,
        source: str
    ) -> List[BusinessInsight]:
        """按类别提取洞察"""
        insights = []
        patterns = self.insight_patterns.get(category, [])

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 提取完整的句子或段落
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()

                # 清理
                context = re.sub(r'^[^，。；]*', '', context)
                context = re.sub(r'[，。；].*$', '', context)

                if context and len(context) > 5:
                    insights.append(BusinessInsight(
                        category=category,
                        title=f"{category}_{len(insights)+1}",
                        description=context,
                        evidence=match.group(0),
                        confidence=0.85
                    ))

        return insights

    def _extract_comparisons(
        self,
        text: str,
        source: str
    ) -> List[ComparisonResult]:
        """提取对比结果"""
        comparisons = []

        for pattern in self.comparison_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        entity_a = match.group(1).strip()
                        entity_b = match.group(2).strip()

                        # 尝试提取数值差异
                        diff_value = None
                        perc_diff = None

                        if len(match.groups()) >= 3:
                            try:
                                diff_value = float(match.group(3))
                            except ValueError:
                                pass

                        # 确定胜者
                        winner = ""
                        insight = ""

                        if '强于' in match.group(0) or '优于' in match.group(0) or '超过' in match.group(0):
                            winner = "A"
                            insight = f"{entity_a}强于{entity_b}"
                        elif '弱于' in match.group(0):
                            winner = "B"
                            insight = f"{entity_a}弱于{entity_b}"
                        elif diff_value:
                            if diff_value > 1:
                                winner = "A"
                                insight = f"{entity_a}是{entity_b}的{diff_value}倍"
                            elif diff_value < 1:
                                winner = "B"
                                insight = f"{entity_a}是{entity_b}的{diff_value}倍"
                            else:
                                winner = "tie"
                                insight = f"{entity_a}与{entity_b}相当"

                        if insight:
                            comparisons.append(ComparisonResult(
                                entity_a=entity_a,
                                entity_b=entity_b,
                                metric="general",
                                difference=diff_value or 0,
                                winner=winner,
                                insight=insight
                            ))

                except Exception as e:
                    logger.debug(f"对比提取失败: {e}")
                    continue

        return comparisons

    def extract_action_items(
        self,
        text: str,
        source: str
    ) -> List[Dict[str, Any]]:
        """提取行动建议"""
        action_items = []

        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 提取完整的建议
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 150)
                context = text[start:end].strip()

                # 清理
                context = re.sub(r'^[^。；\n]*', '', context)
                context = re.split(r'[。；\n]', context)[0].strip()

                if context and len(context) > 10:
                    action_items.append({
                        "type": "action_item",
                        "content": context,
                        "source": source,
                        "trigger": match.group(0)
                    })

        return action_items

# 全局实例
business_logic_analyzer = BusinessLogicAnalyzer()
