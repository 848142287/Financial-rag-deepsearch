"""
跨页表格合并算法
实现智能的跨页表格检测、合并和修复功能
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableFeature:
    """表格特征"""
    table_id: str
    page_num: int
    bbox: List[float]  # [x1, y1, x2, y2]
    row_count: int
    col_count: int
    headers: List[str]
    first_row: List[str]
    last_row: List[str]
    html_content: str
    text_content: str
    position: str  # 'top', 'middle', 'bottom'


@dataclass
class MergeResult:
    """合并结果"""
    merged_table_id: str
    original_tables: List[str]
    merged_html: str
    merged_text: str
    confidence_score: float
    merge_sources: List[Dict[str, Any]]


class CrossPageTableMerger:
    """跨页表格合并器"""

    def __init__(self):
        # 合并配置
        self.header_similarity_threshold = 0.9
        self.structure_consistency_threshold = 0.85
        self.data_continuity_threshold = 0.8
        self.position_weight = 0.2
        self.header_weight = 0.4
        self.structure_weight = 0.3
        self.data_weight = 0.1

        # 表格特征提取规则
        self.table_keywords = ['表', 'Table', '图表', 'Chart']
        self.header_indicators = ['名称', '项目', 'Item', 'Name', '年度', 'Year', '季度', 'Quarter']

    async def merge_cross_page_tables(
        self,
        tables: List[Dict[str, Any]]
    ) -> List[MergeResult]:
        """
        合并跨页表格

        Args:
            tables: 检测到的表格列表

        Returns:
            合并结果列表
        """
        logger.info(f"开始跨页表格合并，检测到 {len(tables)} 个表格")

        # 第一步：提取表格特征
        table_features = []
        for i, table in enumerate(tables):
            feature = self._extract_table_features(table, i)
            table_features.append(feature)

        # 第二步：检测疑似跨页表格
        candidate_groups = self._detect_cross_page_candidates(table_features)
        logger.info(f"检测到 {len(candidate_groups)} 组候选跨页表格")

        # 第三步：计算合并置信度
        merge_groups = []
        for group in candidate_groups:
            confidence = self._calculate_merge_confidence(group)
            if confidence >= 0.85:
                merge_groups.append((group, confidence))
                logger.info(f"找到可合并表格组，置信度: {confidence:.3f}")

        # 第四步：执行表格合并
        merge_results = []
        for group, confidence in merge_groups:
            try:
                merge_result = await self._merge_table_group(group, confidence)
                merge_results.append(merge_result)
            except Exception as e:
                logger.error(f"表格合并失败: {e}")

        logger.info(f"跨页表格合并完成，合并了 {len(merge_results)} 组表格")
        return merge_results

    def _extract_table_features(
        self,
        table: Dict[str, Any],
        index: int
    ) -> TableFeature:
        """提取表格特征"""
        # 获取基本信息
        page_num = table.get('page', 1)
        bbox = table.get('bbox', [0, 0, 0, 0])

        # 解析表格内容
        html_content = table.get('html', '')
        text_content = table.get('text', '')

        # 解析HTML获取表格结构
        headers, first_row, last_row = self._parse_table_html(html_content)

        # 计算行数和列数
        row_count, col_count = self._count_table_rows_cols(html_content)

        # 判断表格在页面中的位置
        position = self._determine_table_position(bbox, page_num)

        return TableFeature(
            table_id=f"table_{index}_{page_num}",
            page_num=page_num,
            bbox=bbox,
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            first_row=first_row,
            last_row=last_row,
            html_content=html_content,
            text_content=text_content,
            position=position
        )

    def _parse_table_html(
        self,
        html_content: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """解析HTML表格"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')

            if not table:
                return [], [], []

            # 提取表头
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))

            # 提取第一行数据
            first_row = []
            rows = table.find_all('tr')
            if len(rows) > 1:
                for td in rows[1].find_all('td'):
                    first_row.append(td.get_text(strip=True))

            # 提取最后一行数据
            last_row = []
            if len(rows) > 1:
                for td in rows[-1].find_all('td'):
                    last_row.append(td.get_text(strip=True))

            return headers, first_row, last_row

        except Exception as e:
            logger.error(f"解析表格HTML失败: {e}")
            return [], [], []

    def _count_table_rows_cols(self, html_content: str) -> Tuple[int, int]:
        """计算表格行列数"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')

            if not table:
                return 0, 0

            rows = table.find_all('tr')
            row_count = len(rows)

            # 计算最大列数
            col_count = 0
            for row in rows:
                cells = row.find_all(['td', 'th'])
                col_count = max(col_count, len(cells))

            return row_count, col_count

        except Exception as e:
            logger.error(f"计算表格行列数失败: {e}")
            return 0, 0

    def _determine_table_position(
        self,
        bbox: List[float],
        page_num: int
    ) -> str:
        """判断表格在页面中的位置"""
        # 简单实现：基于y坐标判断
        _, y1, _, y2 = bbox

        # 假设页面高度为800（标准化）
        page_height = 800
        relative_y = y1 / page_height

        if relative_y < 0.3:
            return 'top'
        elif relative_y > 0.7:
            return 'bottom'
        else:
            return 'middle'

    def _detect_cross_page_candidates(
        self,
        table_features: List[TableFeature]
    ) -> List[List[TableFeature]]:
        """检测疑似跨页表格"""
        candidates = []

        # 按页面排序
        sorted_tables = sorted(table_features, key=lambda x: (x.page_num, x.bbox[1]))

        # 寻找可能的跨页表格序列
        for i, current_table in enumerate(sorted_tables):
            group = [current_table]

            # 检查后续表格
            for next_table in sorted_tables[i+1:]:
                # 检查是否为连续页面
                if next_table.page_num != current_table.page_num + len(group):
                    break

                # 检查位置关系（当前表格在页面底部，下一表格在页面顶部）
                if (current_table.position == 'bottom' and
                    next_table.position == 'top'):
                    group.append(next_table)
                else:
                    break

            # 如果找到多页表格，添加到候选列表
            if len(group) > 1:
                candidates.append(group)

        return candidates

    def _calculate_merge_confidence(self, table_group: List[TableFeature]) -> float:
        """计算合并置信度"""
        if len(table_group) < 2:
            return 0.0

        confidence_scores = []

        # 1. 表头相似度
        header_similarity = self._calculate_header_similarity(table_group)
        confidence_scores.append(('header', header_similarity))

        # 2. 结构一致性
        structure_consistency = self._calculate_structure_consistency(table_group)
        confidence_scores.append(('structure', structure_consistency))

        # 3. 数据连续性
        data_continuity = self._calculate_data_continuity(table_group)
        confidence_scores.append(('data', data_continuity))

        # 4. 位置逻辑性
        position_logic = self._calculate_position_logic(table_group)
        confidence_scores.append(('position', position_logic))

        # 加权平均
        weights = {
            'header': self.header_weight,
            'structure': self.structure_weight,
            'data': self.data_weight,
            'position': self.position_weight
        }

        weighted_score = sum(
            score * weights.get(name, 0.1)
            for name, score in confidence_scores
        )

        logger.info(f"合并置信度计算: {confidence_scores}, 加权分数: {weighted_score:.3f}")

        return weighted_score

    def _calculate_header_similarity(self, table_group: List[TableFeature]) -> float:
        """计算表头相似度"""
        if len(table_group) < 2:
            return 1.0

        # 只比较前两个表格的表头（后续表格通常没有表头）
        headers1 = table_group[0].headers
        headers2 = table_group[1].headers

        if not headers1 or not headers2:
            return 0.5  # 如果没有表头，给予中等分数

        # 计算相似度
        similarity = SequenceMatcher(None, headers1, headers2).ratio()

        # 考虑列数匹配
        col_similarity = 1.0 - abs(len(headers1) - len(headers2)) / max(len(headers1), len(headers2))

        return (similarity + col_similarity) / 2

    def _calculate_structure_consistency(self, table_group: List[TableFeature]) -> float:
        """计算结构一致性"""
        if len(table_group) < 2:
            return 1.0

        # 检查列数一致性
        col_counts = [table.col_count for table in table_group]
        col_consistency = 1.0 - np.std(col_counts) / max(col_counts)

        # 检查表格样式一致性（简化实现）
        style_consistency = 0.8  # 假设样式较为一致

        return (col_consistency + style_consistency) / 2

    def _calculate_data_continuity(self, table_group: List[TableFeature]) -> float:
        """计算数据连续性"""
        if len(table_group) < 2:
            return 1.0

        continuity_scores = []

        for i in range(len(table_group) - 1):
            current = table_group[i]
            next_table = table_group[i + 1]

            # 检查最后一行和第一行的数据类型一致性
            if current.last_row and next_table.first_row:
                # 简单检查：数值类型是否连续
                score = self._check_row_data_type_consistency(
                    current.last_row,
                    next_table.first_row
                )
                continuity_scores.append(score)

        return np.mean(continuity_scores) if continuity_scores else 0.5

    def _check_row_data_type_consistency(
        self,
        row1: List[str],
        row2: List[str]
    ) -> float:
        """检查两行数据类型一致性"""
        if len(row1) != len(row2):
            return 0.5

        consistency_scores = []

        for val1, val2 in zip(row1, row2):
            # 检查是否都是数字
            if self._is_number(val1) and self._is_number(val2):
                consistency_scores.append(1.0)
            # 检查是否都是文本
            elif not self._is_number(val1) and not self._is_number(val2):
                consistency_scores.append(0.8)
            # 混合类型
            else:
                consistency_scores.append(0.3)

        return np.mean(consistency_scores)

    def _is_number(self, value: str) -> bool:
        """检查是否为数字"""
        try:
            float(value.replace(',', '').replace('%', ''))
            return True
        except:
            return False

    def _calculate_position_logic(self, table_group: List[TableFeature]) -> float:
        """计算位置逻辑性"""
        # 检查页面是否连续
        pages = [table.page_num for table in table_group]
        is_consecutive = all(pages[i] + 1 == pages[i + 1] for i in range(len(pages) - 1))

        # 检查位置关系
        position_score = 1.0
        for i in range(len(table_group) - 1):
            current = table_group[i]
            next_table = table_group[i + 1]

            # 第一个表格应该在底部，后续表格应该在顶部
            if i == 0 and current.position != 'bottom':
                position_score *= 0.8
            if next_table.position != 'top':
                position_score *= 0.8

        return position_score if is_consecutive else 0.0

    async def _merge_table_group(
        self,
        table_group: List[TableFeature],
        confidence: float
    ) -> MergeResult:
        """合并表格组"""
        logger.info(f"开始合并表格组: {[t.table_id for t in table_group]}")

        # 获取第一个表格的表头
        headers = table_group[0].headers

        # 收集所有数据行
        all_rows = []

        for table in table_group:
            # 解析HTML表格
            rows_data = self._extract_table_data(table.html_content)

            # 跳过表头行（除了第一个表格）
            if table != table_group[0]:
                rows_data = rows_data[1:] if len(rows_data) > 0 else rows_data

            all_rows.extend(rows_data)

        # 生成合并后的HTML
        merged_html = self._generate_merged_html(headers, all_rows)

        # 生成合并后的文本
        merged_text = self._generate_merged_text(headers, all_rows)

        # 创建合并结果
        merge_result = MergeResult(
            merged_table_id=f"merged_{table_group[0].table_id}_to_{table_group[-1].table_id}",
            original_tables=[t.table_id for t in table_group],
            merged_html=merged_html,
            merged_text=merged_text,
            confidence_score=confidence,
            merge_sources=[
                {
                    'table_id': t.table_id,
                    'page_num': t.page_num,
                    'row_count': t.row_count
                }
                for t in table_group
            ]
        )

        logger.info(f"表格合并完成: {len(all_rows)} 行数据")
        return merge_result

    def _extract_table_data(self, html_content: str) -> List[List[str]]:
        """从HTML提取表格数据"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')

            if not table:
                return []

            rows_data = []
            rows = table.find_all('tr')

            for row in rows:
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    text = cell.get_text(strip=True)
                    row_data.append(text)

                if row_data:  # 忽略空行
                    rows_data.append(row_data)

            return rows_data

        except Exception as e:
            logger.error(f"提取表格数据失败: {e}")
            return []

    def _generate_merged_html(
        self,
        headers: List[str],
        rows: List[List[str]]
    ) -> str:
        """生成合并后的HTML"""
        html_parts = ['<table border="1">', '<thead>', '<tr>']

        # 添加表头
        for header in headers:
            html_parts.append(f'<th>{header}</th>')

        html_parts.extend(['</tr>', '</thead>', '<tbody>'])

        # 添加数据行
        for row in rows:
            html_parts.append('<tr>')
            for cell in row:
                html_parts.append(f'<td>{cell}</td>')
            html_parts.append('</tr>')

        html_parts.extend(['</tbody>', '</table>'])

        return '\n'.join(html_parts)

    def _generate_merged_text(
        self,
        headers: List[str],
        rows: List[List[str]]
    ) -> str:
        """生成合并后的文本"""
        text_parts = []

        # 添加表头
        text_parts.append(' | '.join(headers))
        text_parts.append('-' * len(' | '.join(headers)))

        # 添加数据行
        for row in rows:
            text_parts.append(' | '.join(row))

        return '\n'.join(text_parts)

    def validate_merged_table(self, merge_result: MergeResult) -> Dict[str, Any]:
        """验证合并结果"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            # 解析合并后的表格
            rows_data = self._extract_table_data(merge_result.merged_html)

            # 统计信息
            validation_result['statistics'] = {
                'total_rows': len(rows_data),
                'total_cols': len(rows_data[0]) if rows_data else 0,
                'source_tables': len(merge_result.original_tables)
            }

            # 检查列数一致性
            if rows_data:
                col_counts = [len(row) for row in rows_data]
                if len(set(col_counts)) > 1:
                    validation_result['errors'].append('表格列数不一致')
                    validation_result['is_valid'] = False

            # 检查数据完整性
            empty_rows = sum(1 for row in rows_data if not any(cell.strip() for cell in row))
            if empty_rows > len(rows_data) * 0.1:  # 超过10%的空行
                validation_result['warnings'].append(f'检测到 {empty_rows} 个空行')

            # 检查置信度
            if merge_result.confidence_score < 0.9:
                validation_result['warnings'].append(
                    f'合并置信度较低: {merge_result.confidence_score:.3f}'
                )

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'验证过程出错: {e}')

        return validation_result


# 全局跨页表格合并器实例
cross_page_table_merger = CrossPageTableMerger()