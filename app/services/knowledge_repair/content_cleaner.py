"""
内容清洗器
清洗和修复文本、表格、图片等内容
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """清洗结果"""
    original_content: str
    cleaned_content: str
    issues_found: List[Dict[str, Any]]
    fixes_applied: List[str]
    quality_score: float
    metadata: Dict[str, Any]


class ContentCleaner:
    """内容清洗器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cleaning_rules = {
            'text': self._get_text_cleaning_rules(),
            'table': self._get_table_cleaning_rules(),
            'image': self._get_image_cleaning_rules()
        }

    def _get_text_cleaning_rules(self) -> List[Dict[str, Any]]:
        """获取文本清洗规则"""
        return [
            {
                'name': 'remove_extra_spaces',
                'pattern': r'\s+',
                'replacement': ' ',
                'description': '移除多余空格'
            },
            {
                'name': 'fix_line_breaks',
                'pattern': r'(\n\s*){3,}',
                'replacement': '\n\n',
                'description': '修复多余的换行'
            },
            {
                'name': 'fix_punctuation',
                'pattern': r'([，。！？])([a-zA-Z])',
                'replacement': r'\1 \2',
                'description': '修复中英文标点间空格'
            },
            {
                'name': 'remove_control_chars',
                'pattern': r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',
                'replacement': '',
                'description': '移除控制字符'
            },
            {
                'name': 'fix_brackets',
                'pattern': r'([\(\[\{])\s*([^\)\]\}]*?)\s*([\)\]\}])',
                'replacement': r'\1\2\3',
                'description': '修复括号内空格'
            },
            {
                'name': 'normalize_quotes',
                'pattern': r'[""'''']',
                'replacement': '"',
                'description': '统一引号格式'
            }
        ]

    def _get_table_cleaning_rules(self) -> List[Dict[str, Any]]:
        """获取表格清洗规则"""
        return [
            {
                'name': 'align_columns',
                'pattern': r'\|(\s*)(.*?)(\s*)\|',
                'replacement': r'| \2 |',
                'description': '对齐表格列'
            },
            {
                'name': 'remove_empty_rows',
                'pattern': r'\n\|(\s*\|)+\s*\n',
                'replacement': '\n',
                'description': '移除空行'
            },
            {
                'name': 'fix_table_headers',
                'pattern': r'^\|(.+?)\|\s*\n\|[-\s\|]+\|',
                'replacement': r'|\1|\n|' + '-' * 20 + '|\n',
                'description': '修复表格分隔线'
            }
        ]

    def _get_image_cleaning_rules(self) -> List[Dict[str, Any]]:
        """获取图片清洗规则"""
        return [
            {
                'name': 'fix_image_paths',
                'pattern': r'!\[([^\]]*)\]\(([^)]+)\)',
                'replacement': lambda m: f'![{m.group(1).strip()}]({m.group(2).strip()})',
                'description': '修复图片路径'
            },
            {
                'name': 'standardize_alt_text',
                'pattern': r'!\[\s*\]\(',
                'replacement': '![图片](',
                'description': '标准化替代文本'
            }
        ]

    async def clean_content(self, content: str, content_type: str = 'text') -> CleaningResult:
        """清洗内容"""
        try:
            original_content = content
            issues_found = []
            fixes_applied = []

            # 选择清洗规则
            rules = self.cleaning_rules.get(content_type, self.cleaning_rules['text'])

            # 应用清洗规则
            cleaned_content = content
            for rule in rules:
                if 'pattern' in rule:
                    # 检查是否存在需要修复的问题
                    if re.search(rule['pattern'], cleaned_content):
                        issues_found.append({
                            'type': rule['name'],
                            'description': rule['description'],
                            'severity': 'medium'
                        })

                        # 应用修复
                        if callable(rule['replacement']):
                            cleaned_content = re.sub(rule['pattern'], rule['replacement'], cleaned_content)
                        else:
                            cleaned_content = re.sub(rule['pattern'], rule['replacement'], cleaned_content, flags=re.MULTILINE)

                        fixes_applied.append(rule['description'])

            # 特殊处理
            cleaned_content = self._apply_special_cleaning(cleaned_content, content_type, fixes_applied)

            # 计算质量分数
            quality_score = self._calculate_quality_score(original_content, cleaned_content, issues_found)

            # 生成元数据
            metadata = self._generate_cleaning_metadata(original_content, cleaned_content, issues_found)

            return CleaningResult(
                original_content=original_content,
                cleaned_content=cleaned_content,
                issues_found=issues_found,
                fixes_applied=fixes_applied,
                quality_score=quality_score,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"内容清洗失败: {str(e)}")
            return CleaningResult(
                original_content=content,
                cleaned_content=content,
                issues_found=[{'type': 'error', 'description': str(e), 'severity': 'high'}],
                fixes_applied=[],
                quality_score=0.0,
                metadata={}
            )

    def _apply_special_cleaning(self, content: str, content_type: str, fixes_applied: List[str]) -> str:
        """应用特殊清洗逻辑"""
        # 移除重复段落
        paragraphs = content.split('\n\n')
        unique_paragraphs = []
        seen_hashes = set()

        for para in paragraphs:
            para_hash = hashlib.md5(para.strip().encode()).hexdigest()
            if para_hash not in seen_hashes and para.strip():
                unique_paragraphs.append(para)
                seen_hashes.add(para_hash)
            elif para_hash in seen_hashes and len(unique_paragraphs) > 0:
                fixes_applied.append('移除重复段落')

        content = '\n\n'.join(unique_paragraphs)

        # 修复常见的OCR错误
        ocr_fixes = {
            r'0': 'O',  # 数字0替换为字母O（在适当上下文中）
            r'1': 'l',  # 数字1替换为字母l（在适当上下文中）
            r'rn': 'm',  # rn替换为m
        }

        # 仅在文本环境中应用OCR修复
        if content_type == 'text':
            for wrong, right in ocr_fixes.items():
                # 简单的上下文检查，避免错误替换
                if self._should_apply_ocr_fix(content, wrong, right):
                    content = content.replace(wrong, right)
                    fixes_applied.append(f'修复OCR错误: {wrong} -> {right}')

        return content

    def _should_apply_ocr_fix(self, content: str, wrong: str, right: str) -> bool:
        """判断是否应该应用OCR修复"""
        # 简单的启发式规则
        if wrong == '0' and right == 'O':
            # 检查是否在单词中
            return re.search(r'\b0[A-Za-z]', content) is not None
        elif wrong == '1' and right == 'l':
            return re.search(r'[a-z]1[a-z]', content) is not None
        return False

    def _calculate_quality_score(self, original: str, cleaned: str, issues: List[Dict]) -> float:
        """计算质量分数"""
        base_score = 0.8

        # 根据问题数量调整
        issue_penalty = len(issues) * 0.05
        base_score -= issue_penalty

        # 根据内容长度合理性调整
        if 50 <= len(cleaned) <= 10000:  # 合理的文本长度
            base_score += 0.1

        # 根据字符多样性调整
        unique_chars = len(set(cleaned))
        if unique_chars / max(len(cleaned), 1) > 0.3:  # 足够的字符多样性
            base_score += 0.1

        return max(min(base_score, 1.0), 0.0)

    def _generate_cleaning_metadata(self, original: str, cleaned: str, issues: List[Dict]) -> Dict[str, Any]:
        """生成清洗元数据"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'compression_ratio': len(cleaned) / max(len(original), 1),
            'issue_count': len(issues),
            'issues_by_severity': self._group_issues_by_severity(issues),
            'checksum': hashlib.md5(cleaned.encode()).hexdigest()[:8]
        }

    def _group_issues_by_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """按严重程度分组问题"""
        severity_count = {'low': 0, 'medium': 0, 'high': 0}
        for issue in issues:
            severity = issue.get('severity', 'medium')
            severity_count[severity] += 1
        return severity_count

    async def clean_multimodal_content(self, multimodal_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗多模态内容"""
        cleaned_data = []

        for item in multimodal_data:
            content_type = item.get('type', 'text')
            text_content = item.get('text', '')

            # 清洗文本内容
            cleaning_result = await self.clean_content(text_content, content_type)

            # 更新项目
            cleaned_item = item.copy()
            cleaned_item['text'] = cleaning_result.cleaned_content
            cleaned_item['cleaning_metadata'] = {
                'quality_score': cleaning_result.quality_score,
                'issues_found': cleaning_result.issues_found,
                'fixes_applied': cleaning_result.fixes_applied
            }

            cleaned_data.append(cleaned_item)

        return cleaned_data

    async def standardize_financial_data(self, content: str) -> str:
        """标准化财务数据格式"""
        # 标准化数字格式
        content = re.sub(r'(\d{1,3})(,\d{3})+', lambda m: m.group(0).replace(',', ''), content)

        # 标准化货币单位
        unit_patterns = {
            r'(\d+\.?\d*)\s*万': r'\1W',
            r'(\d+\.?\d*)\s*千': r'\1K',
            r'(\d+\.?\d*)\s*百万': r'\1M',
            r'(\d+\.?\d*)\s*十亿': r'\1B',
            r'(\d+\.?\d*)\s*亿元': r'\1HB'  # Hundred Billion (Chinese)
        }

        for pattern, replacement in unit_patterns.items():
            content = re.sub(pattern, replacement, content)

        # 标准化百分比
        content = re.sub(r'(\d+\.?\d*)\s*%?', r'\1%', content)

        # 标准化日期格式
        date_patterns = [
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3'),
            (r'(\d{4})/(\d{1,2})/(\d{1,2})', r'\1-\2-\3'),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3')
        ]

        for pattern, replacement in date_patterns:
            content = re.sub(pattern, replacement, content)

        return content