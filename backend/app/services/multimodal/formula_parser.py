"""
公式解析器
解析数学公式和财务公式
"""

from app.core.structured_logging import get_structured_logger
import re
from typing import Dict, Any, List

logger = get_structured_logger(__name__)


class FormulaParser:
    """公式解析器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.formula_categories = {
            'valuation': ['估值', 'valuation', 'DCF', 'NPV', 'IRR'],
            'profitability': ['盈利能力', 'profitability', 'ROE', 'ROA', '毛利率', '净利率'],
            'liquidity': ['流动性', 'liquidity', '流动比率', '速动比率', 'current ratio'],
            'leverage': ['杠杆', 'leverage', '负债率', 'debt ratio'],
            'growth': ['增长', 'growth', '增长率', 'growth rate'],
            'efficiency': ['效率', 'efficiency', '周转率', 'turnover']
        }

        # 财务指标模式
        self.financial_patterns = {
            'ROE': r'ROE\s*=?\s*([%/d.-]+)',
            'ROA': r'ROA\s*=?\s*([%/d.-]+)',
            'PE': r'PE\s*=?\s*([d.-]+)',
            'PB': r'PB\s*=?\s*([d.-]+)',
            'EPS': r'EPS\s*=?\s*([d.-]+)',
            'DPS': r'DPS\s*=?\s*([d.-]+)'
        }

    async def parse(self, formula_text: str) -> Dict[str, Any]:
        """解析公式"""
        try:
            # 提取公式表达式
            expression = self._extract_formula_expression(formula_text)

            # 识别变量
            variables = self._extract_variables(expression)

            # 分类公式类型
            category = self._classify_formula_category(formula_text, variables)

            # 解析公式结构
            structure = self._parse_formula_structure(expression)

            # 验证公式
            validation = self._validate_formula(expression, structure)

            # 计算公式复杂度
            complexity = self._calculate_complexity(structure)

            return {
                'expression': expression,
                'category': category,
                'type': structure.get('type', 'unknown'),
                'variables': variables,
                'structure': structure,
                'complexity': complexity,
                'validation': validation,
                'metadata': {
                    'has_special_chars': self._has_special_characters(expression),
                    'is_latex': self._is_latex_formula(expression),
                    'is_financial': self._is_financial_formula(formula_text, variables)
                },
                'confidence': self._calculate_confidence(expression, validation)
            }

        except Exception as e:
            logger.error(f"公式解析失败: {str(e)}")
            return {
                'expression': '',
                'category': 'unknown',
                'type': 'unknown',
                'variables': [],
                'structure': {},
                'complexity': 0,
                'validation': {'valid': False, 'errors': [str(e)]},
                'metadata': {},
                'confidence': 0.0
            }

    def _extract_formula_expression(self, text: str) -> str:
        """提取公式表达式"""
        # LaTeX公式
        latex_patterns = [
            r'\$\$(.*?)\$\$',  # $$...$$
            r'\$(.*?)\$',      # $...$
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}'
        ]

        for pattern in latex_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 标准公式格式
        formula_patterns = [
            r'([A-Za-z]+\s*[+\-*/]\s*[A-Za-z0-9\s\(\)]+\s*=)',
            r'([A-Za-z]+\s*=\s*[A-Za-z0-9\s\(\)\+\-\*/\^]+)',
            r'(\([^)]+\)\s*[+\-*/]\s*[^+\-*/=]+\s*[=])'
        ]

        for pattern in formula_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # 如果没有找到明确的公式，返回整个文本
        return text.strip()

    def _extract_variables(self, expression: str) -> List[Dict[str, Any]]:
        """提取公式变量"""
        variables = []

        # 提取单个字母变量
        single_vars = re.findall(r'\b([A-Za-z])\b(?![A-Za-z])', expression)
        for var in set(single_vars):
            variables.append({
                'name': var,
                'type': 'single_letter',
                'count': expression.count(var)
            })

        # 提取多字母变量
        multi_vars = re.findall(r'\b([A-Za-z]{2,})\b', expression)
        for var in set(multi_vars):
            if var not in ['ROE', 'ROA', 'PE', 'PB', 'EPS', 'DPS']:  # 排除已识别的财务指标
                variables.append({
                    'name': var,
                    'type': 'multi_letter',
                    'count': expression.count(var)
                })

        # 提取希腊字母
        greek_vars = re.findall(r'\\([a-zA-Z]+)', expression)
        for var in set(greek_vars):
            variables.append({
                'name': f'\\{var}',
                'type': 'greek',
                'count': expression.count(f'\\{var}')
            })

        # 提取下标变量
        subscript_vars = re.findall(r'([A-Za-z])_([0-9a-zA-Z]+)', expression)
        for var, sub in subscript_vars:
            variables.append({
                'name': f'{var}_{sub}',
                'type': 'subscript',
                'count': expression.count(f'{var}_{sub}')
            })

        return variables

    def _classify_formula_category(self, text: str, variables: List[Dict]) -> str:
        """分类公式类别"""
        text_lower = text.lower()

        # 检查财务指标
        for metric, pattern in self.financial_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if metric in ['ROE', 'ROA']:
                    return 'profitability'
                elif metric in ['PE', 'PB']:
                    return 'valuation'

        # 检查类别关键词
        for category, keywords in self.formula_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        # 根据变量特征判断
        var_names = [v['name'] for v in variables]

        if any(v in var_names for v in ['π', 'pi', '∫', 'integral']):
            return 'mathematical'

        if any(v in var_names for v in ['∑', 'sum', '∏', 'product']):
            return 'statistical'

        return 'general'

    def _parse_formula_structure(self, expression: str) -> Dict[str, Any]:
        """解析公式结构"""
        structure = {
            'type': 'unknown',
            'operations': [],
            'functions': [],
            'constants': []
        }

        # 识别运算符
        operations = re.findall(r'([+\-*/\^=])', expression)
        structure['operations'] = list(set(operations))

        # 识别函数
        functions = re.findall(r'\\([a-zA-Z]+)|(\w+)\s*\(', expression)
        for func in functions:
            if func[0]:  # LaTeX函数
                structure['functions'].append(f'\\{func[0]}')
            elif func[1]:  # 普通函数
                structure['functions'].append(func[1])

        # 识别常数
        constants = re.findall(r'\b(\d+\.?\d*|π|e|pi)\b', expression)
        structure['constants'] = list(set(constants))

        # 判断公式类型
        if '=' in expression:
            structure['type'] = 'equation'
        elif '∑' in expression or '∫' in expression:
            structure['type'] = 'summation_integral'
        elif any(op in ['+', '-'] for op in operations) and len(operations) > 2:
            structure['type'] = 'polynomial'
        elif '√' in expression or '^' in expression:
            structure['type'] = 'power_root'
        elif '/' in expression:
            structure['type'] = 'fraction'
        else:
            structure['type'] = 'expression'

        return structure

    def _validate_formula(self, expression: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """验证公式"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # 检查括号匹配
        open_count = expression.count('(') + expression.count('{')
        close_count = expression.count(')') + expression.count('}')
        if open_count != close_count:
            validation['valid'] = False
            validation['errors'].append('括号不匹配')

        # 检查运算符
        if '--' in expression or '++' in expression:
            validation['warnings'].append('存在连续运算符')

        # 检查变量定义
        if structure.get('type') == 'equation' and len(expression.split('=')) != 2:
            validation['warnings'].append('等式格式不规范')

        # 检查LaTeX语法
        if self._is_latex_formula(expression):
            if not expression.startswith('\\') and not expression.endswith('\\'):
                validation['warnings'].append('LaTeX公式可能不完整')

        return validation

    def _calculate_complexity(self, structure: Dict[str, Any]) -> int:
        """计算公式复杂度"""
        complexity = 0

        # 基础复杂度
        complexity += len(structure.get('operations', []))
        complexity += len(structure.get('functions', [])) * 2
        complexity += len(structure.get('constants', []))

        # 根据类型调整
        if structure.get('type') == 'polynomial':
            complexity += 3
        elif structure.get('type') == 'summation_integral':
            complexity += 5
        elif structure.get('type') == 'fraction':
            complexity += 2

        return min(complexity, 10)  # 限制最大复杂度为10

    def _has_special_characters(self, expression: str) -> bool:
        """检查是否包含特殊字符"""
        special_chars = ['∑', '∫', '∏', '√', '∞', '±', '≠', '≈', '≤', '≥', 'π', 'α', 'β', 'γ', 'δ']
        return any(char in expression for char in special_chars)

    def _is_latex_formula(self, expression: str) -> bool:
        """检查是否为LaTeX公式"""
        latex_indicators = ['\\', '{', '}', '^', '_']
        return any(indicator in expression for indicator in latex_indicators)

    def _is_financial_formula(self, text: str, variables: List[Dict]) -> bool:
        """检查是否为财务公式"""
        financial_indicators = [
            'ROE', 'ROA', 'PE', 'PB', 'EPS', 'DPS',
            '毛利率', '净利率', '负债率', '周转率'
        ]

        text_lower = text.lower()
        var_names = [v['name'].lower() for v in variables]

        return any(
            indicator.lower() in text_lower or
            any(indicator.lower() in var_name for var_name in var_names)
            for indicator in financial_indicators
        )

    def _calculate_confidence(self, expression: str, validation: Dict[str, Any]) -> float:
        """计算置信度"""
        confidence = 0.7

        # 根据验证结果调整
        if not validation.get('valid', True):
            confidence -= 0.3

        # 根据警告数量调整
        warning_count = len(validation.get('warnings', []))
        confidence -= warning_count * 0.05

        # 根据表达式长度调整
        if len(expression) > 5:
            confidence += 0.1

        # 根据特殊字符调整
        if self._has_special_characters(expression):
            confidence += 0.1

        # 根据LaTeX格式调整
        if self._is_latex_formula(expression):
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)