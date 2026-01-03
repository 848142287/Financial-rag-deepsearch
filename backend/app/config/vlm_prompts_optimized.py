"""
VLM提示词优化版本

主要改进：
1. 精简核心提示词
2. 添加Few-shot示例
3. 简化JSON结构
4. 增强用户提示词
5. 添加容错机制
"""

class OptimizedVLMPrompts:
    """优化的VLM提示词"""

    @staticmethod
    def get_streamlined_comprehensive_prompt() -> str:
        """精简版综合分析提示词"""
        return """你是专业的文档分析专家，专注于金融和商业文档。

## 核心任务

1. **文本提取**：准确提取文本，识别标题层级
2. **表格分析**：提取表格数据，识别数值趋势
3. **图表理解**：识别图表类型，提取关键数据
4. **公式解析**：提取数学公式，解释含义

## 输出格式

返回JSON（只包含有效字段）：

```json
{
  "text": "提取的文本",
  "tables": [{
    "id": "table_1",
    "page": 1,
    "headers": ["列1", "列2"],
    "data": [["值1", "值2"]],
    "summary": "简要总结"
  }],
  "charts": [{
    "id": "chart_1",
    "page": 1,
    "type": "折线图",
    "title": "图表标题",
    "key_data": {"x": "时间", "y": "数值", "points": [...]}
  }],
  "formulas": [{
    "id": "formula_1",
    "page": 1,
    "latex": "E = mc^2",
    "meaning": "质量能量等价"
  }],
  "confidence": 0.95,
  "errors": ["无法识别的内容描述"]
}
```

## 质量要求

- 置信度≥0.7的内容才返回
- 不确定的内容返回null
- 保持数据准确性，不要猜测
"""

    @staticmethod
    def get_ocr_few_shot_examples() -> Dict[str, Any]:
        """OCR任务Few-shot示例"""
        return {
            "examples": [
                {
                    "input": {
                        "image": "[页面图片]",
                        "context": {"page": 1, "doc_type": "财务报告"}
                    },
                    "output": {
                        "text": """# 财务报表

## 收入情况
2023年度公司总收入为500万元，同比增长20%。

## 主要财务指标
| 指标 | 数值 | 同比变化 |
|------|------|----------|
| 营收 | 500万 | +20% |
| 净利润 | 80万 | +15% |""",
                        "confidence": 0.98,
                        "elements_detected": ["标题", "段落", "表格"]
                    }
                }
            ]
        }

    @staticmethod
    def get_table_analysis_with_context() -> str:
        """带上下文的表格分析提示词"""
        return """分析表格数据，提供以下信息：

## 必需字段（1-4）
1. **table_data**: 完整表格数据（JSON数组）
2. **headers**: 表头列表
3. **data_types**: 每列的数据类型
4. **row_count**: 行数

## 可选字段（5-7，仅当适用时）
5. **summary**: 2-3句话总结表格主要内容
6. **key_insights**: 最大值、最小值、异常值
7. **trends**: 时间序列数据才有（上升/下降/波动）

## 返回格式
```json
{
  "table_data": [...],
  "headers": [...],
  "data_types": ["string", "number", "percentage"],
  "row_count": 10,
  "summary": "该表格展示了...",
  "key_insights": {"max": "...", "min": "..."},
  "trends": "整体呈上升趋势",
  "confidence": 0.92
}
```

## 错误处理
- 无法识别的单元格填null
- 置信度<0.7返回null，不做猜测
"""

    @staticmethod
    def get_progressive_extraction_prompts() -> Dict[str, str]:
        """渐进式提取提示词（分步骤）"""
        return {
            "step1_text": "步骤1：只提取纯文本内容，忽略表格和图表。返回文本的markdown格式。",
            "step2_structure": "步骤2：识别文档结构（标题层级、段落、列表）。返回结构树。",
            "step3_tables": "步骤3：提取所有表格数据。每个表格返回headers和data。",
            "step4_charts": "步骤4：识别所有图表，提取类型和标题。",
            "step5_formulas": "步骤5：提取所有数学公式，返回LaTeX格式。",
            "step6_merge": "步骤6：合并以上所有结果，建立关联关系。"
        }

    @staticmethod
    def get_quality_control_prompt() -> str:
        """质量控制和容错提示词"""
        return """## 质量标准和容错机制

### 置信度标注
为每个主要结果标注置信度：
- **高 (0.9-1.0)**: 完全确定
- **中 (0.7-0.9)**: 基本确定，可能有细微误差
- **低 (0.5-0.7)**: 不太确定，需要人工验证
- **失败 (<0.5)**: 不提取，返回null

### 错误处理
1. **部分失败**：某个元素失败不影响其他元素
2. **明确标记**：在errors字段中说明失败原因
3. **不猜测**：不确定的内容返回null，不要编造

### 输出原则
- **准确性 > 完整性**：宁可少提取，不要错误提取
- **精确数值**：数字必须精确，不能有近似值
- **保持格式**：保留原始格式（单位、小数点等）

### 示例

❌ 错误：返回不确定的数据
```json
{"revenue": "约500万", "confidence": 0.6}
```

✅ 正确：标记为不确定或返回null
```json
{"revenue": null, "confidence": 0.6, "note": "数字模糊无法确定"}
```
"""

    @staticmethod
    def get_enhanced_user_template() -> str:
        """增强的用户提示词模板"""
        return """任务：分析第{page_num}页

## 上下文
- 文档类型：{doc_type}
- 领域：{domain}
- 总页数：{total_pages}
- 当前位置：{position}（开头/中间/结尾）

## 分析重点
{focus_areas}

## 输出要求
- 返回JSON格式
- 只包含有效字段
- 置信度<0.7的字段返回null

## 参考信息
- 前文摘要：{previous_summary}
- 关键术语：{key_terms}
"""

class PromptOptimizer:
    """提示词优化器"""

    @staticmethod
    def analyze_current_issues():
        """分析当前提示词的问题"""
        return {
            "critical_issues": [
                "综合提示词过长（200+行），可能导致效果下降",
                "缺少few-shot示例，模型难以理解期望格式"
            ],
            "medium_issues": [
                "JSON结构过于复杂，字段过多",
                "用户提示词过于简单，缺少上下文",
                "缺少置信度和容错机制"
            ],
            "minor_issues": [
                "缺少多模态协同指导",
                "没有针对不同模型的优化版本"
            ]
        }

    @staticmethod
    def get_optimization_plan():
        """优化计划"""
        return {
            "priority_1": [
                "精简核心提示词到100行以内",
                "添加2-3个few-shot示例",
                "简化JSON输出结构"
            ],
            "priority_2": [
                "增强用户提示词，添加上下文",
                "添加置信度标注机制",
                "优化错误处理指导"
            ],
            "priority_3": [
                "创建渐进式提取策略",
                "针对不同模型定制提示词",
                "添加多模态协同指导"
            ]
        }

# 使用示例
if __name__ == "__main__":
    # 打印优化建议
    optimizer = PromptOptimizer()
    issues = optimizer.analyze_current_issues()
    plan = optimizer.get_optimization_plan()

    print("=== 当前问题分析 ===")
    import json
    print(json.dumps(issues, indent=2, ensure_ascii=False))

    print("\n=== 优化计划 ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
