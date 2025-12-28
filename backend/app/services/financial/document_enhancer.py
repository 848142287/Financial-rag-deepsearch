"""
金融文档增强处理器
基于Qwen2.5-VL-Embedding方案的金融研报专用增强功能
"""

import re
import json
import hashlib
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialEntity:
    """金融实体"""
    text: str
    type: str  # COMPANY, STOCK_CODE, FINANCIAL_TERM, METRIC, INDUSTRY
    start_pos: int
    end_pos: int
    normalized_form: str = ""
    confidence: float = 1.0


class FinancialDocumentEnhancer:
    """金融文档增强处理器"""

    def __init__(self):
        # 加载金融领域词典
        self._load_financial_dictionaries()

        # 初始化jieba分词，添加金融专业词汇
        self._enhance_jieba_dict()

        # 公司名称到股票代码的映射
        self.company_stock_map = self._load_company_stock_mapping()

        # 金融术语标准化映射
        self.term_standardization_map = {
            "EPS": "每股收益",
            "ROE": "净资产收益率",
            "PE": "市盈率",
            "PB": "市净率",
            "ROA": "总资产收益率",
            "毛利率": "毛利率",
            "净利率": "净利率",
            "营收": "营业收入",
            "归母净利润": "归属于母公司所有者的净利润",
        }

        # 财务指标模式
        self.metric_patterns = {
            "营业收入": r"营业收入[：:]\s*([\d.,]+)\s*亿元?",
            "净利润": r"(?:净利润|归母净利润)[：:]\s*([\d.,]+)\s*亿元?",
            "毛利率": r"毛利率[：:]\s*([\d.,]+)%",
            "净利率": r"净利率[：:]\s*([\d.,]+)%",
            "ROE": r"(?:净资产收益率|ROE)[：:]\s*([\d.,]+)%",
            "PE": r"(?:市盈率|PE)[：:]\s*([\d.,]+)倍?",
            "PB": r"(?:市净率|PB)[：:]\s*([\d.,]+)倍?",
            "每股收益": r"(?:每股收益|EPS)[：:]\s*([\d.,]+)元?",
        }

        # 标准章节
        self.standard_sections = [
            ("投资要点", "investment_highlights", 0.9),
            ("核心观点", "core_viewpoint", 0.85),
            ("公司概况", "company_overview", 0.7),
            ("行业分析", "industry_analysis", 0.8),
            ("财务分析", "financial_analysis", 0.95),
            ("盈利预测", "earnings_forecast", 0.9),
            ("估值分析", "valuation_analysis", 0.85),
            ("风险提示", "risk_warning", 0.75),
            ("投资建议", "investment_recommendation", 0.9),
        ]

    def _load_financial_dictionaries(self):
        """加载金融词典"""
        # 公司名称词典（示例，实际可从数据库加载）
        self.company_names = set([
            "贵州茅台", "腾讯控股", "阿里巴巴", "美团", "京东",
            "中国平安", "招商银行", "宁德时代", "比亚迪", "隆基绿能",
            "中国石油", "工商银行", "建设银行", "农业银行", "中国银行",
        ])

        # 股票代码模式
        self.stock_code_patterns = [
            r'\b[0-9]{6}\.[SHZ]\b',  # A股：000001.SZ
            r'\b[0-9]{5}\.[HK]\b',   # 港股：00700.HK
            r'\b[A-Z]{1,5}\b',       # 美股：AAPL
        ]

        # 金融术语词典
        self.financial_terms = set([
            "市盈率", "市净率", "市销率", "股息率",
            "净资产收益率", "总资产收益率", "投入资本回报率",
            "毛利率", "净利率", "营业利润率",
            "资产负债率", "流动比率", "速动比率",
            "营收增长率", "净利润增长率", "每股收益增长率",
            "经营活动现金流", "投资活动现金流", "筹资活动现金流",
            "自由现金流", "企业价值", "EV/EBITDA",
        ])

        # 财务指标词典
        self.financial_metrics = set([
            "营业收入", "营业成本", "毛利润", "净利润",
            "总资产", "总负债", "所有者权益", "净资产",
            "经营活动现金流净额", "投资活动现金流净额",
            "筹资活动现金流净额", "期末现金余额",
            "基本每股收益", "稀释每股收益", "每股净资产",
        ])

    def _enhance_jieba_dict(self):
        """增强jieba词典"""
        # 添加公司名称
        for company in self.company_names:
            jieba.add_word(company, freq=1000, tag='nr')

        # 添加金融术语
        for term in self.financial_terms:
            jieba.add_word(term, freq=1000, tag='nz')

        # 添加财务指标
        for metric in self.financial_metrics:
            jieba.add_word(metric, freq=1000, tag='nz')

    def _load_company_stock_mapping(self):
        """加载公司股票代码映射"""
        # 示例数据，实际应从数据库加载
        return {
            "贵州茅台": "600519.SH",
            "腾讯控股": "00700.HK",
            "阿里巴巴": "BABA",
            "中国平安": "601318.SH",
            "招商银行": "600036.SH",
            "宁德时代": "300750.SZ",
            "比亚迪": "002594.SZ",
        }

    def enhance_document(self, document: str, metadata: Dict = None) -> Dict[str, Any]:
        """增强金融文档"""
        try:
            # 1. 基础文本清理
            cleaned_text = self._clean_financial_text(document)

            # 2. 金融实体识别
            entities = self._extract_financial_entities(cleaned_text)

            # 3. 公司名称标准化
            normalized_text = self._normalize_company_names(cleaned_text, entities)

            # 4. 金融术语标准化
            normalized_text = self._standardize_financial_terms(normalized_text)

            # 5. 财务指标提取和标注
            metrics = self._extract_financial_metrics(normalized_text)

            # 6. 时间信息提取
            time_info = self._extract_time_information(normalized_text)

            # 7. 构建增强文档
            enhanced_doc = {
                "original_text": document,
                "cleaned_text": cleaned_text,
                "normalized_text": normalized_text,
                "entities": entities,
                "financial_metrics": metrics,
                "time_info": time_info,
                "metadata": metadata or {},
                "document_hash": hashlib.md5(document.encode()).hexdigest(),
                "enhancement_timestamp": datetime.now().isoformat()
            }

            # 8. 计算文档特征
            enhanced_doc["document_features"] = self._calculate_document_features(enhanced_doc)

            logger.debug(f"文档增强完成，识别到 {len(entities)} 个实体，{len(metrics)} 个财务指标")
            return enhanced_doc

        except Exception as e:
            logger.error(f"文档增强失败: {str(e)}")
            # 返回基础增强结果
            return {
                "original_text": document,
                "normalized_text": document,
                "entities": [],
                "financial_metrics": [],
                "time_info": {},
                "metadata": metadata or {},
                "document_hash": hashlib.md5(document.encode()).hexdigest(),
                "enhancement_timestamp": datetime.now().isoformat(),
                "document_features": {}
            }

    def _clean_financial_text(self, text: str) -> str:
        """清理金融文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 标准化数字格式
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)

        # 标准化百分比
        text = re.sub(r'(\d+)%', r'\1%', text)

        return text.strip()

    def _extract_financial_entities(self, text: str) -> List[FinancialEntity]:
        """提取金融实体"""
        entities = []

        # 使用jieba进行分词和词性标注
        words = pseg.cut(text)

        for word, flag in words:
            # 识别公司名称
            if flag == 'nr' and word in self.company_names:
                entities.append(FinancialEntity(
                    text=word,
                    type="COMPANY",
                    start_pos=text.find(word),
                    end_pos=text.find(word) + len(word),
                    normalized_form=self._get_company_normalized_form(word)
                ))

            # 识别金融术语
            elif word in self.financial_terms:
                entities.append(FinancialEntity(
                    text=word,
                    type="FINANCIAL_TERM",
                    start_pos=text.find(word),
                    end_pos=text.find(word) + len(word),
                    normalized_form=self.term_standardization_map.get(word, word)
                ))

            # 识别财务指标
            elif word in self.financial_metrics:
                entities.append(FinancialEntity(
                    text=word,
                    type="METRIC",
                    start_pos=text.find(word),
                    end_pos=text.find(word) + len(word)
                ))

        # 识别股票代码
        for pattern in self.stock_code_patterns:
            for match in re.finditer(pattern, text):
                stock_code = match.group()
                entities.append(FinancialEntity(
                    text=stock_code,
                    type="STOCK_CODE",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=stock_code
                ))

        # 去重
        unique_entities = []
        seen_positions = set()
        for entity in entities:
            pos_key = (entity.start_pos, entity.end_pos)
            if pos_key not in seen_positions:
                unique_entities.append(entity)
                seen_positions.add(pos_key)

        return unique_entities

    def _get_company_normalized_form(self, company_name: str) -> str:
        """获取公司标准化名称"""
        # 简单的标准化逻辑
        if "集团" in company_name or "控股" in company_name:
            return company_name.replace("集团", "").replace("控股", "")
        return company_name

    def _normalize_company_names(self, text: str, entities: List[FinancialEntity]) -> str:
        """标准化公司名称"""
        normalized_text = text

        for entity in entities:
            if entity.type == "COMPANY" and entity.normalized_form:
                # 添加股票代码
                stock_code = self.company_stock_map.get(entity.text)
                if stock_code and stock_code not in normalized_text:
                    normalized_text = normalized_text.replace(
                        entity.text,
                        f"{entity.text}({stock_code})"
                    )

        return normalized_text

    def _standardize_financial_terms(self, text: str) -> str:
        """标准化金融术语"""
        standardized_text = text

        # 英文术语标准化
        for eng_term, chn_term in self.term_standardization_map.items():
            if eng_term in text and chn_term not in text:
                # 替换英文术语为中文
                standardized_text = standardized_text.replace(
                    eng_term, f"{chn_term}({eng_term})"
                )

        return standardized_text

    def _extract_financial_metrics(self, text: str) -> List[Dict]:
        """提取财务指标及其数值"""
        metrics = []

        for metric_name, pattern in self.metric_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                try:
                    metrics.append({
                        "metric": metric_name,
                        "value": float(value.replace(',', '')),
                        "unit": self._get_metric_unit(metric_name),
                        "context": text[max(0, match.start()-50):match.end()+50],
                        "position": (match.start(), match.end())
                    })
                except ValueError:
                    continue

        return metrics

    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位"""
        unit_mapping = {
            "营业收入": "亿元",
            "净利润": "亿元",
            "毛利率": "%",
            "净利率": "%",
            "ROE": "%",
            "PE": "倍",
            "PB": "倍",
            "每股收益": "元",
        }
        return unit_mapping.get(metric_name, "")

    def _extract_time_information(self, text: str) -> Dict:
        """提取时间信息"""
        time_info = {
            "has_explicit_time": False,
            "specific_dates": [],
            "time_keywords": [],
            "time_range": None,
        }

        # 时间关键词
        time_keywords = ["年", "季度", "月", "同比", "环比", "预测", "预计"]

        for keyword in time_keywords:
            if keyword in text:
                time_info["time_keywords"].append(keyword)

        # 提取具体年份
        year_matches = re.findall(r'(20\d{2})年', text)
        if year_matches:
            time_info["specific_dates"].extend(year_matches)
            time_info["has_explicit_time"] = True

        # 提取季度
        quarter_matches = re.findall(r'(Q[1-4]|第[一二三四]季度)', text)
        if quarter_matches:
            time_info["specific_dates"].extend(quarter_matches)

        return time_info

    def _calculate_document_features(self, enhanced_doc: Dict) -> Dict:
        """计算文档特征"""
        text = enhanced_doc["normalized_text"]

        features = {
            # 文本特征
            "length": len(text),
            "word_count": len(jieba.lcut(text)),
            "sentence_count": len(re.split(r'[。！？；\n]+', text)),

            # 金融特征
            "company_count": len([e for e in enhanced_doc["entities"]
                                if e.type == "COMPANY"]),
            "financial_term_density": len([e for e in enhanced_doc["entities"]
                                         if e.type == "FINANCIAL_TERM"]) / max(len(text), 1),
            "metric_count": len(enhanced_doc["financial_metrics"]),

            # 质量特征
            "numeric_density": len(re.findall(r'\d+', text)) / max(len(text), 1),

            # 时间特征
            "has_forecast": "预测" in text or "预计" in text,
            "has_historical": "同比" in text or "环比" in text,
        }

        return features


# 全局实例
financial_document_enhancer = FinancialDocumentEnhancer()