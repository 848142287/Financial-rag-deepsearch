"""
金融文档OCR增强器
专为金融文档优化的OCR处理，提供高精度的文本提取和表格识别
"""

import os
import re
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """文档类型"""
    FINANCIAL_REPORT = "financial_report"    # 财务报告
    INVOICE = "invoice"                     # 发票
    CONTRACT = "contract"                   # 合同
    BANK_STATEMENT = "bank_statement"       # 银行对账单
    TAX_DOCUMENT = "tax_document"           # 税务文档
    INSURANCE = "insurance"                 # 保险文档
    INVESTMENT = "investment"               # 投资文档
    UNKNOWN = "unknown"                     # 未知类型


class TextRegion(str, Enum):
    """文本区域类型"""
    HEADER = "header"           # 页眉
    TITLE = "title"             # 标题
    PARAGRAPH = "paragraph"     # 段落
    TABLE = "table"             # 表格
    LIST = "list"               # 列表
    FOOTER = "footer"           # 页脚
    SIDEBAR = "sidebar"         # 侧边栏
    WATERMARK = "watermark"     # 水印


@dataclass
class OCRResult:
    """OCR结果"""
    text: str
    confidence: float
    bbox: List[float]  # [x0, y0, x1, y1]
    region_type: TextRegion
    font_size: Optional[int] = None
    font_style: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableResult:
    """表格识别结果"""
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    bbox: List[float]
    table_type: str
    structured_data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FinancialOCREnhancer:
    """金融文档OCR增强器"""

    def __init__(self):
        self.reader = None
        self.financial_patterns = self._load_financial_patterns()
        self.currency_patterns = self._load_currency_patterns()
        self.number_patterns = self._load_number_patterns()

    def _load_financial_patterns(self) -> Dict[str, List[str]]:
        """加载金融文档模式"""
        return {
            # 财务指标
            'financial_metrics': [
                r'营业收入|主营业务收入|营收',
                r'净利润|净收益|纯利润',
                r'毛利润|毛利|毛利率',
                r'营业收入成本|营业成本',
                r'营业收入利润率|营业利润率',
                r'净资产收益率|ROE',
                r'总资产收益率|ROA',
                r'投资回报率|ROI',
                r'市盈率|PE比率|P/E',
                r'每股收益|EPS',
                r'经营活动现金流',
                r'资产负债率',
                r'流动比率|速动比率'
            ],

            # 货币单位
            'currency_units': [
                r'人民币|RMB|￥',
                r'美元|USD|\$',
                r'欧元|EUR|€',
                r'日元|JPY|¥',
                r'英镑|GBP|£',
                r'港元|HKD|HK$',
                r'新台币|NTD|NT$',
                r'元|万元|亿元|百万|十亿|千亿',
                r'千|万|亿|兆'
            ],

            # 百分比
            'percentages': [
                r'%|百分比|年化收益率|回报率|增长率|同比|环比'
            ],

            # 日期格式
            'dates': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{1,2}/\d{1,2}/\d{4}',
                r'Q[1-4]\s*\d{4}',
                r'\d{4}年[上下]半年'
            ],

            # 公司信息
            'company_info': [
                r'有限公司|股份有限公司|集团|公司',
                r'Co\.|Ltd\.|Inc\.|Corp\.|LLC',
                r'董事长|总经理|CEO|CFO',
                r'注册地址|办公地址|联系电话'
            ]
        }

    def _load_currency_patterns(self) -> List[str]:
        """加载货币模式"""
        return [
            r'[￥$€£¥]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # 带符号的货币
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:元|美元|欧元|英镑|日元)',
            r'\d+(?:\.\d+)?\s*(?:万|亿|千万|百)\s*(?:元|美元)',
            r'人民币\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'USD\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ]

    def _load_number_patterns(self) -> List[str]:
        """加载数字模式"""
        return [
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # 千分位数字
            r'\d+\.\d+%?',                  # 小数和百分比
            r'\(\d+\)',                      # 括号数字（常用于负数）
            r'-?\d+,?\d*\.?\d*',              # 可能的负数
        ]

    async def enhance_document_ocr(
        self,
        image_path: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
        config: Dict[str, Any] = None
    ) -> List[OCRResult]:
        """
        增强文档OCR处理

        Args:
            image_path: 图像文件路径
            document_type: 文档类型
            config: 处理配置

        Returns:
            List[OCRResult]: OCR识别结果列表
        """
        try:
            config = config or {}

            # 1. 图像预处理
            processed_image = await self._preprocess_image(image_path, document_type)

            # 2. 检测文本区域
            text_regions = await self._detect_text_regions(processed_image, document_type)

            # 3. 区域级别的OCR处理
            ocr_results = []
            for region in text_regions:
                region_result = await self._process_region_ocr(
                    processed_image, region, document_type
                )
                ocr_results.extend(region_result)

            # 4. 后处理和优化
            enhanced_results = await self._post_process_ocr(ocr_results, document_type)

            # 5. 财务特定处理
            if document_type != DocumentType.UNKNOWN:
                enhanced_results = await self._apply_financial_enhancements(
                    enhanced_results, document_type
                )

            return enhanced_results

        except Exception as e:
            logger.error(f"Error in enhanced OCR processing: {e}")
            return []

    async def _preprocess_image(
        self, image_path: str, document_type: DocumentType
    ) -> np.ndarray:
        """图像预处理"""
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")

            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 文档类型特定的预处理
            if document_type == DocumentType.FINANCIAL_REPORT:
                gray = await self._enhance_financial_report(gray)
            elif document_type == DocumentType.INVOICE:
                gray = await self._enhance_invoice(gray)
            elif document_type == DocumentType.BANK_STATEMENT:
                gray = await self._enhance_bank_statement(gray)

            return gray

        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise

    async def _enhance_financial_report(self, gray: np.ndarray) -> np.ndarray:
        """增强财务报告图像"""
        try:
            # 1. 对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 2. 降噪
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # 3. 锐化
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # 4. 二值化处理
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 5. 形态学操作去除噪点
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.error(f"Error enhancing financial report: {e}")
            return gray

    async def _enhance_invoice(self, gray: np.ndarray) -> np.ndarray:
        """增强发票图像"""
        try:
            # 发票通常有表格和结构化信息
            # 1. 自适应阈值
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 2. 去除水平线（表格线）
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

            # 3. 去除垂直线
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            # 4. 合并处理
            lines = cv2.add(horizontal_lines, vertical_lines)
            cleaned = cv2.subtract(binary, lines)

            return cleaned

        except Exception as e:
            logger.error(f"Error enhancing invoice: {e}")
            return gray

    async def _enhance_bank_statement(self, gray: np.ndarray) -> np.ndarray:
        """增强银行对账单图像"""
        try:
            # 银行对账单通常包含大量的数字和表格
            # 1. 增强对比度
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

            # 2. 高斯模糊降噪
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # 3. 自适应阈值
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
            )

            return binary

        except Exception as e:
            logger.error(f"Error enhancing bank statement: {e}")
            return gray

    async def _detect_text_regions(
        self, image: np.ndarray, document_type: DocumentType
    ) -> List[Dict[str, Any]]:
        """检测文本区域"""
        try:
            regions = []

            # 1. 使用轮廓检测找到文本块
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            height, width = image.shape
            min_area = (height * width) * 0.001  # 最小区域面积
            max_area = (height * width) * 0.8    # 最大区域面积

            for contour in contours:
                area = cv2.contourArea(contour)

                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # 过滤掉不合理的长宽比
                    aspect_ratio = w / h
                    if 0.1 < aspect_ratio < 10:
                        region = {
                            'bbox': [x, y, x + w, y + h],
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour
                        }

                        # 判断区域类型
                        region['type'] = await self._classify_region_type(region, image)
                        regions.append(region)

            # 2. 按位置排序（从上到下，从左到右）
            regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

            return regions

        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []

    async def _classify_region_type(
        self, region: Dict[str, Any], image: np.ndarray
    ) -> TextRegion:
        """分类区域类型"""
        try:
            x, y, x2, y2 = region['bbox']
            w, h = x2 - x, y2 - y
            aspect_ratio = w / h
            area = region['area']

            height, width = image.shape

            # 基于位置和大小判断
            if y < height * 0.1:
                return TextRegion.HEADER
            elif y > height * 0.9:
                return TextRegion.FOOTER
            elif x < width * 0.1:
                return TextRegion.SIDEBAR
            elif aspect_ratio > 5:
                return TextRegion.TITLE
            elif area > (height * width) * 0.1:
                return TextRegion.TABLE
            else:
                return TextRegion.PARAGRAPH

        except Exception as e:
            logger.error(f"Error classifying region type: {e}")
            return TextRegion.PARAGRAPH

    async def _process_region_ocr(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        document_type: DocumentType
    ) -> List[OCRResult]:
        """处理单个区域的OCR"""
        try:
            x, y, x2, y2 = region['bbox']
            region_image = image[y:y2, x:x2]

            # 初始化OCR读取器（如果还没有）
            if self.reader is None:
                self.reader = easyocr.Reader(['ch_sim', 'en'])

            # 使用EasyOCR进行识别
            results = self.reader.readtext(region_image)

            ocr_results = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5 and text.strip():  # 过滤低置信度和空文本
                    # 调整坐标到全图坐标系
                    adjusted_bbox = [
                        x + bbox[0][0], y + bbox[0][1],
                        x + bbox[2][0], y + bbox[2][1]
                    ]

                    ocr_result = OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=adjusted_bbox,
                        region_type=region.get('type', TextRegion.PARAGRAPH),
                        metadata={
                            'region_area': region['area'],
                            'aspect_ratio': region.get('aspect_ratio', 0)
                        }
                    )
                    ocr_results.append(ocr_result)

            # 如果EasyOCR没有结果，尝试Tesseract
            if not ocr_results:
                tesseract_text = pytesseract.image_to_string(
                    region_image, lang='chi_sim+eng'
                ).strip()

                if tesseract_text:
                    ocr_result = OCRResult(
                        text=tesseract_text,
                        confidence=0.6,  # Tesseract默认置信度
                        bbox=[x, y, x2, y2],
                        region_type=region.get('type', TextRegion.PARAGRAPH),
                        metadata={'engine': 'tesseract'}
                    )
                    ocr_results.append(ocr_result)

            return ocr_results

        except Exception as e:
            logger.error(f"Error processing region OCR: {e}")
            return []

    async def _post_process_ocr(
        self, ocr_results: List[OCRResult], document_type: DocumentType
    ) -> List[OCRResult]:
        """OCR结果后处理"""
        try:
            processed_results = []

            for result in ocr_results:
                # 1. 文本清理
                cleaned_text = await self._clean_text(result.text)

                # 2. 数字格式化
                cleaned_text = await self._format_numbers(cleaned_text)

                # 3. 货币符号标准化
                cleaned_text = await self._standardize_currency(cleaned_text)

                # 4. 日期格式标准化
                cleaned_text = await self._standardize_dates(cleaned_text)

                # 5. 金融术语纠正
                cleaned_text = await self._correct_financial_terms(cleaned_text)

                # 更新结果
                result.text = cleaned_text
                processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Error in OCR post-processing: {e}")
            return ocr_results

    async def _clean_text(self, text: str) -> str:
        """清理文本"""
        try:
            # 移除多余的空格
            text = re.sub(r'\s+', ' ', text)

            # 移除特殊字符（保留必要的标点）
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?%$€£¥()-]', '', text)

            # 修正常见OCR错误
            corrections = {
                'O': '0', 'l': '1', 'I': '1', 'S': '5', 'Z': '2',
                '（': '(', '）': ')', '【': '[', '】': ']',
                '，': ',', '。': '.', '；': ';', '：': ':'
            }

            for wrong, correct in corrections.items():
                text = text.replace(wrong, correct)

            return text.strip()

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    async def _format_numbers(self, text: str) -> str:
        """格式化数字"""
        try:
            # 修复千分位分隔符
            text = re.sub(r'(\d+)[，,](\d{3})', r'\1,\2', text)

            # 修复小数点
            text = re.sub(r'(\d+)[。.](\d+)', r'\1.\2', text)

            # 识别并标记财务数字
            for pattern in self.number_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    number = match.group()
                    # 验证数字格式
                    if self._is_valid_financial_number(number):
                        text = text.replace(number, f'[{number}]', 1)

            return text

        except Exception as e:
            logger.error(f"Error formatting numbers: {e}")
            return text

    async def _standardize_currency(self, text: str) -> str:
        """标准化货币符号"""
        try:
            # 统一货币符号
            currency_map = {
                '￥': '¥', '元': '¥', '人民币': '¥',
                '$': 'USD', '美元': 'USD',
                '€': 'EUR', '欧元': 'EUR',
                '£': 'GBP', '英镑': 'GBP',
                '¥': 'JPY', '日元': 'JPY'
            }

            for symbol, standard in currency_map.items():
                text = text.replace(symbol, standard)

            return text

        except Exception as e:
            logger.error(f"Error standardizing currency: {e}")
            return text

    async def _standardize_dates(self, text: str) -> str:
        """标准化日期格式"""
        try:
            # 统一日期格式为 YYYY-MM-DD
            date_patterns = [
                (r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3'),
                (r'(\d{4})/(\d{1,2})/(\d{1,2})', r'\1-\2-\3'),
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),
            ]

            for pattern, replacement in date_patterns:
                text = re.sub(pattern, replacement, text)

            return text

        except Exception as e:
            logger.error(f"Error standardizing dates: {e}")
            return text

    async def _correct_financial_terms(self, text: str) -> str:
        """纠正金融术语"""
        try:
            # 常见OCR错误纠正
            corrections = {
                '营业收入': '营业收入',
                '净利闰': '净利润',
                '资產': '资产',
                '負債': '负债',
                '投資': '投资',
                '収益': '收益',
                '費用': '费用',
                '成本': '成本'
            }

            for wrong, correct in corrections.items():
                text = text.replace(wrong, correct)

            return text

        except Exception as e:
            logger.error(f"Error correcting financial terms: {e}")
            return text

    def _is_valid_financial_number(self, number: str) -> bool:
        """验证是否是有效的财务数字"""
        try:
            # 移除千分位分隔符和括号
            clean_number = re.sub(r'[,\(\)]', '', number)

            # 尝试转换为浮点数
            float(clean_number)

            return True

        except ValueError:
            return False

    async def _apply_financial_enhancements(
        self, ocr_results: List[OCRResult], document_type: DocumentType
    ) -> List[OCRResult]:
        """应用金融特定增强"""
        try:
            enhanced_results = []

            for result in ocr_results:
                # 1. 识别财务指标
                financial_metrics = await self._extract_financial_metrics(result.text)
                if financial_metrics:
                    result.metadata['financial_metrics'] = financial_metrics

                # 2. 识别货币信息
                currency_info = await self._extract_currency_info(result.text)
                if currency_info:
                    result.metadata['currency_info'] = currency_info

                # 3. 识别关键数字
                key_numbers = await self._extract_key_numbers(result.text)
                if key_numbers:
                    result.metadata['key_numbers'] = key_numbers

                enhanced_results.append(result)

            return enhanced_results

        except Exception as e:
            logger.error(f"Error applying financial enhancements: {e}")
            return ocr_results

    async def _extract_financial_metrics(self, text: str) -> List[str]:
        """提取财务指标"""
        try:
            metrics = []

            for category, patterns in self.financial_patterns.items():
                if category == 'financial_metrics':
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        metrics.extend(matches)

            return list(set(metrics))  # 去重

        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return []

    async def _extract_currency_info(self, text: str) -> Dict[str, Any]:
        """提取货币信息"""
        try:
            currency_info = {
                'has_currency': False,
                'currency_symbols': [],
                'amounts': []
            }

            for pattern in self.currency_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    currency_info['has_currency'] = True
                    currency_info['amounts'].extend(matches)

            return currency_info

        except Exception as e:
            logger.error(f"Error extracting currency info: {e}")
            return {}

    async def _extract_key_numbers(self, text: str) -> List[str]:
        """提取关键数字"""
        try:
            key_numbers = []

            # 提取百分比
            percentages = re.findall(r'\d+\.?\d*%?', text)
            key_numbers.extend(percentages)

            # 提取大额数字
            large_numbers = re.findall(r'\d{3,}', text)
            key_numbers.extend(large_numbers)

            return key_numbers

        except Exception as e:
            logger.error(f"Error extracting key numbers: {e}")
            return []


# 全局实例
financial_ocr_enhancer = FinancialOCREnhancer()


# 便捷函数
async def enhance_financial_document_ocr(
    image_path: str,
    document_type: DocumentType = DocumentType.UNKNOWN,
    config: Dict[str, Any] = None
) -> List[OCRResult]:
    """便捷的金融文档OCR增强函数"""
    return await financial_ocr_enhancer.enhance_document_ocr(
        image_path, document_type, config
    )