"""
智能区域检测器
检测 PDF 文档中的不同区域（标题、段落、表格等）
"""

import fitz  # PyMuPDF
from dataclasses import dataclass
import re

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class DetectedRegion:
    """检测到的区域"""
    region_type: str  # title, paragraph, table, image, list, etc.
    text: str
    bbox: tuple  # (x0, y0, x1, y1)
    page: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'type': self.region_type,
            'text': self.text,
            'bbox': self.bbox,
            'page': self.page,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

class SmartRegionDetector:
    """智能区域检测器"""

    def __init__(self):
        self.logger = logger
        # 标题模式
        self.title_patterns = [
            r'^[一二三四五六七八九十]+[\s\.\、]',

            r'^[0-9]+[\s\.\、][\u4e00-\u9fa5]+',
            r'^[A-Z][a-z]+[\s\.].+',
            r'^第[一二三四五六七八九十]+[章节条款]',
        ]

    async def detect_regions(
        self,
        file_path: str,
        detect_tables: bool = True,
        detect_images: bool = True
    ) -> List[DetectedRegion]:
        """
        检测 PDF 文档中的所有区域

        Args:
            file_path: PDF 文件路径
            detect_tables: 是否检测表格
            detect_images: 是否检测图片

        Returns:
            List[DetectedRegion]: 检测到的区域列表
        """
        try:
            self.logger.info(f"开始检测区域: {file_path}")

            doc = fitz.open(file_path)
            all_regions = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_regions = await self._detect_regions_in_page(
                    page,
                    page_num + 1,
                    detect_tables,
                    detect_images
                )
                all_regions.extend(page_regions)

            doc.close()

            self.logger.info(f"检测完成，共发现 {len(all_regions)} 个区域")
            return all_regions

        except Exception as e:
            self.logger.error(f"区域检测失败: {e}")
            return []

    async def _detect_regions_in_page(
        self,
        page,
        page_num: int,
        detect_tables: bool,
        detect_images: bool
    ) -> List[DetectedRegion]:
        """检测单页中的区域"""
        regions = []

        try:
            # 获取文本块
            blocks = page.get_text("blocks")

            for block in blocks:
                if block[6] == 0:  # 文本块
                    region_type, confidence = self._classify_text_block(block, page)

                    if region_type:
                        region = DetectedRegion(
                            region_type=region_type,
                            text=block[4],
                            bbox=block[:4],
                            page=page_num,
                            confidence=confidence
                        )
                        regions.append(region)

            # 检测图片
            if detect_images:
                image_regions = self._detect_image_regions(page, page_num)
                regions.extend(image_regions)

            # 检测表格（简单实现）
            if detect_tables:
                table_regions = self._detect_table_regions(page, page_num)
                regions.extend(table_regions)

        except Exception as e:
            self.logger.warning(f"检测第 {page_num} 页区域失败: {e}")

        return regions

    def _classify_text_block(self, block, page) -> tuple[str, float]:
        """分类文本块"""
        text = block[4].strip()
        bbox = block[:4]

        if not text:
            return "unknown", 0.0

        # 检查是否为标题
        for pattern in self.title_patterns:
            if re.match(pattern, text):
                return "title", 0.9

        # 根据字体大小判断
        try:
            # 获取块的字体信息
            spans = page.get_text("dict")
            for span in spans.get("blocks", []):
                if "lines" in span:
                    for line in span["lines"]:
                        for s in line["spans"]:
                            if text in s.get("text", ""):
                                font_size = s.get("size", 12)
                                if font_size > 16:
                                    return "title", 0.8
                                elif font_size < 10:
                                    return "footnote", 0.7
        except:
            pass

        # 检查是否为列表
        if text.startswith(('•', '-', '*', '1.', '2.', '3.', '①', '②', '③')):
            return "list", 0.85

        # 默认为段落
        return "paragraph", 0.7

    def _detect_image_regions(self, page, page_num: int) -> List[DetectedRegion]:
        """检测图片区域"""
        regions = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # 获取图片位置
                xref = img[0]
                for img_info in page.get_image_rects(xref):
                    region = DetectedRegion(
                        region_type="image",
                        text=f"[Image {img_index + 1}]",
                        bbox=img_info[:4],
                        page=page_num,
                        confidence=0.95,
                        metadata={'xref': xref}
                    )
                    regions.append(region)
            except Exception as e:
                self.logger.warning(f"检测图片失败: {e}")

        return regions

    def _detect_table_regions(self, page, page_num: int) -> List[DetectedRegion]:
        """检测表格区域（简单实现）"""
        # PyMuPDF 不直接支持表格检测
        # 这里返回空列表，实际项目可以集成 camelot 或 pdfplumber
        return []

    def export_regions_to_dict(
        self,
        regions: List[DetectedRegion]
    ) -> List[Dict[str, Any]]:
        """将区域列表导出为字典列表"""
        return [region.to_dict() for region in regions]

# 全局实例
smart_region_detector = SmartRegionDetector()

__all__ = ['SmartRegionDetector', 'smart_region_detector', 'DetectedRegion']
