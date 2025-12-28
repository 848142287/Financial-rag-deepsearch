"""
智能区域检测器
集成了 Multimodal_RAG 的 gptpdf 区域检测功能
使用 Shapely 进行精确的空间位置计算，智能识别表格、图片、文本区域
"""

import logging
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.validation import explain_validity
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectedRegion:
    """检测到的区域"""
    region_type: str  # 'table', 'image', 'text', 'drawing'
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    content: str = ""
    confidence: float = 1.0


class SmartRegionDetector:
    """
    智能区域检测器

    功能：
    1. 智能区域识别：自动识别表格、图片、文本区域
    2. 矩形合并算法：合并相邻的文本块
    3. 精确空间定位：使用 Shapely 进行几何计算
    """

    def __init__(self, merge_distance: float = 20, horizontal_distance: float = 100):
        """
        初始化智能区域检测器

        Args:
            merge_distance: 合并距离阈值
            horizontal_distance: 水平合并距离阈值
        """
        self.merge_distance = merge_distance
        self.horizontal_distance = horizontal_distance

        # 短线过滤阈值
        self.short_line_threshold = 30

        # 小矩形过滤阈值
        self.min_rect_width = 20
        self.min_rect_height = 20

        logger.info(f"初始化智能区域检测器，合并距离: {merge_distance}")

    def _is_near(self, rect1: sg.Polygon, rect2: sg.Polygon, distance: float = None) -> bool:
        """检查两个矩形是否靠近"""
        distance = distance or self.merge_distance
        return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance

    def _is_horizontal_near(
        self,
        rect1: sg.Polygon,
        rect2: sg.Polygon,
        distance: float = None
    ) -> bool:
        """检查两个矩形是否水平靠近"""
        distance = distance or self.horizontal_distance
        result = False
        if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
            if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
                result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
        return result

    def _union_rects(self, rect1: sg.Polygon, rect2: sg.Polygon) -> sg.Polygon:
        """合并两个矩形"""
        return sg.box(*(rect1.union(rect2).bounds))

    def _merge_rects(
        self,
        rect_list: List[sg.Polygon],
        distance: float = None,
        horizontal_distance: float = None
    ) -> List[sg.Polygon]:
        """合并列表中的矩形"""
        distance = distance or self.merge_distance
        horizontal_distance = horizontal_distance or self.horizontal_distance

        merged = True
        while merged:
            merged = False
            new_rect_list = []
            while rect_list:
                rect = rect_list.pop(0)
                for other_rect in rect_list:
                    if self._is_near(rect, other_rect, distance) or (
                        horizontal_distance and self._is_horizontal_near(rect, other_rect, horizontal_distance)
                    ):
                        rect = self._union_rects(rect, other_rect)
                        rect_list.remove(other_rect)
                        merged = True
                new_rect_list.append(rect)
            rect_list = new_rect_list
        return rect_list

    def _adsorb_rects_to_rects(
        self,
        source_rects: List[sg.Polygon],
        target_rects: List[sg.Polygon],
        distance: float = 10
    ) -> Tuple[List[sg.Polygon], List[sg.Polygon]]:
        """当距离小于目标距离时，将一组矩形吸附到另一组矩形"""
        new_source_rects = []
        for text_area_rect in source_rects:
            adsorbed = False
            for index, rect in enumerate(target_rects):
                if self._is_near(text_area_rect, rect, distance):
                    target_rects[index] = self._union_rects(text_area_rect, rect)
                    adsorbed = True
                    break
            if not adsorbed:
                new_source_rects.append(text_area_rect)
        return new_source_rects, target_rects

    def _parse_rects(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        """解析页面中的绘图，并合并相邻的矩形"""
        # 提取画的内容
        drawings = page.get_drawings()

        # 忽略掉长度小于阈值的水平直线
        is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 1 and abs(x['rect'][2] - x['rect'][0]) < self.short_line_threshold
        drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

        # 转换为 shapely 的矩形
        rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

        # 提取图片区域
        images = page.get_image_info()
        image_rects = [sg.box(*image['bbox']) for image in images]

        # 合并 drawings 和 images
        rect_list += image_rects

        merged_rects = self._merge_rects(rect_list, distance=10, horizontal_distance=self.horizontal_distance)
        merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

        # 将大文本区域和小文本区域分开处理
        is_large_content = lambda x: (len(x[4]) / max(1, len(x[4].split('\n')))) > 5
        small_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if not is_large_content(x)]
        large_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if is_large_content(x)]

        _, merged_rects = self._adsorb_rects_to_rects(large_text_area_rects, merged_rects, distance=0.1)
        _, merged_rects = self._adsorb_rects_to_rects(small_text_area_rects, merged_rects, distance=5)

        # 再次自身合并
        merged_rects = self._merge_rects(merged_rects, distance=10)

        # 过滤比较小的矩形
        merged_rects = [
            rect for rect in merged_rects
            if rect.bounds[2] - rect.bounds[0] > self.min_rect_width and
            rect.bounds[3] - rect.bounds[1] > self.min_rect_height
        ]

        return [rect.bounds for rect in merged_rects]

    def detect_regions(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[DetectedRegion]:
        """
        检测 PDF 中的智能区域

        Args:
            pdf_path: PDF 文件路径
            pages: 要检测的页码列表（None 表示全部）

        Returns:
            List[DetectedRegion]: 检测到的区域列表
        """
        doc = fitz.open(pdf_path)
        all_regions = []

        pages_to_process = pages if pages else range(len(doc))

        for page_num in pages_to_process:
            try:
                page = doc[page_num]
                rects = self._parse_rects(page)

                # 获取文本内容
                text_blocks = page.get_text('blocks')

                # 分类区域
                for rect in rects:
                    region = self._classify_region(rect, page, text_blocks, page_num)
                    if region:
                        all_regions.append(region)

            except Exception as e:
                logger.error(f"处理页面 {page_num+1} 时出错: {e}")

        doc.close()

        logger.info(f"智能区域检测完成，共检测到 {len(all_regions)} 个区域")
        return all_regions

    def _classify_region(
        self,
        rect: Tuple[float, float, float, float],
        page: fitz.Page,
        text_blocks: List,
        page_num: int
    ) -> Optional[DetectedRegion]:
        """
        分类检测到的区域

        Args:
            rect: 区域边界框
            page: PyMuPDF 页面对象
            text_blocks: 文本块列表
            page_num: 页码

        Returns:
            DetectedRegion 或 None
        """
        # 获取区域内的文本
        rect_fitz = fitz.Rect(rect)
        text = page.get_text('text', clip=rect_fitz)

        # 检查是否包含表格特征
        table_keywords = ['|', '表', 'Table', '表格']
        if any(keyword in text for keyword in table_keywords):
            return DetectedRegion(
                region_type='table',
                bbox=rect,
                page_num=page_num,
                content=text,
                confidence=0.8
            )

        # 检查是否是图片
        images = page.get_image_info()
        for img in images:
            img_bbox = img['bbox']
            if self._rects_overlap(rect, img_bbox):
                return DetectedRegion(
                    region_type='image',
                    bbox=rect,
                    page_num=page_num,
                    content=text,
                    confidence=0.9
                )

        # 默认为文本区域
        if text.strip():
            return DetectedRegion(
                region_type='text',
                bbox=rect,
                page_num=page_num,
                content=text,
                confidence=0.7
            )

        return None

    def _rects_overlap(
        self,
        rect1: Tuple[float, float, float, float],
        rect2: Tuple[float, float, float, float]
    ) -> bool:
        """检查两个矩形是否重叠"""
        return not (
            rect1[2] <= rect2[0] or  # rect1 在 rect2 左侧
            rect1[0] >= rect2[2] or  # rect1 在 rect2 右侧
            rect1[3] <= rect2[1] or  # rect1 在 rect2 上方
            rect1[1] >= rect2[3]     # rect1 在 rect2 下方
        )

    def extract_tables_from_regions(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """从检测到的区域中提取表格"""
        return [r for r in regions if r.region_type == 'table']

    def extract_images_from_regions(
        self,
        regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """从检测到的区域中提取图片"""
        return [r for r in regions if r.region_type == 'image']


# 全局实例
smart_region_detector = SmartRegionDetector()
