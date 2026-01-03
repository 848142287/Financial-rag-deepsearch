"""
OCR 服务
使用 GLM-4.6V 云端 API 进行文字识别
"""

from app.core.structured_logging import get_structured_logger
from typing import Dict, Any, Optional
from enum import Enum

from app.services.multimodal.vlm_integration import GLM46VModel, ImageInput

logger = get_structured_logger(__name__)

class OCRModelType(str, Enum):
    """OCR 模型类型"""
    GLM_4_6V = "glm_4.6v"  # GLM-4.6V云端OCR

class OCRService:
    """
    OCR服务

    使用 GLM-4.6V 云端API进行高精度文字识别
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        model_kwargs: Optional[dict] = None
    ):
        """
        初始化OCR服务

        Args:
            enable_ocr: 是否启用OCR
            model_kwargs: GLM-4.6V模型参数
        """
        self.enable_ocr = enable_ocr
        self.model_kwargs = model_kwargs or {}

        # GLM-4.6V模型实例
        self._model: Optional[GLM46VModel] = None

        logger.info(f"初始化OCR服务: 启用={enable_ocr}")

    def _get_model(self) -> Optional[GLM46VModel]:
        """获取GLM-4.6V模型实例"""
        if not self.enable_ocr:
            return None

        if self._model is None:
            try:
                # 如果没有提供配置，从系统配置读取
                if not self.model_kwargs:

                    if not OCR_CONFIG["enable_ocr"]:
                        logger.warning("OCR功能未启用")
                        return None

                    self.model_kwargs = OCR_CONFIG["model_kwargs"]

                self._model = GLM46VModel(self.model_kwargs)
                logger.info("✅ GLM-4.6V OCR模型初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ GLM-4.6V OCR模型初始化失败: {e}")
                self._model = None

        return self._model

    async def extract_text_from_image(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从图片中提取文本

        Args:
            image_bytes: 图片字节内容
            prompt: 提示词（可选，默认使用GLM-4.6V内置OCR提示词）

        Returns:
            包含识别结果的字典
        """
        try:
            model = self._get_model()
            if model is None:
                return {
                    'success': False,
                    'error': 'OCR模型未初始化',
                    'text': ''
                }

            # 使用GLM-4.6V的专用OCR功能
            if prompt is None:
                # 使用内置的优化OCR提示词
                text = await model.extract_text_from_image(
                    ImageInput(image_bytes=image_bytes)
                )
            else:
                # 使用自定义提示词
                result = await model.analyze_image(
                    ImageInput(image_bytes=image_bytes),
                    question=prompt
                )
                text = result.text_description

            return {
                'success': True,
                'text': text,
                'model': 'glm-4.6v',
                'confidence': 0.95
            }

        except Exception as e:
            logger.error(f"GLM-4.6V OCR提取失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    async def extract_text_from_pdf_page(
        self,
        pdf_bytes: bytes,
        page_num: int = 0
    ) -> Dict[str, Any]:
        """
        从 PDF 页面提取文本

        Args:
            pdf_bytes: PDF 文件字节
            page_num: 页码（从 0 开始）

        Returns:
            提取结果
        """
        import io

        try:
            # 首先尝试使用 pdfplumber 提取文本
            try:
                import pdfplumber

                pdf_file = io.BytesIO(pdf_bytes)

                with pdfplumber.open(pdf_file) as pdf:
                    if page_num >= len(pdf.pages):
                        return {
                            'success': False,
                            'error': f'页码 {page_num} 超出范围 (共 {len(pdf.pages)} 页)',
                            'text': ''
                        }

                    page = pdf.pages[page_num]
                    text = page.extract_text()

                    # 如果文本质量足够（长度>100），直接返回
                    if text and len(text.strip()) > 100:
                        logger.info(f"✅ PDF第{page_num}页文本提取成功（pdfplumber）")
                        return {
                            'success': True,
                            'text': text,
                            'source': 'pdfplumber',
                            'model': 'none'
                        }

            except ImportError:
                logger.warning("pdfplumber未安装，跳过直接文本提取")
            except Exception as e:
                logger.warning(f"pdfplumber提取失败: {e}")

            # 文本质量不足或提取失败，使用GLM-4.6V OCR
            logger.info(f"PDF第{page_num}页使用GLM-4.6V OCR识别")

            # 将PDF页面转换为图像
            try:
                from pdf2image import convert_from_bytes

                images = convert_from_bytes(pdf_bytes, first_page=page_num + 1, last_page=page_num + 1)

                if not images:
                    return {
                        'success': False,
                        'error': f'无法转换PDF第{page_num}页为图像',
                        'text': ''
                    }

                # 将PIL图像转换为字节
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()

                # 使用GLM-4.6V OCR
                result = await self.extract_text_from_image(image_bytes)

                if result['success']:
                    result['source'] = 'glm_4.6v_ocr'

                return result

            except ImportError:
                logger.error("pdf2image未安装，无法进行OCR识别")
                return {
                    'success': False,
                    'error': 'pdf2image未安装，无法进行PDF图像转换',
                    'text': ''
                }
            except Exception as e:
                logger.error(f"PDF转图像失败: {e}")
                return {
                    'success': False,
                    'error': f'PDF转图像失败: {str(e)}',
                    'text': ''
                }

        except Exception as e:
            logger.error(f"PDF页面提取失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

# 兼容性：保留旧名称
HybridOCRService = OCRService
