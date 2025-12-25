"""
视觉模型实现
集成各种先进的视觉理解模型
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union
import asyncio
from pathlib import Path
import base64
import io
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import DetrImageProcessor, DetrForObjectDetection
    VISION_MODELS_AVAILABLE = True
except ImportError:
    VISION_MODELS_AVAILABLE = False
    logging.warning("PyTorch或Transformers未安装，视觉模型功能不可用")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR未安装，OCR功能不可用")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，OCR功能不可用")

from .base_model import (
    BaseVisionTextModel,
    ModelType,
    TaskType,
    ModelInput,
    ModelOutput
)

logger = logging.getLogger(__name__)


class VisionModel(BaseVisionTextModel):
    """视觉模型"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not VISION_MODELS_AVAILABLE:
            raise ImportError("需要安装PyTorch和Transformers: pip install torch torchvision transformers")

        super().__init__(config)
        self.model_type = ModelType.VISION

        # 模型配置
        self.model_name_or_path = self.config.get('model_name', 'microsoft/detr-resnet-50')
        self.task_config = self.config.get('task_config', {})

        # 模型实例
        self.models = {}
        self.processors = {}

        # EasyOCR实例
        self.easyocr_reader = None

        # PaddleOCR实例
        self.paddleocr_reader = None

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_model_type(self) -> ModelType:
        return ModelType.VISION

    async def load_model(self) -> bool:
        """加载视觉模型"""
        try:
            # 加载分类模型
            if TaskType.IMAGE_CLASSIFICATION in self.get_supported_tasks():
                await self._load_classification_model()

            # 加载检测模型
            if TaskType.OBJECT_DETECTION in self.get_supported_tasks():
                await self._load_detection_model()

            # 初始化OCR
            if TaskType.OCR in self.get_supported_tasks():
                await self._init_ocr_models()

            self.is_loaded = True
            logger.info("视觉模型加载成功")
            return True

        except Exception as e:
            logger.error(f"视觉模型加载失败: {str(e)}")
            return False

    async def _load_classification_model(self):
        """加载图像分类模型"""
        model_name = self.task_config.get('classification', 'google/vit-base-patch16-224')

        logger.info(f"加载分类模型: {model_name}")

        # 加载处理器和模型
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.processors['classification'] = processor
        self.models['classification'] = model

    async def _load_detection_model(self):
        """加载目标检测模型"""
        model_name = self.task_config.get('detection', 'facebook/detr-resnet-50')

        logger.info(f"加载检测模型: {model_name}")

        # 加载处理器和模型
        processor = DetrImageProcessor.from_pretrained(model_name)
        model = DetrForObjectDetection.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.processors['detection'] = processor
        self.models['detection'] = model

    async def _init_ocr_models(self):
        """初始化OCR模型"""
        # 初始化EasyOCR
        if EASYOCR_AVAILABLE:
            langs = self.task_config.get('ocr_langs', ['ch_sim', 'en'])
            self.easyocr_reader = easyocr.Reader(langs, gpu=self.device != 'cpu')
            logger.info(f"EasyOCR初始化成功，支持语言: {langs}")

        # 初始化PaddleOCR
        if PADDLEOCR_AVAILABLE:
            use_angle_cls = self.task_config.get('use_angle_cls', True)
            lang = self.task_config.get('paddle_lang', 'ch')
            self.paddleocr_reader = paddleocr.PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=self.device != 'cpu'
            )
            logger.info(f"PaddleOCR初始化成功，语言: {lang}")

    async def unload_model(self) -> bool:
        """卸载模型"""
        try:
            # 清理GPU内存
            for model in self.models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model

            self.models.clear()
            self.processors.clear()

            # 清理OCR实例
            self.easyocr_reader = None
            self.paddleocr_reader = None

            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("视觉模型卸载成功")
            return True

        except Exception as e:
            logger.error(f"视觉模型卸载失败: {str(e)}")
            return False

    def get_supported_tasks(self) -> List[TaskType]:
        """获取支持的任务类型"""
        return [
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.OBJECT_DETECTION,
            TaskType.OCR,
            TaskType.IMAGE_CAPTIONING,
            TaskType.VISUAL_QUESTION_ANSWERING,
            TaskType.CHART_UNDERSTANDING,
            TaskType.TABLE_RECOGNITION,
            TaskType.FORMULA_RECOGNITION
        ]

    async def process(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """处理视觉输入"""
        if not self.is_loaded:
            await self.load_model()

        # 验证输入
        if isinstance(inputs, list):
            for inp in inputs:
                if not self.validate_input(inp, task_type):
                    return ModelOutput(
                        results={},
                        error_message="输入验证失败"
                    )
        else:
            if not self.validate_input(inputs, task_type):
                return ModelOutput(
                    results={},
                    error_message="输入验证失败"
                )

        # 预处理
        preprocessed = await self.preprocess(inputs, task_type)

        # 推理
        if task_type == TaskType.IMAGE_CLASSIFICATION:
            outputs = await self._classify(preprocessed)
        elif task_type == TaskType.OBJECT_DETECTION:
            outputs = await self._detect_objects(preprocessed)
        elif task_type == TaskType.OCR:
            outputs = await self._ocr(preprocessed)
        elif task_type == TaskType.IMAGE_CAPTIONING:
            outputs = await self._caption_image(preprocessed)
        elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
            question = kwargs.get('question', '')
            outputs = await self._vqa(preprocessed, question)
        else:
            outputs = await self._generic_process(preprocessed, task_type, **kwargs)

        # 后处理
        if isinstance(inputs, list):
            return [await self.postprocess(out, task_type, **kwargs) for out in outputs]
        else:
            return await self.postprocess(outputs, task_type, **kwargs)

    async def _preprocess_single(self, input_data: ModelInput, task_type: TaskType) -> Any:
        """预处理单个输入"""
        # 转换输入为numpy数组
        image = self._load_image(input_data.data)

        if task_type == TaskType.IMAGE_CLASSIFICATION:
            if 'classification' in self.processors:
                processor = self.processors['classification']
                return processor(image, return_tensors="pt")
            else:
                # 使用默认预处理
                image = Image.fromarray(image)
                return self.transform(image).unsqueeze(0)

        elif task_type == TaskType.OBJECT_DETECTION:
            if 'detection' in self.processors:
                processor = self.processors['detection']
                return processor(image, return_tensors="pt")
            else:
                # 返回原始图像用于其他处理
                return image

        else:
            # 其他任务返回原始图像
            return image

    async def _postprocess_single(self, output: Any, task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """后处理单个输出"""
        if task_type == TaskType.IMAGE_CLASSIFICATION:
            return await self._postprocess_classification(output)
        elif task_type == TaskType.OBJECT_DETECTION:
            return await self._postprocess_detection(output)
        elif task_type == TaskType.OCR:
            return await self._postprocess_ocr(output)
        else:
            return {"raw_output": str(output)}

    def _load_image(self, image_data: Any) -> np.ndarray:
        """加载图像数据"""
        if isinstance(image_data, str):
            # 路径或base64
            if os.path.exists(image_data):
                image = cv2.imread(image_data)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 尝试base64解码
                try:
                    img_data = base64.b64decode(image_data)
                    image = np.array(Image.open(io.BytesIO(img_data)))
                    return image
                except:
                    raise ValueError("无法解析图像数据")

        elif isinstance(image_data, np.ndarray):
            return image_data

        elif isinstance(image_data, Image.Image):
            return np.array(image_data)

        else:
            raise ValueError(f"不支持的图像数据类型: {type(image_data)}")

    async def _classify(self, inputs: Any) -> Any:
        """图像分类"""
        if 'classification' not in self.models:
            raise ValueError("分类模型未加载")

        model = self.models['classification']
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs

    async def _detect_objects(self, inputs: Any) -> Any:
        """目标检测"""
        if 'detection' not in self.models:
            raise ValueError("检测模型未加载")

        model = self.models['detection']
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs

    async def _ocr(self, inputs: Any) -> List[str]:
        """OCR文字识别"""
        if isinstance(inputs, torch.Tensor):
            # 转换为numpy
            image = inputs.squeeze().permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
        else:
            image = inputs

        results = []

        # 使用EasyOCR
        if self.easyocr_reader:
            easy_results = self.easyocr_reader.readtext(image)
            results.extend([result[1] for result in easy_results])

        # 使用PaddleOCR
        if self.paddleocr_reader:
            paddle_results = self.paddleocr_reader.ocr(image, cls=True)
            for line in paddle_results:
                if line:
                    for word_info in line:
                        results.append(word_info[1][0])

        return results

    async def _caption_image(self, inputs: Any) -> str:
        """图像描述生成"""
        # 这里可以使用BLIP等模型
        # 暂时返回简单描述
        return "这是一张图片"

    async def _vqa(self, inputs: Any, question: str) -> str:
        """视觉问答"""
        # 这里可以使用VQA模型
        # 暂时返回简单回答
        return f"关于图片'{question}'的答案"

    async def _generic_process(self, inputs: Any, task_type: TaskType, **kwargs) -> Any:
        """通用处理"""
        return inputs

    async def _postprocess_classification(self, output: Any) -> Dict[str, Any]:
        """后处理分类结果"""
        if isinstance(output, dict) and 'logits' in output:
            logits = output['logits']
        else:
            logits = output

        if isinstance(logits, torch.Tensor):
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)

            return {
                'class_id': predicted_class.item(),
                'confidence': confidence.item(),
                'probabilities': probs.tolist()
            }

        return {"class_id": -1, "confidence": 0.0}

    async def _postprocess_detection(self, output: Any) -> Dict[str, Any]:
        """后处理检测结果"""
        if isinstance(output, dict):
            results = []
            if 'logits' in output and 'pred_boxes' in output:
                logits = output['logits']
                boxes = output['pred_boxes']

                # 处理每个检测
                probs = torch.nn.functional.softmax(logits, -1)
                scores, labels = torch.max(probs, -1)

                for i in range(len(boxes[0])):
                    box = boxes[0][i].cpu().numpy()
                    score = scores[0][i].item()
                    label = labels[0][i].item()

                    if score > 0.5:  # 置信度阈值
                        results.append({
                            'box': box.tolist(),
                            'score': score,
                            'label': label,
                            'label_name': f"class_{label}"
                        })

            return {'detections': results}

        return {"detections": []}

    async def _postprocess_ocr(self, output: Any) -> Dict[str, Any]:
        """后处理OCR结果"""
        if isinstance(output, list):
            return {
                'texts': output,
                'total_texts': len(output)
            }

        return {"texts": [], "total_texts": 0}