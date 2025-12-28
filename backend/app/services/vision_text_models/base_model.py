"""
视觉文本模型基类
定义视觉和文本处理的通用接口
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型"""
    VISION = "vision"
    TEXT = "text"
    MULTIMODAL = "multimodal"


class TaskType(Enum):
    """任务类型"""
    # 视觉任务
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    IMAGE_CAPTIONING = "image_captioning"
    VISUAL_QUESTION_ANSWERING = "vqa"
    CHART_UNDERSTANDING = "chart_understanding"
    TABLE_RECOGNITION = "table_recognition"
    FORMULA_RECOGNITION = "formula_recognition"

    # 文本任务
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "ner"
    RELATION_EXTRACTION = "relation_extraction"
    TEXT_SUMMARIZATION = "text_summarization"
    QUESTION_ANSWERING = "qa"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

    # 多模态任务
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VISUAL_DOCUMENT_ANALYSIS = "visual_document_analysis"
    CROSS_MODAL_RETRIEVAL = "cross_modal_retrieval"


@dataclass
class ModelInput:
    """模型输入"""
    data: Union[str, np.ndarray, List[str], List[np.ndarray]]
    data_type: str  # text, image, audio, video
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_options: Optional[Dict[str, Any]] = None


@dataclass
class ModelOutput:
    """模型输出"""
    results: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BaseVisionTextModel(ABC):
    """视觉文本模型基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.__class__.__name__
        self.model_type = self._get_model_type()
        self.is_loaded = False
        self.device = self.config.get('device', 'cpu')

    @abstractmethod
    def _get_model_type(self) -> ModelType:
        """获取模型类型"""
        pass

    @abstractmethod
    async def load_model(self) -> bool:
        """加载模型"""
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """卸载模型"""
        pass

    @abstractmethod
    def get_supported_tasks(self) -> List[TaskType]:
        """获取支持的任务类型"""
        pass

    @abstractmethod
    async def process(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """处理输入数据"""
        pass

    async def preprocess(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType
    ) -> Union[Any, List[Any]]:
        """预处理输入数据"""
        if isinstance(inputs, list):
            return [await self._preprocess_single(inp, task_type) for inp in inputs]
        else:
            return await self._preprocess_single(inputs, task_type)

    async def postprocess(
        self,
        outputs: Any,
        task_type: TaskType,
        **kwargs
    ) -> ModelOutput:
        """后处理输出数据"""
        start_time = asyncio.get_event_loop().time()

        processed_result = await self._postprocess_single(outputs, task_type, **kwargs)

        processing_time = asyncio.get_event_loop().time() - start_time

        return ModelOutput(
            results=processed_result,
            processing_time=processing_time,
            metadata={'model': self.model_name, 'task': task_type.value}
        )

    @abstractmethod
    async def _preprocess_single(self, input_data: ModelInput, task_type: TaskType) -> Any:
        """预处理单个输入"""
        pass

    @abstractmethod
    async def _postprocess_single(self, output: Any, task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """后处理单个输出"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.model_name,
            'type': self.model_type.value,
            'tasks': [task.value for task in self.get_supported_tasks()],
            'device': self.device,
            'loaded': self.is_loaded,
            'config': self.config
        }

    def validate_input(self, input_data: ModelInput, task_type: TaskType) -> bool:
        """验证输入数据"""
        # 基本验证
        if not input_data.data:
            return False

        # 根据模型类型验证
        if self.model_type == ModelType.VISION:
            return self._validate_vision_input(input_data, task_type)
        elif self.model_type == ModelType.TEXT:
            return self._validate_text_input(input_data, task_type)
        elif self.model_type == ModelType.MULTIMODAL:
            return self._validate_multimodal_input(input_data, task_type)

        return False

    def _validate_vision_input(self, input_data: ModelInput, task_type: TaskType) -> bool:
        """验证视觉输入"""
        valid_types = ['image', 'video', 'numpy']
        if input_data.data_type not in valid_types:
            return False

        # 根据任务类型进行特定验证
        if task_type == TaskType.OCR:
            # OCR需要图像输入
            return input_data.data_type in ['image', 'numpy']
        elif task_type == TaskType.IMAGE_CLASSIFICATION:
            # 图像分类需要图像
            return input_data.data_type in ['image', 'numpy']
        elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
            # VQA需要图像和问题
            if not input_data.metadata or 'question' not in input_data.metadata:
                return False

        return True

    def _validate_text_input(self, input_data: ModelInput, task_type: TaskType) -> bool:
        """验证文本输入"""
        if input_data.data_type != 'text':
            return False

        # 根据任务类型进行特定验证
        if task_type == TaskType.QUESTION_ANSWERING:
            # QA需要问题和上下文
            if not input_data.metadata or ('question' not in input_data.metadata and 'context' not in input_data.metadata):
                return False

        return True

    def _validate_multimodal_input(self, input_data: ModelInput, task_type: TaskType) -> bool:
        """验证多模态输入"""
        # 多模态输入更复杂，这里做基本验证
        if input_data.data_type not in ['multimodal', 'document']:
            return False

        # 检查必要的元数据
        if not input_data.metadata:
            return False

        return True

    async def batch_process(
        self,
        inputs: List[ModelInput],
        task_type: TaskType,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """批量处理"""
        if not batch_size:
            batch_size = self.config.get('batch_size', 8)

        results = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"处理批次 {batch_num}/{total_batches}")

            # 处理当前批次
            if len(batch) == 1:
                batch_result = await self.process(batch[0], task_type, **kwargs)
                results.append(batch_result)
            else:
                batch_results = await self.process(batch, task_type, **kwargs)
                results.extend(batch_results)

        return results

    def calculate_confidence(self, output: Any, task_type: TaskType) -> float:
        """计算置信度"""
        # 默认实现，子类可以重写
        return 1.0

    def extract_features(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        layer_name: Optional[str] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """提取特征"""
        # 默认实现，子类可以重写
        raise NotImplementedError("特征提取功能需要子类实现")

    async def warm_up(self):
        """模型预热"""
        logger.info(f"预热模型 {self.model_name}")
        # 创建简单的测试输入
        if self.model_type == ModelType.TEXT:
            test_input = ModelInput(
                data="test",
                data_type="text"
            )
            await self.process(test_input, TaskType.TEXT_CLASSIFICATION)
        elif self.model_type == ModelType.VISION:
            # 创建简单的测试图像
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_input = ModelInput(
                data=test_image,
                data_type="numpy"
            )
            await self.process(test_input, TaskType.IMAGE_CLASSIFICATION)

        logger.info(f"模型 {self.model_name} 预热完成")