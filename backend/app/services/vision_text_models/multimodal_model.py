"""
多模态模型实现
集成视觉和文本的联合理解模型
"""

import logging
from typing import Dict, Any, Optional, List, Union
import asyncio
import torch
import numpy as np
from PIL import Image

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        BlipForQuestionAnswering,
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        AutoProcessor, AutoModelForCausalLM
    )
    MULTIMODAL_MODELS_AVAILABLE = True
except ImportError:
    MULTIMODAL_MODELS_AVAILABLE = False
    logging.warning("多模态模型未安装，相关功能不可用")

from .base_model import (
    BaseVisionTextModel,
    ModelType,
    TaskType,
    ModelInput,
    ModelOutput
)

logger = logging.getLogger(__name__)


class MultimodalModel(BaseVisionTextModel):
    """多模态模型"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not MULTIMODAL_MODELS_AVAILABLE:
            raise ImportError("需要安装Transformers和Pillow: pip install transformers pillow")

        super().__init__(config)
        self.model_type = ModelType.MULTIMODAL

        # 模型配置
        self.model_configs = {
            'image_captioning': self.config.get('captioning_model', 'Salesforce/blip-image-captioning-base'),
            'vqa': self.config.get('vqa_model', 'Salesforce/blip-vqa-base'),
            'document_understanding': self.config.get('document_model', 'microsoft/layoutlmv3-base'),
            'multimodal_llm': self.config.get('multimodal_llm', 'llava-hf/llava-1.5-7b-hf')
        }

        # 模型和处理器实例
        self.models = {}
        self.processors = {}

        # 图像预处理
        self.image_processor = None

    def _get_model_type(self) -> ModelType:
        return ModelType.MULTIMODAL

    async def load_model(self) -> bool:
        """加载多模态模型"""
        try:
            # 加载图像描述模型
            if TaskType.IMAGE_CAPTIONING in self.get_supported_tasks():
                await self._load_captioning_model()

            # 加载VQA模型
            if TaskType.VISUAL_QUESTION_ANSWERING in self.get_supported_tasks():
                await self._load_vqa_model()

            # 加载文档理解模型
            if TaskType.DOCUMENT_UNDERSTANDING in self.get_supported_tasks():
                await self._load_document_model()

            self.is_loaded = True
            logger.info("多模态模型加载成功")
            return True

        except Exception as e:
            logger.error(f"多模态模型加载失败: {str(e)}")
            return False

    async def _load_captioning_model(self):
        """加载图像描述模型"""
        model_name = self.model_configs['image_captioning']
        logger.info(f"加载图像描述模型: {model_name}")

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.processors['captioning'] = processor
        self.models['captioning'] = model

    async def _load_vqa_model(self):
        """加载视觉问答模型"""
        model_name = self.model_configs['vqa']
        logger.info(f"加载VQA模型: {model_name}")

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.processors['vqa'] = processor
        self.models['vqa'] = model

    async def _load_document_model(self):
        """加载文档理解模型"""
        model_name = self.model_configs['document_understanding']
        logger.info(f"加载文档理解模型: {model_name}")

        processor = LayoutLMv3Processor.from_pretrained(model_name)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.processors['document'] = processor
        self.models['document'] = model

    async def unload_model(self) -> bool:
        """卸载模型"""
        try:
            # 清理所有模型
            for model in self.models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model

            self.models.clear()
            self.processors.clear()

            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("多模态模型卸载成功")
            return True

        except Exception as e:
            logger.error(f"多模态模型卸载失败: {str(e)}")
            return False

    def get_supported_tasks(self) -> List[TaskType]:
        """获取支持的任务类型"""
        return [
            TaskType.IMAGE_CAPTIONING,
            TaskType.VISUAL_QUESTION_ANSWERING,
            TaskType.DOCUMENT_UNDERSTANDING,
            TaskType.VISUAL_DOCUMENT_ANALYSIS,
            TaskType.CHART_UNDERSTANDING,
            TaskType.TABLE_RECOGNITION,
            TaskType.FORMULA_RECOGNITION,
            TaskType.CROSS_MODAL_RETRIEVAL
        ]

    async def process(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """处理多模态输入"""
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
        if task_type == TaskType.IMAGE_CAPTIONING:
            outputs = await self._generate_caption(preprocessed)
        elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
            question = kwargs.get('question', '')
            outputs = await self._answer_visual_question(preprocessed, question)
        elif task_type == TaskType.DOCUMENT_UNDERSTANDING:
            outputs = await self._understand_document(preprocessed)
        elif task_type == TaskType.CHART_UNDERSTANDING:
            outputs = await self._understand_chart(preprocessed)
        elif task_type == TaskType.TABLE_RECOGNITION:
            outputs = await self._recognize_table(preprocessed)
        else:
            outputs = await self._generic_process(preprocessed, task_type, **kwargs)

        # 后处理
        if isinstance(inputs, list):
            return [await self.postprocess(out, task_type, **kwargs) for out in outputs]
        else:
            return await self.postprocess(outputs, task_type, **kwargs)

    async def _preprocess_single(self, input_data: ModelInput, task_type: TaskType) -> Any:
        """预处理单个输入"""
        # 获取图像和文本
        image = None
        text = ""

        if isinstance(input_data.data, dict):
            # 多模态输入
            image = input_data.data.get('image')
            text = input_data.data.get('text', '')
        elif input_data.data_type == 'multimodal' and input_data.metadata:
            image = input_data.metadata.get('image')
            text = input_data.metadata.get('text', '')

        # 转换图像格式
        if image is not None:
            if isinstance(image, str):
                # 路径
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                # numpy数组
                image = Image.fromarray(image).convert('RGB')

        # 根据任务类型进行特定预处理
        if task_type == TaskType.IMAGE_CAPTIONING:
            if 'captioning' in self.processors:
                processor = self.processors['captioning']
                if image:
                    return processor(image, return_tensors="pt")
        elif task_type == TaskType.VISUAL_QUESTION_ANSWERING:
            question = input_data.metadata.get('question', '') if input_data.metadata else ''
            if 'vqa' in self.processors and image:
                processor = self.processors['vqa']
                return processor(image, question, return_tensors="pt")
        elif task_type == TaskType.DOCUMENT_UNDERSTANDING:
            if 'document' in self.processors and image:
                processor = self.processors['document']
                encoding = processor(
                    image,
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                return encoding

        return {'image': image, 'text': text}

    async def _postprocess_single(self, output: Any, task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """后处理单个输出"""
        if isinstance(output, dict):
            return output
        elif isinstance(output, str):
            return {'result': output}
        else:
            return {'output': str(output)}

    async def _generate_caption(self, inputs: Any) -> str:
        """生成图像描述"""
        if 'captioning' not in self.models:
            return "无法生成描述"

        model = self.models['captioning']
        model.eval()

        with torch.no_grad():
            if isinstance(inputs, dict) and 'pixel_values' in inputs:
                # 移动到设备
                pixel_values = inputs['pixel_values']
                if self.device != 'cpu':
                    pixel_values = pixel_values.to(self.device)

                # 生成描述
                generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
                caption = self.processors['captioning'].decode(generated_ids[0], skip_special_tokens=True)
                return caption

        return "无法生成描述"

    async def _answer_visual_question(self, inputs: Any, question: str) -> str:
        """回答视觉问题"""
        if not question or 'vqa' not in self.models:
            return "无法回答问题"

        model = self.models['vqa']
        model.eval()

        with torch.no_grad():
            if isinstance(inputs, dict):
                # 移动到设备
                if self.device != 'cpu':
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}

                # 生成答案
                generated_ids = model.generate(**inputs)
                answer = self.processors['vqa'].decode(generated_ids[0], skip_special_tokens=True)
                return answer

        return "无法回答问题"

    async def _understand_document(self, inputs: Any) -> Dict[str, Any]:
        """文档理解"""
        if 'document' not in self.models:
            return {"entities": [], "text": ""}

        model = self.models['document']
        model.eval()

        with torch.no_grad():
            if isinstance(inputs, dict):
                # 移动到设备
                if self.device != 'cpu':
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}

                # 前向传播
                outputs = model(**inputs)

                # 处理输出
                predictions = outputs.logits.argmax(-1)
                labels = [model.config.id2label[p.item()] for p in predictions[0]]

                # 提取文本和实体
                tokens = self.processors['document'].tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                entities = []

                for token, label in zip(tokens, labels):
                    if label != 'O':  # 不是其他标签
                        entities.append({
                            'text': token.replace('Ġ', ''),
                            'label': label,
                            'confidence': 1.0
                        })

                return {
                    'entities': entities,
                    'text': ' '.join([t.replace('Ġ', '') for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']])
                }

        return {"entities": [], "text": ""}

    async def _understand_chart(self, inputs: Any) -> Dict[str, Any]:
        """图表理解"""
        # 这里可以使用专门的图表理解模型
        # 暂时返回基本信息
        return {
            'chart_type': 'unknown',
            'title': '',
            'data_points': [],
            'insights': []
        }

    async def _recognize_table(self, inputs: Any) -> Dict[str, Any]:
        """表格识别"""
        # 这里可以使用专门的表格识别模型
        # 暂时返回基本信息
        return {
            'rows': [],
            'headers': [],
            'cells': [],
            'structure': 'unknown'
        }

    async def _generic_process(self, inputs: Any, task_type: TaskType, **kwargs) -> Any:
        """通用处理"""
        return inputs