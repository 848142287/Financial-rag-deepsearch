"""
视觉语言模型（VLM）集成模块
支持多种视觉语言模型的统一接口和图像理解功能
主要使用通义千问VL模型替代OpenAI GPT-4V
"""

import asyncio
import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import requests

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    """图像输入数据结构"""
    image_path: Optional[str] = None
    image_array: Optional[np.ndarray] = None
    image_bytes: Optional[bytes] = None
    base64_data: Optional[str] = None

    def to_base64(self) -> str:
        """转换为base64格式"""
        if self.base64_data:
            return self.base64_data
        elif self.image_path:
            with open(self.image_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode()
        elif self.image_bytes:
            return base64.b64encode(self.image_bytes).decode()
        elif self.image_array is not None:
            _, buffer = cv2.imencode('.jpg', self.image_array)
            return base64.b64encode(buffer).decode()
        else:
            raise ValueError("No image data provided")


@dataclass
class VLMResult:
    """VLM处理结果"""
    text_description: str
    confidence: float
    extracted_entities: List[Dict[str, Any]]
    extracted_relations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float


class BaseVLMModel(ABC):
    """视觉语言模型基类"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.1)
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))

    @abstractmethod
    async def analyze_image(
        self,
        image: ImageInput,
        question: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """分析图像"""
        pass

    @abstractmethod
    async def extract_text_from_image(self, image: ImageInput) -> str:
        """从图像中提取文本"""
        pass

    @abstractmethod
    async def understand_document_structure(self, image: ImageInput) -> Dict[str, Any]:
        """理解文档结构"""
        pass

    async def batch_analyze_images(
        self,
        images: List[Tuple[ImageInput, str, Optional[Dict[str, Any]]]]
    ) -> List[VLMResult]:
        """批量分析图像"""
        tasks = []
        for image, question, context in images:
            task = self.analyze_image(image, question, context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing image {i}: {str(result)}")
                processed_results.append(VLMResult(
                    text_description="",
                    confidence=0.0,
                    extracted_entities=[],
                    extracted_relations=[],
                    metadata={"error": str(result)},
                    processing_time=0.0
                ))
            else:
                processed_results.append(result)

        return processed_results


class QwenVLModel(BaseVLMModel):
    """通义千问VL模型集成 - 主要使用的VLM模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("qwen-vl-max", config)
        self.api_key = config["api_key"]
        self.api_base = config.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = config.get("model", "qwen-vl-max")

    async def analyze_image(
        self,
        image: ImageInput,
        question: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """使用通义千问VL模型分析图像"""
        start_time = datetime.now()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 构建系统提示
            system_prompt = self._build_system_prompt(context)

            # 构建消息 - 使用OpenAI兼容格式
            messages = []

            # 添加系统提示
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

            # 添加图像和问题
            content = []
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image.to_base64()}",
                    "detail": "high"
                }
            })

            if question:
                content.append({
                    "type": "text",
                    "text": question
                })
            else:
                content.append({
                    "type": "text",
                    "text": "请详细描述这张图片的内容，特别是其中的金融数据、图表、表格等重要信息。"
                })

            messages.append({
                "role": "user",
                "content": content
            })

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # 发送请求到通义千问VL
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            )

            response.raise_for_status()
            result = response.json()

            # 解析响应
            if result.get("choices") and len(result["choices"]) > 0:
                text_description = result["choices"][0]["message"]["content"]
            else:
                text_description = ""

            # 提取实体和关系
            entities, relations = self._extract_entities_and_relations(text_description)

            processing_time = (datetime.now() - start_time).total_seconds()

            return VLMResult(
                text_description=text_description,
                confidence=0.9,  # Qwen-VL通常有较高的置信度
                extracted_entities=entities,
                extracted_relations=relations,
                metadata={
                    "model": self.model,
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                    "response_id": result.get("id")
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Qwen-VL analysis failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return VLMResult(
                text_description="",
                confidence=0.0,
                extracted_entities=[],
                extracted_relations=[],
                metadata={"error": str(e)},
                processing_time=processing_time
            )

    def _build_system_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """构建系统提示"""
        base_prompt = """你是一个专业的金融文档分析助手。请仔细观察图片并提供准确的描述。

如果是金融研报页面，请识别其结构、内容类型和关键信息。
如果包含图表，请分析图表的类型、数据和趋势。
如果是表格，请识别表格的结构和数据内容。

请特别关注：
- 银行业务数据和分析
- 保险行业相关指标
- 证券市场信息
- 财务数据和统计
- 风险指标和评级
- 企业经营数据

请保持客观和准确，以专业金融分析的视角进行分析。"""

        if context:
            if context.get("document_type"):
                base_prompt += f"\n文档类型：{context['document_type']}"
            if context.get("analysis_focus"):
                base_prompt += f"\n分析重点：{context['analysis_focus']}"
            if context.get("financial_domain"):
                base_prompt += f"\n金融领域：{context['financial_domain']}"

        return base_prompt

    async def extract_text_from_image(self, image: ImageInput) -> str:
        """从图像中提取文本"""
        question = "请提取图片中的所有文字内容，保持原有的格式和结构。"
        result = await self.analyze_image(image, question)
        return result.text_description

    async def understand_document_structure(self, image: ImageInput) -> Dict[str, Any]:
        """理解文档结构"""
        question = """请分析这个文档页面的结构，包括：
1. 标题层级
2. 段落结构
3. 表格识别
4. 图片和图表位置
5. 页眉页脚
请以JSON格式返回结构化信息。"""

        result = await self.analyze_image(image, question)

        # 尝试解析JSON结构
        try:
            structure_info = json.loads(result.text_description)
        except:
            structure_info = {
                "raw_description": result.text_description,
                "entities": result.extracted_entities
            }

        return structure_info

    def _extract_entities_and_relations(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取实体和关系"""
        entities = []
        relations = []

        # 简单的实体提取逻辑
        import re

        # 提取数字、日期、货币等实体
        patterns = {
            "数字": r'\b\d+(?:\.\d+)?\b',
            "日期": r'\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?\b',
            "货币": r'[￥¥$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?',
            "百分比": r'\d+(?:\.\d+)?%'
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": entity_type,
                    "confidence": 0.8
                })

        return entities, relations


class OpenAIVisionModel(BaseVLMModel):
    """OpenAI GPT-4V模型集成（已弃用，仅作为备用）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("gpt-4-vision-preview", config)
        self.api_key = config["api_key"]
        self.api_base = config.get("api_base", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4-vision-preview")

    async def analyze_image(
        self,
        image: ImageInput,
        question: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """使用OpenAI Vision模型分析图像（已弃用）"""
        logger.warning("OpenAI GPT-4V model is deprecated. Using Qwen-VL instead.")

        # 返回弃用提示
        return VLMResult(
            text_description="OpenAI GPT-4V model is deprecated. Please use Qwen-VL model.",
            confidence=0.0,
            extracted_entities=[],
            extracted_relations=[],
            metadata={"error": "deprecated", "suggestion": "Use Qwen-VL model"},
            processing_time=0.0
        )

    async def extract_text_from_image(self, image: ImageInput) -> str:
        """从图像中提取文本"""
        result = await self.analyze_image(image, "请提取图片中的所有文本内容。")
        return result.text_description

    async def understand_document_structure(self, image: ImageInput) -> Dict[str, Any]:
        """理解文档结构"""
        result = await self.analyze_image(image, "请分析这个文档的结构。")
        return {
            "raw_description": result.text_description,
            "entities": result.extracted_entities
        }

    def _build_system_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """构建系统提示"""
        return "OpenAI GPT-4V model is deprecated. Please use Qwen-VL model."

    def _extract_entities_and_relations(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取实体和关系"""
        return [], []


class LocalVLMModel(BaseVLMModel):
    """本地VLM模型集成（支持Transformers）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("local_vlm", config)
        self.model_path = config["model_path"]
        self.device = config.get("device", "cpu")
        self.model = None
        self.processor = None
        self._model_loaded = False

    async def _load_model(self):
        """加载模型"""
        if self._model_loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )

            self._model_loaded = True
            logger.info(f"Local VLM model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load local VLM model: {str(e)}")
            raise

    def _load_model_sync(self):
        """同步加载模型"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self.device
        )

    async def analyze_image(
        self,
        image: ImageInput,
        question: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """使用本地VLM模型分析图像"""
        start_time = datetime.now()

        try:
            await self._load_model()

            # 准备输入
            if image.image_path:
                from PIL import Image as PILImage
                pil_image = PILImage.open(image.image_path).convert('RGB')
            elif image.image_array is not None:
                pil_image = Image.fromarray(image.image_array).convert('RGB')
            elif image.image_bytes:
                pil_image = Image.open(io.BytesIO(image.image_bytes)).convert('RGB')
            else:
                raise ValueError("Invalid image input")

            # 构建提示
            prompt = question or "请描述这张图片的内容"
            if context:
                prompt += f"\n上下文：{context}"

            # 处理输入
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                self.executor,
                lambda: self.processor(prompt, pil_image, return_tensors="pt")
            )

            # 移动到设备
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成响应
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )

            # 解码结果
            text_description = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # 提取实体和关系
            entities, relations = self._extract_entities_and_relations(text_description)

            processing_time = (datetime.now() - start_time).total_seconds()

            return VLMResult(
                text_description=text_description,
                confidence=0.8,
                extracted_entities=entities,
                extracted_relations=relations,
                metadata={
                    "model_path": self.model_path,
                    "device": self.device
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Local VLM analysis failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return VLMResult(
                text_description="",
                confidence=0.0,
                extracted_entities=[],
                extracted_relations=[],
                metadata={"error": str(e)},
                processing_time=processing_time
            )

    async def extract_text_from_image(self, image: ImageInput) -> str:
        """从图像中提取文本"""
        question = "请提取图片中的所有文本。"
        result = await self.analyze_image(image, question)
        return result.text_description

    async def understand_document_structure(self, image: ImageInput) -> Dict[str, Any]:
        """理解文档结构"""
        question = """请分析这个文档的结构，识别各个部分。"""
        result = await self.analyze_image(image, question)

        return {
            "raw_description": result.text_description,
            "entities": result.extracted_entities
        }

    def _extract_entities_and_relations(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """从文本中提取实体和关系"""
        entities = []
        relations = []

        import re

        # 提取实体
        patterns = {
            "number": r'\b\d+(?:\.\d+)?\b',
            "date": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": entity_type,
                    "confidence": 0.7
                })

        return entities, relations


class VLMManager:
    """VLM模型管理器 - 默认使用Qwen-VL"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        # 默认使用Qwen-VL而不是OpenAI
        self.default_model = config.get("default_model", "qwen")
        self.fallback_enabled = config.get("fallback_enabled", True)

        # 初始化模型
        self._initialize_models()

    def _initialize_models(self):
        """初始化VLM模型"""
        # 优先初始化Qwen-VL
        if "qwen" in self.config:
            self.models["qwen"] = QwenVLModel(self.config["qwen"])
            logger.info("Qwen-VL model initialized successfully")

        # OpenAI Vision（已弃用，仅作为备用）
        if "openai" in self.config:
            self.models["openai"] = OpenAIVisionModel(self.config["openai"])
            logger.warning("OpenAI GPT-4V model is deprecated. Qwen-VL is recommended.")

        # 本地模型
        if "local" in self.config:
            self.models["local"] = LocalVLMModel(self.config["local"])

    async def analyze_image(
        self,
        image: ImageInput,
        question: str = "",
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """分析图像"""
        model_name = model or self.default_model

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        vlm_model = self.models[model_name]

        try:
            result = await vlm_model.analyze_image(image, question, context)
            return result
        except Exception as e:
            logger.error(f"Model {model_name} failed: {str(e)}")

            # 尝试使用备用模型
            if self.fallback_enabled and model_name != self.default_model:
                logger.info(f"Falling back to default model {self.default_model}")
                return await self.analyze_image(
                    image, question, self.default_model, context
                )

            raise

    async def batch_analyze_images(
        self,
        images: List[Tuple[ImageInput, str]],
        model: Optional[str] = None,
        batch_size: int = 5
    ) -> List[VLMResult]:
        """批量分析图像"""
        results = []

        # 分批处理
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = await self.models.get(
                model or self.default_model
            ).batch_analyze_images(batch)
            results.extend(batch_results)

            # 添加延迟以避免速率限制
            if i + batch_size < len(images):
                await asyncio.sleep(0.1)

        return results

    async def extract_text_from_document_pages(
        self,
        page_images: List[ImageInput],
        model: Optional[str] = None
    ) -> List[str]:
        """从文档页面批量提取文本"""
        # 构建批次输入
        batch_input = [(img, "请提取页面中的所有文本内容", None) for img in page_images]

        results = await self.batch_analyze_images(
            batch_input,
            model,
            batch_size=3  # 文档处理使用较小的批次
        )

        # 提取文本
        texts = [result.text_description for result in results]
        return texts

    async def understand_document_structure(
        self,
        page_images: List[ImageInput],
        model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """理解多页文档结构"""
        structures = []

        for image in page_images:
            structure = await self.models.get(
                model or self.default_model
            ).understand_document_structure(image)
            structures.append(structure)

        return structures

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.models.keys())

    def close(self):
        """关闭所有模型"""
        for model in self.models.values():
            if hasattr(model, 'executor'):
                model.executor.shutdown(wait=True)


# 便利函数
async def create_vlm_manager(config_path: Optional[str] = None) -> VLMManager:
    """创建VLM管理器"""
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置 - 使用Qwen-VL作为主要模型
        config = {
            "default_model": "qwen",  # 默认使用Qwen-VL
            "fallback_enabled": True,
            "qwen": {
                "api_key": "your-qwen-api-key",
                "model": "qwen-vl-max",
                "max_tokens": 512,
                "temperature": 0.1,
                "max_workers": 4
            },
            "openai": {
                "api_key": "your-openai-api-key",
                "model": "gpt-4-vision-preview",
                "max_tokens": 512,
                "temperature": 0.1,
                "max_workers": 4
            }
        }

    return VLMManager(config)