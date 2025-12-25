"""
文本模型实现
集成各种先进的文本处理模型
"""

import logging
from typing import Dict, Any, Optional, List, Union
import asyncio

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM, pipeline
    )
    TEXT_MODELS_AVAILABLE = True
except ImportError:
    TEXT_MODELS_AVAILABLE = False
    logging.warning("Transformers未安装，文本模型功能不可用")

from .base_model import (
    BaseVisionTextModel,
    ModelType,
    TaskType,
    ModelInput,
    ModelOutput
)

logger = logging.getLogger(__name__)


class TextModel(BaseVisionTextModel):
    """文本模型"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not TEXT_MODELS_AVAILABLE:
            raise ImportError("需要安装Transformers: pip install transformers")

        super().__init__(config)
        self.model_type = ModelType.TEXT

        # 模型配置
        self.model_configs = {
            'classification': self.config.get('classification_model', 'bert-base-chinese'),
            'ner': self.config.get('ner_model', 'ckiplab/bert-base-chinese-ner'),
            'qa': self.config.get('qa_model', 'deepset/roberta-base-squad2'),
            'summarization': self.config.get('summarization_model', 'facebook/bart-large-cnn'),
            'sentiment': self.config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        }

        # 模型和分词器实例
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}

    def _get_model_type(self) -> ModelType:
        return ModelType.TEXT

    async def load_model(self) -> bool:
        """加载文本模型"""
        try:
            # 加载分类模型
            if TaskType.TEXT_CLASSIFICATION in self.get_supported_tasks():
                await self._load_classification_model()

            # 加载NER模型
            if TaskType.NAMED_ENTITY_RECOGNITION in self.get_supported_tasks():
                await self._load_ner_model()

            # 加载问答模型
            if TaskType.QUESTION_ANSWERING in self.get_supported_tasks():
                await self._load_qa_model()

            # 加载摘要模型
            if TaskType.TEXT_SUMMARIZATION in self.get_supported_tasks():
                await self._load_summarization_model()

            # 加载情感分析模型
            if TaskType.SENTIMENT_ANALYSIS in self.get_supported_tasks():
                await self._load_sentiment_model()

            self.is_loaded = True
            logger.info("文本模型加载成功")
            return True

        except Exception as e:
            logger.error(f"文本模型加载失败: {str(e)}")
            return False

    async def _load_classification_model(self):
        """加载文本分类模型"""
        model_name = self.model_configs['classification']
        logger.info(f"加载分类模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.tokenizers['classification'] = tokenizer
        self.models['classification'] = model

        # 创建pipeline
        self.pipelines['classification'] = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device != 'cpu' else -1
        )

    async def _load_ner_model(self):
        """加载命名实体识别模型"""
        model_name = self.model_configs['ner']
        logger.info(f"加载NER模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.tokenizers['ner'] = tokenizer
        self.models['ner'] = model

        # 创建pipeline
        self.pipelines['ner'] = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if self.device != 'cpu' else -1
        )

    async def _load_qa_model(self):
        """加载问答模型"""
        model_name = self.model_configs['qa']
        logger.info(f"加载问答模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.tokenizers['qa'] = tokenizer
        self.models['qa'] = model

        # 创建pipeline
        self.pipelines['qa'] = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device != 'cpu' else -1
        )

    async def _load_summarization_model(self):
        """加载摘要模型"""
        model_name = self.model_configs['summarization']
        logger.info(f"加载摘要模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.tokenizers['summarization'] = tokenizer
        self.models['summarization'] = model

        # 创建pipeline
        self.pipelines['summarization'] = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device != 'cpu' else -1
        )

    async def _load_sentiment_model(self):
        """加载情感分析模型"""
        model_name = self.model_configs['sentiment']
        logger.info(f"加载情感分析模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if self.device != 'cpu':
            model = model.to(self.device)

        self.tokenizers['sentiment'] = tokenizer
        self.models['sentiment'] = model

        # 创建pipeline
        self.pipelines['sentiment'] = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device != 'cpu' else -1
        )

    async def unload_model(self) -> bool:
        """卸载模型"""
        try:
            # 清理所有模型
            for model in self.models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model

            self.models.clear()
            self.tokenizers.clear()
            self.pipelines.clear()

            # 清理PyTorch缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("文本模型卸载成功")
            return True

        except Exception as e:
            logger.error(f"文本模型卸载失败: {str(e)}")
            return False

    def get_supported_tasks(self) -> List[TaskType]:
        """获取支持的任务类型"""
        return [
            TaskType.TEXT_CLASSIFICATION,
            TaskType.NAMED_ENTITY_RECOGNITION,
            TaskType.RELATION_EXTRACTION,
            TaskType.TEXT_SUMMARIZATION,
            TaskType.QUESTION_ANSWERING,
            TaskType.SENTIMENT_ANALYSIS
        ]

    async def process(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """处理文本输入"""
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
        if task_type == TaskType.TEXT_CLASSIFICATION:
            outputs = await self._classify_text(preprocessed)
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            outputs = await self._extract_entities(preprocessed)
        elif task_type == TaskType.QUESTION_ANSWERING:
            question = kwargs.get('question', '')
            context = kwargs.get('context', '')
            outputs = await self._answer_question(preprocessed, question, context)
        elif task_type == TaskType.TEXT_SUMMARIZATION:
            outputs = await self._summarize_text(preprocessed)
        elif task_type == TaskType.SENTIMENT_ANALYSIS:
            outputs = await self._analyze_sentiment(preprocessed)
        else:
            outputs = await self._generic_process(preprocessed, task_type, **kwargs)

        # 后处理
        if isinstance(inputs, list):
            return [await self.postprocess(out, task_type, **kwargs) for out in outputs]
        else:
            return await self.postprocess(outputs, task_type, **kwargs)

    async def _preprocess_single(self, input_data: ModelInput, task_type: TaskType) -> Any:
        """预处理单个输入"""
        text = input_data.data

        # 基本文本清理
        if isinstance(text, str):
            text = text.strip()

        # 根据任务类型进行特定预处理
        if task_type == TaskType.QUESTION_ANSWERING:
            # QA需要问题和上下文
            question = input_data.metadata.get('question', '') if input_data.metadata else ''
            context = input_data.metadata.get('context', '') if input_data.metadata else ''
            return {'question': question, 'context': context}
        else:
            return text

    async def _postprocess_single(self, output: Any, task_type: TaskType, **kwargs) -> Dict[str, Any]:
        """后处理单个输出"""
        if isinstance(output, dict):
            return output
        elif isinstance(output, list):
            return {'results': output}
        else:
            return {'output': str(output)}

    async def _classify_text(self, inputs: Any) -> Any:
        """文本分类"""
        if 'classification' in self.pipelines:
            pipeline = self.pipelines['classification']
            if isinstance(inputs, str):
                return pipeline(inputs)
            elif isinstance(inputs, list):
                return pipeline(inputs)
        return {"label": "unknown", "score": 0.0}

    async def _extract_entities(self, inputs: Any) -> Any:
        """命名实体识别"""
        if 'ner' in self.pipelines:
            pipeline = self.pipelines['ner']
            if isinstance(inputs, str):
                return pipeline(inputs)
            elif isinstance(inputs, list):
                return pipeline(inputs)
        return []

    async def _answer_question(self, inputs: Any, question: str, context: str) -> Any:
        """问答"""
        if 'qa' in self.pipelines and question and context:
            pipeline = self.pipelines['qa']
            return pipeline(question=question, context=context)
        return {"answer": "无法回答", "score": 0.0}

    async def _summarize_text(self, inputs: Any) -> Any:
        """文本摘要"""
        if 'summarization' in self.pipelines:
            pipeline = self.pipelines['summarization']
            if isinstance(inputs, str):
                return pipeline(inputs, max_length=150, min_length=30, do_sample=False)
            elif isinstance(inputs, list):
                text = ' '.join(inputs)
                return pipeline(text, max_length=150, min_length=30, do_sample=False)
        return {"summary_text": ""}

    async def _analyze_sentiment(self, inputs: Any) -> Any:
        """情感分析"""
        if 'sentiment' in self.pipelines:
            pipeline = self.pipelines['sentiment']
            if isinstance(inputs, str):
                return pipeline(inputs)
            elif isinstance(inputs, list):
                return pipeline(inputs)
        return {"label": "neutral", "score": 0.0}

    async def _generic_process(self, inputs: Any, task_type: TaskType, **kwargs) -> Any:
        """通用处理"""
        return inputs