"""
金融领域语言模型服务
集成多个金融领域预训练模型，提供专业的金融NLP能力
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
import jieba
import jieba.posseg as pseg
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FinancialTaskType(Enum):
    """金融任务类型"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"  # 情感分析
    NER = "ner"  # 命名实体识别
    TEXT_CLASSIFICATION = "text_classification"  # 文本分类
    KEYWORD_EXTRACTION = "keyword_extraction"  # 关键词提取
    SUMMARIZATION = "summarization"  # 摘要生成
    QA = "qa"  # 问答
    RISK_PREDICTION = "risk_prediction"  # 风险预测
    MARKET_PREDICTION = "market_prediction"  # 市场预测


class ModelType(Enum):
    """模型类型"""
    FINBERT = "finbert"  # 金融情感分析模型
    CHINESE_FINBERT = "chinese_finbert"  # 中文金融BERT
    FINNLP = "finnlp"  # 金融NLP模型
    CUSTOM_FINANCIAL = "custom_financial"  # 自定义金融模型


@dataclass
class FinancialEntity:
    """金融实体"""
    entity: str
    entity_type: str  # 公司、股票、人名、地点等
    confidence: float
    start_pos: int
    end_pos: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SentimentResult:
    """情感分析结果"""
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]  # 各情感类型的详细分数


@dataclass
class ModelResult:
    """模型结果"""
    task_type: FinancialTaskType
    model_type: ModelType
    result: Any
    confidence: float
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


class FinancialLLMService:
    """金融领域语言模型服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化金融语言模型服务

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[ModelType, Any] = {}
        self.tokenizers: Dict[ModelType, Any] = {}
        self.pipelines: Dict[str, Any] = {}

        # 模型配置
        self.model_configs = {
            ModelType.FINBERT: {
                "model_name": "yiyanghkust/finbert-tone",
                "task": "sentiment-analysis",
                "max_length": 512
            },
            ModelType.CHINESE_FINBERT: {
                "model_name": "hltc/finbert-chinese",
                "task": "feature-extraction",
                "max_length": 512
            },
            ModelType.FINNLP: {
                "model_name": "chatglm3-6b",  # 如果可用
                "task": "text-generation",
                "max_length": 2048
            }
        }

        # 初始化jieba金融词典
        self._init_financial_jieba()

    def _init_financial_jieba(self):
        """初始化jieba金融词典"""
        try:
            # 添加金融领域词汇
            financial_terms = [
                "年报", "季报", "半年报", "招股说明书", "上市", "退市",
                "并购", "重组", "收购", "兼并", "投资", "融资",
                "股票", "债券", "基金", "期货", "期权", "外汇",
                "市盈率", "市净率", "ROE", "ROA", "毛利率", "净利率",
                "营业收入", "净利润", "资产总额", "负债总额", "现金流",
                "A股", "B股", "H股", "科创板", "创业板", "新三板",
                "上交所", "深交所", "港交所", "纳斯达克", "纽交所",
                "证监会", "银保监会", "央行", "美联储", "欧洲央行",
                "宏观经济", "微观经济", "货币政策", "财政政策",
                "牛市", "熊市", "震荡市", "涨停", "跌停", "停牌"
            ]

            for term in financial_terms:
                jieba.add_word(term)

            logger.info("金融词典初始化完成")

        except Exception as e:
            logger.warning(f"金融词典初始化失败: {str(e)}")

    async def load_model(self, model_type: ModelType) -> bool:
        """
        加载指定模型

        Args:
            model_type: 模型类型

        Returns:
            是否加载成功
        """
        try:
            if model_type in self.models:
                logger.info(f"模型已加载: {model_type.value}")
                return True

            config = self.model_configs.get(model_type)
            if not config:
                logger.error(f"未找到模型配置: {model_type.value}")
                return False

            logger.info(f"开始加载模型: {model_type.value}")

            if model_type == ModelType.FINBERT:
                await self._load_finbert()
            elif model_type == ModelType.CHINESE_FINBERT:
                await self._load_chinese_finbert()
            elif model_type == ModelType.FINNLP:
                await self._load_finnlp()
            else:
                logger.error(f"不支持的模型类型: {model_type.value}")
                return False

            logger.info(f"模型加载成功: {model_type.value}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败 {model_type.value}: {str(e)}")
            return False

    async def _load_finbert(self):
        """加载FinBERT模型"""
        try:
            model_name = self.model_configs[ModelType.FINBERT]["model_name"]

            # 加载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            # 创建pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )

            self.models[ModelType.FINBERT] = model
            self.tokenizers[ModelType.FINBERT] = tokenizer
            self.pipelines["sentiment_analysis"] = sentiment_pipeline

        except Exception as e:
            logger.error(f"FinBERT加载失败: {str(e)}")
            raise

    async def _load_chinese_finbert(self):
        """加载中文金融BERT模型"""
        try:
            model_name = self.model_configs[ModelType.CHINESE_FINBERT]["model_name"]

            # 加载tokenizer和模型
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            # 创建句子嵌入模型
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            self.models[ModelType.CHINESE_FINBERT] = model
            self.tokenizers[ModelType.CHINESE_FINBERT] = tokenizer
            self.models["chinese_embedding"] = embedding_model

        except Exception as e:
            logger.error(f"中文金融BERT加载失败: {str(e)}")
            raise

    async def _load_finnlp(self):
        """加载FinNLP模型（如果可用）"""
        try:
            # 这里可以加载更复杂的金融NLP模型
            # 由于资源限制，这里使用轻量级替代方案
            logger.info("使用轻量级金融NLP模型")

            # 创建通用文本生成pipeline
            generator = pipeline(
                "text2text-generation",
                model="t5-small",
                device=0 if self.device.type == "cuda" else -1
            )

            self.pipelines["text_generation"] = generator

        except Exception as e:
            logger.error(f"FinNLP加载失败: {str(e)}")
            raise

    async def analyze_sentiment(self, text: str, model_type: ModelType = ModelType.FINBERT) -> ModelResult:
        """
        金融情感分析

        Args:
            text: 输入文本
            model_type: 使用的模型类型

        Returns:
            情感分析结果
        """
        import time
        start_time = time.time()

        try:
            # 确保模型已加载
            if not await self.load_model(model_type):
                raise RuntimeError(f"模型加载失败: {model_type.value}")

            if model_type == ModelType.FINBERT:
                result = await self._sentiment_analysis_finbert(text)
            else:
                result = await self._sentiment_analysis_chinese(text)

            processing_time = time.time() - start_time

            return ModelResult(
                task_type=FinancialTaskType.SENTIMENT_ANALYSIS,
                model_type=model_type,
                result=result,
                confidence=result.confidence,
                processing_time=processing_time,
                metadata={"text_length": len(text)}
            )

        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return ModelResult(
                task_type=FinancialTaskType.SENTIMENT_ANALYSIS,
                model_type=model_type,
                result=SentimentResult("neutral", 0.0, {}),
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _sentiment_analysis_finbert(self, text: str) -> SentimentResult:
        """使用FinBERT进行情感分析"""
        try:
            pipeline = self.pipelines.get("sentiment_analysis")
            if not pipeline:
                raise RuntimeError("情感分析pipeline未初始化")

            # 预处理文本
            text = text[:512]  # 限制长度

            # 执行推理
            result = pipeline(text)[0]

            # 映射标签
            label_mapping = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral",
                "POSITIVE": "positive"
            }

            sentiment = label_mapping.get(result["label"], "neutral")
            confidence = result["score"]

            # 构建详细分数
            scores = {
                "positive": confidence if sentiment == "positive" else 0.0,
                "negative": confidence if sentiment == "negative" else 0.0,
                "neutral": confidence if sentiment == "neutral" else 0.0
            }

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )

        except Exception as e:
            logger.error(f"FinBERT情感分析失败: {str(e)}")
            return SentimentResult("neutral", 0.0, {})

    async def _sentiment_analysis_chinese(self, text: str) -> SentimentResult:
        """使用中文模型进行情感分析"""
        try:
            # 简单的基于词典的情感分析
            positive_words = ["上涨", "增长", "盈利", "利好", "乐观", "强势", "突破"]
            negative_words = ["下跌", "亏损", "利空", "悲观", "弱势", "跌破", "风险"]

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count > neg_count:
                sentiment = "positive"
                confidence = min(0.8, pos_count / (pos_count + neg_count + 1))
            elif neg_count > pos_count:
                sentiment = "negative"
                confidence = min(0.8, neg_count / (pos_count + neg_count + 1))
            else:
                sentiment = "neutral"
                confidence = 0.5

            scores = {
                "positive": confidence if sentiment == "positive" else 0.1,
                "negative": confidence if sentiment == "negative" else 0.1,
                "neutral": 0.8 if sentiment == "neutral" else 0.1
            }

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )

        except Exception as e:
            logger.error(f"中文情感分析失败: {str(e)}")
            return SentimentResult("neutral", 0.0, {})

    async def extract_entities(self, text: str) -> ModelResult:
        """
        金融实体识别

        Args:
            text: 输入文本

        Returns:
            实体识别结果
        """
        import time
        start_time = time.time()

        try:
            entities = await self._extract_financial_entities(text)
            processing_time = time.time() - start_time

            return ModelResult(
                task_type=FinancialTaskType.NER,
                model_type=ModelType.CHINESE_FINBERT,
                result=entities,
                confidence=0.8,
                processing_time=processing_time,
                metadata={"entity_count": len(entities)}
            )

        except Exception as e:
            logger.error(f"实体识别失败: {str(e)}")
            return ModelResult(
                task_type=FinancialTaskType.NER,
                model_type=ModelType.CHINESE_FINBERT,
                result=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _extract_financial_entities(self, text: str) -> List[FinancialEntity]:
        """提取金融实体"""
        entities = []

        try:
            # 使用jieba进行分词和词性标注
            words = pseg.cut(text)

            # 金融实体模式
            patterns = {
                "COMPANY": [r"[\u4e00-\u9fff]+公司", r"[\u4e00-\u9fff]+集团", r"[\u4e00-\u9fff]+股份"],
                "STOCK": [r"[0-9]{6}", r"[0-9]{4}\.[A-Z]{2}", r"[A-Z]{1,4}"],
                "PERSON": [r"[\u4e00-\u9fff]{2,4}(先生|女士|董事长|CEO|总裁|总经理)"],
                "AMOUNT": [r"[0-9,]+\.?[0-9]*万", r"[0-9,]+\.?[0-9]*亿", r"[0-9,]+\.?[0-9]*元"],
                "TIME": [r"[0-9]{4}年", r"[0-9]{1,2}月", r"[0-9]{1,2}日", r"今年|去年|明年"]
            }

            import re

            for word, flag in words:
                start_pos = text.find(word)
                if start_pos == -1:
                    continue

                end_pos = start_pos + len(word)

                # 检查各种实体类型
                if flag in ["nr", "nt"]:  # 人名、机构名
                    entity_type = "PERSON" if flag == "nr" else "ORGANIZATION"
                    entities.append(FinancialEntity(
                        entity=word,
                        entity_type=entity_type,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=end_pos
                    ))

                # 使用正则表达式进行模式匹配
                for entity_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if re.match(pattern, word):
                            entities.append(FinancialEntity(
                                entity=word,
                                entity_type=entity_type,
                                confidence=0.7,
                                start_pos=start_pos,
                                end_pos=end_pos
                            ))

            # 去重
            entities = list({(e.entity, e.start_pos): e for e in entities}.values())
            entities.sort(key=lambda x: x.start_pos)

        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")

        return entities

    async def extract_keywords(self, text: str, top_k: int = 10) -> ModelResult:
        """
        关键词提取

        Args:
            text: 输入文本
            top_k: 返回关键词数量

        Returns:
            关键词提取结果
        """
        import time
        start_time = time.time()

        try:
            keywords = await self._extract_financial_keywords(text, top_k)
            processing_time = time.time() - start_time

            return ModelResult(
                task_type=FinancialTaskType.KEYWORD_EXTRACTION,
                model_type=ModelType.CHINESE_FINBERT,
                result=keywords,
                confidence=0.7,
                processing_time=processing_time,
                metadata={"keyword_count": len(keywords)}
            )

        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return ModelResult(
                task_type=FinancialTaskType.KEYWORD_EXTRACTION,
                model_type=ModelType.CHINESE_FINBERT,
                result=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _extract_financial_keywords(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """提取金融关键词"""
        keywords = []

        try:
            # 使用TF-IDF算法提取关键词
            from sklearn.feature_extraction.text import TfidfVectorizer
            import jieba.analyse

            # 使用jieba的TF-IDF关键词提取
            jieba_keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)

            # 过滤和排序
            financial_keywords = []
            for keyword, weight in jieba_keywords:
                # 检查是否为金融相关词汇
                if self._is_financial_term(keyword):
                    financial_keywords.append((keyword, weight))

            # 如果提取的金融关键词不足，补充其他关键词
            if len(financial_keywords) < top_k:
                remaining = top_k - len(financial_keywords)
                for keyword, weight in jieba_keywords:
                    if keyword not in [kw for kw, _ in financial_keywords]:
                        financial_keywords.append((keyword, weight))
                        remaining -= 1
                        if remaining <= 0:
                            break

            keywords = financial_keywords[:top_k]

        except Exception as e:
            logger.error(f"金融关键词提取失败: {str(e)}")

        return keywords

    def _is_financial_term(self, term: str) -> bool:
        """判断是否为金融术语"""
        financial_domains = [
            "股票", "债券", "基金", "期货", "期权", "外汇", "黄金",
            "银行", "保险", "证券", "信托", "租赁", "投资", "融资",
            "并购", "重组", "上市", "退市", "分红", "配股", "增发",
            "宏观经济", "货币政策", "财政政策", "利率", "汇率",
            "营业收入", "净利润", "资产", "负债", "现金流", "ROE", "ROA"
        ]

        return any(domain in term for domain in financial_domains)

    async def generate_summary(self, text: str, max_length: int = 150) -> ModelResult:
        """
        生成金融摘要

        Args:
            text: 输入文本
            max_length: 摘要最大长度

        Returns:
            摘要生成结果
        """
        import time
        start_time = time.time()

        try:
            summary = await self._generate_financial_summary(text, max_length)
            processing_time = time.time() - start_time

            return ModelResult(
                task_type=FinancialTaskType.SUMMARIZATION,
                model_type=ModelType.FINNLP,
                result=summary,
                confidence=0.7,
                processing_time=processing_time,
                metadata={"original_length": len(text), "summary_length": len(summary)}
            )

        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return ModelResult(
                task_type=FinancialTaskType.SUMMARIZATION,
                model_type=ModelType.FINNLP,
                result="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _generate_financial_summary(self, text: str, max_length: int) -> str:
        """生成金融摘要"""
        try:
            # 使用TextRank算法进行抽取式摘要
            sentences = text.split('。')
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 3:
                return text[:max_length] + "..." if len(text) > max_length else text

            # 计算句子重要性
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0

                # 包含金融关键词的句子得分更高
                for keyword in ["业绩", "收入", "利润", "增长", "下降", "投资", "风险"]:
                    if keyword in sentence:
                        score += 1

                # 包含数字的句子得分更高
                import re
                if re.search(r'\d+', sentence):
                    score += 0.5

                # 长度适中的句子得分更高
                if 20 <= len(sentence) <= 100:
                    score += 0.5

                sentence_scores.append((i, score, sentence))

            # 按得分排序并选择前几个句子
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sorted(sentence_scores[:3], key=lambda x: x[0])

            summary = '。'.join([s[2] for s in top_sentences])

            # 限制长度
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."

            return summary

        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text

    async def get_embeddings(self, texts: List[str]) -> ModelResult:
        """
        获取文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量结果
        """
        import time
        start_time = time.time()

        try:
            # 确保中文模型已加载
            await self.load_model(ModelType.CHINESE_FINBERT)

            embedding_model = self.models.get("chinese_embedding")
            if not embedding_model:
                raise RuntimeError("嵌入模型未加载")

            # 生成嵌入
            embeddings = embedding_model.encode(texts, convert_to_tensor=True)
            embeddings_list = embeddings.cpu().numpy().tolist()

            processing_time = time.time() - start_time

            return ModelResult(
                task_type=FinancialTaskType.TEXT_CLASSIFICATION,  # 使用现有类型
                model_type=ModelType.CHINESE_FINBERT,
                result=embeddings_list,
                confidence=1.0,
                processing_time=processing_time,
                metadata={
                    "text_count": len(texts),
                    "embedding_dim": len(embeddings_list[0]) if embeddings_list else 0
                }
            )

        except Exception as e:
            logger.error(f"嵌入生成失败: {str(e)}")
            return ModelResult(
                task_type=FinancialTaskType.TEXT_CLASSIFICATION,
                model_type=ModelType.CHINESE_FINBERT,
                result=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        status = {
            "device": str(self.device),
            "loaded_models": [],
            "available_pipelines": list(self.pipelines.keys()),
            "model_configs": {k.value: v for k, v in self.model_configs.items()}
        }

        for model_type in ModelType:
            if model_type in self.models:
                status["loaded_models"].append(model_type.value)

        return status

    async def unload_model(self, model_type: ModelType):
        """卸载模型以释放内存"""
        try:
            if model_type in self.models:
                del self.models[model_type]
                logger.info(f"模型已卸载: {model_type.value}")

            if model_type in self.tokenizers:
                del self.tokenizers[model_type]

            # 清理相关pipeline
            pipelines_to_remove = []
            for pipeline_name, pipeline in self.pipelines.items():
                if model_type.value in str(type(pipeline)):
                    pipelines_to_remove.append(pipeline_name)

            for pipeline_name in pipelines_to_remove:
                del self.pipelines[pipeline_name]

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"模型卸载失败: {str(e)}")


# 全局金融语言模型服务实例
financial_llm_service = FinancialLLMService()