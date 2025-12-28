"""
Qwen嵌入和重排序模型服务
集成阿里云的Qwen2.5-VL-Embedding和Text-Embedding-V4模型
使用dashscope SDK进行API调用
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import httpx
import json

try:
    import dashscope
    from dashscope import MultiModalEmbedding, TextEmbedding, Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("dashscope SDK not installed, falling back to HTTP API")

from app.core.config import settings
from app.core.vector_validator import VectorQualityValidator, validate_embeddings
from app.core.vector_config import get_dimension

logger = logging.getLogger(__name__)


@dataclass
class QwenEmbeddingConfig:
    """Qwen嵌入模型配置"""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    primary_embedding_model: str = "qwen2.5-vl-embedding"
    backup_embedding_model: str = "text-embedding-v4"
    reranker_model: str = "qwen3-rerank"
    timeout: int = 60
    max_retries: int = 5  # 增加重试次数
    batch_size: int = 8  # 减小批量大小以避免500错误
    normalize_embeddings: bool = True
    batch_delay: float = 0.1  # 批量之间的延迟(秒)

    # 缓存配置
    cache_size: int = 1000
    enable_cache: bool = True

    # 向量质量验证配置
    enable_validation: bool = True  # 启用向量质量验证
    skip_invalid_embeddings: bool = True  # 跳过无效向量而非返回零向量
    raise_on_all_failed: bool = True  # 如果所有向量都失败，抛出异常


class QwenEmbeddingService:
    """Qwen嵌入和重排序模型服务类"""

    def __init__(self, config: Optional[QwenEmbeddingConfig] = None):
        # 从settings读取配置或使用默认值
        if config is None:
            config = QwenEmbeddingConfig(
                api_key="sk-5233a3a4b1a24426b6846a432794bbe2",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                primary_embedding_model="qwen2.5-vl-embedding",
                backup_embedding_model="text-embedding-v4",
                reranker_model="qwen3-rerank",
                batch_size=8,  # 减小批量大小
                normalize_embeddings=True,
                cache_size=1000,
                enable_cache=True,
                enable_validation=True,
                skip_invalid_embeddings=True,
                raise_on_all_failed=True
            )

        self.config = config
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.primary_model_failures = 0
        self.max_primary_failures = 5

        # 初始化向量质量验证器
        if self.config.enable_validation:
            self.vector_validator = VectorQualityValidator(
                expected_dim=get_dimension(self.config.primary_embedding_model),
                check_zero_vector=True,
                check_nan=True,
                check_inf=True,
                check_dimension=True
            )
            logger.info("Vector quality validation enabled")
        else:
            self.vector_validator = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _call_multimodal_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """使用dashscope MultiModalEmbedding API - 批量处理优化"""
        if not DASHSCOPE_AVAILABLE:
            return await self._fallback_embedding_request("qwen2.5-vl-embedding", texts)

        try:
            dashscope.api_key = self.config.api_key

            # 批量处理: 将所有文本合并到一个API调用中
            batch_size = self.config.batch_size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_success = False

                # 重试逻辑 - 特别针对500错误
                for attempt in range(self.config.max_retries):
                    try:
                        # 构建批量输入
                        input_data = [{'text': text} for text in batch_texts]

                        # 在线程池中执行同步调用
                        loop = asyncio.get_event_loop()
                        resp = await loop.run_in_executor(
                            None,
                            lambda: MultiModalEmbedding.call(
                                model="qwen2.5-vl-embedding",
                                input=input_data
                            )
                        )

                        if resp.status_code == 200:
                            # 批量提取所有嵌入向量
                            batch_embeddings = [
                                np.array(emb['embedding'], dtype=np.float32)
                                for emb in resp.output['embeddings']
                            ]
                            all_embeddings.extend(batch_embeddings)
                            batch_success = True
                            break
                        else:
                            # 检查是否是500错误
                            if resp.status_code == 500:
                                error_msg = resp.get('message', '')
                                if 'batching backend' in error_msg.lower():
                                    logger.warning(f"500批量错误 (尝试 {attempt + 1}/{self.config.max_retries}): 减小批次大小重试")
                                    # 500错误时等待更长时间
                                    await asyncio.sleep(2 ** attempt)
                                    # 如果是第一次失败,尝试减小批次
                                    if attempt == 0 and len(batch_texts) > 1:
                                        logger.info(f"将批次从 {len(batch_texts)} 减小到 {max(1, len(batch_texts) // 2)}")
                                        # 递归处理更小的批次
                                        half = len(batch_texts) // 2
                                        all_embeddings.extend(await self._call_multimodal_embedding(batch_texts[:half]))
                                        all_embeddings.extend(await self._call_multimodal_embedding(batch_texts[half:]))
                                        batch_success = True
                                        break
                                    continue
                            else:
                                error_msg = resp.get('message', 'Unknown error')
                                logger.error(f"MultiModalEmbedding API error: {resp}")
                                # 不再返回零向量，而是让重试机制处理
                                if attempt == self.config.max_retries - 1:
                                    # 最后一次尝试失败，抛出异常
                                    raise RuntimeError(
                                        f"MultiModalEmbedding API failed after {self.config.max_retries} attempts: "
                                        f"status={resp.status_code}, error={error_msg}"
                                    )
                                continue

                    except Exception as e:
                        logger.error(f"MultiModalEmbedding API调用异常 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(1 * (attempt + 1))  # 递增延迟
                            continue
                        else:
                            # 最后一次尝试失败，抛出异常
                            raise RuntimeError(
                                f"MultiModalEmbedding API failed after {self.config.max_retries} attempts: {str(e)}"
                            ) from e

                # 批次之间添加延迟以避免触发速率限制
                if i + batch_size < len(texts) and self.config.batch_delay > 0:
                    await asyncio.sleep(self.config.batch_delay)

            return all_embeddings

        except Exception as e:
            logger.error(f"MultiModalEmbedding API call failed: {e}")
            return await self._fallback_embedding_request("qwen2.5-vl-embedding", texts)

    async def _call_text_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """使用dashscope TextEmbedding API - 批量处理优化"""
        if not DASHSCOPE_AVAILABLE:
            return await self._fallback_embedding_request("text-embedding-v4", texts)

        try:
            dashscope.api_key = self.config.api_key

            # 批量处理: 将所有文本合并到一个API调用中
            batch_size = self.config.batch_size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_success = False

                # 重试逻辑 - 特别针对500错误
                for attempt in range(self.config.max_retries):
                    try:
                        # 在线程池中执行同步调用 - 批量发送
                        loop = asyncio.get_event_loop()
                        resp = await loop.run_in_executor(
                            None,
                            lambda: TextEmbedding.call(
                                model="text-embedding-v4",
                                input=batch_texts  # 发送整个批次
                            )
                        )

                        if resp.status_code == 200:
                            # 批量提取所有嵌入向量
                            batch_embeddings = [
                                np.array(emb['embedding'], dtype=np.float32)
                                for emb in resp.output['embeddings']
                            ]
                            all_embeddings.extend(batch_embeddings)
                            batch_success = True
                            break
                        else:
                            # 检查是否是500错误
                            if resp.status_code == 500:
                                error_msg = resp.get('message', '')
                                if 'batching backend' in error_msg.lower():
                                    logger.warning(f"TextEmbedding 500批量错误 (尝试 {attempt + 1}/{self.config.max_retries}): 减小批次大小重试")
                                    # 500错误时等待更长时间
                                    await asyncio.sleep(2 ** attempt)
                                    # 如果是第一次失败,尝试减小批次
                                    if attempt == 0 and len(batch_texts) > 1:
                                        logger.info(f"将批次从 {len(batch_texts)} 减小到 {max(1, len(batch_texts) // 2)}")
                                        # 递归处理更小的批次
                                        half = len(batch_texts) // 2
                                        all_embeddings.extend(await self._call_text_embedding(batch_texts[:half]))
                                        all_embeddings.extend(await self._call_text_embedding(batch_texts[half:]))
                                        batch_success = True
                                        break
                                    continue
                            else:
                                error_msg = resp.get('message', 'Unknown error')
                                logger.error(f"TextEmbedding API error: {resp}")
                                # 不再返回零向量
                                if attempt == self.config.max_retries - 1:
                                    raise RuntimeError(
                                        f"TextEmbedding API failed after {self.config.max_retries} attempts: "
                                        f"status={resp.status_code}, error={error_msg}"
                                    )
                                continue

                    except Exception as e:
                        logger.error(f"TextEmbedding API调用异常 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(1 * (attempt + 1))  # 递增延迟
                            continue
                        else:
                            raise RuntimeError(
                                f"TextEmbedding API failed after {self.config.max_retries} attempts: {str(e)}"
                            ) from e

                # 批次之间添加延迟以避免触发速率限制
                if i + batch_size < len(texts) and self.config.batch_delay > 0:
                    await asyncio.sleep(self.config.batch_delay)

            return all_embeddings

        except Exception as e:
            logger.error(f"TextEmbedding API call failed: {e}")
            return await self._fallback_embedding_request("text-embedding-v4", texts)

    async def _fallback_embedding_request(self, model: str, texts: List[str]) -> List[np.ndarray]:
        """HTTP API fallback for embedding requests - 批量处理优化"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        # 批量处理: 将所有文本合并到一个HTTP请求中
        batch_size = self.config.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 批量请求格式
            payload = {
                "model": model,
                "input": batch_texts,  # 发送整个批次
                "encoding_format": "float"
            }

            batch_success = False
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.post(
                        "embeddings",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "data" in result and len(result["data"]) > 0:
                        # 批量提取所有嵌入向量
                        batch_embeddings = [
                            np.array(item["embedding"], dtype=np.float32)
                            for item in result["data"]
                        ]
                        all_embeddings.extend(batch_embeddings)
                        batch_success = True
                        break

                except httpx.HTTPStatusError as e:
                    # 检查是否是500错误
                    if e.response.status_code == 500:
                        logger.warning(f"HTTP fallback 500错误 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                        # 如果是第一次失败且批次大于1,尝试减小批次
                        if attempt == 0 and len(batch_texts) > 1:
                            logger.info(f"将批次从 {len(batch_texts)} 减小到 {max(1, len(batch_texts) // 2)}")
                            half = len(batch_texts) // 2
                            all_embeddings.extend(await self._fallback_embedding_request(model, batch_texts[:half]))
                            all_embeddings.extend(await self._fallback_embedding_request(model, batch_texts[half:]))
                            batch_success = True
                            break
                        await asyncio.sleep(2 ** attempt)
                        continue
                    elif e.response.status_code == 429:  # Rate limit
                        logger.warning(f"HTTP fallback 速率限制 (尝试 {attempt + 1}/{self.config.max_retries})")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Qwen embedding API HTTP error (attempt {attempt + 1}): {e}")
                        raise

                except Exception as e:
                    logger.error(f"Qwen embedding API request error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    # 不再返回零向量，抛出异常
                    raise RuntimeError(
                        f"HTTP fallback failed after {self.config.max_retries} attempts: {str(e)}"
                    ) from e

            # 批次之间添加延迟
            if i + batch_size < len(texts) and self.config.batch_delay > 0:
                await asyncio.sleep(self.config.batch_delay)

        return all_embeddings

    async def _call_qwen_rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """使用dashscope qwen3-rerank API"""
        if top_k is None:
            top_k = len(documents)

        if DASHSCOPE_AVAILABLE:
            try:
                dashscope.api_key = self.config.api_key

                # 构建请求参数
                rerank_data = {
                    "model": "qwen3-rerank",
                    "input": {
                        "query": query,
                        "documents": documents
                    },
                    "parameters": {
                        "return_documents": False,
                        "top_n": top_k,
                        "instruct": "Given a web search query, retrieve relevant passages that answer the query."
                    }
                }

                # 在线程池中执行同步调用
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: dashscope.TextReRank.call(
                        model=rerank_data["model"],
                        query=rerank_data["input"]["query"],
                        documents=rerank_data["input"]["documents"],
                        top_n=rerank_data["parameters"]["top_n"],
                        return_documents=rerank_data["parameters"]["return_documents"],
                        instruct=rerank_data["parameters"]["instruct"]
                    )
                )

                if resp.status_code == 200:
                    results = []
                    for result in resp.output.results:
                        # 转换为 (index, score) 格式
                        doc_index = result.index
                        relevance_score = result.relevance_score
                        results.append((doc_index, relevance_score))
                    return results
                else:
                    logger.error(f"Qwen Rerank API error: {resp}")
                    return [(i, 1.0) for i in range(len(documents))]  # 返回原始顺序

            except Exception as e:
                logger.error(f"Qwen Rerank SDK call failed: {e}")
                # fallback to HTTP API
                return await self._fallback_rerank_request(query, documents, top_k)
        else:
            # fallback to HTTP API
            return await self._fallback_rerank_request(query, documents, top_k)

    async def _fallback_rerank_request(self, query: str, documents: List[str], top_k: int) -> List[Tuple[int, float]]:
        """HTTP API fallback for reranking requests"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "qwen3-rerank",
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "return_documents": False,
                "top_n": top_k,
                "instruct": "Given a web search query, retrieve relevant passages that answer the query."
            }
        }

        for attempt in range(self.config.max_retries):
            try:
                # 使用正确的API端点
                response = await self.client.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                if "output" in result and "results" in result["output"]:
                    rerank_results = []
                    for item in result["output"]["results"]:
                        doc_index = item["index"]
                        relevance_score = item["relevance_score"]
                        rerank_results.append((doc_index, relevance_score))
                    return rerank_results

            except httpx.HTTPStatusError as e:
                logger.error(f"Qwen reranking HTTP error (attempt {attempt + 1}): {e}")
                if e.response.status_code == 429:  # Rate limit
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

            except Exception as e:
                logger.error(f"Qwen reranking HTTP request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise

        # 如果所有尝试都失败，返回原始顺序
        return [(i, 1.0) for i in range(len(documents))]

    def _get_cache_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        return f"{model}:{hash(text)}"

    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从缓存中获取嵌入"""
        if not self.config.enable_cache:
            return None

        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key].copy()

        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """保存嵌入到缓存"""
        if not self.config.enable_cache:
            return

        if len(self.embedding_cache) >= self.config.cache_size:
            # 简单的LRU策略：删除第一个元素
            first_key = next(iter(self.embedding_cache))
            del self.embedding_cache[first_key]

        self.embedding_cache[cache_key] = embedding.copy()
        self.cache_misses += 1

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """归一化嵌入向量"""
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    async def encode(self, texts: List[str], use_backup: bool = False) -> List[np.ndarray]:
        """编码文本为嵌入向量"""
        if not texts:
            return []

        # 批量处理
        results = []
        use_multimodal = not use_backup  # 主模型使用multimodal，备用模型使用text embedding

        # 检查缓存
        cached_results = {}
        uncached_texts = []
        uncached_indices = []

        model_name = self.config.primary_embedding_model if not use_backup else self.config.backup_embedding_model

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, model_name)
            cached_embedding = self._get_from_cache(cache_key)
            if cached_embedding is not None:
                cached_results[i] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 处理未缓存的文本
        if uncached_texts:
            try:
                # 根据模型类型选择调用方式
                if not use_backup:
                    # 使用 qwen2.5-vl-embedding (MultimodalEmbedding API)
                    embeddings = await self._call_multimodal_embedding(uncached_texts)
                else:
                    # 使用 text-embedding-v4 (TextEmbedding API)
                    embeddings = await self._call_text_embedding(uncached_texts)

                # 向量质量验证（如果启用）
                if self.vector_validator and self.config.enable_validation:
                    valid_embeddings, invalid_indices = self._validate_and_filter(
                        embeddings, uncached_texts, model_name
                    )

                    # 记录统计
                    if invalid_indices:
                        logger.warning(
                            f"Filtered out {len(invalid_indices)} invalid embeddings "
                            f"out of {len(embeddings)} using {model_name}"
                        )

                    # 检查是否所有向量都无效
                    if len(valid_embeddings) == 0 and self.config.raise_on_all_failed:
                        raise RuntimeError(
                            f"All embeddings failed validation for {len(uncached_texts)} texts. "
                            f"Model: {model_name}"
                        )

                    # 使用验证后的向量
                    embeddings = valid_embeddings

                # 处理结果 - 只处理有效向量
                for i, embedding in enumerate(embeddings):
                    # 归一化
                    embedding = self._normalize_embedding(embedding)

                    original_index = uncached_indices[i]
                    cached_results[original_index] = embedding

                    # 保存到缓存
                    cache_key = self._get_cache_key(uncached_texts[i], model_name)
                    self._save_to_cache(cache_key, embedding)

                # 如果使用主模型失败，尝试备用模型
                if not use_backup and len(cached_results) != len(texts):
                    self.primary_model_failures += 1
                    if self.primary_model_failures >= self.max_primary_failures:
                        logger.warning(f"Primary model failed {self.primary_model_failures} times, switching to backup")
                        return await self.encode(texts, use_backup=True)
                else:
                    self.primary_model_failures = 0

            except Exception as e:
                logger.error(f"Error encoding texts with {model_name}: {e}")
                if not use_backup:
                    logger.info("Falling back to backup embedding model")
                    return await self.encode(texts, use_backup=True)
                raise

        # 构建最终结果 - 不再使用零向量作为默认值
        results = []
        for i in range(len(texts)):
            if i in cached_results:
                results.append(cached_results[i])
            else:
                # 如果没有缓存且跳过无效向量，则跳过
                if self.config.skip_invalid_embeddings:
                    logger.warning(f"No valid embedding for text at index {i}, skipping")
                    # 可以选择抛出异常或添加占位符
                    if self.config.raise_on_all_failed and len(results) == 0:
                        raise RuntimeError(f"Failed to generate any valid embeddings")
                else:
                    # 兼容旧逻辑：返回零向量（不推荐）
                    results.append(np.zeros(get_dimension(model_name), dtype=np.float32))

        logger.info(f"Encoded {len(results)} texts using {model_name} (requested: {len(texts)})")
        logger.info(f"Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}")

        return results

    def _validate_and_filter(
        self,
        embeddings: List[np.ndarray],
        texts: List[str],
        model_name: str
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        验证并过滤向量

        Args:
            embeddings: 嵌入向量列表
            texts: 对应的文本列表
            model_name: 模型名称

        Returns:
            (有效向量列表, 无效索引列表)
        """
        if not self.vector_validator:
            return embeddings, []

        valid_embeddings = []
        invalid_indices = []

        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            result = self.vector_validator.validate_embedding(embedding, i)

            if result.is_valid:
                valid_embeddings.append(embedding)
            else:
                invalid_indices.append(i)
                text_preview = text[:50] if text else "N/A"
                logger.warning(
                    f"Invalid embedding at index {i} (text: '{text_preview}...'): {result.reason}"
                )

                # 如果不跳过无效向量，则记录但不过滤
                if not self.config.skip_invalid_embeddings:
                    valid_embeddings.append(embedding)

        return valid_embeddings, invalid_indices

    async def encode_single(self, text: str, use_backup: bool = False) -> np.ndarray:
        """编码单个文本"""
        embeddings = await self.encode([text], use_backup=use_backup)
        return embeddings[0] if embeddings else np.zeros(1024, dtype=np.float32)

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            embeddings = await self.encode([text1, text2])
            if len(embeddings) >= 2:
                # 计算余弦相似度
                similarity = np.dot(embeddings[0], embeddings[1])
                return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")

        return 0.0

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """重排序文档"""
        if not documents:
            return []

        try:
            results = await self._call_qwen_rerank(query, documents, top_k)
            return results

        except Exception as e:
            logger.error(f"Error reranking documents: {e}")

        # 如果重排序失败，返回原始顺序
        return [(i, 1.0) for i in range(len(documents))]

    async def batch_rerank(self, queries: List[str], documents_list: List[List[str]]) -> List[List[Tuple[int, float]]]:
        """批量重排序"""
        results = []

        for query, documents in zip(queries, documents_list):
            try:
                rerank_result = await self.rerank(query, documents)
                results.append(rerank_result)
            except Exception as e:
                logger.error(f"Error in batch reranking for query '{query[:50]}...': {e}")
                # 返回原始顺序
                results.append([(i, 1.0) for i in range(len(documents))])

        return results

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        # Qwen嵌入模型的维度通常是1024或1536，这里返回1024
        return 1024

    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "service": "QwenEmbeddingService",
            "primary_model": self.config.primary_embedding_model,
            "backup_model": self.config.backup_embedding_model,
            "reranker_model": self.config.reranker_model,
            "embedding_dimension": self.get_embedding_dimension(),
            "cache_enabled": self.config.enable_cache,
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "primary_model_failures": self.primary_model_failures
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试主模型
            test_embedding = await self.encode_single("test", use_backup=False)
            primary_status = "healthy" if np.any(test_embedding) else "failed"

            # 测试备用模型
            test_embedding_backup = await self.encode_single("test", use_backup=True)

            backup_status = "healthy" if np.any(test_embedding_backup) else "failed"

            return {
                "status": "healthy" if primary_status == "healthy" or backup_status == "healthy" else "unhealthy",
                "primary_model": primary_status,
                "backup_model": backup_status,
                "embedding_dimension": self.get_embedding_dimension(),
                "cache_size": len(self.embedding_cache),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "primary_model_failures": self.primary_model_failures
            }
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入 - 向后兼容的别名方法

        Args:
            texts: 待编码的文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = await self.encode(texts)
        # 将numpy数组转换为列表
        return [emb.tolist() for emb in embeddings]


# 全局服务实例
qwen_embedding_service = QwenEmbeddingService()


async def get_qwen_embedding_service() -> QwenEmbeddingService:
    """获取Qwen嵌入服务实例"""
    return qwen_embedding_service


# 兼容性别名 - 用于向后兼容
RerankService = QwenEmbeddingService
EmbeddingService = QwenEmbeddingService