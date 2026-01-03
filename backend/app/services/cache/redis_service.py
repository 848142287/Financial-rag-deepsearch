"""
Redis缓存服务 - 用于存储文档解析中间结果
"""
import json
import hashlib
from typing import Dict, List, Any, Optional

from app.core.structured_logging import get_structured_logger
from app.core.config import settings

logger = get_structured_logger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not available")

class DocumentRedisCache:
    """
    文档解析Redis缓存服务

    功能：
    - 存储DeepSeek处理后的完整解析结果
    - 提供快速查询接口
    - 支持TTL自动过期（默认30天）
    - 提供批量操作

    注意：
    - 默认TTL为30天（2592000秒），可根据需要调整
    - 数据在Redis中持久化存储，便于Milvus和Neo4j随时读取
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Redis缓存服务

        Args:
            config: 配置参数
                - host: Redis主机地址
                - port: Redis端口
                - db: 数据库编号
                - password: 密码
                - default_ttl: 默认过期时间（秒）
                - prefix: 键前缀
        """
        self.config = config or {}
        self.host = self.config.get('host', getattr(settings, 'REDIS_HOST', 'localhost'))
        self.port = self.config.get('port', getattr(settings, 'REDIS_PORT', 6379))
        self.db = self.config.get('db', getattr(settings, 'REDIS_DB', 0))
        self.password = self.config.get('password', getattr(settings, 'REDIS_PASSWORD', None))
        self.default_ttl = self.config.get('default_ttl', 2592000)  # 默认30天（30*24*3600秒）
        self.prefix = self.config.get('prefix', 'doc_parse:')

        self._client = None
        self._connect()

    def _connect(self):
        """连接Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis不可用，缓存功能将被禁用")
            return

        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # 测试连接
            self._client.ping()
            logger.info(f"✅ Redis连接成功: {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"❌ Redis连接失败: {e}")
            self._client = None

    def is_available(self) -> bool:
        """检查Redis是否可用"""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False

    def _generate_key(self, file_path: str, suffix: str = '') -> str:
        """生成Redis键"""
        # 使用文件路径的MD5作为唯一标识
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
        key = f"{self.prefix}{file_hash}"
        if suffix:
            key += f":{suffix}"
        return key

    async def store_document_parse(
        self,
        file_path: str,
        parse_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        存储文档解析结果

        Args:
            file_path: 文件路径
            parse_data: 解析数据（包含DeepSeek处理后的完整结果）
            ttl: 过期时间（秒），None表示使用默认值

        Returns:
            是否存储成功
        """
        if not self.is_available():
            logger.warning("Redis不可用，跳过缓存存储")
            return False

        try:
            key = self._generate_key(file_path)

            # 准备存储的数据
            cache_data = {
                'file_path': file_path,
                'stored_at': datetime.utcnow().isoformat(),
                'parse_data': parse_data
            }

            # 序列化
            json_data = json.dumps(cache_data, ensure_ascii=False)

            # 存储到Redis
            ttl = ttl or self.default_ttl
            self._client.setex(key, ttl, json_data)

            logger.info(f"✅ 文档解析结果已存储到Redis: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"❌ 存储到Redis失败: {e}")
            return False

    async def get_document_parse(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取文档解析结果

        Args:
            file_path: 文件路径

        Returns:
            解析数据，如果不存在返回None
        """
        if not self.is_available():
            return None

        try:
            key = self._generate_key(file_path)
            json_data = self._client.get(key)

            if not json_data:
                logger.debug(f"Redis中未找到: {key}")
                return None

            cache_data = json.loads(json_data)
            logger.info(f"✅ 从Redis获取文档解析结果: {key}")
            return cache_data.get('parse_data')

        except Exception as e:
            logger.error(f"❌ 从Redis获取失败: {e}")
            return None

    async def store_chunks(
        self,
        file_path: str,
        chunks: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        存储文档分块（用于Milvus向量化）

        Args:
            file_path: 文件路径
            chunks: 分块列表
            ttl: 过期时间

        Returns:
            是否存储成功
        """
        if not self.is_available():
            return False

        try:
            key = self._generate_key(file_path, 'chunks')

            cache_data = {
                'file_path': file_path,
                'stored_at': datetime.utcnow().isoformat(),
                'chunks_count': len(chunks),
                'chunks': chunks
            }

            json_data = json.dumps(cache_data, ensure_ascii=False)
            ttl = ttl or self.default_ttl

            self._client.setex(key, ttl, json_data)

            logger.info(f"✅ 文档分块已存储到Redis: {len(chunks)} 个分块")
            return True

        except Exception as e:
            logger.error(f"❌ 存储分块到Redis失败: {e}")
            return False

    async def get_chunks(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取文档分块

        Args:
            file_path: 文件路径

        Returns:
            分块列表
        """
        if not self.is_available():
            return None

        try:
            key = self._generate_key(file_path, 'chunks')
            json_data = self._client.get(key)

            if not json_data:
                return None

            cache_data = json.loads(json_data)
            return cache_data.get('chunks')

        except Exception as e:
            logger.error(f"❌ 从Redis获取分块失败: {e}")
            return None

    async def store_markdown(
        self,
        file_path: str,
        markdown: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        存储Markdown内容

        Args:
            file_path: 文件路径
            markdown: Markdown内容
            metadata: 元数据
            ttl: 过期时间

        Returns:
            是否存储成功
        """
        if not self.is_available():
            return False

        try:
            key = self._generate_key(file_path, 'markdown')

            cache_data = {
                'file_path': file_path,
                'stored_at': datetime.utcnow().isoformat(),
                'markdown': markdown,
                'metadata': metadata
            }

            json_data = json.dumps(cache_data, ensure_ascii=False)
            ttl = ttl or self.default_ttl

            self._client.setex(key, ttl, json_data)

            logger.info(f"✅ Markdown已存储到Redis (长度: {len(markdown)})")
            return True

        except Exception as e:
            logger.error(f"❌ 存储Markdown到Redis失败: {e}")
            return False

    async def get_markdown(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取Markdown内容

        Args:
            file_path: 文件路径

        Returns:
            包含markdown和metadata的字典
        """
        if not self.is_available():
            return None

        try:
            key = self._generate_key(file_path, 'markdown')
            json_data = self._client.get(key)

            if not json_data:
                return None

            cache_data = json.loads(json_data)
            return {
                'markdown': cache_data.get('markdown'),
                'metadata': cache_data.get('metadata')
            }

        except Exception as e:
            logger.error(f"❌ 从Redis获取Markdown失败: {e}")
            return None

    async def store_multimodal_analysis(
        self,
        file_path: str,
        analysis_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        存储多模态分析结果

        Args:
            file_path: 文件路径
            analysis_data: 多模态分析数据
            ttl: 过期时间

        Returns:
            是否存储成功
        """
        if not self.is_available():
            return False

        try:
            key = self._generate_key(file_path, 'multimodal')

            cache_data = {
                'file_path': file_path,
                'stored_at': datetime.utcnow().isoformat(),
                'analysis': analysis_data
            }

            json_data = json.dumps(cache_data, ensure_ascii=False)
            ttl = ttl or self.default_ttl

            self._client.setex(key, ttl, json_data)

            logger.info(f"✅ 多模态分析结果已存储到Redis")
            return True

        except Exception as e:
            logger.error(f"❌ 存储多模态分析到Redis失败: {e}")
            return False

    async def get_multimodal_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取多模态分析结果

        Args:
            file_path: 文件路径

        Returns:
            多模态分析数据
        """
        if not self.is_available():
            return None

        try:
            key = self._generate_key(file_path, 'multimodal')
            json_data = self._client.get(key)

            if not json_data:
                return None

            cache_data = json.loads(json_data)
            return cache_data.get('analysis')

        except Exception as e:
            logger.error(f"❌ 从Redis获取多模态分析失败: {e}")
            return None

    async def delete_document(self, file_path: str) -> bool:
        """
        删除文档的所有缓存数据

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        if not self.is_available():
            return False

        try:
            # 删除所有相关键
            patterns = ['', 'chunks', 'markdown', 'multimodal']
            keys = [self._generate_key(file_path, suffix) for suffix in patterns]

            for key in keys:
                self._client.delete(key)

            logger.info(f"✅ 已删除文档的Redis缓存: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ 删除Redis缓存失败: {e}")
            return False

    async def exists(self, file_path: str) -> bool:
        """
        检查文档是否已缓存

        Args:
            file_path: 文件路径

        Returns:
            是否存在
        """
        if not self.is_available():
            return False

        try:
            key = self._generate_key(file_path)
            return bool(self._client.exists(key))
        except:
            return False

    def close(self):
        """关闭Redis连接"""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis连接已关闭")
            except:
                pass

# 全局单例
_redis_cache: Optional[DocumentRedisCache] = None

def get_redis_cache(config: Optional[Dict[str, Any]] = None) -> DocumentRedisCache:
    """获取Redis缓存服务单例"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = DocumentRedisCache(config)
    return _redis_cache

__all__ = [
    'DocumentRedisCache',
    'get_redis_cache'
]
