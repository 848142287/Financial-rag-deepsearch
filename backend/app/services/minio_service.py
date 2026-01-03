"""
MinIO对象存储服务
"""

from app.core.structured_logging import get_structured_logger
from typing import List, Optional, Dict, Any
from minio import Minio
from minio.error import S3Error
import io
from datetime import datetime, timedelta

from app.core.config import settings

logger = get_structured_logger(__name__)


class MinIOService:
    """MinIO服务类"""

    def __init__(self):
        """初始化MinIO客户端"""
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_name = settings.minio_bucket_name

    async def init_buckets(self):
        """初始化存储桶"""
        try:
            # 检查存储桶是否存在
            if not self.client.bucket_exists(self.bucket_name):
                # 创建存储桶
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists")

            # 创建其他必要的存储桶
            additional_buckets = [
                "processed-files",
                "embeddings",
                "thumbnails",
                "temp"
            ]

            for bucket in additional_buckets:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")

        except S3Error as e:
            logger.error(f"Failed to initialize MinIO buckets: {e}")
            raise

    async def upload_file(self, file_path: str, file_data: bytes,
                         content_type: str = "application/octet-stream",
                         bucket: Optional[str] = None) -> bool:
        """上传文件"""
        try:
            bucket = bucket or self.bucket_name
            file_obj = io.BytesIO(file_data)

            # 添加元数据
            metadata = {
                "uploaded_at": datetime.utcnow().isoformat(),
                "content_type": content_type
            }

            result = self.client.put_object(
                bucket_name=bucket,
                object_name=file_path,
                data=file_obj,
                length=len(file_data),
                content_type=content_type,
                metadata=metadata
            )

            logger.info(f"File uploaded successfully: {file_path}")
            return True

        except S3Error as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            return False

    async def download_file(self, file_path: str,
                           bucket: Optional[str] = None) -> Optional[bytes]:
        """下载文件"""
        try:
            bucket = bucket or self.bucket_name
            response = self.client.get_object(bucket, file_path)
            file_data = response.read()
            response.close()
            response.release_conn()

            return file_data

        except S3Error as e:
            logger.error(f"Failed to download file {file_path}: {e}")
            return None

    async def delete_file(self, file_path: str,
                         bucket: Optional[str] = None) -> bool:
        """删除文件"""
        try:
            bucket = bucket or self.bucket_name
            self.client.remove_object(bucket, file_path)
            logger.info(f"File deleted successfully: {file_path}")
            return True

        except S3Error as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    async def list_files(self, prefix: str = "",
                        bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出文件"""
        try:
            bucket = bucket or self.bucket_name
            objects = self.client.list_objects(bucket, prefix=prefix)

            files = []
            for obj in objects:
                file_info = {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag,
                    "content_type": obj.content_type if hasattr(obj, 'content_type') else None
                }
                files.append(file_info)

            return files

        except S3Error as e:
            logger.error(f"Failed to list files: {e}")
            return []

    async def get_file_url(self, file_path: str,
                          expires_in: int = 3600,
                          bucket: Optional[str] = None) -> Optional[str]:
        """获取文件的预签名URL"""
        try:
            bucket = bucket or self.bucket_name
            url = self.client.presigned_get_object(
                bucket_name=bucket,
                object_name=file_path,
                expires=timedelta(seconds=expires_in)
            )
            return url

        except S3Error as e:
            logger.error(f"Failed to get file URL for {file_path}: {e}")
            return None

    async def file_exists(self, file_path: str,
                         bucket: Optional[str] = None) -> bool:
        """检查文件是否存在"""
        try:
            bucket = bucket or self.bucket_name
            self.client.stat_object(bucket, file_path)
            return True

        except S3Error:
            return False

    async def copy_file(self, source_path: str, destination_path: str,
                       source_bucket: Optional[str] = None,
                       dest_bucket: Optional[str] = None) -> bool:
        """复制文件"""
        try:
            source_bucket = source_bucket or self.bucket_name
            dest_bucket = dest_bucket or self.bucket_name

            # 获取源文件信息
            source_info = self.client.stat_object(source_bucket, source_path)

            # 复制文件
            result = self.client.copy_object(
                bucket_name=dest_bucket,
                object_name=destination_path,
                source=f"{source_bucket}/{source_path}"
            )

            logger.info(f"File copied from {source_path} to {destination_path}")
            return True

        except S3Error as e:
            logger.error(f"Failed to copy file from {source_path} to {destination_path}: {e}")
            return False

    async def get_bucket_usage(self, bucket: Optional[str] = None) -> Dict[str, Any]:
        """获取存储桶使用情况"""
        try:
            bucket = bucket or self.bucket_name
            objects = self.client.list_objects(bucket)

            total_files = 0
            total_size = 0

            for obj in objects:
                total_files += 1
                total_size += obj.size

            return {
                "bucket": bucket,
                "total_files": total_files,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }

        except S3Error as e:
            logger.error(f"Failed to get bucket usage: {e}")
            return {
                "bucket": bucket or self.bucket_name,
                "total_files": 0,
                "total_size": 0,
                "total_size_mb": 0
            }

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 测试连接
            self.client.list_buckets()
            return True
        except S3Error as e:
            logger.error(f"MinIO健康检查失败: {e}")
            return False

    async def upload_document_content(
        self,
        document_id: int,
        content_type: str,
        file_data: bytes,
        filename: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """上传文档相关内容"""
        try:
            # 构建对象名称
            object_name = f"documents/{document_id}/{content_type}/{filename}"

            # 设置元数据
            if not metadata:
                metadata = {}

            metadata.update({
                "document_id": str(document_id),
                "content_type": content_type,
                "upload_time": datetime.now().isoformat(),
                "filename": filename
            })

            # 上传文件
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                metadata=metadata
            )

            logger.info(f"Successfully uploaded {content_type} for document {document_id}")
            return object_name

        except S3Error as e:
            logger.error(f"Failed to upload {content_type} for document {document_id}: {e}")
            return None

    async def generate_presigned_url(
        self,
        object_name: str,
        expires_in: int = 3600,
        method: str = "GET"
    ) -> Optional[str]:
        """生成预签名URL"""
        try:
            from datetime import timedelta

            if method.upper() == "GET":
                url = self.client.presigned_get_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    expires=timedelta(seconds=expires_in)
                )
            elif method.upper() == "PUT":
                url = self.client.presigned_put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    expires=timedelta(seconds=expires_in)
                )
            else:
                logger.error(f"Unsupported method: {method}")
                return None

            return url

        except S3Error as e:
            logger.error(f"Failed to generate presigned URL for {object_name}: {e}")
            return None

    async def upload_temp_image(
        self,
        image_data: bytes,
        filename: str,
        expires_in_hours: int = 24
    ) -> Optional[str]:
        """
        上传临时图片并返回可公开访问的 URL

        Args:
            image_data: 图片字节数据
            filename: 文件名
            expires_in_hours: 过期时间（小时）

        Returns:
            可公开访问的图片 URL，失败返回 None
        """
        try:
            import uuid
            from datetime import timedelta

            # 使用 temp bucket
            bucket = "temp"

            # 确保 bucket 存在
            if not self.client.bucket_exists(bucket):
                try:
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
                except S3Error as e:
                    logger.error(f"Failed to create bucket {bucket}: {e}")
                    return None

            # 生成唯一的对象名称
            ext = filename.split('.')[-1] if '.' in filename else 'png'
            unique_id = str(uuid.uuid4())
            object_name = f"qwen_analysis/{datetime.now().strftime('%Y%m%d')}/{unique_id}.{ext}"

            # 上传图片
            file_obj = io.BytesIO(image_data)
            content_type = f"image/{ext}"

            result = self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_obj,
                length=len(image_data),
                content_type=content_type,
                metadata={
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(hours=expires_in_hours)).isoformat(),
                    "purpose": "qwen_multimodal_analysis"
                }
            )

            logger.info(f"Temp image uploaded: {object_name}")

            # 生成预签名 URL（24小时有效）
            url = self.client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=timedelta(hours=expires_in_hours)
            )

            # 检查 URL 是否为公网可访问地址
            # 如果包含 localhost、127.0.0.1 或内网 IP，则不可公网访问
            public_url = self._ensure_public_url(url)

            if public_url != url:
                logger.warning(f"MinIO URL is not publicly accessible, using public URL: {public_url}")

            return public_url

        except S3Error as e:
            logger.error(f"Failed to upload temp image: {e}")
            return None

    def _ensure_public_url(self, url: str) -> str:
        """
        确保URL是公网可访问的

        如果是内网地址，尝试替换为公网地址
        如果无法替换，返回原 URL（调用方需要处理）
        """
        import os

        # 从环境变量获取公网地址
        public_host = os.getenv("MINIO_PUBLIC_HOST", os.getenv("MINIO_ENDPOINT", "localhost:9000"))

        # 替换为公网地址
        if "localhost" in url or "127.0.0.1" in url or "minio:" in url:
            # 提取路径部分
            if "?" in url:
                path, params = url.split("?", 1)
                path = path[path.index("/", 8):]  # 跳过协议和主机部分
                return f"http://{public_host}{path}?{params}"
            else:
                path = url[url.index("/", 8):]
                return f"http://{public_host}{path}"

        return url

    async def cleanup_temp_images(
        self,
        prefix: str = "qwen_analysis",
        older_than_hours: int = 24
    ) -> int:
        """
        清理过期的临时图片

        Args:
            prefix: 对象名前缀
            older_than_hours: 清理多少小时前的文件

        Returns:
            删除的文件数量
        """
        try:
            from datetime import timedelta

            bucket = "temp"
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

            # 列出所有对象
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)

            deleted_count = 0
            for obj in objects:
                # 检查是否过期
                if obj.last_modified and obj.last_modified < cutoff_time:
                    self.client.remove_object(bucket, obj.object_name)
                    deleted_count += 1
                    logger.info(f"Deleted expired temp image: {obj.object_name}")

            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count

        except S3Error as e:
            logger.error(f"Failed to cleanup temp images: {e}")
            return 0
# 兼容性别名
MinioService = MinIOService

__all__ = ['MinIOService', 'MinioService']
