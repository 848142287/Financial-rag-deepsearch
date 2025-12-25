"""
MinIO对象存储服务
"""

import logging
from typing import List, Optional, Dict, Any
from minio import Minio
from minio.error import S3Error
import io
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)


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