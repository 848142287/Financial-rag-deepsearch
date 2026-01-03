"""
Mineru引擎集成（优化版）
提供基于MinerU HTTP API的高质量文档解析能力
参考Multimodal_RAG_OCR项目最佳实践
"""

import os
from app.core.structured_logging import get_structured_logger
import json
import tempfile
from pathlib import Path

import requests

logger = get_structured_logger(__name__)

class MineruEngine:
    """MinerU解析引擎 - HTTP API版本"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化MinerU引擎

        Args:
            config: 配置字典，包含：
                - api_url: MinerU API地址
                - vllm_url: VLLM服务器地址（可选）
                - backend: 解析后端 (pipeline/vlm)
                - timeout: 请求超时时间
        """
        self.engine_name = "mineru"
        self.config = config or {}

        # API配置
        self.api_url = self.config.get('mineru_api_url', 'http://localhost:8001/file_parse')
        self.vllm_url = self.config.get('vllm_server_url', '')
        self.backend = self.config.get('mineru_backend', 'pipeline')
        self.timeout = int(self.config.get('mineru_timeout', 600))

        # 解析选项
        self.extract_images = self.config.get('extract_images', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        self.language = self.config.get('language', 'zh')

        # 返回选项
        self.return_md = self.config.get('return_md', True)
        self.return_middle_json = self.config.get('return_middle_json', True)
        self.return_model_output = self.config.get('return_model_output', True)
        self.return_content_list = self.config.get('return_content_list', True)

        # 临时目录（用于图片保存）
        self.temp_dir = tempfile.mkdtemp(prefix="mineru_")

        logger.info(f"MinerU引擎初始化完成 - API: {self.api_url}")

    async def parse_document(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        使用MinerU HTTP API解析文档

        Args:
            file_path: 文件路径
            output_dir: 输出目录

        Returns:
            解析结果字典
        """
        try:
            logger.info(f"开始MinerU解析: {file_path}")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 准备输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 调用MinerU HTTP API
            result = await self._call_mineru_api(file_path)

            # 处理解析结果
            processed_result = await self._process_mineru_result(result, output_dir)

            logger.info(
                f"MinerU解析完成: "
                f"{len(processed_result.get('content_blocks', []))}个内容块, "
                f"{len(processed_result.get('images', []))}个图片, "
                f"{len(processed_result.get('tables', []))}个表格"
            )
            return processed_result

        except Exception as e:
            logger.error(f"MinerU解析失败: {str(e)}")
            raise

    async def _call_mineru_api(self, file_path: str) -> Dict[str, Any]:
        """
        调用MinerU HTTP API

        Args:
            file_path: 文件路径

        Returns:
            API响应结果
        """
        try:
            # 准备文件上传
            with open(file_path, 'rb') as f:
                files = [('files', (Path(file_path).name, f, 'application/pdf'))]

                # 构建请求数据（对齐Multimodal_RAG_OCR的参数）
                data = {
                    'backend': self.backend,
                    'parse_method': 'auto',
                    'lang_list': self._map_language(self.language),
                    'return_md': 'true' if self.return_md else 'false',
                    'return_middle_json': 'true' if self.return_middle_json else 'false',
                    'return_model_output': 'true' if self.return_model_output else 'false',
                    'return_content_list': 'true' if self.return_content_list else 'false',
                    'start_page_id': '0',
                    'end_page_id': '99999',
                }

                # 添加VLLM URL（如果配置了）
                if self.vllm_url:
                    data['server_url'] = self.vllm_url

                logger.info(f"调用MinerU API: {self.api_url}")
                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

            # 检查响应状态
            if response.status_code != 200:
                raise RuntimeError(
                    f"MinerU API返回错误: {response.status_code}, {response.text}"
                )

            # 解析JSON响应
            if response.headers.get("content-type", "").startswith("application/json"):
                file_json = response.json()
            else:
                file_json = json.loads(response.text)

            return file_json

        except requests.exceptions.Timeout:
            raise RuntimeError(f"MinerU API请求超时 (timeout: {self.timeout}s)")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"MinerU API连接失败。请确保MinerU服务正在运行: {self.api_url}"
            )
        except Exception as e:
            logger.error(f"MinerU API调用失败: {str(e)}")
            raise

    async def _process_mineru_result(
        self,
        api_result: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        处理MinerU API返回结果

        Args:
            api_result: API返回的原始结果
            output_dir: 输出目录

        Returns:
            处理后的结果字典
        """
        try:
            processed_result = {
                "engine": self.engine_name,
                "backend": api_result.get("backend", self.backend),
                "version": api_result.get("version", "unknown"),
                "total_pages": 0,
                "content_blocks": [],
                "images": [],
                "tables": [],
                "formulas": [],
                "metadata": {}
            }

            # 提取结果数据
            results = api_result.get("results", {})
            if not results:
                logger.warning("MinerU返回空结果")
                return processed_result

            # 获取第一个结果
            file_key = list(results.keys())[0]
            res = results[file_key]

            # 提取markdown内容
            md_content = res.get("md_content", "")

            # 解析JSON字段
            middle_json = self._safe_json_loads(res.get("middle_json"))
            model_output = self._safe_json_loads(res.get("model_output"))
            content_list = self._safe_json_loads(res.get("content_list"))

            # 统计总页数
            if middle_json and "pdf_info" in middle_json:
                processed_result["total_pages"] = len(middle_json["pdf_info"])

            # 处理content_list（结构化内容）
            if content_list and isinstance(content_list, list):
                for idx, item in enumerate(content_list):
                    if not isinstance(item, dict):
                        continue

                    content_block = {
                        "id": f"{item.get('type', 'unknown')}_{idx}",
                        "content_type": item.get("type", "text"),
                        "content": item.get("text", ""),
                        "bbox": item.get("bbox", []),
                        "page_number": item.get("page_idx", 0) + 1,
                        "confidence": 1.0,
                        "source_engine": self.engine_name
                    }

                    # 处理特殊类型
                    if item.get("type") == "table":
                        table_data = self._extract_table_data(item)
                        processed_result["tables"].append(table_data)
                        content_block.update(table_data)

                    elif item.get("type") == "image":
                        image_data = await self._extract_image_data(item, output_dir, idx)
                        if image_data:
                            processed_result["images"].append(image_data)
                            content_block.update(image_data)

                    elif item.get("type") in ("isolate_formula", "inline_formula"):
                        formula_data = self._extract_formula_data(item)
                        processed_result["formulas"].append(formula_data)
                        content_block.update(formula_data)

                    processed_result["content_blocks"].append(content_block)

            # 添加元数据
            processed_result["metadata"] = {
                "engine": self.engine_name,
                "backend": processed_result["backend"],
                "version": processed_result["version"],
                "total_pages": processed_result["total_pages"],
                "total_images": len(processed_result["images"]),
                "total_tables": len(processed_result["tables"]),
                "total_formulas": len(processed_result["formulas"]),
                "md_content": md_content,
                "raw_data": {
                    "middle_json": middle_json,
                    "model_output": model_output,
                    "content_list": content_list,
                },
                "config": {
                    "extract_images": self.extract_images,
                    "extract_tables": self.extract_tables,
                    "ocr_enabled": self.ocr_enabled,
                    "language": self.language,
                }
            }

            return processed_result

        except Exception as e:
            logger.error(f"MinerU结果处理失败: {str(e)}")
            raise

    def _extract_table_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """提取表格数据"""
        table_data = {
            "id": f"table_{item.get('page_idx', 0)}",
            "content_type": "table",
            "html": item.get("table_body", ""),
            "text": "",
        }

        # 从HTML提取文本
        if table_data["html"]:
            import re
            text_content = re.sub(r'<[^>]+>', ' ', table_data["html"])
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            table_data["text"] = text_content

        return table_data

    async def _extract_image_data(
        self,
        item: Dict[str, Any],
        output_dir: str,
        idx: int
    ) -> Optional[Dict[str, Any]]:
        """提取图片数据"""
        try:
            # MinerU HTTP API返回的图片路径
            img_path = item.get("img_path", "")

            image_data = {
                "id": f"image_{item.get('page_idx', 0)}_{idx}",
                "content_type": "image",
                "file_path": img_path,
                "description": "",
            }

            # 提取图片标题
            captions = item.get("image_caption", [])
            if captions and isinstance(captions, list):
                image_data["description"] = " ".join(str(c) for c in captions)

            return image_data

        except Exception as e:
            logger.warning(f"提取图片数据失败: {str(e)}")
            return None

    def _extract_formula_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """提取公式数据"""
        text = item.get("text", "")

        return {
            "id": f"formula_{item.get('page_idx', 0)}",
            "content_type": "formula",
            "latex": text,
            "text": text
        }

    def _safe_json_loads(self, text: Any) -> Optional[Dict]:
        """安全地解析JSON字符串"""
        if not isinstance(text, str):
            return text
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def _map_language(self, lang: str) -> str:
        """映射语言代码到MinerU格式"""
        lang_map = {
            'zh': 'ch',
            'chinese': 'ch',
            'en': 'en',
            'english': 'en',
            'ja': 'ja',
            'japanese': 'ja',
            'ko': 'ko',
            'korean': 'ko',
            'auto': 'auto'
        }
        return lang_map.get(lang.lower(), 'ch')

    def is_available(self) -> bool:
        """检查MinerU API是否可用"""
        try:
            response = requests.get(
                self.api_url.replace('/file_parse', '/health')
                if '/file_parse' in self.api_url
                else f"{self.api_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            # 尝试通过API调用判断
            try:
                response = requests.post(
                    self.api_url,
                    files=[],
                    data={},
                    timeout=5
                )
                # 即使返回错误，只要能连接就说明服务在运行
                return True
            except Exception:
                return False

    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("MinerU引擎临时文件已清理")
        except Exception as e:
            logger.error(f"MinerU引擎清理失败: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.cleanup()
