"""
模型管理器
统一管理所有视觉和文本模型
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
from enum import Enum

from .base_model import BaseVisionTextModel, ModelType, TaskType, ModelInput, ModelOutput
from .vision_model import VisionModel
from .text_model import TextModel
from .multimodal_model import MultimodalModel

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models: Dict[str, BaseVisionTextModel] = {}
        self.model_status: Dict[str, Dict[str, Any]] = {}

        # 支持的模型类型
        self.supported_model_types = {
            ModelType.VISION: VisionModel,
            ModelType.TEXT: TextModel,
            ModelType.MULTIMODAL: MultimodalModel
        }

    async def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """注册模型"""
        try:
            if model_name in self.models:
                logger.warning(f"模型 {model_name} 已存在，将被替换")
                await self.unregister_model(model_name)

            # 创建模型实例
            model_class = self.supported_model_types.get(model_type)
            if not model_class:
                logger.error(f"不支持的模型类型: {model_type}")
                return False

            config = model_config or self.config.get(model_type.value, {})
            model = model_class(config)

            # 注册模型
            self.models[model_name] = model
            self.model_status[model_name] = {
                'registered': True,
                'loaded': False,
                'type': model_type.value,
                'config': config
            }

            logger.info(f"模型 {model_name} 注册成功")
            return True

        except Exception as e:
            logger.error(f"注册模型 {model_name} 失败: {str(e)}")
            return False

    async def unregister_model(self, model_name: str) -> bool:
        """注销模型"""
        try:
            if model_name in self.models:
                # 卸载模型
                await self.unload_model(model_name)

                # 删除模型
                del self.models[model_name]
                del self.model_status[model_name]

                logger.info(f"模型 {model_name} 注销成功")
                return True

            logger.warning(f"模型 {model_name} 不存在")
            return False

        except Exception as e:
            logger.error(f"注销模型 {model_name} 失败: {str(e)}")
            return False

    async def load_model(self, model_name: str) -> bool:
        """加载模型"""
        try:
            if model_name not in self.models:
                logger.error(f"模型 {model_name} 不存在")
                return False

            model = self.models[model_name]
            if model.is_loaded:
                logger.info(f"模型 {model_name} 已加载")
                return True

            # 加载模型
            success = await model.load_model()
            if success:
                self.model_status[model_name]['loaded'] = True
                self.model_status[model_name]['load_time'] = asyncio.get_event_loop().time()
                logger.info(f"模型 {model_name} 加载成功")
            else:
                logger.error(f"模型 {model_name} 加载失败")

            return success

        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {str(e)}")
            return False

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        try:
            if model_name not in self.models:
                logger.warning(f"模型 {model_name} 不存在")
                return False

            model = self.models[model_name]
            if not model.is_loaded:
                logger.info(f"模型 {model_name} 未加载")
                return True

            # 卸载模型
            success = await model.unload_model()
            if success:
                self.model_status[model_name]['loaded'] = False
                logger.info(f"模型 {model_name} 卸载成功")
            else:
                logger.error(f"模型 {model_name} 卸载失败")

            return success

        except Exception as e:
            logger.error(f"卸载模型 {model_name} 失败: {str(e)}")
            return False

    async def load_all_models(self) -> Dict[str, bool]:
        """加载所有模型"""
        results = {}
        tasks = []

        for model_name in self.models:
            task = self.load_model(model_name)
            tasks.append((model_name, task))

        for model_name, task in tasks:
            results[model_name] = await task

        return results

    async def unload_all_models(self) -> Dict[str, bool]:
        """卸载所有模型"""
        results = {}
        tasks = []

        for model_name in self.models:
            task = self.unload_model(model_name)
            tasks.append((model_name, task))

        for model_name, task in tasks:
            results[model_name] = await task

        return results

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return list(self.models.keys())

    def get_loaded_models(self) -> List[str]:
        """获取已加载模型列表"""
        return [
            name for name, model in self.models.items()
            if model.is_loaded
        ]

    def get_models_by_type(self, model_type: ModelType) -> List[str]:
        """根据类型获取模型"""
        return [
            name for name, model in self.models.items()
            if model.model_type == model_type
        ]

    def get_models_by_task(self, task_type: TaskType) -> List[str]:
        """根据任务获取模型"""
        return [
            name for name, model in self.models.items()
            if task_type in model.get_supported_tasks()
        ]

    async def process_with_model(
        self,
        model_name: str,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """使用指定模型处理"""
        try:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")

            model = self.models[model_name]

            # 检查模型是否支持该任务
            if task_type not in model.get_supported_tasks():
                raise ValueError(f"模型 {model_name} 不支持任务 {task_type}")

            # 处理输入
            return await model.process(inputs, task_type, **kwargs)

        except Exception as e:
            logger.error(f"使用模型 {model_name} 处理失败: {str(e)}")
            return ModelOutput(
                results={},
                error_message=str(e)
            )

    async def process_auto(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        preferred_model: Optional[str] = None,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """自动选择模型处理"""
        # 获取支持该任务的模型
        available_models = self.get_models_by_task(task_type)

        if not available_models:
            raise ValueError(f"没有模型支持任务 {task_type}")

        # 优先使用指定的模型
        if preferred_model and preferred_model in available_models:
            model_name = preferred_model
        else:
            # 选择第一个可用的模型
            model_name = available_models[0]

        logger.info(f"自动选择模型 {model_name} 处理任务 {task_type}")

        return await self.process_with_model(model_name, inputs, task_type, **kwargs)

    async def batch_process(
        self,
        model_name: str,
        inputs: List[ModelInput],
        task_type: TaskType,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """批量处理"""
        try:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 不存在")

            model = self.models[model_name]

            # 检查模型是否支持该任务
            if task_type not in model.get_supported_tasks():
                raise ValueError(f"模型 {model_name} 不支持任务 {task_type}")

            # 批量处理
            return await model.batch_process(inputs, task_type, batch_size, **kwargs)

        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            # 返回错误结果
            return [ModelOutput(
                results={},
                error_message=str(e)
            ) for _ in inputs]

    async def compare_models(
        self,
        inputs: Union[ModelInput, List[ModelInput]],
        task_type: TaskType,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Union[ModelOutput, List[ModelOutput]]]:
        """比较多个模型的结果"""
        # 获取要比较的模型
        if model_names:
            models_to_compare = [
                name for name in model_names
                if name in self.get_models_by_task(task_type)
            ]
        else:
            models_to_compare = self.get_models_by_task(task_type)

        if not models_to_compare:
            raise ValueError(f"没有模型支持任务 {task_type}")

        logger.info(f"比较模型: {models_to_compare}")

        # 并行处理
        results = {}
        tasks = []

        for model_name in models_to_compare:
            task = self.process_with_model(model_name, inputs, task_type)
            tasks.append((model_name, task))

        for model_name, task in tasks:
            try:
                result = await task
                results[model_name] = result
            except Exception as e:
                logger.error(f"模型 {model_name} 处理失败: {str(e)}")
                results[model_name] = ModelOutput(
                    results={},
                    error_message=str(e)
                )

        return results

    def get_model_info(self, model_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """获取模型信息"""
        if model_name:
            if model_name not in self.models:
                return {}

            model = self.models[model_name]
            info = model.get_model_info()
            info.update(self.model_status.get(model_name, {}))
            return info
        else:
            # 返回所有模型信息
            all_info = {}
            for name in self.models:
                all_info[name] = self.get_model_info(name)
            return all_info

    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'total_models': len(self.models),
            'loaded_models': len(self.get_loaded_models()),
            'models_by_type': {
                model_type.value: self.get_models_by_type(model_type)
                for model_type in ModelType
            },
            'supported_tasks': list(TaskType),
            'model_status': self.model_status.copy()
        }

    async def warm_up_models(self, model_names: Optional[List[str]] = None):
        """预热模型"""
        if model_names:
            models_to_warm = [
                name for name in model_names
                if name in self.models
            ]
        else:
            models_to_warm = list(self.models.keys())

        logger.info(f"预热模型: {models_to_warm}")

        # 并行预热
        tasks = []
        for model_name in models_to_warm:
            model = self.models[model_name]
            if not model.is_loaded:
                # 先加载模型
                task = self.load_model(model_name)
                tasks.append(task)

        # 等待加载完成
        await asyncio.gather(*tasks, return_exceptions=True)

        # 预热模型
        for model_name in models_to_warm:
            model = self.models[model_name]
            if model.is_loaded:
                try:
                    await model.warm_up()
                    logger.info(f"模型 {model_name} 预热完成")
                except Exception as e:
                    logger.error(f"模型 {model_name} 预热失败: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'manager_status': 'healthy',
            'models': {}
        }

        # 检查每个模型
        for model_name, model in self.models.items():
            try:
                # 简单测试
                test_input = self._create_test_input(model.model_type)
                if test_input:
                    test_task = model.get_supported_tasks()[0] if model.get_supported_tasks() else None
                    if test_task:
                        # 快速测试
                        result = await model.process(test_input, test_task)
                        health_status['models'][model_name] = {
                            'status': 'healthy' if not result.error_message else 'unhealthy',
                            'loaded': model.is_loaded,
                            'last_check': asyncio.get_event_loop().time()
                        }
                    else:
                        health_status['models'][model_name] = {
                            'status': 'no_supported_tasks',
                            'loaded': model.is_loaded
                        }
                else:
                    health_status['models'][model_name] = {
                        'status': 'no_test_input',
                        'loaded': model.is_loaded
                    }

            except Exception as e:
                health_status['models'][model_name] = {
                    'status': 'error',
                    'error': str(e),
                    'loaded': model.is_loaded
                }

        return health_status

    def _create_test_input(self, model_type: ModelType) -> Optional[ModelInput]:
        """创建测试输入"""
        if model_type == ModelType.TEXT:
            return ModelInput(
                data="这是一个测试文本",
                data_type="text"
            )
        elif model_type == ModelType.VISION:
            # 创建简单的测试图像
            import numpy as np
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return ModelInput(
                data=test_image,
                data_type="numpy"
            )
        elif model_type == ModelType.MULTIMODAL:
            import numpy as np
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return ModelInput(
                data={'image': test_image, 'text': '测试'},
                data_type="multimodal"
            )

        return None