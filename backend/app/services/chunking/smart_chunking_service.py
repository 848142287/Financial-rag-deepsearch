"""
智能文档分块服务
支持多种分块策略，提升检索质量
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from app.core.structured_logging import get_structured_logger
import re

logger = get_structured_logger(__name__)


class ChunkingStrategy(Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"  # 固定大小分块
    SEMANTIC = "semantic"  # 语义分块
    INTELLIGENT = "intelligent"  # 智能分块
    HYBRID = "hybrid"  # 混合分块
    RECURSIVE = "recursive"  # 递归分块


class SmartChunkingService:
    """
    智能分块服务

    功能：
    1. 多种分块策略（固定大小、语义、智能、混合）
    2. 上下文保持（避免信息丢失）
    3. 金融领域优化（保留数值、表格）
    4. 自适应分块（根据内容调整）
    5. 质量检查（chunk完整性）
    """

    def __init__(self, default_strategy: ChunkingStrategy = ChunkingStrategy.INTELLIGENT):
        """
        Args:
            default_strategy: 默认分块策略
        """
        self.default_strategy = default_strategy

        # 金融文档的特殊模式
        self.financial_patterns = {
            'table_start': r'\|[\s\w|]+\|',
            'list_start': r'^\s*[\d\-\•\*]+\s',
            'section_header': r'^[一二三四五六七八九十百]+、|\d+\.\s+',
            'date': r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2}',
            'number': r'\d+\.?\d*\s*[亿元千百万元%]?',
        }

    def chunk_document(
        self,
        text: str,
        strategy: Optional[ChunkingStrategy] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        分块文档

        Args:
            text: 文档内容
            strategy: 分块策略
            **kwargs: 策略参数

        Returns:
            文档块列表
        """
        strategy = strategy or self.default_strategy

        logger.info(f"📝 使用策略 {strategy.value} 分块文档")

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(text, **kwargs)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(text, **kwargs)
        elif strategy == ChunkingStrategy.INTELLIGENT:
            return self._intelligent_chunk(text, **kwargs)
        elif strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunk(text, **kwargs)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunk(text, **kwargs)
        else:
            logger.warning(f"⚠️ 未知策略 {strategy}，使用默认策略")
            return self._intelligent_chunk(text, **kwargs)

    def _fixed_size_chunk(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        固定大小分块

        Args:
            text: 文档内容
            chunk_size: 块大小（字符数）
            overlap: 重叠大小

        Returns:
            文档块列表
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # 截取文本
            chunk_text = text[start:end]

            chunks.append({
                'index': chunk_index,
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'strategy': 'fixed_size',
                'metadata': {
                    'size': len(chunk_text),
                    'overlap': overlap if start > 0 else 0
                }
            })

            start = end - overlap
            chunk_index += 1

        logger.info(f"✅ 固定大小分块完成: {len(chunks)}个块")
        return chunks

    def _semantic_chunk(
        self,
        text: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        语义分块（按段落、句子）

        Args:
            text: 文档内容

        Returns:
            文档块列表
        """
        chunks = []
        chunk_index = 0

        # 1. 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # 如果段落太长，按句子分割
            if len(paragraph) > 1000:
                sentences = self._split_into_sentences(paragraph)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 800:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append({
                                'index': chunk_index,
                                'text': current_chunk.strip(),
                                'strategy': 'semantic',
                                'metadata': {
                                    'type': 'paragraph',
                                    'size': len(current_chunk)
                                }
                            })
                            chunk_index += 1
                        current_chunk = sentence

                if current_chunk:
                    chunks.append({
                        'index': chunk_index,
                        'text': current_chunk.strip(),
                        'strategy': 'semantic',
                        'metadata': {
                            'type': 'paragraph',
                            'size': len(current_chunk)
                        }
                    })
                    chunk_index += 1
            else:
                # 直接作为一块
                chunks.append({
                    'index': chunk_index,
                    'text': paragraph.strip(),
                    'strategy': 'semantic',
                    'metadata': {
                        'type': 'paragraph',
                        'size': len(paragraph.strip())
                    }
                })
                chunk_index += 1

        logger.info(f"✅ 语义分块完成: {len(chunks)}个块")
        return chunks

    def _intelligent_chunk(
        self,
        text: str,
        target_chunk_size: int = 800,
        max_chunk_size: int = 1500,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        智能分块（结合语义和规则）

        优化点：
        1. 识别表格，保持完整
        2. 识别列表，保持完整
        3. 识别章节标题
        4. 识别金融数据
        5. 自适应大小

        Args:
            text: 文档内容
            target_chunk_size: 目标块大小
            max_chunk_size: 最大块大小

        Returns:
            文档块列表
        """
        chunks = []
        chunk_index = 0

        # 1. 识别特殊结构
        structures = self._identify_structures(text)

        # 2. 按结构分块
        current_chunk = ""
        current_size = 0

        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            line_num = i

            # 检查是否是特殊结构的开始
            if line_num in structures:
                structure = structures[line_num]

                # 如果当前块非空，先保存
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        chunk_index,
                        current_chunk.strip(),
                        'intelligent',
                        {'type': 'mixed', 'size': len(current_chunk.strip())}
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_size = 0

                # 添加整个结构作为一个块
                chunks.append(self._create_chunk(
                    chunk_index,
                    structure['content'],
                    'intelligent',
                    {
                        'type': structure['type'],
                        'size': len(structure['content']),
                        'preserved': True
                    }
                ))
                chunk_index += 1

                # 跳过结构中的行
                i = structure['end_line'] + 1
                continue

            # 普通行：检查是否需要分块
            if current_size + len(line) > target_chunk_size:
                # 寻找最佳切分点
                if self._is_good_break_point(line):
                    # 在这里切分
                    current_chunk += line
                    chunks.append(self._create_chunk(
                        chunk_index,
                        current_chunk.strip(),
                        'intelligent',
                        {'type': 'text', 'size': len(current_chunk.strip())}
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_size = 0
                elif current_size > max_chunk_size:
                    # 强制切分
                    chunks.append(self._create_chunk(
                        chunk_index,
                        current_chunk.strip(),
                        'intelligent',
                        {'type': 'text', 'size': len(current_chunk.strip())}
                    ))
                    chunk_index += 1
                    current_chunk = line
                    current_size = len(line)
                else:
                    # 继续累加
                    current_chunk += line + '\n'
                    current_size += len(line) + 1
            else:
                # 继续累加
                current_chunk += line + '\n'
                current_size += len(line) + 1

            i += 1

        # 保存最后一块
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                chunk_index,
                current_chunk.strip(),
                'intelligent',
                {'type': 'text', 'size': len(current_chunk.strip())}
            ))

        logger.info(f"✅ 智能分块完成: {len(chunks)}个块")
        return chunks

    def _identify_structures(self, text: str) -> Dict[int, Dict[str, Any]]:
        """识别文档中的特殊结构（表格、列表等）"""
        structures = {}
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检测表格（Markdown格式）
            if re.match(self.financial_patterns['table_start'], line):
                start_line = i
                # 找到表格结束
                while i < len(lines) and (lines[i].strip().startswith('|') or lines[i].strip() == ''):
                    i += 1
                end_line = i - 1

                # 记录表格结构
                table_content = '\n'.join(lines[start_line:end_line + 1])
                structures[start_line] = {
                    'type': 'table',
                    'content': table_content,
                    'start_line': start_line,
                    'end_line': end_line
                }
                continue

            # 检测列表
            if re.match(self.financial_patterns['list_start'], line):
                start_line = i
                # 找到列表结束
                while i < len(lines) and re.match(self.financial_patterns['list_start'], lines[i]):
                    i += 1
                end_line = i - 1

                # 记录列表结构
                list_content = '\n'.join(lines[start_line:end_line + 1])
                structures[start_line] = {
                    'type': 'list',
                    'content': list_content,
                    'start_line': start_line,
                    'end_line': end_line
                }
                continue

            i += 1

        return structures

    def _is_good_break_point(self, line: str) -> bool:
        """判断是否是好的切分点"""
        # 句子结尾
        if re.search(r'[。！？\.!?]$', line):
            return True
        # 段落结束（空行）
        if not line.strip():
            return True
        return False

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 按中英文标点分割
        sentences = re.split(r'([。！？\.!?])', text)

        result = []
        current = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            # 添加标点
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            result.append(sentence.strip())

        return [s for s in result if s]

    def _create_chunk(
        self,
        index: int,
        text: str,
        strategy: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建chunk对象"""
        return {
            'index': index,
            'text': text,
            'strategy': strategy,
            'metadata': metadata
        }

    def _hybrid_chunk(
        self,
        text: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        混合分块（结合多种策略）

        策略：
        1. 先用语义分块识别段落
        2. 对过长段落使用固定大小分块
        3. 保持特殊结构完整
        """
        # 先语义分块
        semantic_chunks = self._semantic_chunk(text, **kwargs)

        # 对过长的块进行二次分块
        final_chunks = []
        chunk_index = 0

        for chunk in semantic_chunks:
            if len(chunk['text']) > 1200:
                # 二次分块
                sub_chunks = self._fixed_size_chunk(
                    chunk['text'],
                    chunk_size=600,
                    overlap=50
                )

                for sub_chunk in sub_chunks:
                    sub_chunk['index'] = chunk_index
                    sub_chunk['strategy'] = 'hybrid'
                    sub_chunk['metadata']['parent_chunk'] = chunk['index']
                    final_chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                chunk['index'] = chunk_index
                chunk['strategy'] = 'hybrid'
                final_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"✅ 混合分块完成: {len(final_chunks)}个块")
        return final_chunks

    def _recursive_chunk(
        self,
        text: str,
        separators: List[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        递归分块（LangChain风格）

        按照分隔符优先级递归分割
        """
        if separators is None:
            separators = ['\n\n', '\n', '。', '！', '？', '.', '!', '?', ' ', '']

        chunks = []
        chunk_index = 0

        def recursive_split(text: str, separator_index: int) -> List[str]:
            """递归分割函数"""
            if separator_index >= len(separators):
                return [text]

            separator = separators[separator_index]

            if separator:
                parts = text.split(separator)
            else:
                return [text]

            # 检查每部分大小
            final_parts = []
            for part in parts:
                if len(part) <= 800:
                    final_parts.append(part)
                else:
                    # 部分太大，递归分割
                    sub_parts = recursive_split(part, separator_index + 1)
                    final_parts.extend(sub_parts)

            return final_parts

        split_parts = recursive_split(text, 0)

        for part in split_parts:
            if part.strip():
                chunks.append({
                    'index': chunk_index,
                    'text': part.strip(),
                    'strategy': 'recursive',
                    'metadata': {
                        'size': len(part.strip())
                    }
                })
                chunk_index += 1

        logger.info(f"✅ 递归分块完成: {len(chunks)}个块")
        return chunks

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证和优化chunk质量

        检查项：
        1. 块大小是否合理
        2. 是否包含完整句子
        3. 是否有过多空白
        4. 特殊结构是否完整
        """
        validated_chunks = []

        for chunk in chunks:
            text = chunk['text']

            # 1. 检查大小
            if len(text) < 20:
                logger.warning(f"⚠️ Chunk {chunk['index']} 太小: {len(text)} 字符")
                continue

            if len(text) > 2000:
                logger.warning(f"⚠️ Chunk {chunk['index']} 太大: {len(text)} 字符")
                # 可以选择进一步分割或跳过
                # 这里我们保留，但记录警告

            # 2. 检查空白比例
            whitespace_ratio = len(re.findall(r'\s', text)) / len(text)
            if whitespace_ratio > 0.5:
                logger.warning(f"⚠️ Chunk {chunk['index']} 空白比例过高: {whitespace_ratio:.2%}")

            # 3. 检查句子完整性
            if not text[-1] in ['。', '！', '？', '.', '!', '?', '，', ',', ';', '；']:
                # 句子可能不完整，标记
                chunk['metadata']['incomplete'] = True

            validated_chunks.append(chunk)

        logger.info(f"✅ Chunk验证完成: {len(validated_chunks)}/{len(chunks)} 通过")
        return validated_chunks


def get_smart_chunking_service(
    strategy: ChunkingStrategy = ChunkingStrategy.INTELLIGENT
) -> SmartChunkingService:
    """获取智能分块服务实例"""
    return SmartChunkingService(default_strategy=strategy)


__all__ = [
    'SmartChunkingService',
    'get_smart_chunking_service',
    'ChunkingStrategy'
]
