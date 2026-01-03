"""
统一文本分块工具
整合所有分块策略到单一模块
"""
from enum import Enum
import re

class ChunkingStrategy(Enum):
    """分块策略枚举"""
    FIXED = "fixed"         # 固定大小分块
    SEMANTIC = "semantic"   # 语义分块
    TITLE = "title"        # 基于标题分块
    RECURSIVE = "recursive" # 递归分块

class TextChunker:
    """统一的文本分块工具"""

    @staticmethod
    def chunk(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    ) -> List[str]:
        """
        统一的分块接口

        Args:
            text: 待分块的文本
            chunk_size: 块大小（字符数）
            overlap: 重叠大小
            strategy: 分块策略

        Returns:
            List[str]: 分块结果列表
        """
        if not text or not text.strip():
            return []

        if strategy == ChunkingStrategy.SEMANTIC:
            return TextChunker._semantic_chunk(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.TITLE:
            return TextChunker._title_based_chunk(text)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return TextChunker._recursive_chunk(text, chunk_size)
        else:
            return TextChunker._fixed_chunk(text, chunk_size, overlap)

    @staticmethod
    def _fixed_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        固定大小分块

        从unified_document_pipeline.py抽取并优化
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # 寻找合适的断点（句号、问号、感叹号等）
            if end < text_length:
                for delimiter in ['.', '。', '?', '？', '!', '！', '\n']:
                    delimiter_pos = text.rfind(delimiter, start, end)
                    if delimiter_pos > start:
                        end = delimiter_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 移动起始位置，考虑重叠
            start = max(start + 1, end - overlap)

        return chunks

    @staticmethod
    def _semantic_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        语义分块

        基于句子边界进行分块，保持语义完整性
        """
        # 分割句子
        sentence_endings = r'(?<=[.。.!?！?])\s+'
        sentences = re.split(sentence_endings, text)

        chunks = []
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # 如果单个句子超过chunk_size，强制分割
            if sentence_size > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0

                # 长句子按字符分割
                while sentence_size > chunk_size:
                    chunks.append(sentence[:chunk_size])
                    sentence = sentence[chunk_size:]
                    sentence_size = len(sentence)

                current_chunk = sentence
                current_size = sentence_size
            else:
                # 如果加上这个句子会超过chunk_size
                if current_size + sentence_size > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # 考虑重叠
                    words = current_chunk.split()
                    if len(words) > overlap // 2:
                        overlap_text = " ".join(words[-(overlap // 2):])
                        current_chunk = overlap_text + " " + sentence
                        current_size = len(overlap_text) + sentence_size
                    else:
                        current_chunk = sentence
                        current_size = sentence_size
                else:
                    current_chunk += (" " if current_chunk else "") + sentence
                    current_size += sentence_size + (1 if current_chunk else 0)

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def _title_based_chunk(text: str) -> List[str]:
        """
        基于标题的分块

        按照标题（如#、##等）分割文本
        """
        # 匹配Markdown标题或数字标题
        title_pattern = r'(\n|^)[#]+\s+[^\n]+|(\n|^)\d+\.\s+[^\n]+'

        # 找到所有标题位置
        matches = list(re.finditer(title_pattern, text))

        if not matches:
            return [text]

        chunks = []
        start = 0

        for i, match in enumerate(matches):
            # 跳过第一个标题（通常是文档标题）
            if i == 0:
                start = match.end()
                continue

            chunk = text[start:match.start()].strip()
            if chunk:
                chunks.append(chunk)

            start = match.end()

        # 添加最后一个块
        if start < len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def _recursive_chunk(text: str, chunk_size: int) -> List[str]:
        """
        递归分块

        递归地将文本分割成更小的块
        """
        if len(text) <= chunk_size:
            return [text]

        # 尝试在中间分割
        mid = len(text) // 2

        # 寻找最近的断点
        for i in range(mid, max(0, mid - 100), -1):
            if text[i] in '.。!！?？\n':
                left = text[:i+1]
                right = text[i+1:]
                return TextChunker._recursive_chunk(left, chunk_size) + \
                       TextChunker._recursive_chunk(right, chunk_size)

        # 如果找不到断点，强制分割
        left = text[:mid]
        right = text[mid:]
        return TextChunker._recursive_chunk(left, chunk_size) + \
               TextChunker._recursive_chunk(right, chunk_size)

    @staticmethod
    def get_chunk_metadata(
        text: str,
        chunks: List[str]
    ) -> List[dict]:
        """
        获取分块元数据

        Args:
            text: 原始文本
            chunks: 分块结果

        Returns:
            List[dict]: 每个块的元数据
        """
        metadata = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            # 找到块在原文中的位置
            chunk_start = text.find(chunk, current_pos)
            chunk_end = chunk_start + len(chunk)

            metadata.append({
                'chunk_id': i,
                'start_char': chunk_start,
                'end_char': chunk_end,
                'char_count': len(chunk),
                'word_count': len(chunk.split()),
                'sentence_count': len(re.split(r'[.。.!?！?]', chunk))
            })

            current_pos = chunk_end

        return metadata
