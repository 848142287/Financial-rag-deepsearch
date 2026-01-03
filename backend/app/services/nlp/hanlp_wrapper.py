"""
HanLP分词器包装器
提供与jieba兼容的接口，便于从jieba迁移到hanlp
"""

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# 全局HanLP实例
_hanlp_instance = None

class HanLPWrapper:
    """HanLP包装器 - 提供jieba兼容接口"""
    
    def __init__(self):
        self._tokenizer = None
        self._tagger = None
        self._custom_words = set()
        self._initialize()
    
    def _initialize(self):
        """初始化HanLP"""
        try:
            import hanlp
            # 加载中文分词器 (CTB6_CONVSEG)
            self._tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            # 加载词性标注器
            self._tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN)
            logger.info("✅ HanLP初始化成功")
        except Exception as e:
            logger.error(f"❌ HanLP初始化失败: {e}")
            raise
    
    def initialize(self):
        """初始化方法（jieba兼容）"""
        # HanLP在构造函数中已初始化
        pass
    
    def lcut(self, text: str, cut_all: bool = False) -> List[str]:
        """
        分词并返回列表（jieba兼容）
        
        Args:
            text: 待分词文本
            cut_all: 是否使用全模式（HanLP不支持，忽略）
        
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        try:
            if self._tokenizer:
                result = self._tokenizer(text)
                # HanLP可能返回列表或嵌套列表，展平处理
                if isinstance(result, list):
                    if all(isinstance(item, list) for item in result):
                        # 嵌套列表，展平
                        return [token for sublist in result for token in sublist]
                    return result
                return list(result)
            else:
                # 降级为简单分词
                return list(text)
        except Exception as e:
            logger.error(f"HanLP分词失败: {e}")
            # 降级为按字符分词
            return list(text)
    
    def cut(self, text: str, cut_all: bool = False):
        """
        分词生成器（jieba兼容）
        
        Args:
            text: 待分词文本
            cut_all: 是否使用全模式
        
        Yields:
            分词结果
        """
        for word in self.lcut(text, cut_all):
            yield word
    
    def add_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None):
        """
        添加自定义词（jieba兼容）
        
        Args:
            word: 词语
            freq: 词频（HanLP中通过用户词典实现）
            tag: 词性标签
        """
        self._custom_words.add(word)
        # HanLP通过配置文件添加自定义词，这里记录到集合中
        # 实际使用时可以通过加载用户词典实现
        logger.debug(f"添加自定义词: {word}, freq={freq}, tag={tag}")
    
    def suggest_freq(self, segment: str, tune: bool = False) -> float:
        """
        建议词频（jieba兼容，简化实现）
        
        Args:
            segment: 词语
            tune: 是否调整
        
        Returns:
            词频值
        """
        # 简化实现，返回固定值
        return 100.0
    
    def del_word(self, word: str):
        """
        删除自定义词（jieba兼容）
        
        Args:
            word: 词语
        """
        if word in self._custom_words:
            self._custom_words.remove(word)
    
    def load_userdict(self, file_path: str):
        """
        加载用户词典（jieba兼容）
        
        Args:
            file_path: 词典文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            word = parts[0]
                            self._custom_words.add(word)
            logger.info(f"加载用户词典: {file_path}")
        except Exception as e:
            logger.error(f"加载用户词典失败: {e}")
    
    def posseg(self):
        """
        词性标注模块（jieba兼容）
        
        Returns:
            词性标注器
        """
        return HanLPPosseg(self._tagger, self._tokenizer)
    
    def analyse(self):
        """
        关键词提取模块（jieba兼容）
        
        Returns:
            关键词提取器
        """
        return HanLPAnalyse(self)

class HanLPPosseg:
    """HanLP词性标注器（jieba.posseg兼容）"""
    
    def __init__(self, tagger, tokenizer):
        self._tagger = tagger
        self._tokenizer = tokenizer
    
    def cut(self, text: str):
        """
        分词并词性标注
        
        Args:
            text: 待分词文本
        
        Yields:
            pair对象，包含word和flag
        """
        try:
            # 先分词
            if self._tokenizer:
                tokens = self._tokenizer(text)
                if isinstance(tokens, list) and all(isinstance(item, list) for item in tokens):
                    tokens = [token for sublist in tokens for token in sublist]
            else:
                tokens = list(text)
            
            # 词性标注
            if self._tagger:
                pos_result = self._tagger(tokens)
                # 返回pair对象
                for i, token in enumerate(tokens):
                    pos_tag = pos_result[i] if i < len(pos_result) else 'n'
                    yield Pair(token, pos_tag)
            else:
                for token in tokens:
                    yield Pair(token, 'n')
                    
        except Exception as e:
            logger.error(f"词性标注失败: {e}")
            for token in tokens:
                yield Pair(token, 'n')

class Pair:
    """词对（jieba.posseg.pair兼容）"""
    
    def __init__(self, word: str, flag: str):
        self.word = word
        self.flag = flag
    
    def __str__(self):
        return f"{self.word}/{self.flag}"
    
    def __repr__(self):
        return f"Pair(word='{self.word}', flag='{self.flag}')"

class HanLPAnalyse:
    """HanLP关键词提取器（jieba.analyse兼容）"""
    
    def __init__(self, wrapper: HanLPWrapper):
        self._wrapper = wrapper
    
    def extract_tags(self, text: str, topK: int = 20, withWeight: bool = False, allowPOS: Tuple = ()):
        """
        提取关键词（jieba.analyse.extract_tags兼容）
        
        Args:
            text: 待分析文本
            topK: 返回前K个关键词
            withWeight: 是否返回权重
            allowPOS: 允许的词性
        
        Returns:
            关键词列表
        """
        try:
            # 使用TF-IDF或TextRank提取关键词
            # 这里简化实现：使用词频统计
            words = self._wrapper.lcut(text)
            
            # 过滤停用词和单字
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            
            # 词频统计
            freq = {}
            for word in words:
                if len(word) > 1 and word not in stopwords:
                    freq[word] = freq.get(word, 0) + 1
            
            # 排序
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            
            # 返回结果
            if withWeight:
                return sorted_words[:topK]
            else:
                return [word for word, _ in sorted_words[:topK]]
                
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []

# 全局实例
_hanlp_wrapper = None

def get_hanlp_wrapper() -> HanLPWrapper:
    """获取HanLP包装器单例"""
    global _hanlp_wrapper
    if _hanlp_wrapper is None:
        _hanlp_wrapper = HanLPWrapper()
    return _hanlp_wrapper

# 导出jieba兼容的接口
initialize = lambda: get_hanlp_wrapper().initialize()
lcut = lambda text, cut_all=False: get_hanlp_wrapper().lcut(text, cut_all)
cut = lambda text, cut_all=False: get_hanlp_wrapper().cut(text, cut_all)
add_word = lambda word, freq=None, tag=None: get_hanlp_wrapper().add_word(word, freq, tag)
del_word = lambda word: get_hanlp_wrapper().del_word(word)
load_userdict = lambda file_path: get_hanlp_wrapper().load_userdict(file_path)

# 词性标注模块
posseg = type('posseg', (), {
    'cut': lambda text: get_hanlp_wrapper().posseg().cut(text),
    'lcut': lambda text: list(get_hanlp_wrapper().posseg().cut(text))
})

# 关键词提取模块
analyse = type('analyse', (), {
    'extract_tags': lambda text, topK=20, withWeight=False, allowPOS=(): get_hanlp_wrapper().analyse().extract_tags(text, topK, withWeight, allowPOS)
})
