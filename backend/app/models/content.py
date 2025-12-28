"""
专用内容数据模型
包含图像、图表、表格、公式等专用内容表
"""

from sqlalchemy import Column, Integer, String, Text, BigInteger, DateTime, ForeignKey, Enum, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class ContentType(str, enum.Enum):
    IMAGE = "image"
    CHART = "chart"
    TABLE = "table"
    FORMULA = "formula"


class ImageType(str, enum.Enum):
    PHOTOGRAPH = "photograph"
    DIAGRAM = "diagram"
    ILLUSTRATION = "illustration"
    SCREENSHOT = "screenshot"
    SCAN = "scan"
    ICON = "icon"
    LOGO = "logo"


class ChartType(str, enum.Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    RADAR = "radar"
    CANDLESTICK = "candlestick"
    WATERFALL = "waterfall"


class TableType(str, enum.Enum):
    DATA_TABLE = "data_table"
    FINANCIAL_REPORT = "financial_report"
    COMPARISON_TABLE = "comparison_table"
    SCHEDULE_TABLE = "schedule_table"
    MATRIX_TABLE = "matrix_table"


class FormulaType(str, enum.Enum):
    MATHEMATICAL = "mathematical"
    FINANCIAL = "financial"
    STATISTICAL = "statistical"
    CHEMICAL = "chemical"
    PHYSICS = "physics"


class ImageContent(Base):
    """图像内容表"""
    __tablename__ = "image_contents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True, index=True)

    # 图像基本信息
    image_type = Column(Enum(ImageType), nullable=False, index=True)
    title = Column(String(1000))
    caption = Column(Text)
    alt_text = Column(String(2000))  # 替代文本

    # 图像属性
    width = Column(Integer)  # 宽度（像素）
    height = Column(Integer)  # 高度（像素）
    resolution = Column(Float)  # 分辨率（DPI）
    color_mode = Column(String(20))  # RGB, CMYK, Grayscale
    file_size = Column(BigInteger)  # 文件大小（字节）
    format = Column(String(10))  # JPEG, PNG, GIF, etc.

    # 位置信息
    page_number = Column(Integer)
    position_x = Column(Float)  # 页面中的X坐标
    position_y = Column(Float)  # 页面中的Y坐标
    bbox = Column(JSON)  # 边界框 [x1, y1, x2, y2]

    # 存储信息
    original_path = Column(String(1000))  # 原始文件路径
    processed_path = Column(String(1000))  # 处理后文件路径
    thumbnail_path = Column(String(1000))  # 缩略图路径
    storage_key = Column(String(500), index=True)  # MinIO存储键

    # 分析结果
    description = Column(Text)  # AI生成的描述
    tags = Column(JSON)  # 标签列表
    objects = Column(JSON)  # 检测到的对象
    text_content = Column(Text)  # 图像中的文本（OCR）
    confidence = Column(Float)  # 检测置信度

    # 元数据
    content_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="images")
    chapter = relationship("Chapter", backref="images")
    chunk = relationship("DocumentChunk", backref="images")

    def __repr__(self):
        return f"<ImageContent(id={self.id}, document_id={self.document_id}, image_type='{self.image_type}')>"


class ChartContent(Base):
    """图表内容表"""
    __tablename__ = "chart_contents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True, index=True)
    image_id = Column(Integer, ForeignKey("image_contents.id"), nullable=True, index=True)

    # 图表基本信息
    chart_type = Column(Enum(ChartType), nullable=False, index=True)
    title = Column(String(1000), nullable=False)
    subtitle = Column(String(1000))
    caption = Column(Text)

    # 图表数据
    data_series = Column(JSON)  # 数据系列
    axes = Column(JSON)  # 坐标轴信息
    legend = Column(JSON)  # 图例
    grid = Column(JSON)  # 网格信息

    # 位置信息
    page_number = Column(Integer)
    position = Column(JSON)  # 在页面中的位置
    size = Column(JSON)  # 图表尺寸

    # 分析结果
    insights = Column(JSON)  # AI生成的洞察
    key_trends = Column(JSON)  # 关键趋势
    anomalies = Column(JSON)  # 异常点
    interpretation = Column(Text)  # 图表解释

    # 数据提取
    extracted_data = Column(JSON)  # 提取的原始数据
    data_quality_score = Column(Float)  # 数据质量评分
    extraction_confidence = Column(Float)  # 提取置信度

    # 元数据
    content_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="charts")
    chapter = relationship("Chapter", backref="charts")
    image = relationship("ImageContent", backref="charts")

    def __repr__(self):
        return f"<ChartContent(id={self.id}, document_id={self.document_id}, chart_type='{self.chart_type}')>"


class TableContent(Base):
    """表格内容表"""
    __tablename__ = "table_contents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True, index=True)
    image_id = Column(Integer, ForeignKey("image_contents.id"), nullable=True, index=True)

    # 表格基本信息
    table_type = Column(Enum(TableType), nullable=False, index=True)
    title = Column(String(1000))
    caption = Column(Text)
    description = Column(Text)

    # 表格结构
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    has_header = Column(Boolean, default=True)
    has_footer = Column(Boolean, default=False)

    # 表格数据
    headers = Column(JSON)  # 表头列表
    rows = Column(JSON)     # 行数据
    cells = Column(JSON)    # 单元格详细信息
    structure = Column(JSON)  # 表格结构信息

    # 位置信息
    page_number = Column(Integer)
    position = Column(JSON)  # 在页面中的位置
    size = Column(JSON)  # 表格尺寸

    # 分析结果
    summary = Column(Text)  # 表格摘要
    key_findings = Column(JSON)  # 关键发现
    data_insights = Column(JSON)  # 数据洞察
    confidence = Column(Float)  # 识别置信度

    # 数据质量
    completeness_score = Column(Float)  # 完整性评分
    accuracy_score = Column(Float)  # 准确性评分
    consistency_score = Column(Float)  # 一致性评分

    # 元数据
    content_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="tables")
    chapter = relationship("Chapter", backref="tables")
    image = relationship("ImageContent", backref="tables")

    def __repr__(self):
        return f"<TableContent(id={self.id}, document_id={self.document_id}, rows={self.row_count}, cols={self.column_count})>"


class FormulaContent(Base):
    """公式内容表"""
    __tablename__ = "formula_contents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True, index=True)
    image_id = Column(Integer, ForeignKey("image_contents.id"), nullable=True, index=True)

    # 公式基本信息
    formula_type = Column(Enum(FormulaType), nullable=False, index=True)
    name = Column(String(500))  # 公式名称
    notation = Column(String(500))  # 公式符号
    description = Column(Text)  # 公式描述

    # 公式内容
    latex = Column(Text)  # LaTeX格式
    mathml = Column(Text)  # MathML格式
    ascii_math = Column(Text)  # ASCII数学格式
    plain_text = Column(Text)  # 纯文本表示

    # 公式变量
    variables = Column(JSON)  # 变量列表
    parameters = Column(JSON)  # 参数列表
    constants = Column(JSON)  # 常量列表

    # 位置信息
    page_number = Column(Integer)
    position = Column(JSON)  # 在页面中的位置
    size = Column(JSON)  # 公式尺寸

    # 分析结果
    explanation = Column(Text)  # 公式解释
    interpretation = Column(Text)  # 公式含义
    applications = Column(JSON)  # 应用场景
    related_formulas = Column(JSON)  # 相关公式

    # 计算相关
    is_computable = Column(Boolean, default=False)  # 是否可计算
    computation_result = Column(JSON)  # 计算结果（如果可计算）
    computation_method = Column(Text)  # 计算方法

    # 金融特定信息
    financial_context = Column(Text)  # 金融上下文
    financial_meaning = Column(Text)  # 金融含义
    industry_usage = Column(Text)  # 行业应用

    # 元数据
    content_metadata = Column(JSON)
    confidence = Column(Float)  # 识别置信度
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    document = relationship("Document", backref="formulas")
    chapter = relationship("Chapter", backref="formulas")
    image = relationship("ImageContent", backref="formulas")

    def __repr__(self):
        return f"<FormulaContent(id={self.id}, document_id={self.document_id}, formula_type='{self.formula_type}')>"


class ContentRelationship(Base):
    """内容关系表"""
    __tablename__ = "content_relationships"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    # 关系主体
    source_type = Column(Enum(ContentType), nullable=False)
    source_id = Column(Integer, nullable=False)
    target_type = Column(Enum(ContentType), nullable=False)
    target_id = Column(Integer, nullable=False)

    # 关系信息
    relationship_type = Column(String(50), nullable=False)  # references, contains, explains, illustrates
    relationship_description = Column(Text)
    confidence = Column(Float)

    # 上下文信息
    context = Column(Text)
    page_number = Column(Integer)
    position = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    document = relationship("Document")

    def __repr__(self):
        return f"<ContentRelationship(source={self.source_type}:{self.source_id}, target={self.target_type}:{self.target_id})>"