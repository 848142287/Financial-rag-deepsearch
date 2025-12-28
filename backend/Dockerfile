# Financial RAG Deepsearch Unified Dockerfile
# 基于Python 3.11的轻量级镜像，适用于backend和celery-worker

FROM python:3.11-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 配置Debian阿里云镜像源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装基础系统依赖 (添加重试机制)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    wget \
    unzip \
    gcc \
    g++ \
    make \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libxml2 \
    libxslt1.1 \
    || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-missing --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    wget \
    unzip \
    gcc \
    g++ \
    make \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libxml2 \
    libxslt1.1) \
    && rm -rf /var/lib/apt/lists/*

# 升级pip并配置镜像源
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 复制requirements文件
COPY requirements.txt requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 创建必要的目录
RUN mkdir -p logs uploads temp cache/disk cache/uploads module_backups && \
    chown -R appuser:appuser /app

# 复制应用代码
COPY . .

# 再次确保目录权限正确
RUN chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health-check || exit 1

# 默认启动命令 (backend服务)
# docker-compose.yml中会为celery-worker覆盖此命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]