#!/bin/bash

# Financial RAG Docker 部署脚本
# 用于一键部署所有服务，包含数据持久化配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数: 打印信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数: 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 函数: 检查Docker和Docker Compose
check_prerequisites() {
    print_info "检查系统环境..."

    check_command docker
    check_command docker-compose

    # 检查Docker是否运行
    if ! docker info &> /dev/null; then
        print_error "Docker 未运行，请先启动 Docker"
        exit 1
    fi

    print_success "系统环境检查通过"
}

# 函数: 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."

    mkdir -p storage/{parsed_docs,cache,uploads}
    mkdir -p logs
    mkdir -p temp
    mkdir -p config

    print_success "目录创建完成"
}

# 函数: 检查数据卷
check_volumes() {
    print_info "检查数据卷..."

    # 列出所有相关数据卷
    VOLUMES=(
        "financial-rag-mysql-data"
        "financial-rag-minio-data"
        "financial-rag-etcd-data"
        "financial-rag-minio-milvus-data"
        "financial-rag-milvus-data"
        "financial-rag-neo4j-data"
        "financial-rag-neo4j-logs"
        "financial-rag-neo4j-plugins"
        "financial-rag-neo4j-import"
        "financial-rag-redis-data"
    )

    EXISTING_VOLUMES=()
    for volume in "${VOLUMES[@]}"; do
        if docker volume ls -q | grep -q "^${volume}$"; then
            EXISTING_VOLUMES+=("$volume")
        fi
    done

    if [ ${#EXISTING_VOLUMES[@]} -gt 0 ]; then
        print_warning "发现已存在的数据卷:"
        for volume in "${EXISTING_VOLUMES[@]}"; do
            echo "  - $volume"
        done
        echo ""
        read -p "是否继续部署？数据将被保留 (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "部署已取消"
            exit 0
        fi
    fi
}

# 函数: 拉取最新镜像
pull_images() {
    print_info "拉取Docker镜像..."

    docker-compose pull

    print_success "镜像拉取完成"
}

# 函数: 构建应用镜像
build_app() {
    print_info "构建应用镜像..."

    docker-compose build backend

    print_success "应用镜像构建完成"
}

# 函数: 启动服务
start_services() {
    print_info "启动所有服务..."

    docker-compose up -d

    print_success "服务启动完成"
}

# 函数: 等待服务就绪
wait_for_services() {
    print_info "等待服务就绪..."

    # 等待MySQL
    print_info "等待 MySQL 就绪..."
    for i in {1..60}; do
        if docker-compose exec -T mysql mysqladmin ping -h localhost -uroot -proot123456 &> /dev/null; then
            print_success "MySQL 已就绪"
            break
        fi
        sleep 2
    done

    # 等待Milvus
    print_info "等待 Milvus 就绪..."
    for i in {1..60}; do
        if docker-compose exec -T milvus-standalone curl -f http://localhost:9091/healthz &> /dev/null; then
            print_success "Milvus 已就绪"
            break
        fi
        sleep 2
    done

    # 等待Neo4j
    print_info "等待 Neo4j 就绪..."
    for i in {1..60}; do
        if docker-compose exec -T neo4j cypher-shell -u neo4j -p neo4j123456 "RETURN 1" &> /dev/null; then
            print_success "Neo4j 已就绪"
            break
        fi
        sleep 2
    done

    # 等待Backend
    print_info "等待 Backend 就绪..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/api/v1/health-check &> /dev/null; then
            print_success "Backend 已就绪"
            break
        fi
        sleep 2
    done

    print_success "所有服务已就绪"
}

# 函数: 显示服务状态
show_status() {
    print_info "服务状态:"
    echo ""
    docker-compose ps
    echo ""

    print_info "数据卷:"
    echo ""
    docker volume ls | grep financial-rag
    echo ""
}

# 函数: 显示访问信息
show_access_info() {
    echo ""
    echo "==================================================================="
    echo -e "${GREEN}服务访问信息${NC}"
    echo "==================================================================="
    echo ""
    echo "🌐 Backend API:"
    echo "   URL:  http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    echo ""
    echo "🗄️  MySQL:"
    echo "   Host: localhost:3306"
    echo "   User: root"
    echo "   Pass: root123456"
    echo "   DB:   financial_rag"
    echo ""
    echo "🪣 MinIO Console:"
    echo "   URL:  http://localhost:9001"
    echo "   User: minioadmin"
    echo "   Pass: minioadmin"
    echo ""
    echo "🔍 Milvus:"
    echo "   Port: 19530"
    echo "   UI:   http://localhost:9001 (attu)"
    echo ""
    echo "🕸️  Neo4j:"
    echo "   Browser: http://localhost:7474"
    echo "   User:    neo4j"
    echo "   Pass:    neo4j123456"
    echo "   Bolt:    bolt://localhost:7687"
    echo ""
    echo "📦 Redis:"
    echo "   Port: 6379"
    echo ""
    echo "==================================================================="
    echo ""
}

# 主函数
main() {
    echo ""
    echo "==================================================================="
    echo "           Financial RAG Docker 部署脚本"
    echo "==================================================================="
    echo ""

    # 检查环境
    check_prerequisites

    # 创建目录
    create_directories

    # 检查数据卷
    check_volumes

    # 拉取镜像
    pull_images

    # 构建应用
    build_app

    # 启动服务
    start_services

    # 等待服务就绪
    wait_for_services

    # 显示状态
    show_status

    # 显示访问信息
    show_access_info

    print_success "部署完成！"
    echo ""
}

# 运行主函数
main "$@"
