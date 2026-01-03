#!/bin/bash

# 修复存储目录权限
echo "Fixing storage directory permissions..."

# 确保所有需要的目录存在且有正确权限
mkdir -p /app/storage/parsed_docs
mkdir -p /app/storage/cache
mkdir -p /app/logs
mkdir -p /app/uploads
mkdir -p /app/temp
mkdir -p /app/data
mkdir -p /app/.cache/fontconfig
mkdir -p ~/.cache/fontconfig

# 设置字体缓存环境变量
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf
export XDG_CACHE_HOME=/app/.cache

# 如果是root用户，修复权限（忽略错误）
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, fixing ownership..."
    chown -R appuser:appuser /app/storage 2>/dev/null || true
    chown -R appuser:appuser /app/logs 2>/dev/null || true
    chown -R appuser:appuser /app/uploads 2>/dev/null || true
    chown -R appuser:appuser /app/temp 2>/dev/null || true
    chown -R appuser:appuser /app/.cache 2>/dev/null || true
    # 注意：/app/data 包含只读挂载的markdown目录，跳过chown
    # 只确保目录存在，不修改权限
    chmod -R 755 /app/storage 2>/dev/null || true
    chmod -R 755 /app/logs 2>/dev/null || true
    chmod -R 755 /app/uploads 2>/dev/null || true
    chmod -R 755 /app/temp 2>/dev/null || true
    chmod -R 755 /app/.cache 2>/dev/null || true
fi

# 切换到appuser执行命令
if [ "$(id -u)" = "0" ]; then
    echo "Switching to appuser..."
    exec gosu appuser "$@"
else
    echo "Already running as non-root user"
    exec "$@"
fi
